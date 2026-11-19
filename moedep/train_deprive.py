"""Deprivation phase: text-only continued training of Qwen3-VL-30B-A3B with LoRA on all routed experts and trainable
routers, under five balancing conditions (--cond):
  aux_on       : load-balancing auxiliary loss of the Hugging Face model with --aux_coef
  aux_off      : coefficient 0
  freeze_V10   : aux on, LoRA disabled on the vision-exclusive experts (the routers can still send text to them)
  block_V10    : aux on, router logits of the vision-exclusive experts masked during training
  bias_balance : no auxiliary loss, aux-loss-free bias balancing with update rate --bias_gamma
Every --eval_every steps: held-out text NLL, ARC-Easy accuracy, routing loads of text and image tokens on the expert sets,
vision probe accuracy (POPE, MMVP, V*, TextVQA), LoRA delta norms per expert set. Saves metrics.jsonl and the LoRA + router state."""
import argparse, json, math, random, time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch
from moedep.model import MoeVLM
from moedep import data as D
from moedep.lora_experts import install, set_enabled, lora_state, update_balance_bias
from moedep import paths

DEP = paths.DATA


def load_dialogues(split, n, seed, max_turns=6):
    files = sorted((DEP / "ultrachat_200k" / "data").glob(f"{split}-*.parquet"))
    rows = []
    for f in files:
        t = pq.read_table(f, columns=["messages"]).to_pydict()["messages"]
        rows += [m[:max_turns] for m in t]
        if len(rows) >= 4 * n:
            break
    random.Random(seed).shuffle(rows)
    return rows[:n]


def tokenize_dialogue(m, msgs, max_len):
    """input_ids + labels (-100 outside assistant content) for one dialogue, using chat-template prefix boundaries."""
    tok = m.tok; proc = m.processor
    msgs = [{"role": x["role"], "content": x["content"]} for x in msgs if x["role"] in ("user", "assistant")]
    if not msgs or msgs[0]["role"] != "user":
        return None
    full = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    ids = tok(full, add_special_tokens=False)["input_ids"]
    labels = [-100] * len(ids)
    for i, x in enumerate(msgs):
        if x["role"] != "assistant":
            continue
        pre = len(tok(proc.apply_chat_template(msgs[:i], tokenize=False, add_generation_prompt=True), add_special_tokens=False)["input_ids"])
        post = len(tok(proc.apply_chat_template(msgs[:i + 1], tokenize=False, add_generation_prompt=False), add_special_tokens=False)["input_ids"])
        for j in range(pre, min(post, len(ids))):
            labels[j] = ids[j]
    ids, labels = ids[:max_len], labels[:max_len]
    if sum(l != -100 for l in labels) < 8:
        return None
    return ids, labels


def collate(batch, pad_id):
    L = max(len(b[0]) for b in batch)
    ids = torch.full((len(batch), L), pad_id); lab = torch.full((len(batch), L), -100); att = torch.zeros((len(batch), L), dtype=torch.long)
    for k, (i, l) in enumerate(batch):
        ids[k, :len(i)] = torch.tensor(i); lab[k, :len(l)] = torch.tensor(l); att[k, :len(i)] = 1
    return {"input_ids": ids, "attention_mask": att, "labels": lab}


def arc_items(n, seed):
    f = sorted((DEP / "ai2_arc" / "ARC-Easy").glob("test-*.parquet"))[0]
    t = pq.read_table(f).to_pydict(); rng = random.Random(seed); items = []
    for i in rng.sample(range(len(t["question"])), min(n, len(t["question"]))):
        ch = t["choices"][i]; labels = ch["label"]; texts = ch["text"]
        letters = [chr(ord("A") + k) for k in range(len(texts))]
        gold = letters[labels.index(t["answerKey"][i])]
        opts = "\n".join(f"{L}. {x}" for L, x in zip(letters, texts))
        items.append(dict(prompt=f"{t['question'][i].strip()}\n{opts}\n{D.MCQ_SUFFIX}", letters=letters, gold=gold))
    return items


@torch.no_grad()
def evaluate(m, probes, masks, loras, step):
    m.model.eval(); res = {"step": step}
    # held-out text NLL over assistant tokens
    tot, n = 0.0, 0
    for ids, labels in probes["heldout"]:
        batch = collate([(ids, labels)], m.tok.pad_token_id); batch = {k: v.to(m.model.device) for k, v in batch.items()}
        out = m.model(**batch, use_cache=False); nt = int((batch["labels"] != -100).sum()); tot += float(out.loss) * nt; n += nt
    res["heldout_nll"] = tot / max(n, 1)
    # ARC-Easy
    acc = []
    for it in probes["arc"]:
        inputs = m.build_inputs(it["prompt"], None, None); r = m.forward(inputs, capture=False)
        acc.append(m.score(r["lp_last"], "mcq", it["letters"], it["gold"])["correct"])
    res["arc_acc"] = float(np.mean(acc))
    # routing loads on expert sets
    def loads(items):
        agg = {}
        for it in items:
            inputs = m.build_inputs(it["prompt"], it["images"], it.get("assistant")); ids, tags = m.tags(inputs)
            r = m.forward(inputs, capture=True); idx = r["topk_idx"].astype(np.int64)
            for tag, name in ((0, "text"), (1, "image")):
                sel = idx[:, tags == tag, :]
                if sel.shape[1] == 0:
                    continue
                for sname, M in masks.items():
                    hit = M[np.arange(m.L)[:, None, None], sel]
                    a = agg.setdefault(f"{name}_on_{sname}", [0.0, 0]); a[0] += hit.sum(); a[1] += hit.size
        return {k: v[0] / max(v[1], 1) for k, v in agg.items()}
    for k, v in loads(probes["route_text"]).items():
        res[f"alpaca_{k}"] = v
    for k, v in loads(probes["route_mm"]).items():
        res[f"mm_{k}"] = v
    # vision probes
    for src, items in probes["vision"].items():
        accs = []
        for it in items:
            inputs = m.build_inputs(it["prompt"], it["images"], None); r = m.forward(inputs, capture=False)
            accs.append(m.score(r["lp_last"], it["qtype"], it["letters"], it["gold"])["correct"])
        res[f"vis_{src}"] = float(np.mean(accs))
    # LoRA delta norms per set
    dn = torch.stack([lo.delta_norms() for lo in loras]).cpu().numpy()        # (L, E)
    for sname, M in masks.items():
        res[f"delta_{sname}"] = float(dn[M].mean())
    res["delta_all"] = float(dn.mean())
    m.model.train()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(paths.MODELS / "Qwen3-VL-30B-A3B-Instruct")); ap.add_argument("--out_dir", required=True)
    ap.add_argument("--analysis", default=str(paths.RUNS / "qwen3vl" / "analysis"))
    ap.add_argument("--cond", choices=["aux_on", "aux_off", "freeze_V10", "block_V10", "bias_balance"], required=True)
    ap.add_argument("--bias_gamma", type=float, default=0.001)
    ap.add_argument("--aux_coef", type=float, default=0.01); ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--micro_bs", type=int, default=2); ap.add_argument("--accum", type=int, default=4); ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--router_lr", type=float, default=5e-5); ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--r", type=int, default=8); ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--eval_every", type=int, default=100); ap.add_argument("--n_train", type=int, default=14000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_probe", type=int, default=40)
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True); torch.manual_seed(a.seed)
    masks = {n: np.array(json.load(open(Path(a.analysis) / f"substrate_{f}.json"))["mask"], dtype=bool)
             for n, f in (("V10", "ratio10"), ("TOPIMG16_noV10", "topimg16_noV10"), ("TOPIMG16_V10part", "topimg16_V10part"), ("T10", "text10"), ("SHARED", "shared"))}
    m = MoeVLM(a.model); L, E = m.L, m.E
    loras = install(m, r=a.r, alpha=a.alpha, train_routers=True)
    coef = 0.0 if a.cond in ("aux_off", "bias_balance") else a.aux_coef
    if a.cond == "bias_balance":
        for rt in m.routers:
            rt._bias = torch.zeros(E, device=rt.weight.device)
    m.model.config.text_config.router_aux_loss_coef = coef
    if a.cond == "freeze_V10":
        set_enabled(loras, ~masks["V10"])
    train_block = a.cond == "block_V10"
    m.model.gradient_checkpointing_enable(); m.model.config.use_cache = False
    params = [p for lo in loras for p in lo.parameters()]; rparams = [rt.weight for rt in m.routers]
    opt = torch.optim.AdamW([{"params": params, "lr": a.lr}, {"params": rparams, "lr": a.router_lr}], weight_decay=0.0, betas=(0.9, 0.95))
    print(f"cond={a.cond} coef={coef} trainable LoRA params={sum(p.numel() for p in params)/1e6:.1f}M routers={sum(p.numel() for p in rparams)/1e6:.1f}M", flush=True)
    # data
    t0 = time.time()
    train = [x for x in (tokenize_dialogue(m, d, a.max_len) for d in load_dialogues("train_sft", a.n_train, a.seed)) if x]
    heldout = [x for x in (tokenize_dialogue(m, d, 1024) for d in load_dialogues("test_sft", 60, a.seed + 1)) if x][:40]
    print(f"train dialogues={len(train)} heldout={len(heldout)} tokenized in {time.time()-t0:.0f}s", flush=True)
    mm = D.mm_items(120, 0)
    vision = {src: [it for it in mm if it["source"] == src][:a.n_probe] for src in ("pope", "mmvp", "vstar", "textvqa")}
    probes = {"heldout": heldout, "arc": arc_items(100, a.seed),
              "route_text": D.text_items(30, None, 0, a.seed)[:30], "route_mm": [it for src in ("pope", "vstar") for it in vision[src][:10]],
              "vision": vision}
    metrics = open(out / "metrics.jsonl", "a")
    def run_eval(step):
        if train_block:
            m.set_ablation(None)                                  # evaluate with the blocked experts available again
        res = evaluate(m, probes, masks, loras, step); res["cond"] = a.cond; res["elapsed"] = time.time() - t0
        metrics.write(json.dumps(res) + "\n"); metrics.flush()
        print("EVAL " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}), flush=True)
        if train_block:
            m.set_ablation(masks["V10"], "reroute")
            for rt in m.routers:
                rt._report_masked = True
    run_eval(0)
    if train_block:
        m.set_ablation(masks["V10"], "reroute")
        for rt in m.routers:
            rt._report_masked = True
    m.model.train(); rng = random.Random(a.seed); order = list(range(len(train))); rng.shuffle(order); ptr = 0
    step_t = time.time()
    for step in range(1, a.steps + 1):
        lr_scale = min(1.0, step / a.warmup)
        for g, base in zip(opt.param_groups, (a.lr, a.router_lr)):
            g["lr"] = base * lr_scale
        opt.zero_grad(set_to_none=True); lm_sum = aux_sum = 0.0
        for _ in range(a.accum):
            batch = [train[order[(ptr + k) % len(order)]] for k in range(a.micro_bs)]; ptr += a.micro_bs
            batch = collate(batch, m.tok.pad_token_id); batch = {k: v.to(m.model.device) for k, v in batch.items()}
            outp = m.model(**batch, use_cache=False, output_router_logits=True)
            loss = outp.loss / a.accum; loss.backward()
            lm_sum += float(outp.loss) - coef * float(outp.aux_loss); aux_sum += float(outp.aux_loss)
            if a.cond == "bias_balance":
                update_balance_bias(m.routers, outp.router_logits, a.bias_gamma)
        torch.nn.utils.clip_grad_norm_(params + rparams, 1.0); opt.step()
        if step % 10 == 0 or step == 1:
            print(f"step {step} lm_loss {lm_sum/a.accum:.4f} aux {aux_sum/a.accum:.3f} {(time.time()-step_t)/ (10 if step>1 else 1):.1f}s/step mem {torch.cuda.max_memory_allocated()/2**30:.1f}GiB", flush=True); step_t = time.time()
        if step % a.eval_every == 0 or step == a.steps:
            run_eval(step)
            torch.save({"lora": lora_state(loras), "routers": [rt.weight.detach().cpu() for rt in m.routers],
                        "bias": [getattr(rt, "_bias", None) for rt in m.routers] if a.cond == "bias_balance" else None,
                        "args": vars(a), "step": step}, out / "state.pt")
    print("done", flush=True)


if __name__ == "__main__":
    main()
