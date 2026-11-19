"""Re-exposure phase: starting from the LoRA + router state of a deprivation run, continue training on image-caption
pairs (COCO val2014 captions, probe images excluded) and track the vision probes, routing loads and LoRA delta norms
with the same evaluate() as the deprivation phase. Loss on the caption tokens only."""
import argparse, collections, json, random, re, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from moedep.model import MoeVLM
from moedep import data as D
from moedep.lora_experts import install, lora_state
from moedep.train_deprive import evaluate, load_dialogues, tokenize_dialogue, arc_items, DEP
from moedep import paths

COCO_IMG = DEP / "coco_val2014" / "val2014"; COCO_ANN = DEP / "coco_ann" / "annotations" / "captions_val2014.json"
CAP_PROMPT = "Describe this image in one sentence."


def caption_pairs(n, seed, exclude):
    caps = collections.defaultdict(list)
    for a in json.load(open(COCO_ANN))["annotations"]:
        caps[a["image_id"]].append(a["caption"].strip())
    ids = [i for i in sorted(caps) if i not in exclude]; random.Random(seed).shuffle(ids)
    return [(i, caps[i]) for i in ids[:n]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(paths.MODELS / "Qwen3-VL-30B-A3B-Instruct")); ap.add_argument("--out_dir", required=True)
    ap.add_argument("--state", required=True, help="state.pt of a deprivation run (or 'none' for the undeprived control)")
    ap.add_argument("--analysis", default=str(paths.RUNS / "qwen3vl" / "analysis"))
    ap.add_argument("--steps", type=int, default=400); ap.add_argument("--micro_bs", type=int, default=4); ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--router_lr", type=float, default=5e-5); ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--r", type=int, default=8); ap.add_argument("--alpha", type=float, default=16.0); ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--n_images", type=int, default=6000); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--n_probe", type=int, default=40)
    ap.add_argument("--max_pixels", type=int, default=640 * 32 * 32)
    ap.add_argument("--keep_bias", type=int, default=1, help="apply the saved balancing bias (bias_balance runs) during restoration/eval")
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True); torch.manual_seed(a.seed); random.seed(a.seed)
    masks = {n: np.array(json.load(open(Path(a.analysis) / f"substrate_{f}.json"))["mask"], dtype=bool)
             for n, f in (("V10", "ratio10"), ("TOPIMG16_noV10", "topimg16_noV10"), ("TOPIMG16_V10part", "topimg16_V10part"), ("T10", "text10"), ("SHARED", "shared"))}
    m = MoeVLM(a.model, max_pixels=a.max_pixels)
    loras = install(m, r=a.r, alpha=a.alpha, train_routers=True)
    if a.state != "none":
        st = torch.load(a.state, map_location="cpu")
        for li, lo in enumerate(loras):
            lo.load_state_dict({k.split(".", 1)[1]: v for k, v in st["lora"].items() if k.startswith(f"layer{li}.")})
        for rt, w in zip(m.routers, st["routers"]):
            rt.weight.data.copy_(w.to(rt.weight.device))
        if a.keep_bias and st.get("bias") and st["bias"][0] is not None:
            for rt, b in zip(m.routers, st["bias"]):
                rt._bias = b.to(rt.weight.device).float()
            print("balancing bias loaded and kept active", flush=True)
        print(f"loaded deprivation state from {a.state} (step {st.get('step')}, cond {st['args'].get('cond')})", flush=True)
    for lo in loras:
        lo.enabled.fill_(1.0)                                     # all experts trainable during restoration
    m.model.config.text_config.router_aux_loss_coef = 0.0         # plain re-alignment, no balancing pressure
    m.model.gradient_checkpointing_enable(); m.model.config.use_cache = False
    params = [p for lo in loras for p in lo.parameters()]; rparams = [rt.weight for rt in m.routers]
    opt = torch.optim.AdamW([{"params": params, "lr": a.lr}, {"params": rparams, "lr": a.router_lr}], weight_decay=0.0, betas=(0.9, 0.95))
    # probes (same as deprivation)
    t0 = time.time(); mm = D.mm_items(120, 0)
    vision = {src: [it for it in mm if it["source"] == src][:a.n_probe] for src in ("pope", "mmvp", "vstar", "textvqa")}
    probe_ids = {int(re.search(r"(\d+)$", it["meta"]["image_source"]).group(1)) for it in mm if it["source"] == "pope"}
    heldout = [x for x in (tokenize_dialogue(m, d, 1024) for d in load_dialogues("test_sft", 60, a.seed + 1)) if x][:40]
    probes = {"heldout": heldout, "arc": arc_items(100, a.seed), "route_text": D.text_items(30, None, 0, a.seed)[:30],
              "route_mm": [it for src in ("pope", "vstar") for it in vision[src][:10]], "vision": vision}
    pairs = caption_pairs(a.n_images, a.seed, probe_ids); print(f"restoration pairs={len(pairs)} probes built in {time.time()-t0:.0f}s", flush=True)
    metrics = open(out / "metrics.jsonl", "a")
    def run_eval(step):
        res = evaluate(m, probes, masks, loras, step); res["phase"] = "restore"; res["elapsed"] = time.time() - t0
        metrics.write(json.dumps(res) + "\n"); metrics.flush()
        print("EVAL " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}), flush=True)
    run_eval(0)
    if a.steps == 0:
        print("eval-only run; done", flush=True); return
    m.model.train(); ptr = 0; step_t = time.time()
    for step in range(1, a.steps + 1):
        for g, base in zip(opt.param_groups, (a.lr, a.router_lr)):
            g["lr"] = base * min(1.0, step / a.warmup)
        opt.zero_grad(set_to_none=True); lsum = 0.0
        for _ in range(a.accum):
            losses = []
            for k in range(a.micro_bs):
                iid, caps = pairs[(ptr + k) % len(pairs)]
                img = Image.open(COCO_IMG / f"COCO_val2014_{iid:012d}.jpg").convert("RGB"); cap = random.choice(caps)
                inputs = m.build_inputs(CAP_PROMPT, [img], cap); pre = m.build_inputs(CAP_PROMPT, [img], None)["input_ids"].shape[1]
                labels = inputs["input_ids"].clone(); labels[:, :pre] = -100
                outp = m.model(**inputs, labels=labels, use_cache=False)
                (outp.loss / (a.accum * a.micro_bs)).backward(); losses.append(float(outp.loss))
            ptr += a.micro_bs; lsum += float(np.mean(losses))
        torch.nn.utils.clip_grad_norm_(params + rparams, 1.0); opt.step()
        if step % 10 == 0 or step == 1:
            print(f"step {step} cap_loss {lsum/a.accum:.4f} {(time.time()-step_t)/(10 if step>1 else 1):.1f}s/step mem {torch.cuda.max_memory_allocated()/2**30:.1f}GiB", flush=True); step_t = time.time()
        if step % a.eval_every == 0 or step == a.steps:
            run_eval(step); torch.save({"lora": lora_state(loras), "routers": [rt.weight.detach().cpu() for rt in m.routers], "args": vars(a), "step": step}, out / "state.pt")
    print("done", flush=True)


if __name__ == "__main__":
    main()
