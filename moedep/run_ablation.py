"""Lesions: remove named expert sets and score every probe item under each condition.
--mask NAME=path.json (repeatable) names a set. --cond (repeatable) is one of
  none                 the intact model
  NAME:MODE            MODE = reroute (mask the router logits of the set before the top-k selection) or
                       zero (keep the routing, drop the contribution of the set)
  NAME:MODE:SCOPE      scoped lesion: SCOPE = image (image-token positions only) or text (text positions only,
                       generated tokens included)
  random:NAME:SEED     re-route a random set with as many experts per layer as NAME, drawn outside NAME
Writes results.jsonl (one row per item and condition, resumable) and summary.md."""
import argparse, json, time, traceback
from pathlib import Path
import numpy as np
import torch
from moedep.kimi_model import load_runner
from moedep import data as D
from moedep import paths


def random_masks(V, seed):
    rng = np.random.default_rng(seed); M = np.zeros_like(V)
    for l in range(V.shape[0]):
        k = int(V[l].sum()); pool = np.flatnonzero(~V[l])
        if k and len(pool):
            M[l, rng.choice(pool, size=min(k, len(pool)), replace=False)] = True
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(paths.MODELS / "Qwen3-VL-30B-A3B-Instruct"))
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--mask", action="append", default=[], help="NAME=path/to/substrate.json")
    ap.add_argument("--cond", action="append", default=[], help="none | NAME:reroute | NAME:zero | random:NAME:SEED")
    ap.add_argument("--sets", default="ascii,shapes"); ap.add_argument("--n_mm", type=int, default=120)
    ap.add_argument("--n_concepts", type=int, default=300); ap.add_argument("--n_var", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_pixels", type=int, default=1280 * 32 * 32); ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bias_scale", type=float, default=None, help="Kimi-VL only: scale of the native balancing bias")
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    masks = {}
    for spec in a.mask:
        name, path = spec.split("=", 1); masks[name] = np.array(json.load(open(path))["mask"], dtype=bool)
    conds = []
    for c in a.cond:
        if c == "none":
            conds.append(("none", None, "reroute", None))
        elif c.startswith("random:"):
            _, name, seed = c.split(":"); conds.append((f"random_{name}_{seed}", random_masks(masks[name], 100 + int(seed)), "reroute", None))
        else:
            parts = c.split(":"); name, mode = parts[0], parts[1]; scope = parts[2] if len(parts) > 2 else None
            conds.append((f"{name}_{mode}" + (f"_{scope}" if scope else ""), masks[name], mode, scope))
    items = [it for it in D.build(a.sets.split(","), n_mm=a.n_mm, n_concepts=a.n_concepts, n_var=a.n_var, seed=a.seed)
             if it["qtype"] in ("mcq", "yn") or (it["qtype"] == "free" and it["gold"])]
    if a.limit:
        keep, seen = [], {}
        for it in items:
            k = (it["set"], it["source"], it["fmt"]); seen[k] = seen.get(k, 0) + 1
            if seen[k] <= a.limit:
                keep.append(it)
        items = keep
    index = out / "results.jsonl"
    done = {(json.loads(l)["uid"], json.loads(l)["condition"]) for l in open(index) if l.strip()} if index.exists() else set()
    print(f"items={len(items)} conditions={[c[0] for c in conds]} masks={ {k: int(v.sum()) for k, v in masks.items()} } done={len(done)}", flush=True)
    m = load_runner(a.model, attn=a.attn, max_pixels=a.max_pixels)
    if a.bias_scale is not None and hasattr(m, "set_bias_scale"):
        m.set_bias_scale(a.bias_scale)
    t0 = time.time(); n = 0
    with open(index, "a") as fo:
        for it in items:
            try:
                inputs = m.build_inputs(it["prompt"], it["images"], it["assistant"])
                for cname, mask, mode, scope in conds:
                    if (it["uid"], cname) in done:
                        continue
                    m.set_ablation(mask, mode, scope)
                    res = m.forward(inputs, capture=False)
                    sc = m.score(res["lp_last"], it["qtype"], it["letters"], it["gold"])
                    row = {k: it[k] for k in ("uid", "set", "source", "fmt", "concept", "qtype", "gold")}
                    row.update(condition=cname, pred=sc["pred"], correct=sc["correct"], gold_lp=sc["gold_lp"],
                               gold_logodds=sc["gold_logodds"], top1=res["top1"], n_tokens=int(inputs["input_ids"].shape[1]))
                    fo.write(json.dumps(row) + "\n"); n += 1
                fo.flush(); m.set_ablation(None)
            except Exception as e:
                print(f"[FAIL] {it['uid']}: {type(e).__name__}: {e}", flush=True); traceback.print_exc(); m.set_ablation(None)
            finally:
                it["images"] = None
            if n and n % 300 == 0:
                print(f"{n} forwards, {time.time()-t0:.0f}s, mem={torch.cuda.max_memory_allocated()/2**30:.1f}GiB", flush=True)
    print(f"done {n} forwards in {time.time()-t0:.0f}s", flush=True)
    summarize(index, out / "summary.md")


def summarize(index, dst):
    from collections import defaultdict
    rows, seen = [], set()
    for l in open(index):
        if l.strip():
            r = json.loads(l)
            if (r["uid"], r["condition"]) not in seen:
                seen.add((r["uid"], r["condition"])); rows.append(r)
    g = defaultdict(list)
    for r in rows:
        key = (r["set"], r["fmt"] if r.get("fmt") else r["source"], r["condition"])
        g[key].append(r)
    base = {}
    for (s, f, c), rs in g.items():
        if c == "none":
            base[(s, f)] = {r["uid"]: r for r in rs}
    lines = ["| set | source/fmt | condition | n | acc | d_acc vs none | mean gold log-odds | d_logodds vs none |", "|---|---|---|---|---|---|---|---|"]
    for k in sorted(g):
        rs = g[k]; acc = np.mean([r["correct"] for r in rs]); lo = np.mean([r["gold_logodds"] for r in rs])
        b = base.get((k[0], k[1]), {})
        if b and k[2] != "none":
            pairs = [(r, b[r["uid"]]) for r in rs if r["uid"] in b]
            dacc = np.mean([r["correct"] - bb["correct"] for r, bb in pairs]); dlo = np.mean([r["gold_logodds"] - bb["gold_logodds"] for r, bb in pairs])
            lines.append(f"| {k[0]} | {k[1]} | {k[2]} | {len(rs)} | {acc:.3f} | {dacc:+.3f} | {lo:+.2f} | {dlo:+.2f} |")
        else:
            lines.append(f"| {k[0]} | {k[1]} | {k[2]} | {len(rs)} | {acc:.3f} |  | {lo:+.2f} |  |")
    Path(dst).write_text("\n".join(lines)); print("\n".join(lines))


if __name__ == "__main__":
    main()
