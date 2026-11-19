"""Routing pass: one forward per probe item. Saves the top-k experts of every token in every MoE layer with the token
tags, and scores the answer. Resumable (skips uids present in index.jsonl)."""
import argparse, json, time, traceback
from pathlib import Path
import numpy as np
import torch
from moedep.kimi_model import load_runner
from moedep import data as D
from moedep import paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(paths.MODELS / "Qwen3-VL-30B-A3B-Instruct"))
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--sets", default="mm,text,ascii,shapes")
    ap.add_argument("--n_mm", type=int, default=100); ap.add_argument("--n_alpaca", type=int, default=400)
    ap.add_argument("--n_qonly", type=int, default=300); ap.add_argument("--n_concepts", type=int, default=300)
    ap.add_argument("--n_var", type=int, default=6); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_pixels", type=int, default=1280 * 32 * 32)
    ap.add_argument("--attn", default="sdpa"); ap.add_argument("--bias_scale", type=float, default=None, help="Kimi-VL only: scale of the native balancing bias")
    ap.add_argument("--limit", type=int, default=0, help="debug: stop after this many items")
    a = ap.parse_args()

    out = Path(a.out_dir); (out / "routing").mkdir(parents=True, exist_ok=True)
    index = out / "index.jsonl"
    done = set()
    if index.exists():
        done = {json.loads(l)["uid"] for l in open(index) if l.strip()}
    sets = a.sets.split(",")
    t0 = time.time()
    items = D.build(sets, a.n_mm, a.n_alpaca, a.n_qonly, a.n_concepts, a.n_var, a.seed)
    if a.limit:
        # keep a few of every (set, fmt) for smoke runs
        keep, seen = [], {}
        for it in items:
            k = (it["set"], it["source"], it["fmt"]); seen[k] = seen.get(k, 0) + 1
            if seen[k] <= a.limit:
                keep.append(it)
        items = keep
    print(f"items={len(items)} (done={len(done)}) built in {time.time()-t0:.0f}s", flush=True)
    from collections import Counter
    print(Counter((it["set"], it["fmt"]) for it in items), flush=True)

    m = load_runner(a.model, attn=a.attn, max_pixels=a.max_pixels)
    if a.bias_scale is not None and hasattr(m, "set_bias_scale"):
        m.set_bias_scale(a.bias_scale)
    print(f"model loaded: L={m.L} E={m.E} K={m.K} moe_layers={m.moe_layers[:3]}..{m.moe_layers[-1]}", flush=True)
    json.dump({"L": m.L, "E": m.E, "K": m.K, "moe_layers": m.moe_layers, "model": a.model, "max_pixels": a.max_pixels,
               "image_token_id": m.image_token_id}, open(out / "model_meta.json", "w"))

    n_ok = n_fail = 0; t1 = time.time()
    with open(index, "a") as fo:
        for k, it in enumerate(items):
            if it["uid"] in done:
                continue
            try:
                inputs = m.build_inputs(it["prompt"], it["images"], it["assistant"])
                ids, tags = m.tags(inputs)
                res = m.forward(inputs, capture=True)
                sc = m.score(res["lp_last"], it["qtype"], it["letters"], it["gold"])
                np.savez_compressed(out / "routing" / f"{it['uid']}.npz", topk_idx=res["topk_idx"], topk_w=res["topk_w"],
                                    input_ids=ids, tags=tags)
                row = {k2: it[k2] for k2 in ("uid", "set", "source", "fmt", "concept", "qtype", "letters", "gold", "meta")}
                row.update(n_tokens=int(len(ids)), n_image=int((tags == 1).sum()), n_text=int((tags == 0).sum()),
                           n_vspecial=int((tags == 2).sum()), aux_loss=res["aux_loss"], top1=res["top1"],
                           pred=sc["pred"], correct=sc["correct"], gold_lp=sc["gold_lp"], gold_logodds=sc["gold_logodds"],
                           n_images=len(it["images"] or []))
                fo.write(json.dumps(row) + "\n"); fo.flush(); n_ok += 1
            except Exception as e:
                n_fail += 1; print(f"[FAIL] {it['uid']}: {type(e).__name__}: {e}", flush=True); traceback.print_exc()
                if n_fail > 20:
                    raise
            finally:
                it["images"] = None
            if (n_ok + n_fail) % 25 == 0:
                el = time.time() - t1
                print(f"{k+1}/{len(items)} ok={n_ok} fail={n_fail} {el:.0f}s ({el/max(1,n_ok+n_fail):.2f}s/item) "
                      f"mem={torch.cuda.max_memory_allocated()/2**30:.1f}GiB", flush=True)
    print(f"done ok={n_ok} fail={n_fail} total={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
