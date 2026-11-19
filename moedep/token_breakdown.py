"""Which text tokens land on the vision-exclusive experts (V10)? Per-token V10 load (mean over layers of the fraction of
its top-k experts that are in V10), split into art-like tokens (whitespace/punctuation/digits only) vs word-like
tokens, for several text groups. Uniform expectation = mean_l |V10_l| / E.
Usage: python -m moedep.token_breakdown RUN_DIR [MODEL_DIR]     (CPU, needs the tokenizer; the analysis runs at import)"""
import json, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
from moedep import paths

run = Path(sys.argv[1]); model = sys.argv[2] if len(sys.argv) > 2 else str(paths.MODELS / "Qwen3-VL-30B-A3B-Instruct")
sub = json.load(open(run / "analysis" / "substrate_ratio10.json")); V = np.array(sub["mask"], dtype=bool)
L, E = V.shape
tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
rows = [json.loads(l) for l in open(run / "index.jsonl") if l.strip()]
GROUPS = {
    "ascii/ascii_text": lambda r: r["set"] == "ascii" and r["fmt"] == "ascii_text",
    "ascii/ascii_shuffled": lambda r: r["set"] == "ascii" and r["fmt"] == "ascii_shuffled",
    "ascii/caption": lambda r: r["set"] == "ascii" and r["fmt"] == "caption",
    "shapes/ascii_text": lambda r: r["set"] == "shapes" and r["fmt"] == "ascii_text",
    "shapes/svg": lambda r: r["set"] == "shapes" and r["fmt"] == "svg",
    "shapes/caption": lambda r: r["set"] == "shapes" and r["fmt"] == "caption",
    "text/alpaca": lambda r: r["source"] == "alpaca",
    "text/qonly": lambda r: r["set"] == "text" and r["fmt"] == "qonly",
    "mm/text": lambda r: r["set"] == "mm",
}
art_re = re.compile(r"^[\s/\\|_\-.'`,:;()\[\]{}<>=+*#~^\"!?@$%&0-9]+$")
cache = {}
def dec(i):
    if i not in cache:
        cache[i] = tok.decode([int(i)])
    return cache[i]

uni = V.sum(1).mean() / E
print(f"uniform expectation of V10 load = {uni:.4f}   (|V10| per layer mean {V.sum(1).mean():.1f} / {E})\n")
print(f"{'group':22s} {'n_tok':>7s} {'load_all':>8s} | {'art-like n':>10s} {'load':>6s} | {'word-like n':>11s} {'load':>6s}")
tables = {}
for g, pred in GROUPS.items():
    rs = [r for r in rows if pred(r)]
    if not rs:
        continue
    stats = {"art": [0.0, 0], "word": [0.0, 0]}; per = defaultdict(lambda: [0.0, 0])
    for r in rs:
        z = np.load(run / "routing" / f"{r['uid']}.npz"); idx = z["topk_idx"].astype(np.int64); tags = z["tags"]; ids = z["input_ids"]
        hit = V[np.arange(L)[:, None, None], idx]            # (L, S, K)
        load = hit.mean(axis=(0, 2))                         # (S,)
        for s in np.flatnonzero(tags == 0):
            t = dec(ids[s]); kind = "art" if art_re.match(t) else "word"
            stats[kind][0] += load[s]; stats[kind][1] += 1; per[t][0] += load[s]; per[t][1] += 1
    tot = stats["art"][0] + stats["word"][0]; n = stats["art"][1] + stats["word"][1]
    print(f"{g:22s} {n:7d} {tot/max(n,1):8.4f} | {stats['art'][1]:10d} {stats['art'][0]/max(stats['art'][1],1):6.4f} | "
          f"{stats['word'][1]:11d} {stats['word'][0]/max(stats['word'][1],1):6.4f}")
    tables[g] = sorted(per.items(), key=lambda kv: -kv[1][1])[:12]
for g in ("ascii/ascii_text", "shapes/svg", "text/alpaca"):
    if g in tables:
        print(f"\nmost frequent text tokens in {g}: token -> (mean V10 load, count)")
        print("  " + "  ".join(f"{t!r}:{v[0]/v[1]:.3f}({v[1]})" for t, v in tables[g]))
