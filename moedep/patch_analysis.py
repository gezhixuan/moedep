"""Where do the vision-exclusive experts' image tokens come from? Maps every image token of the mm probe set back to
its 32x32 patch on the resized image and relates per-token expert-set load to (a) target bbox membership (V*, ZoomBench),
(b) patch intensity std (information proxy), (c) border vs interior position. CPU only; uses the saved routing logs.
Usage: python -m moedep.patch_analysis RUN_DIR     (Qwen3-VL-30B-A3B routing pass; the analysis runs at import)"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
from moedep import data as D
from moedep.benchmarks import load_vstar, load_zoombench

run = Path(sys.argv[1]); out = run / "analysis"
MASKS = {n: np.array(json.load(open(out / f"substrate_{f}.json"))["mask"], dtype=bool)
         for n, f in (("V10", "ratio10"), ("TOPIMG16_noV10", "topimg16_noV10"), ("TOPIMG16_V10part", "topimg16_V10part"), ("T10", "text10"))}
L, E = MASKS["V10"].shape
rows = {json.loads(l)["uid"]: json.loads(l) for l in open(run / "index.jsonl") if l.strip()}
boxes = {s.uid: s.bbox for s in load_vstar()}; boxes.update({s.uid: s.bbox for s in load_zoombench()})
FACTOR, MINP, MAXP = 32, 64 * 32 * 32, 1280 * 32 * 32

def rank(a):
    r = np.empty(len(a)); r[np.argsort(a)] = np.arange(len(a)); return r

tok = defaultdict(list)          # per-token records: load per mask, std, border, box class, source, layer-third loads
for it in D.mm_items(120, 0):
    r = rows.get(it["uid"])
    if r is None:
        continue
    img = it["images"][0]; W, H = img.size
    rh, rw = smart_resize(H, W, factor=FACTOR, min_pixels=MINP, max_pixels=MAXP); gh, gw = rh // FACTOR, rw // FACTOR
    z = np.load(run / "routing" / f"{it['uid']}.npz"); idx = z["topk_idx"].astype(np.int64); tags = z["tags"]
    pos = np.flatnonzero(tags == 1); assert len(pos) == gh * gw, (it["uid"], len(pos), gh, gw)
    g = np.asarray(img.convert("L").resize((rw, rh), Image.BICUBIC), dtype=np.float32)
    std = g.reshape(gh, FACTOR, gw, FACTOR).std(axis=(1, 3)).reshape(-1)
    loads = {n: M[np.arange(L)[:, None, None], idx[:, pos, :]].mean(axis=(0, 2)) for n, M in MASKS.items()}   # (n_img,)
    thirds = {n: [M[np.arange(a, b)[:, None, None], idx[a:b][:, pos, :]].mean(axis=(0, 2)) for a, b in ((0, L // 3), (L // 3, 2 * L // 3), (2 * L // 3, L))] for n, M in MASKS.items()}
    yy, xx = np.divmod(np.arange(gh * gw), gw)
    border = (yy == 0) | (yy == gh - 1) | (xx == 0) | (xx == gw - 1)
    cls = np.full(gh * gw, -1)
    if it["uid"] in boxes:
        x0, y0, x1, y1 = boxes[it["uid"]]; sx, sy = rw / W, rh / H
        bx0, by0, bx1, by1 = x0 * sx, y0 * sy, x1 * sx, y1 * sy
        cx0, cy0 = xx * FACTOR, yy * FACTOR; cx1, cy1 = cx0 + FACTOR, cy0 + FACTOR
        inter = (cx1 > bx0) & (cx0 < bx1) & (cy1 > by0) & (cy0 < by1)
        near = (cx1 > bx0 - FACTOR) & (cx0 < bx1 + FACTOR) & (cy1 > by0 - FACTOR) & (cy0 < by1 + FACTOR) & ~inter
        cls[inter] = 0; cls[near] = 1; cls[~inter & ~near] = 2
    for k in range(gh * gw):
        tok["src"].append(it["source"]); tok["std"].append(std[k]); tok["border"].append(border[k]); tok["cls"].append(cls[k])
        for n in MASKS:
            tok[n].append(loads[n][k])
            for t in range(3):
                tok[f"{n}_t{t}"].append(thirds[n][t][k])
    it["images"] = None
T = {k: np.array(v) for k, v in tok.items()}
uni = {n: M.sum(1).mean() / E for n, M in MASKS.items()}
R = [f"# Patch-level analysis of expert-set load ({run.name}); {len(T['std'])} image tokens from 600 mm items\n",
     "Per-token load = mean over layers of the fraction of its top-k experts in the set. Uniform expectation: " + ", ".join(f"{n} {u:.3f}" for n, u in uni.items()) + "\n"]
R.append("## (a) Target box membership (V* and ZoomBench, 240 items): mean load by token class\n\n| source | class | n tokens | " + " | ".join(MASKS) + " | V10 early/mid/late |\n|---|---|---|" + "---|" * (len(MASKS) + 1))
for src in ("vstar", "zoombench"):
    for c, name in ((0, "inside box"), (1, "ring (1 patch)"), (2, "far")):
        m = (T["src"] == src) & (T["cls"] == c)
        if m.sum():
            R.append(f"| {src} | {name} | {int(m.sum())} | " + " | ".join(f"{T[n][m].mean():.4f}" for n in MASKS) + " | " + "/".join(f"{T[f'V10_t{t}'][m].mean():.3f}" for t in range(3)) + " |")
R.append("\n## (b) Patch intensity std (information proxy), all 600 images: mean load by std quintile\n\n| std quintile | std range | n | " + " | ".join(MASKS) + " |\n|---|---|---|" + "---|" * len(MASKS))
q = np.quantile(T["std"], [0, .2, .4, .6, .8, 1.0])
for i in range(5):
    m = (T["std"] >= q[i]) & (T["std"] <= q[i + 1] if i == 4 else T["std"] < q[i + 1])
    R.append(f"| Q{i+1} | {q[i]:.1f}-{q[i+1]:.1f} | {int(m.sum())} | " + " | ".join(f"{T[n][m].mean():.4f}" for n in MASKS) + " |")
rs = rank(T["std"])
R.append("\nSpearman rank correlation of patch std with load: " + ", ".join(f"{n} {np.corrcoef(rs, rank(T[n]))[0,1]:+.3f}" for n in MASKS))
R.append("Flat patches (std < 2): " + ", ".join(f"{n} {T[n][T['std']<2].mean():.4f}" for n in MASKS) + f" (n={int((T['std']<2).sum())})")
R.append("\n## (c) Border vs interior patches\n\n| position | n | " + " | ".join(MASKS) + " |\n|---|---|" + "---|" * len(MASKS))
for name, m in (("border", T["border"]), ("interior", ~T["border"])):
    R.append(f"| {name} | {int(m.sum())} | " + " | ".join(f"{T[n][m].mean():.4f}" for n in MASKS) + " |")
R.append("\n## (d) By source\n\n| source | n | mean std | " + " | ".join(MASKS) + " |\n|---|---|---|" + "---|" * len(MASKS))
for src in ("pope", "mmvp", "vstar", "textvqa", "zoombench"):
    m = T["src"] == src
    R.append(f"| {src} | {int(m.sum())} | {T['std'][m].mean():.1f} | " + " | ".join(f"{T[n][m].mean():.4f}" for n in MASKS) + " |")
(out / "patch_report.md").write_text("\n".join(R)); np.savez_compressed(out / "patch_tokens.npz", **T)
print("\n".join(R))
