"""Expert sets derived from the routing counts of one run folder. CPU only.

Usage: python -m moedep.expert_sets --run_dir RUN_DIR [--affinity_count N] [--overlap_with OTHER_RUN_DIR]

Reads RUN_DIR/analysis/counts.npz and substrate_ratio10.json, both written by moedep.analyze_routing, and writes one
mask file RUN_DIR/analysis/substrate_<name>.json per set, in the format that run_ablation, run_generation and the
training scripts read. The image load P(e | image) and the text load P(e | text) of expert e are its shares of the
image-token and text-token selections of its layer on the mm probe set. E is the number of routed experts per layer.

  topimg16          image top-16 set: per layer, the k experts with the highest image load, with
                    k = max(4, round(16 E / 128)), which is 16 of 128 experts and 8 of 64
  toptxt16          text top-16 set: the same with the text load
  topimg16_V10part  vision-exclusive part of the image top-16 set
  topimg16_noV10    dual-use part of the image top-16 set
  topimg32          image top-32 set: 2k experts per layer
  shared            above-uniform set: image load and text load both at least 1/E
  all               every routed expert
  lowload           least-used set: per layer, as many experts as the layer has vision-exclusive experts, those with
                    the fewest selections over the image and text tokens of the mm set, the Alpaca chats and the
                    questions without their image
  union             vision-exclusive or least-used
  el<N>             with --affinity_count N: the N experts with the lowest raw text share C_text / (C_text + C_image)
                    over all layers (N = 462 for Qwen3-VL-30B-A3B in the paper)
"""
import argparse, json
from pathlib import Path
import numpy as np


def dist(c):
    c = c.astype(float); return c / np.maximum(c.sum(1, keepdims=True), 1)       # P(e | group, layer)


def load_mask(run, name):
    return np.array(json.load(open(Path(run) / "analysis" / f"substrate_{name}.json"))["mask"], bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--affinity_count", type=int, default=0, help="also write el<N>: the N experts with the lowest raw text share")
    ap.add_argument("--overlap_with", default=None, help="other run folder: print the overlap of the two vision-exclusive sets")
    a = ap.parse_args()
    run = Path(a.run_dir); A = run / "analysis"
    z = np.load(A / "counts.npz"); meta = json.load(open(run / "model_meta.json")); L, E = meta["L"], meta["E"]
    Pi = dist(z["mm__image"]); Pt = dist(z["mm__text"]); V10 = load_mask(run, "ratio10")
    k = max(4, round(16 * E / 128))
    T = np.zeros((L, E), bool); T32 = np.zeros((L, E), bool); X = np.zeros((L, E), bool)
    for l in range(L):
        T[l, np.argsort(-Pi[l])[:k]] = True; T32[l, np.argsort(-Pi[l])[:2 * k]] = True; X[l, np.argsort(-Pt[l])[:k]] = True
    masks = {"topimg16": T, "toptxt16": X, "topimg16_noV10": T & ~V10, "topimg16_V10part": T & V10, "topimg32": T32,
             "shared": (Pi >= 1 / E) & (Pt >= 1 / E), "all": np.ones((L, E), bool)}
    if "text__alpaca" in z.files and "text__qonly" in z.files:
        tot = z["mm__image"] + z["mm__text"] + z["text__alpaca"] + z["text__qonly"]; LOW = np.zeros((L, E), bool)
        for l in range(L):
            LOW[l, np.argsort(tot[l])[:int(V10[l].sum())]] = True
        masks["lowload"] = LOW; masks["union"] = V10 | LOW
    if a.affinity_count:
        Ci, Ct = z["mm__image"].astype(float), z["mm__text"].astype(float)
        share = Ct / np.maximum(Ci + Ct, 1)                                       # raw text share per (l, e)
        EL = np.zeros(L * E, bool); EL[np.argsort(share.reshape(-1))[:a.affinity_count]] = True
        masks[f"el{a.affinity_count}"] = EL.reshape(L, E)
    for n, M in masks.items():
        json.dump({"definition": n, "L": L, "E": E, "mask": M.astype(int).tolist(), "per_layer": M.sum(1).tolist(),
                   "source_run": str(run)}, open(A / f"substrate_{n}.json", "w"))
        print(n, int(M.sum()), "image load %.3f text load %.3f" % (np.mean([Pi[l][M[l]].sum() for l in range(L)]),
                                                                    np.mean([Pt[l][M[l]].sum() for l in range(L)])))
    if a.overlap_with:
        Vo = load_mask(a.overlap_with, "ratio10")
        print("V10 of", Path(a.overlap_with).name, int(Vo.sum()), "| V10 of", run.name, int(V10.sum()), "| overlap", int((Vo & V10).sum()))


if __name__ == "__main__":
    main()
