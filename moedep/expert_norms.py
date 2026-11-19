"""What do the vision-exclusive experts compute? For sampled image and text tokens of mm items, evaluate ALL experts of
every MoE layer on the token's MoE-block input and record, per (layer, expert): mean ||y_e|| / ||x||, mean cosine of y_e with the
realized MoE output y* (sum of the token's routed experts), and within-set mean pairwise cosine among expert outputs.
Qwen3-VL-MoE only (HF Qwen3VLMoeTextExperts weights: gate_up_proj (E, 2I, D), down_proj (E, D, I))."""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from moedep.model import MoeVLM
from moedep import data as D
from moedep import paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(paths.MODELS / "Qwen3-VL-30B-A3B-Instruct")); ap.add_argument("--out_dir", required=True)
    ap.add_argument("--analysis", required=True, help="analysis dir with substrate_*.json")
    ap.add_argument("--n_items", type=int, default=40); ap.add_argument("--n_img_tok", type=int, default=64); ap.add_argument("--n_txt_tok", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    masks = {n: np.array(json.load(open(Path(a.analysis) / f"substrate_{f}.json"))["mask"], dtype=bool)
             for n, f in (("V10", "ratio10"), ("TOPIMG16_noV10", "topimg16_noV10"), ("TOPIMG16_V10part", "topimg16_V10part"), ("T10", "text10"), ("SHARED", "shared"))}
    m = MoeVLM(a.model)
    L, E = m.L, m.E
    cap = {}                                       # layer -> (x, idx, w) captured at the sparse block input
    def make_hook(li):
        def hook(mod, inp, outp):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1])
            _, w, idx = mod.gate(x)
            cap[li] = (x, idx.detach(), w.detach())
        return hook
    hs = [m.layers[i].mlp.register_forward_hook(make_hook(k)) for k, i in enumerate(m.moe_layers)]
    rng = np.random.default_rng(a.seed)
    items = [it for it in D.mm_items(120, 0) if it["source"] in ("pope", "mmvp", "vstar", "textvqa")]
    rng.shuffle(items); items = items[:a.n_items]
    acc = {k: np.zeros((L, E)) for k in ("ratio_img", "ratio_txt", "cos_img", "cos_txt", "n_img", "n_txt")}
    pair = {s: np.zeros(L) for s in masks}; pair_n = {s: np.zeros(L) for s in masks}; pair_all = np.zeros(L)
    t0 = time.time()
    with torch.no_grad():
        for it in items:
            inputs = m.build_inputs(it["prompt"], it["images"], None); ids, tags = m.tags(inputs)
            cap.clear(); m.model(**inputs, use_cache=False, logits_to_keep=1)
            img_pos = np.flatnonzero(tags == 1); txt_pos = np.flatnonzero(tags == 0)
            sel_img = rng.choice(img_pos, min(a.n_img_tok, len(img_pos)), replace=False); sel_txt = rng.choice(txt_pos, min(a.n_txt_tok, len(txt_pos)), replace=False)
            for li in range(L):
                x, idx, w = cap[li]; ex = m.layers[m.moe_layers[li]].mlp.experts
                for kind, sel in (("img", sel_img), ("txt", sel_txt)):
                    xs = x[sel].float()                                                   # (n, D)
                    gu = torch.einsum("nd,eid->eni", xs, ex.gate_up_proj.float())        # (E, n, 2I)
                    g, u = gu.chunk(2, dim=-1); h = ex.act_fn(g) * u
                    y = torch.einsum("eni,edi->end", h, ex.down_proj.float())           # (E, n, D)
                    xn = xs.norm(dim=-1).clamp_min(1e-6)
                    ratio = y.norm(dim=-1) / xn[None]                                     # (E, n)
                    ystar = torch.zeros_like(xs)
                    ar = torch.arange(len(sel), device=xs.device)
                    for k in range(idx.shape[1]):
                        ystar += w[sel, k].float()[:, None] * y[idx[sel, k], ar]
                    cos = F.cosine_similarity(y, ystar[None].expand_as(y), dim=-1)       # (E, n)
                    acc[f"ratio_{kind}"][li] += ratio.sum(1).cpu().numpy(); acc[f"cos_{kind}"][li] += cos.sum(1).cpu().numpy(); acc[f"n_{kind}"][li] += len(sel)
                    if kind == "img":
                        yn = F.normalize(y, dim=-1)
                        C = torch.einsum("end,fnd->nef", yn, yn).mean(0).cpu().numpy()   # (E, E)
                        off = ~np.eye(E, dtype=bool); pair_all[li] += C[off].mean()
                        for s, M in masks.items():
                            sel_e = np.flatnonzero(M[li])
                            if len(sel_e) > 1:
                                sub = C[np.ix_(sel_e, sel_e)]; pair[s][li] += sub[~np.eye(len(sel_e), dtype=bool)].mean(); pair_n[s][li] += 1
            it["images"] = None
            print(f"{it['uid']} done, {time.time()-t0:.0f}s", flush=True)
    n_items = len(items)
    res = {k: acc[k] / np.maximum(acc["n_" + k.split("_")[1]], 1) for k in ("ratio_img", "ratio_txt", "cos_img", "cos_txt")}
    np.savez_compressed(out / "expert_norms.npz", **res, pair_all=pair_all / n_items, **{f"pair_{s}": pair[s] / np.maximum(pair_n[s], 1) for s in masks})
    R = [f"# Expert output statistics ({n_items} mm items, {a.n_img_tok} image + {a.n_txt_tok} text tokens each, all {E} experts x {L} layers)\n",
         "| expert set | n experts | ||y||/||x|| image tokens | text tokens | cos(y_e, y*) image | cos(y_e, y*) text | within-set pairwise cos (image) |", "|---|---|---|---|---|---|---|"]
    allm = np.ones((L, E), bool); rest = allm & ~np.any([masks[s] for s in masks], axis=0)
    for name, M in list(masks.items()) + [("all", allm), ("none of the sets", rest)]:
        row = [f"{res[k][M].mean():.4f}" for k in ("ratio_img", "ratio_txt", "cos_img", "cos_txt")]
        pc = (pair[name] / np.maximum(pair_n[name], 1)).mean() if name in pair else ((pair_all / n_items).mean() if name == "all" else float("nan"))
        R.append(f"| {name} | {int(M.sum())} | " + " | ".join(row) + f" | {pc:.3f} |")
    R.append("\nPer-layer ||y||/||x|| on image tokens, V10 / TOPIMG16_noV10 / all: " + "; ".join(
        f"L{l}: {res['ratio_img'][l][masks['V10'][l]].mean():.3f}/{res['ratio_img'][l][masks['TOPIMG16_noV10'][l]].mean():.3f}/{res['ratio_img'][l].mean():.3f}" for l in range(0, L, 6)))
    (out / "expert_norms.md").write_text("\n".join(R)); print("\n".join(R))


if __name__ == "__main__":
    main()
