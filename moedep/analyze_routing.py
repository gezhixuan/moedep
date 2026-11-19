"""CPU analysis of a routing pass.

Inputs : <run_dir>/index.jsonl, <run_dir>/routing/*.npz, <run_dir>/model_meta.json
Outputs: <run_dir>/analysis/{counts.npz, report.md, substrate_*.json}

Groups (token populations) over which routing counts C[l, e] are accumulated:
  mm/image, mm/text, mm/vspecial            (image+question items)
  text/alpaca, text/qonly                   (text-only items)
  ascii/<fmt>/{text,image}, shapes/<fmt>/{text,image}

Expert-set masks written here (layers x experts). P(e | image) and P(e | text) are the image load and the text load of
expert e in its layer on the mm set, E is the number of routed experts per layer.
  substrate_ratio10.json    V10, the vision-exclusive experts: P(e | image) >= 10 P(e | text) and P(e | image) >= 1/E
  substrate_text10.json     T10, the text-exclusive experts: the mirror definition
  substrate_bshare05.json   balanced text share P(e | text) / (P(e | image) + P(e | text)) of at most 5%
  substrate_tbshare05.json  balanced text share of at least 95%
  substrate_share05.json    raw text share C_text / (C_text + C_image) of at most 5%
The other sets are derived from counts.npz by moedep.expert_sets.
"""
import argparse, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np

EPS = 1e-12


def jsd(p, q):
    p = p / max(p.sum(), EPS); q = q / max(q.sum(), EPS); m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / np.maximum(b[mask], EPS))))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def load(run_dir):
    rows = [json.loads(l) for l in open(run_dir / "index.jsonl") if l.strip()]
    meta = json.load(open(run_dir / "model_meta.json"))
    L, E = meta["L"], meta["E"]
    counts = defaultdict(lambda: np.zeros((L, E), np.int64))
    ntok = defaultdict(int)
    for r in rows:
        z = np.load(run_dir / "routing" / f"{r['uid']}.npz")
        idx, tags = z["topk_idx"].astype(np.int64), z["tags"]
        if r["set"] == "mm":
            keys = {0: "mm/text", 1: "mm/image", 2: "mm/vspecial"}
        elif r["set"] == "text":
            keys = {0: f"text/{'alpaca' if r['source']=='alpaca' else 'qonly'}"}
        else:
            keys = {0: f"{r['set']}/{r['fmt']}/text", 1: f"{r['set']}/{r['fmt']}/image"}
        for tag, key in keys.items():
            sel = idx[:, tags == tag, :]          # (L, n, K)
            n = sel.shape[1]
            if n == 0:
                continue
            ntok[key] += n
            flat = sel.reshape(L, -1)
            for l in range(L):
                counts[key][l] += np.bincount(flat[l], minlength=E)
    return rows, meta, dict(counts), dict(ntok)


def dist(c):
    return c / np.maximum(c.sum(1, keepdims=True), 1)       # P(e | group, layer)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run_dir", required=True); a = ap.parse_args()
    run = Path(a.run_dir); out = run / "analysis"; out.mkdir(exist_ok=True)
    rows, meta, C, N = load(run)
    L, E, K = meta["L"], meta["E"], meta["K"]
    np.savez_compressed(out / "counts.npz", **{k.replace("/", "__"): v for k, v in C.items()}, ntok=json.dumps(N))
    R = []
    R.append(f"# Routing analysis: {run.name}\n\nmodel={meta['model']} L={L} E={E} K={K}; items={len(rows)}\n")
    R.append("## Token counts per group\n\n| group | tokens | top-k selections |\n|---|---|---|")
    for k in sorted(N):
        R.append(f"| {k} | {N[k]} | {N[k]*K} |")

    # ---------- behavioral ----------
    R.append("\n## Behavioral accuracy (first-token scoring)\n\n| set | source | fmt | qtype | n | acc | mean gold log-odds |\n|---|---|---|---|---|---|---|")
    grp = defaultdict(list)
    for r in rows:
        if r["correct"] is not None:
            grp[(r["set"], r["source"], r["fmt"], r["qtype"])].append(r)
    for k in sorted(grp, key=str):
        rs = grp[k]; acc = np.mean([r["correct"] for r in rs]); lo = np.mean([r["gold_logodds"] for r in rs])
        R.append(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {len(rs)} | {acc:.3f} | {lo:+.2f} |")

    if "mm/image" not in C or "mm/text" not in C:
        (out / "report.md").write_text("\n".join(R)); print("no mm set; wrote behavioral report only"); return

    # ---------- affinity map (ExpertLens definition) ----------
    Ci, Ct = C["mm/image"].astype(float), C["mm/text"].astype(float)
    share = Ct / np.maximum(Ci + Ct, 1)                     # text share per (l, e)
    seen = (Ci + Ct) > 0
    bins = np.arange(0, 1.0001, 0.05)
    hist, _ = np.histogram(share[seen], bins=bins)
    R.append(f"\n## Affinity map (mm set): text share = C_text / (C_text + C_image) per expert\n\n"
             f"image tokens={N['mm/image']}, text tokens={N['mm/text']} (ratio {N['mm/image']/max(N['mm/text'],1):.1f}:1); "
             f"experts never selected: {int((~seen).sum())} / {L*E}\n\n| text-share bin | experts |\n|---|---|")
    for b0, h in zip(bins[:-1], hist):
        R.append(f"| {b0:.2f}-{b0+0.05:.2f} | {int(h)} |")
    n05 = int(((share <= 0.05) & seen).sum()); n95 = int(((share >= 0.95) & seen).sum())
    R.append(f"\nexperts with text share <= 5%: **{n05}** (ExpertLens reports 462 on this model); >= 95%: {n95}")

    # ---------- normalized loads ----------
    Pi, Pt = dist(Ci), dist(Ct)                              # P(e | image, l), P(e | text, l)
    ratio = Pi / np.maximum(Pt, EPS)
    # balanced text share: what the ExpertLens share would be with a 1:1 image:text token mix
    bshare = Pt / np.maximum(Pi + Pt, EPS)
    bhist, _ = np.histogram(bshare[seen], bins=bins)
    R.append("\n### Balanced text share (token mix re-weighted to 1:1): share_b = P(e|text) / (P(e|text) + P(e|image))\n\n| bin | experts |\n|---|---|")
    for b0, h in zip(bins[:-1], bhist):
        R.append(f"| {b0:.2f}-{b0+0.05:.2f} | {int(h)} |")
    nb05 = int(((bshare <= 0.05) & seen).sum()); nb95 = int(((bshare >= 0.95) & seen).sum())
    R.append(f"\nbalanced share <= 5% (image-leaning): **{nb05}**; >= 95% (text-leaning): **{nb95}**; "
             f"modes: {np.argmax(bhist)*5}-{np.argmax(bhist)*5+5}% and second peak at "
             f"{int(np.argsort(bhist)[-2])*5}-{int(np.argsort(bhist)[-2])*5+5}%")
    V05 = (share <= 0.05) & seen
    V10 = (Pi >= 10 * Pt) & (Pi >= 1.0 / E)
    T10 = (Pt >= 10 * Pi) & (Pt >= 1.0 / E)
    V05b = (bshare <= 0.05) & seen
    T05b = (bshare >= 0.95) & seen
    R.append("\n## Substrate definitions per layer\n\nV05: text share <= 5%. V10: P(e|image) >= 10 x P(e|text) and >= uniform. T10: the text mirror.\n\n"
             "| layer | V05 | V10 | T10 | JSD(image, text) bits | image load on V10 | text load on V10 |\n|---|---|---|---|---|---|---|")
    jsd_it = np.array([jsd(Pi[l], Pt[l]) for l in range(L)])
    for l in range(L):
        R.append(f"| {l} | {int(V05[l].sum())} | {int(V10[l].sum())} | {int(T10[l].sum())} | {jsd_it[l]:.3f} | "
                 f"{Pi[l][V10[l]].sum():.3f} | {Pt[l][V10[l]].sum():.4f} |")
    R.append(f"\ntotals: V05={int(V05.sum())}, V10={int(V10.sum())}, T10={int(T10.sum())}; mean JSD(image,text)={jsd_it.mean():.3f} "
             f"(early third {jsd_it[:L//3].mean():.3f}, middle {jsd_it[L//3:2*L//3].mean():.3f}, late {jsd_it[2*L//3:].mean():.3f})")
    R.append(f"\nbalanced-share substrates: V05b (share_b <= 5%) = {int(V05b.sum())} experts, T05b (share_b >= 95%) = {int(T05b.sum())} experts; "
             f"per-layer V05b: {V05b.sum(1).tolist()}")
    for name, V in (("share05", V05), ("ratio10", V10), ("bshare05", V05b), ("text10", T10), ("tbshare05", T05b)):
        json.dump({"definition": name, "L": L, "E": E, "mask": V.astype(int).tolist(), "per_layer": V.sum(1).tolist(),
                   "source_run": str(run)}, open(out / f"substrate_{name}.json", "w"))

    # ---------- load of every group on the vision-exclusive experts (V10) ----------
    def load_on(V, c):
        P = dist(c.astype(float)); return np.array([P[l][V[l]].sum() for l in range(L)])
    uni = V10.sum(1) / E
    R.append("\n## Load on the vision substrate V10 (fraction of a group's top-k selections that land on V10 experts)\n\n"
             "uniform expectation = |V10_l| / E. Rows: group; columns: mean over layer thirds and all layers.\n\n"
             "| group | early | middle | late | all | all / uniform |\n|---|---|---|---|---|---|")
    thirds = lambda v: (v[:L//3].mean(), v[L//3:2*L//3].mean(), v[2*L//3:].mean(), v.mean())
    R.append("| uniform |" + "|".join(f" {x:.4f} " for x in thirds(uni)) + "| 1.00 |")
    loads = {}
    for g in sorted(C):
        v = load_on(V10, C[g]); loads[g] = v
        e, m_, la, al = thirds(v)
        R.append(f"| {g} | {e:.4f} | {m_:.4f} | {la:.4f} | {al:.4f} | {al/max(uni.mean(),EPS):.2f} |")

    # ---------- JSD between groups ----------
    pairs = [("mm/image", "mm/text"), ("mm/text", "text/alpaca"), ("mm/text", "text/qonly"), ("mm/image", "text/alpaca")]
    for s in ("ascii", "shapes"):
        fm = sorted({k.split("/")[1] for k in C if k.startswith(s + "/")})
        for f in fm:
            for sub in ("text", "image"):
                k = f"{s}/{f}/{sub}"
                if k in C:
                    pairs += [(k, "mm/image"), (k, "mm/text"), (k, "text/alpaca")]
        for f1 in fm:
            for f2 in fm:
                if f1 < f2:
                    k1, k2 = f"{s}/{f1}/text", f"{s}/{f2}/text"
                    if k1 in C and k2 in C:
                        pairs.append((k1, k2))
    R.append("\n## Routing divergence between groups: JSD(P(e|g1,l), P(e|g2,l)) in bits\n\n| g1 | g2 | early | middle | late | all |\n|---|---|---|---|---|---|")
    J = {}
    for g1, g2 in pairs:
        if g1 in C and g2 in C and g1 != g2:
            P1, P2 = dist(C[g1].astype(float)), dist(C[g2].astype(float))
            v = np.array([jsd(P1[l], P2[l]) for l in range(L)]); J[(g1, g2)] = v
            e, m_, la, al = thirds(v)
            R.append(f"| {g1} | {g2} | {e:.3f} | {m_:.3f} | {la:.3f} | {al:.3f} |")

    # ---------- per-layer table for key groups ----------
    key = [g for g in ("mm/image", "mm/text", "text/alpaca", "text/qonly", "ascii/ascii_text/text", "ascii/caption/text",
                       "ascii/ascii_image/image", "ascii/photo/image", "shapes/svg/text", "shapes/ascii_text/text",
                       "shapes/raster/image", "shapes/caption/text") if g in loads]
    R.append("\n## Per-layer load on V10 by group\n\n| layer | uniform | " + " | ".join(key) + " |\n|---|---|" + "---|" * len(key))
    for l in range(L):
        R.append(f"| {l} | {uni[l]:.4f} | " + " | ".join(f"{loads[g][l]:.4f}" for g in key) + " |")

    (out / "report.md").write_text("\n".join(R))
    print("\n".join(R[:60])); print(f"... report: {out/'report.md'}")


if __name__ == "__main__":
    main()
