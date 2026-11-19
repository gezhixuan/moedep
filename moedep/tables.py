"""Print the tables of the paper from the per-item results of the lesion and generation runs. CPU only.

Usage: python -m moedep.tables [RUNS_DIR]        (default: MOEDEP_RUNS)

Reads results.jsonl of the run folders that scripts/paper_runs.sh writes, and for Table 2 the routing counts, and
prints Table 1, Table 2, Tables S3 to S6 and the five-probe means. Accuracy is in percent, a change is in percentage
points against the intact model ("none") of the same run folder, caption NLL is in nats per token. The five probes are
POPE, MMVP, V*, ZoomBench and TextVQA. A cell prints as n/a when its run folder or condition is missing. Row and column
labels give the paper's term, followed in brackets by the condition name in results.jsonl.
"""
import argparse
import collections
import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np

from moedep import paths

PROBES = [("pope", "POPE"), ("mmvp", "MMVP"), ("vstar", "V*"), ("zoombench", "ZoomBench"), ("textvqa", "TextVQA"),
          ("hr_bench_4k", "HR-4K"), ("hr_bench_8k", "HR-8K"), ("mmerw_lite", "MME-RW"), ("arc_easy", "ARC")]
P9 = [p for p, _ in PROBES]
Q, K, KNB, G = "qwen3vl", "kimivl", "kimivl_nobias", "gemma4"


def mean(xs):
    xs = list(xs)
    return None if not xs or any(x is None for x in xs) else sum(xs) / len(xs)


class Results:
    """Per-item results under one runs directory, read on first use."""

    def __init__(self, base):
        self.base, self._lesion, self._gen = Path(base), {}, {}

    def lesion(self, run):
        """(condition, probe) -> {uid: correct} of a moedep.run_ablation folder."""
        if run not in self._lesion:
            d = collections.defaultdict(dict); f = self.base / run / "results.jsonl"
            for l in (open(f) if f.exists() else []):
                if l.strip():
                    r = json.loads(l); d[(r["condition"], r["source"])][r["uid"]] = float(r["correct"])
            self._lesion[run] = d
        return self._lesion[run]

    def generation(self, run):
        """(condition, task) -> {uid: caption NLL or TextVQA soft accuracy} of a moedep.run_generation folder."""
        if run not in self._gen:
            d = collections.defaultdict(dict); f = self.base / run / "results.jsonl"
            for l in (open(f) if f.exists() else []):
                if l.strip():
                    r = json.loads(l); d[(r["condition"], r["task"])][r["uid"]] = r["nll"] if r["task"] == "caption" else 100 * r["soft_acc"]
            self._gen[run] = d
        return self._gen[run]

    def acc(self, run, cond, probe):
        v = self.lesion(run).get((cond, probe))
        return None if not v else 100 * sum(v.values()) / len(v)

    def delta(self, run, cond):
        """Change against the intact model on the nine probes, then the five-probe mean."""
        out = []
        for p in P9:
            a, b = self.acc(run, cond, p), self.acc(run, "none", p)
            out.append(None if a is None or b is None else a - b)
        return out + [mean(out[:5])]

    def gen(self, run, cond, task):
        v = self.generation(run).get((cond, task))
        return None if not v else sum(v.values()) / len(v)

    def gen_delta(self, runs, cond, task):
        """Paired change against the intact model, in the first of the folders that holds the condition."""
        for run in runs:
            v, b = self.generation(run).get((cond, task)), self.generation(run).get(("none", task))
            if v and b:
                return mean(v[u] - b[u] for u in v if u in b)
        return None


def table1(R):
    A2, S, GEN, GS = f"{Q}/ablation2", f"{Q}/ablation_scope", f"{Q}/generation", f"{Q}/generation_scope"
    four = P9[:4]
    rand = [(A2, "random_V10_0"), (A2, "random_V10_1"), (S, "random_V10_2"), (S, "random_V10_3"), (S, "random_V10_4")]
    spec = [("none", S, "none", GEN), ("vision-exclusive [V10_reroute]", A2, "V10_reroute", GEN),
            ("random, mean of five draws [random_V10_0 to 4]", None, None, None),
            ("image top-16, vision-exclusive part [TOPIMG16_V10part_reroute]", A2, "TOPIMG16_V10part_reroute", GEN),
            ("image top-16, dual-use part [TOPIMG16_noV10_reroute]", S, "TOPIMG16_noV10_reroute", GEN),
            ("above uniform [SHARED_reroute]", A2, "SHARED_reroute", GEN), ("text top-16 [TOPTXT16_reroute]", A2, "TOPTXT16_reroute", GEN),
            ("dual-use part, image [TOPIMG16_noV10_reroute_image]", S, "TOPIMG16_noV10_reroute_image", GS),
            ("dual-use part, text [TOPIMG16_noV10_reroute_text]", S, "TOPIMG16_noV10_reroute_text", GS),
            ("every expert, image, zeroed [ALL_zero_image]", S, "ALL_zero_image", GS)]
    rows = []
    for label, run, cond, gen in spec:
        if run is None:                                   # generation metrics of the random control: one draw (seed 0)
            vals = [mean(R.acc(r, c, p) for r, c in rand) for p in four] + [R.gen(GEN, "random_V10_0", "textvqa"), R.gen(GEN, "random_V10_0", "caption")]
        else:
            vals = [R.acc(run, cond, p) for p in four] + [R.gen(gen, cond, "textvqa"), R.gen(gen, cond, "caption")]
        rows.append((label, vals))
    return ["Removed set", "POPE", "MMVP", "V*", "ZoomBench", "TVQA", "NLL"], rows, [1, 1, 1, 1, 1, 2]


def by_probe(R, run, gen_run, cols):
    """Rows = nine probes, caption NLL and greedy TextVQA. cols: (label, lesion conditions to average, generation condition)."""
    rows = [(name, [mean(R.acc(run, c, p) for c in conds) for _, conds, _ in cols]) for p, name in PROBES]
    rows.append(("Caption NLL", [R.gen(gen_run, g, "caption") for _, _, g in cols], 2))
    rows.append(("TextVQA (generated)", [R.gen(gen_run, g, "textvqa") for _, _, g in cols]))
    return ["Probe"] + [c[0] for c in cols], rows


def table_s3(R):
    cols = [("none", ["none"], "none"), ("vision-exclusive [V10]", ["V10_reroute"], "V10_reroute"),
            ("least-used [LOWLOAD]", ["LOWLOAD_reroute"], "LOWLOAD_reroute"),
            ("random [random_V10_0, 1]", ["random_V10_0", "random_V10_1"], "random_V10_0"), ("union [UNION]", ["UNION_reroute"], "UNION_reroute"),
            ("random [random_UNION_0]", ["random_UNION_0"], "random_UNION_0"), ("affinity-462 [EL462]", ["EL462_reroute"], "EL462_reroute")]
    return by_probe(R, f"{Q}/ablation_prune", f"{Q}/generation", cols)


def table_s4(R):
    cols = [("none", ["none"], "none"), ("vision-exclusive [V10]", ["V10_reroute"], "V10_reroute"), ("random [random_V10_0]", ["random_V10_0"], "random_V10_0"),
            ("least-used [LOWLOAD]", ["LOWLOAD_reroute"], "LOWLOAD_reroute"), ("top-16, excl. part [TOPIMG16_V10part]", ["TOPIMG16_V10part_reroute"], "TOPIMG16_V10part_reroute"),
            ("top-16, dual-use part [TOPIMG16_noV10]", ["TOPIMG16_noV10_reroute"], "TOPIMG16_noV10_reroute")]
    return by_probe(R, f"{G}/ablation", f"{G}/generation2", cols)


SCOPE_HEAD = ["Lesion"] + [n for _, n in PROBES] + ["Mean 5"]


def table_s5(R):
    S, gens = f"{Q}/ablation_scope", [f"{Q}/generation_scope", f"{Q}/generation_scope2"]
    spec = [("dual-use part, all", "TOPIMG16_noV10_reroute"), ("dual-use part, image", "TOPIMG16_noV10_reroute_image"),
            ("dual-use part, text", "TOPIMG16_noV10_reroute_text"), ("dual-use part, image, zeroed", "TOPIMG16_noV10_zero_image"),
            ("vision-exclusive, image", "V10_reroute_image"), ("vision-exclusive, text", "V10_reroute_text"),
            ("above uniform, image", "SHARED_reroute_image"), ("above uniform, text", "SHARED_reroute_text"),
            ("image top-16, image", "TOPIMG16_reroute_image"), ("image top-32, image", "TOPIMG32_reroute_image"),
            ("image top-16, text", "TOPIMG16_reroute_text"), ("every expert, image, zeroed", "ALL_zero_image"),
            ("every expert, text, zeroed", "ALL_zero_text")]
    rows = [(f"{label} [{c}]", R.delta(S, c) + [R.gen_delta(gens, c, "caption"), R.gen_delta(gens, c, "textvqa")]) for label, c in spec]
    return SCOPE_HEAD + ["Cap. NLL", "TextVQA gen."], rows, [1] * 10 + [2, 1]


def table_s6(R):
    rows = []
    for model, run, run2 in (("Kimi-VL", f"{K}/ablation_scope", f"{K}/ablation_scope2"), ("Gemma-4", f"{G}/ablation_scope", f"{G}/ablation_scope2")):
        block = [(f"{model}: {label} [{c}]", R.delta(r, c)) for label, r, c in (
            ("dual-use part, all", run, "TOPIMG16_noV10_reroute"), ("dual-use part, image", run, "TOPIMG16_noV10_reroute_image"),
            ("dual-use part, text", run, "TOPIMG16_noV10_reroute_text"), ("vision-exclusive, image", run, "V10_reroute_image"),
            ("vision-exclusive, text", run, "V10_reroute_text"), ("above uniform, image", run, "SHARED_reroute_image"),
            ("above uniform, text", run, "SHARED_reroute_text"), ("every expert, image, zeroed", run, "ALL_zero_image"),
            ("every expert, text, zeroed", run2, "ALL_zero_text"))]
        rows += block
        if model == "Kimi-VL":
            KB = f"{K}/ablation_scope_bias0"
            rows += [(f"Kimi-VL, no bias: {label} [{c}]", R.delta(KB, c)) for label, c in (
                ("dual-use part, image", "TOPIMG16_noV10_reroute_image"), ("dual-use part, text", "TOPIMG16_noV10_reroute_text"),
                ("vision-exclusive, image", "V10nb_reroute_image"), ("vision-exclusive, text", "V10nb_reroute_text"),
                ("every expert, image, zeroed", "ALL_zero_image"))]
    return SCOPE_HEAD, rows


def spread(base, run):
    """Effective experts per layer for image tokens (mean over layers of the exponential of the routing entropy), share
    of image routing on the 16 highest-load experts per layer (%), share of experts that are vision-exclusive (%)."""
    a = Path(base) / run / "analysis"
    if not (a / "counts.npz").exists() or not (a / "substrate_ratio10.json").exists():
        return [None, None, None]
    c = np.load(a / "counts.npz")["mm__image"].astype(float); p = c / np.maximum(c.sum(1, keepdims=True), 1)
    h = -(p * np.log(np.where(p > 0, p, 1.0))).sum(1)
    top = np.sort(p, axis=1)[:, ::-1][:, :16].sum(1).mean()
    v = np.array(json.load(open(a / "substrate_ratio10.json"))["mask"], dtype=bool)
    return [float(np.exp(h).mean()), 100 * float(top), 100 * float(v.mean())]


def five_probe(R):
    """Five-probe mean change per routing regime: vision-exclusive lesion at all positions, size-matched random control,
    every expert zeroed at image positions and at text positions."""
    m5 = lambda run, cond: R.delta(run, cond)[-1]
    qrand = [(f"{Q}/ablation_prune", "random_V10_0"), (f"{Q}/ablation_prune", "random_V10_1"), (f"{Q}/ablation_scope", "random_V10_2"),
             (f"{Q}/ablation_scope", "random_V10_3"), (f"{Q}/ablation_scope", "random_V10_4")]
    return [("Qwen3-VL", Q, [m5(f"{Q}/ablation2", "V10_reroute"), mean(m5(r, c) for r, c in qrand),
                             m5(f"{Q}/ablation_scope", "ALL_zero_image"), m5(f"{Q}/ablation_scope", "ALL_zero_text")]),
            ("Kimi-VL", K, [m5(f"{K}/ablation", "V10_reroute"), m5(f"{K}/ablation", "random_V10_0"),
                            m5(f"{K}/ablation_scope", "ALL_zero_image"), m5(f"{K}/ablation_scope2", "ALL_zero_text")]),
            ("Kimi-VL, no bias", KNB, [m5(f"{K}/ablation_bias0", "V10nb_reroute"), None, m5(f"{K}/ablation_scope_bias0", "ALL_zero_image"), None]),
            ("Gemma-4", G, [m5(f"{G}/ablation", "V10_reroute"), m5(f"{G}/ablation", "random_V10_0"),
                            m5(f"{G}/ablation_scope", "ALL_zero_image"), m5(f"{G}/ablation_scope2", "ALL_zero_text")])]


def table2(R):
    rows = [(name, spread(R.base, run) + [v[0], v[2]]) for name, run, v in five_probe(R)]
    return ["Regime", "N_eff", "Top-16 (%)", "Excl. (%)", "Lesion", "Silenced"], rows, [0, 0, 1, 1, 1], [False, False, False, True, True]


def fmt(x, nd=1, signed=False):
    if x is None:
        return "n/a"
    q = Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)     # ties round away from zero
    return f"{0:.{nd}f}" if q == 0 else (f"{q:+f}" if signed else f"{q:f}")


def show(title, header, rows, nd=None, signed=False):
    n = len(header) - 1
    nd = nd or [1] * n; signed = signed if isinstance(signed, list) else [signed] * n
    cells = [[r[0]] + [fmt(v, r[2] if len(r) > 2 else d, s) for v, d, s in zip(r[1], nd, signed)] for r in rows]   # r[2]: decimals of that row
    w = [max(len(r[i]) for r in [header] + cells) for i in range(n + 1)]
    print("\n" + title)
    for r in [header] + cells:
        print("  " + r[0].ljust(w[0]) + "".join("  " + c.rjust(w[i]) for i, c in enumerate(r[1:], 1)))


def main():
    ap = argparse.ArgumentParser(description="Print Table 1, Table 2, Tables S3 to S6 and the five-probe means of the paper.")
    ap.add_argument("runs", nargs="?", default=str(paths.RUNS), help="runs directory (default: MOEDEP_RUNS)")
    R = Results(ap.parse_args().runs)
    show("Table 1. Lesions on Qwen3-VL: accuracy (%), greedy TextVQA soft accuracy (TVQA, %), caption NLL", *table1(R))
    show("Table 2. Routing spread, exclusivity and five-probe change (points) of the vision-exclusive lesion and of every "
         "expert zeroed at image positions", *table2(R))
    show("Table S3. Pruning by routing affinity on Qwen3-VL: accuracy (%), re-routed sets", *table_s3(R))
    show("Table S4. Lesions on Gemma-4: accuracy (%), re-routed sets", *table_s4(R))
    show("Table S5. Scoped lesions on Qwen3-VL: change (points) at all, image or text positions", *table_s5(R), signed=True)
    show("Table S6. Scoped lesions on Kimi-VL, Kimi-VL without its balancing bias, and Gemma-4: change (points)", *table_s6(R), signed=True)
    show("Five-probe means: change (points) against the intact model",
         ["Regime", "vision-exclusive lesion", "random control", "every expert zeroed, image", "every expert zeroed, text"],
         [(name, v) for name, _, v in five_probe(R)], signed=True)


if __name__ == "__main__":
    main()
