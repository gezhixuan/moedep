"""Print paired 95% bootstrap intervals for the lesions of one run folder. CPU only.

Usage: python -m moedep.intervals RUN_FOLDER [--cond NAME ...]        e.g. $MOEDEP_RUNS/qwen3vl/ablation_scope

For every condition in results.jsonl of a moedep.run_ablation folder, prints the change in accuracy (points) against
the intact model ("none") on each probe with a percentile interval from 2,000 resamples of the items, each item keeping
its pair of outcomes. The mean over the five main probes, and over the eight image probes when the folder holds them,
resamples the items of each probe independently. A star marks an interval that excludes zero. These are the intervals
of Fig. 3 and Fig. S1 and the ones quoted in the text.
"""
import argparse
from pathlib import Path

import numpy as np

from moedep.tables import PROBES, Results

N_BOOT, SEED = 2000, 0


def paired(cond, base):
    """Per-item differences (points) of two {uid: correct} dicts over their shared items."""
    return np.array([100 * (cond[u] - base[u]) for u in sorted(set(cond) & set(base))])


def interval(diffs):
    """Mean and 95% interval of the unweighted mean over the given probes (one array of paired differences each)."""
    rng = np.random.default_rng(SEED); boots = np.zeros(N_BOOT)
    for d in diffs:
        boots += d[rng.integers(0, len(d), size=(N_BOOT, len(d)))].mean(1)
    lo, hi = np.percentile(boots / len(diffs), [2.5, 97.5])
    return float(np.mean([d.mean() for d in diffs])), float(lo), float(hi)


def cell(diffs):
    if not diffs or any(len(d) == 0 for d in diffs):
        return "n/a"
    m, lo, hi = interval(diffs)
    return f"{m:+.1f} [{lo:+.1f}, {hi:+.1f}]" + ("*" if lo > 0 or hi < 0 else " ")


def main():
    ap = argparse.ArgumentParser(description="Paired 95% bootstrap intervals of the lesions in one run_ablation folder.")
    ap.add_argument("run", help="folder with results.jsonl")
    ap.add_argument("--cond", action="append", help="condition name as in results.jsonl (default: all)")
    a = ap.parse_args()
    run = Path(a.run).resolve(); t = Results(run.parent).lesion(run.name)
    conds = a.cond or sorted({c for c, _ in t} - {"none"})
    probes = [(p, n) for p, n in PROBES if ("none", p) in t]
    means = [("Mean 5", [p for p, _ in PROBES[:5]]), ("Mean 8", [p for p, _ in PROBES[:8]])]
    means = [(n, ps) for n, ps in means if all(("none", p) in t for p in ps)]
    for c in conds:
        print("\n" + c)
        d = {p: paired(t.get((c, p), {}), t[("none", p)]) for p, _ in probes}
        for p, n in probes:
            print(f"  {n:10s} {cell([d[p]])}")
        for n, ps in means:
            print(f"  {n:10s} {cell([d[p] for p in ps])}")


if __name__ == "__main__":
    main()
