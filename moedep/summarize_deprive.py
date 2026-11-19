"""Summarize deprivation and re-exposure runs: metrics.jsonl per run -> markdown tables of the key quantities vs step.
Usage: python -m moedep.summarize_deprive [PATTERN]     (run folders under MOEDEP_RUNS, default "deprive_*")"""
import json, sys
from pathlib import Path
import numpy as np
from moedep import paths

RUNS = paths.RUNS
KEYS = [("heldout_nll", "held-out text NLL"), ("arc_acc", "ARC-Easy acc"),
        ("alpaca_text_on_V10", "Alpaca text load on V10"), ("alpaca_text_on_TOPIMG16_noV10", "Alpaca text load on shared-part"),
        ("mm_image_on_V10", "image load on V10"), ("mm_image_on_TOPIMG16_noV10", "image load on shared-part"),
        ("vis_pope", "POPE"), ("vis_mmvp", "MMVP"), ("vis_vstar", "V*"), ("vis_textvqa", "TextVQA"),
        ("delta_V10", "LoRA delta V10"), ("delta_TOPIMG16_noV10", "LoRA delta shared-part"), ("delta_T10", "LoRA delta T10")]


def main(pattern):
    runs = sorted(p for p in RUNS.glob(pattern) if (p / "metrics.jsonl").exists())
    out = []
    for run in runs:
        rows = [json.loads(l) for l in open(run / "metrics.jsonl") if l.strip()]
        steps = [r["step"] for r in rows]
        out.append(f"\n### {run.name} ({len(rows)} evals, last step {steps[-1]})\n")
        out.append("| metric | " + " | ".join(str(s) for s in steps) + " |"); out.append("|---|" + "---|" * len(steps))
        for k, name in KEYS:
            if k in rows[0]:
                out.append(f"| {name} | " + " | ".join(f"{r[k]:.3f}" for r in rows) + " |")
    txt = "\n".join(out); print(txt)
    (RUNS / "deprive_summary.md").write_text(txt)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "deprive_*")
