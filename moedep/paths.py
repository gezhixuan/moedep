"""Where datasets, model weights and run outputs live.

Each location is read from an environment variable. The defaults are folders in the repository root.

    MOEDEP_DATA    datasets, one folder per dataset (layout in the README)      default: data/
    MOEDEP_MODELS  model weights, one folder per model                           default: models/
    MOEDEP_RUNS    outputs of every run                                          default: runs/
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _dir(var, default):
    return Path(os.environ.get(var) or REPO / default)


DATA = _dir("MOEDEP_DATA", "data")
MODELS = _dir("MOEDEP_MODELS", "models")
RUNS = _dir("MOEDEP_RUNS", "runs")
