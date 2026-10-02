"""Aggregate the per-run tables into the numbers quoted by the figures, report and README."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from mtl_eurosat import metrics
from mtl_eurosat.config import RESULTS
from mtl_eurosat.experiment import HEADLINE, VARIANTS

# The single-seed run of the original notebook (notebooks/pierre/results/metrics_summary.json).
NOTEBOOK = Path(__file__).resolve().parents[2] / "notebooks/pierre/results/metrics_summary.json"


def _num(v: str) -> float | int | str:
    if v == "":
        return float("nan")
    try:
        f = float(v)
    except ValueError:
        return v
    return int(f) if f.is_integer() and "." not in v else f


def read_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return [{k: _num(v) for k, v in row.items()} for row in csv.DictReader(f)]


def runs(results: Path = RESULTS) -> list[dict]:
    return read_csv(results / "runs.csv")


def column(rows: list[dict], variant: str, key: str) -> np.ndarray:
    """``key`` for every seed of ``variant``, ordered by seed."""
    sel = sorted((r for r in rows if r["variant"] == variant), key=lambda r: r["seed"])
    return np.array([r.get(key, np.nan) for r in sel], dtype=float)


def summary(rows: list[dict]) -> list[dict]:
    """One line per variant: mean and 95% CI across seeds of every metric."""
    out = []
    for name, v in VARIANTS.items():
        if not any(r["variant"] == name for r in rows):
            continue
        line: dict = {"variant": name, "label": v.label, "n_seeds": len(column(rows, name, "seed"))}
        line["n_params"] = int(column(rows, name, "n_params")[0])
        for key in (
            "val_acc",
            "ood_acc",
            "ood_auroc",
            "ood_acc_Forest",
            "ood_acc_DenseResidential",
            "ood_acc_MediumResidential",
            "val_recon_mse",
            "ood_recon_mse",
            "val_acc_masked",
            "ood_acc_masked",
            "ood_auroc_masked",
            "ood_acc_Forest_masked",
        ):
            values = column(rows, name, key)  # NaN where a metric does not apply
            if np.isnan(values).all():
                continue
            m, lo, hi = metrics.mean_ci(values)
            line[key] = {
                "mean": m,
                "lo": lo,
                "hi": hi,
                "min": float(values.min()),
                "max": float(values.max()),
            }
        out.append(line)
    return out


def comparisons(rows: list[dict]) -> dict[str, dict]:
    """Paired (same seed = same split) differences in OOD accuracy for the questions asked."""

    def paired(a: str, b: str, key: str = "ood_acc") -> dict:
        seeds = sorted(
            set(column(rows, a, "seed").astype(int)) & set(column(rows, b, "seed").astype(int))
        )
        va = np.array([_get(rows, a, s, key) for s in seeds])
        vb = np.array([_get(rows, b, s, key) for s in seeds])
        return {**metrics.paired_difference(va, vb), "n": len(seeds)}

    return {
        # Effect of the auxiliary loss, architecture held fixed.
        "aux_loss": paired("hard_a0.6", "hard_a1.0"),
        "aux_loss_forest": paired("hard_a0.6", "hard_a1.0", "ood_acc_Forest"),
        # What the course project compared: MTL against the small single-task CNN.
        "mtl_vs_cnn": paired("hard_a0.6", "cnn"),
        # Effect of the architecture alone, no reconstruction.
        "arch": paired("hard_a1.0", "cnn"),
    }


def _get(rows: list[dict], variant: str, seed: int, key: str) -> float:
    return next(r[key] for r in rows if r["variant"] == variant and r["seed"] == seed)


def notebook_run() -> dict[str, float]:
    """OOD accuracies of the original single-seed notebook run."""
    raw = json.loads(NOTEBOOK.read_text())
    return {name: raw[name]["acc"] for name in ("baseline", "mtl", "softshare")}


def write_summary(results: Path = RESULTS) -> dict:
    rows = runs(results)
    out = {
        "variants": summary(rows),
        "comparisons": comparisons(rows),
        "notebook_single_seed": notebook_run(),
        "headline": list(HEADLINE),
    }
    (results / "summary.json").write_text(json.dumps(out, indent=2))
    return out
