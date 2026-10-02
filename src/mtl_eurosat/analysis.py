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

    def paired(a: str, b: str, key: str = "ood_acc", key_a: str | None = None) -> dict:
        seeds = sorted(
            set(column(rows, a, "seed").astype(int)) & set(column(rows, b, "seed").astype(int))
        )
        if not seeds:  # a variant that was not run
            return {}
        va = np.array([_get(rows, a, s, key_a or key) for s in seeds])
        vb = np.array([_get(rows, b, s, key) for s in seeds])
        return {**metrics.paired_difference(va, vb), "n": len(seeds)}

    return {
        # Effect of the auxiliary loss, architecture held fixed.
        "aux_loss": paired("hard_a0.6", "hard_a1.0"),
        "aux_loss_forest": paired("hard_a0.6", "hard_a1.0", "ood_acc_Forest"),
        "aux_loss_auroc": paired("hard_a0.6", "hard_a1.0", "ood_auroc"),
        # What the course project compared: MTL against the small single-task CNN.
        "mtl_vs_cnn": paired("hard_a0.6", "cnn"),
        "mtl_vs_cnn_auroc": paired("hard_a0.6", "cnn", "ood_auroc"),
        # Effect of the architecture alone, no reconstruction.
        "arch": paired("hard_a1.0", "cnn"),
        # Soft sharing scored as trained (masked inputs) vs the plain CNN ...
        "soft_masked_vs_cnn": paired("soft", "cnn", key_a="ood_acc_masked"),
        "soft_masked_vs_cnn_auroc": paired("soft", "cnn", "ood_auroc", key_a="ood_auroc_masked"),
        # ... and the same masking without any reconstruction.
        "cnn_masked_vs_cnn": paired("cnn_masked", "cnn", key_a="ood_acc_masked"),
        "cnn_masked_vs_cnn_auroc": paired(
            "cnn_masked", "cnn", "ood_auroc", key_a="ood_auroc_masked"
        ),
        # Soft sharing against the masked-input CNN: same inputs, only the model differs.
        "soft_masked_vs_cnn_masked": paired("soft", "cnn_masked", "ood_acc_masked"),
        "soft_masked_vs_cnn_masked_auroc": paired("soft", "cnn_masked", "ood_auroc_masked"),
        # Masking as plain augmentation: the masked-input CNN scored on clean images.
        "cnn_masked_clean_vs_cnn": paired("cnn_masked", "cnn"),
        "cnn_masked_clean_vs_cnn_forest": paired("cnn_masked", "cnn", "ood_acc_Forest"),
    }


# Each model scored on the kind of input it was trained on: the masked-input models
# on masked images, the others on clean ones.
AS_TRAINED = {"soft": "_masked", "cnn_masked": "_masked"}


def as_trained(rows: list[dict], variant: str, key: str) -> np.ndarray:
    """``key`` (e.g. ``val_acc``) for ``variant``, on the inputs the model was trained on."""
    return column(rows, variant, key + AS_TRAINED.get(variant, ""))


def as_trained_range(rows: list[dict]) -> dict[str, float]:
    """Worst EuroSAT validation run and the span of AID mean accuracies, every model as trained."""
    names = [n for n in VARIANTS if any(r["variant"] == n for r in rows)]
    val = [as_trained(rows, n, "val_acc").min() for n in names]
    ood = [as_trained(rows, n, "ood_acc").mean() for n in names]
    return {"val_min": float(min(val)), "ood_lo": float(min(ood)), "ood_hi": float(max(ood))}


def epoch_swing(history: list[dict], variant: str, key: str, first_epoch: int = 3) -> float:
    """Median over seeds of the range (max - min) of ``key`` across epochs >= ``first_epoch``.

    It measures how much a metric moves within one training run once training has settled,
    i.e. how much the choice of checkpoint alone can change it.
    """
    spans = []
    for seed in sorted({h["seed"] for h in history if h["variant"] == variant}):
        v = [
            h[key]
            for h in history
            if h["variant"] == variant and h["seed"] == seed and h["epoch"] >= first_epoch
        ]
        spans.append(max(v) - min(v))
    return float(np.median(spans)) if spans else float("nan")


def _get(rows: list[dict], variant: str, seed: int, key: str) -> float:
    return next(r.get(key, np.nan) for r in rows if r["variant"] == variant and r["seed"] == seed)


def edge_strength(x: np.ndarray) -> float:
    """Mean absolute grey-level step between horizontal neighbours (0-255 scale).

    A crude texture measure: smooth canopy scores low, roofs and streets score high.
    """
    grey = x.astype(float).mean(-1)
    return float(np.abs(np.diff(grey, axis=2)).mean())


def shift_stats(images) -> dict[str, float]:
    """Edge strength per image group: where AID forests sit between the training classes."""
    groups = {
        "eurosat_forest": images.x_id[images.y_id == 0],
        "eurosat_residential": images.x_id[images.y_id == 1],
        "aid_forest": images.x_ood[images.group_ood == "Forest"],
        "aid_residential": images.x_ood[images.y_ood == 1],
    }
    return {k: edge_strength(v) for k, v in groups.items()}


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
