"""The multi-seed grid: every variant, every seed, scored in- and out-of-distribution.

Each (variant, seed) run is cached as JSON in ``data/interim/runs/`` so an interrupted
grid resumes where it stopped; :func:`collect` then writes the small tables under
``results/`` that the figures, the report and the README are built from.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from mtl_eurosat import data, metrics
from mtl_eurosat.config import ALPHAS, INTERIM, OOD_GROUPS, RESULTS, SEEDS, VAL_FRACTION
from mtl_eurosat.models import n_parameters
from mtl_eurosat.train import Variant, fit, mask_pixels, predict

RUNS = INTERIM / "runs"
N_EXAMPLES = 8  # images per reconstruction strip


def _alpha_name(a: float) -> str:
    return f"hard_a{a:.1f}"


VARIANTS: dict[str, Variant] = {
    "cnn": Variant("cnn", "cnn", label="Single-task CNN"),
    **{
        _alpha_name(a): Variant(_alpha_name(a), "hard", alpha=a, label=f"Hard sharing, α={a:.1f}")
        for a in ALPHAS
    },
    # The notebook's soft-sharing settings (alpha, beta, gamma, mask ratio), unchanged.
    "soft": Variant(
        "soft", "soft", alpha=0.5, beta=0.005, gamma=0.01, mask_ratio=0.3, label="Soft sharing"
    ),
}
# The comparison that answers the question gets the full set of seeds; the alpha sweep
# around it gets half, which is enough to see a trend and halves the compute.
HEADLINE = ("cnn", _alpha_name(1.0), _alpha_name(0.6), "soft")


def seeds_for(name: str) -> tuple[int, ...]:
    return SEEDS if name in HEADLINE else SEEDS[: len(SEEDS) // 2]


def pick_device(device: str) -> torch.device:
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    return torch.device(device)


def run_one(v: Variant, seed: int, images: data.Images, device: torch.device, epochs: int) -> dict:
    train_idx, val_idx = data.stratified_split(images.y_id, VAL_FRACTION, seed)
    x = data.to_tensor(images.x_id).to(device)
    y = torch.from_numpy(images.y_id).to(device)
    x_ood = data.to_tensor(images.x_ood).to(device)
    t_train, t_val = torch.from_numpy(train_idx).to(device), torch.from_numpy(val_idx).to(device)
    result = fit(
        v, x[t_train], y[t_train], x[t_val], y[t_val], seed, x_ood, images.y_ood, epochs=epochs
    )
    model = result.model
    p_val, mse_val = predict(model, x[t_val])
    p_ood, mse_ood = predict(model, x_ood)
    y_val = images.y_id[val_idx]
    row = {
        "variant": v.name,
        "seed": seed,
        "n_params": n_parameters(model),
        "best_epoch": result.best_epoch,
        "seconds": round(result.seconds, 1),
        "val_acc": metrics.accuracy(y_val, p_val),
        "val_recon_mse": float(np.nanmean(mse_val)) if not np.isnan(mse_val).all() else None,
        "ood_acc": metrics.accuracy(images.y_ood, p_ood),
        "ood_auroc": metrics.auroc(images.y_ood, p_ood),
        "ood_recon_mse": float(np.nanmean(mse_ood)) if not np.isnan(mse_ood).all() else None,
    }
    for group in OOD_GROUPS:
        sel = images.group_ood == group
        row[f"ood_acc_{group}"] = metrics.accuracy(images.y_ood[sel], p_ood[sel])
    if v.mask_ratio > 0:
        # The soft-sharing model only ever sees masked images during training (as in the
        # notebook), so it is also scored on masked images: one fixed mask per seed.
        gen = torch.Generator().manual_seed(seed)
        pm_val, _ = predict(model, mask_pixels(x[t_val], v.mask_ratio, gen)[0])
        pm_ood, _ = predict(model, mask_pixels(x_ood, v.mask_ratio, gen)[0])
        forest = images.group_ood == "Forest"
        row["val_acc_masked"] = metrics.accuracy(y_val, pm_val)
        row["ood_acc_masked"] = metrics.accuracy(images.y_ood, pm_ood)
        row["ood_auroc_masked"] = metrics.auroc(images.y_ood, pm_ood)
        row["ood_acc_Forest_masked"] = metrics.accuracy(images.y_ood[forest], pm_ood[forest])
    out = {"summary": row, "history": result.history, "p_ood": p_ood.round(5).tolist()}
    if seed == 0 and v.arch != "cnn" and v.alpha < 1.0:
        out["examples"] = _examples(model, x[t_val], x_ood)
    return out


def _examples(model, x_val: torch.Tensor, x_ood: torch.Tensor) -> dict[str, list]:
    """Reconstructions of fixed validation and OOD images, as uint8 lists."""
    idx_val = torch.linspace(0, len(x_val) - 1, N_EXAMPLES).long()
    idx_ood = torch.linspace(0, len(x_ood) - 1, N_EXAMPLES).long()
    with torch.no_grad():
        r_val = model(x_val[idx_val]).recon
        r_ood = model(x_ood[idx_ood]).recon
    return {
        "val_in": data.to_uint8(x_val[idx_val].cpu()).tolist(),
        "val_out": data.to_uint8(r_val.cpu()).tolist(),
        "ood_in": data.to_uint8(x_ood[idx_ood].cpu()).tolist(),
        "ood_out": data.to_uint8(r_ood.cpu()).tolist(),
    }


def run_grid(
    names: list[str] | None = None, device: str = "auto", epochs: int | None = None
) -> None:
    from mtl_eurosat.config import EPOCHS

    images = data.load()
    dev = pick_device(device)
    RUNS.mkdir(parents=True, exist_ok=True)
    for name in names or list(VARIANTS):
        for seed in seeds_for(name):
            path = RUNS / f"{name}_s{seed}.json"
            if path.exists():
                continue
            out = run_one(VARIANTS[name], seed, images, dev, epochs or EPOCHS)
            path.write_text(json.dumps(out))
            s = out["summary"]
            print(
                f"{name:<10} seed {seed}  val {s['val_acc']:.4f}  ood {s['ood_acc']:.3f}"
                f"  auroc {s['ood_auroc']:.3f}  forest {s['ood_acc_Forest']:.3f}"
                f"  ({s['seconds']:.0f}s)",
                flush=True,
            )


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(k for row in rows for k in row))  # union, first-seen order
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def collect(runs: Path = RUNS, results: Path = RESULTS, images: data.Images | None = None) -> None:
    """Cached runs -> ``results/{runs,history,ood_scores}.csv`` and reconstruction examples."""
    files = sorted(runs.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no runs in {runs}; run `mtl-eurosat run` first")
    images = images or data.load()
    summary, history, scores, examples = [], [], [], {}
    for path in files:
        out = json.loads(path.read_text())
        s = out["summary"]
        summary.append(s)
        history += [{"variant": s["variant"], "seed": s["seed"], **h} for h in out["history"]]
        scores += [
            {
                "variant": s["variant"],
                "seed": s["seed"],
                "image": i,
                "group": g,
                "label": int(y),
                "p_residential": p,
            }
            for i, (g, y, p) in enumerate(
                zip(images.group_ood, images.y_ood, out["p_ood"], strict=True)
            )
        ]
        if "examples" in out:
            examples[s["variant"]] = out["examples"]
    order = {name: i for i, name in enumerate(VARIANTS)}
    summary.sort(key=lambda r: (order[r["variant"]], r["seed"]))
    results.mkdir(parents=True, exist_ok=True)
    _write_csv(results / "runs.csv", summary)
    _write_csv(results / "history.csv", history)
    _write_csv(results / "ood_scores.csv", scores)
    np.savez_compressed(
        results / "reconstructions.npz",
        **{
            f"{v}__{k}": np.asarray(a, dtype=np.uint8)
            for v, ex in examples.items()
            for k, a in ex.items()
        },
    )
    print(f"collected {len(summary)} runs into {results}")
