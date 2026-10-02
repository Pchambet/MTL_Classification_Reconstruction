"""Static figures (``docs/figures/*.png``), all drawn from ``results/``."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from mtl_eurosat import analysis, data
from mtl_eurosat.config import AMBER, FIGURES, GRID, INK, RESULTS, SLATE, TEAL

SHORT = {
    "cnn": "Single-task\nCNN",
    "hard_a1.0": "Hard sharing\nα = 1 (no recon.)",
    "hard_a0.6": "Hard sharing\nα = 0.6",
    "soft": "Soft sharing",
}
GROUPS = {
    "Forest": "AID forest",
    "DenseResidential": "AID dense\nresidential",
    "MediumResidential": "AID medium\nresidential",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 10,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "font.size": 10,
            "text.color": INK,
            "legend.frameon": False,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
        }
    )


def _save(fig: plt.Figure, name: str, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    fig.savefig(path)
    plt.close(fig)
    return path


def _pct(ax: plt.Axes) -> None:
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))


def _strip(ax, x: float, values: np.ndarray, colour: str, width: float = 0.12) -> None:
    """Seeds as dots, mean as a bar, 95% CI as a vertical line."""
    from mtl_eurosat.metrics import mean_ci

    jitter = np.linspace(-width, width, len(values)) if len(values) > 1 else np.zeros(1)
    ax.scatter(x + jitter, values, s=16, color=colour, alpha=0.55, linewidths=0, zorder=3)
    m, lo, hi = mean_ci(values)
    ax.plot([x - 1.6 * width, x + 1.6 * width], [m, m], color=colour, lw=2.4, zorder=4)
    if not np.isnan(lo):
        ax.plot([x, x], [lo, hi], color=colour, lw=1.2, zorder=4)


# (variant, validation metric, AID metric, label)
HERO = (
    ("cnn", "val_acc", "ood_acc", SHORT["cnn"]),
    ("hard_a1.0", "val_acc", "ood_acc", SHORT["hard_a1.0"]),
    ("hard_a0.6", "val_acc", "ood_acc", SHORT["hard_a0.6"]),
    ("soft", "val_acc", "ood_acc", "Soft sharing\nclean inputs"),
    ("soft", "val_acc_masked", "ood_acc_masked", "Soft sharing\nmasked inputs"),
)


def hero(rows: list[dict], out: Path, title: str) -> Path:
    """In-distribution vs AID accuracy for the headline variants, every seed shown."""
    nb = analysis.notebook_run()
    notebook = {("cnn", "ood_acc"): nb["baseline"], ("hard_a0.6", "ood_acc"): nb["mtl"]}
    notebook[("soft", "ood_acc")] = nb["softshare"]
    cols = [c for c in HERO if any(r["variant"] == c[0] for r in rows)]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for i, (name, val_key, ood_key, _) in enumerate(cols):
        _strip(ax, i - 0.2, analysis.column(rows, name, val_key), SLATE)
        _strip(ax, i + 0.2, analysis.column(rows, name, ood_key), TEAL)
        if (name, ood_key) in notebook:
            ax.scatter(i + 0.2, notebook[(name, ood_key)], marker="D", s=34, color=AMBER, zorder=5)
    ax.set_xticks(range(len(cols)), [c[3] for c in cols])
    ax.set_ylim(0.4, 1.04)
    ax.axhline(0.5, color=SLATE, lw=0.8, ls=":")
    ax.text(len(cols) - 0.45, 0.505, "chance", color=SLATE, fontsize=8, ha="right", va="bottom")
    _pct(ax)
    ax.set_ylabel("accuracy")
    ax.set_xlim(-0.6, len(cols) - 0.4)
    ax.text(-0.2, 1.013, "EuroSAT validation", color=SLATE, fontsize=8.5, ha="center")
    ax.text(0.2, 0.42, "AID, out of distribution", color=TEAL, fontsize=8.5, ha="left")
    ax.scatter([], [], marker="D", s=30, color=AMBER, label="original single-seed notebook run")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.set_title(title, fontsize=11.5)
    return _save(fig, "hero.png", out)


def alpha_sweep(rows: list[dict], out: Path) -> Path:
    s = {line["variant"]: line for line in analysis.summary(rows)}
    alphas = sorted((float(n.split("_a")[1]) for n in s if n.startswith("hard_a")), reverse=True)
    names = [f"hard_a{a:.1f}" for a in alphas]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1.6, 1]})
    for key, colour, label in (
        ("ood_acc", TEAL, "AID, all 120 images"),
        ("ood_acc_Forest", AMBER, "AID forest only"),
    ):
        m = np.array([s[n][key]["mean"] for n in names])
        lo = np.array([s[n][key]["lo"] for n in names])
        hi = np.array([s[n][key]["hi"] for n in names])
        ax.fill_between(alphas, lo, hi, color=colour, alpha=0.15, lw=0)
        ax.plot(alphas, m, "-o", color=colour, ms=4, lw=2)
        ax.text(alphas[-1] - 0.02, m[-1], label, color=colour, fontsize=8.5, va="center")
        base = s["cnn"][key]["mean"]
        ax.axhline(base, color=colour, lw=1, ls="--", alpha=0.7)
    ax.text(
        alphas[0], s["cnn"]["ood_acc"]["mean"] + 0.012, "single-task CNN", color=SLATE, fontsize=8
    )
    ax.set_xlim(alphas[0] + 0.05, alphas[-1] - 0.32)
    ax.set_xlabel("α (weight on classification; 1 − α on reconstruction)")
    ax.set_ylabel("out-of-distribution accuracy")
    _pct(ax)
    first, last = s[names[0]]["ood_acc"]["mean"], s[names[-1]]["ood_acc"]["mean"]
    ax.set_title(
        f"AID accuracy, α = {alphas[0]:g} → {alphas[-1]:g}: {first:.1%} → {last:.1%}", fontsize=11
    )
    # alpha = 1 trains no decoder: its reconstruction error is meaningless and left out.
    rec_alphas = [a for a in alphas if a < 1]
    mse = [s[f"hard_a{a:.1f}"]["val_recon_mse"]["mean"] for a in rec_alphas]
    ax2.plot(rec_alphas, mse, "-o", color=SLATE, ms=4, lw=2)
    ax2.set_xlim(alphas[0] + 0.05, alphas[-1] - 0.05)
    ax2.set_xlabel("α")
    ax2.set_ylabel("validation reconstruction MSE\n(pixels in [−1, 1])")
    ax2.set_title(
        f"Reconstruction MSE, α = {rec_alphas[0]:g} → {rec_alphas[-1]:g}:"
        f" {mse[0]:.4f} → {mse[-1]:.4f}",
        fontsize=11,
    )
    fig.tight_layout()
    return _save(fig, "alpha_sweep.png", out)


def per_group(rows: list[dict], out: Path) -> Path:
    s = {line["variant"]: line for line in analysis.summary(rows)}
    names = [n for n in SHORT if n in s]
    colours = [SLATE, "#94a3b8", TEAL, AMBER]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    width = 0.8 / len(names)
    for j, name in enumerate(names):
        for i, g in enumerate(GROUPS):
            v = s[name][f"ood_acc_{g}"]
            x = i + (j - (len(names) - 1) / 2) * width
            ax.bar(
                x,
                v["mean"],
                width * 0.92,
                color=colours[j],
                label=SHORT[name].replace("\n", " ") if i == 0 else None,
            )
            ax.plot([x, x], [v["lo"], v["hi"]], color=INK, lw=1)
    ax.set_xticks(range(len(GROUPS)), list(GROUPS.values()))
    ax.set_ylim(0, 1.05)
    _pct(ax)
    ax.set_ylabel("accuracy (mean, 95% CI)")
    ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    forest = [s[n]["ood_acc_Forest"]["mean"] for n in names]
    resid = [s[n][f"ood_acc_{g}"]["mean"] for n in names for g in list(GROUPS)[1:]]
    ax.set_title(
        f"AID residential scenes: ≥ {min(resid):.0%} right; AID forests:"
        f" {min(forest):.0%}–{max(forest):.0%} right",
        fontsize=11,
    )
    return _save(fig, "per_group.png", out)


def epochs(history: list[dict], out: Path) -> Path:
    """OOD accuracy after every epoch: the in-distribution checkpoint choice is a lottery."""
    panels = [n for n in ("cnn", "hard_a0.6") if any(h["variant"] == n for h in history)]
    fig, axes = plt.subplots(1, len(panels), figsize=(10, 3.8), sharey=True, squeeze=False)
    for ax, name in zip(axes[0], panels, strict=True):
        rows = [h for h in history if h["variant"] == name]
        for seed in sorted({h["seed"] for h in rows}):
            r = sorted((h for h in rows if h["seed"] == seed), key=lambda h: h["epoch"])
            ep = [h["epoch"] for h in r]
            ax.plot(ep, [h["ood_acc"] for h in r], color=TEAL, lw=1, alpha=0.45)
            ax.plot(ep, [h["val_acc"] for h in r], color=SLATE, lw=1, alpha=0.45)
        ax.text(10.2, 0.995, "EuroSAT val", color=SLATE, fontsize=8, va="center")
        ax.text(10.2, 0.68, "AID", color=TEAL, fontsize=8, va="center")
        ax.set_xlim(1, 12)
        ax.set_xticks(range(1, 11))
        ax.set_xlabel("epoch")
        ax.set_title(SHORT[name].replace("\n", " "), fontsize=10.5)
        _pct(ax)
    axes[0][0].set_ylabel("accuracy (one line per seed)")
    ood = np.array([h["ood_acc"] for h in history if h["variant"] in panels and h["epoch"] >= 3])
    val = np.array([h["val_acc"] for h in history if h["variant"] in panels and h["epoch"] >= 3])
    fig.suptitle(
        f"From epoch 3 on, validation stays within {val.min():.1%}–{val.max():.1%}"
        f" while AID accuracy ranges {ood.min():.0%}–{ood.max():.0%}",
        x=0.01,
        ha="left",
        fontsize=11.5,
        fontweight="bold",
        color=INK,
    )
    fig.tight_layout()
    return _save(fig, "epochs.png", out)


def scores(score_rows: list[dict], out: Path) -> Path:
    """Distribution of P(residential) on AID, pooled over seeds."""
    names = [n for n in ("cnn", "hard_a0.6") if any(r["variant"] == n for r in score_rows)]
    fig, axes = plt.subplots(1, len(names), figsize=(10, 3.4), sharey=True, squeeze=False)
    bins = np.linspace(0, 1, 21)
    for ax, name in zip(axes[0], names, strict=True):
        rows = [r for r in score_rows if r["variant"] == name]
        forest = np.array([r["p_residential"] for r in rows if r["label"] == 0])
        resid = np.array([r["p_residential"] for r in rows if r["label"] == 1])
        ax.hist(resid, bins=bins, color=SLATE, alpha=0.6, label="AID residential")
        ax.hist(forest, bins=bins, color=AMBER, alpha=0.75, label="AID forest")
        ax.axvline(0.5, color=INK, lw=0.8, ls=":")
        ax.set_xlabel("predicted P(residential)")
        ax.set_title(
            f"{SHORT[name].replace(chr(10), ' ')}: {np.mean(forest >= 0.5):.0%} of forests above 0.5",
            fontsize=10.5,
        )
    axes[0][0].set_ylabel("images × seeds")
    axes[0][0].legend(fontsize=8, loc="upper center")
    fig.tight_layout()
    return _save(fig, "scores.png", out)


def reconstructions(npz: Path, out: Path, variant: str = "hard_a0.6") -> Path:
    with np.load(npz) as z:
        rows = [z[f"{variant}__{k}"] for k in ("val_in", "val_out", "ood_in", "ood_out")]
    labels = ["EuroSAT input", "reconstruction", "AID input", "reconstruction"]
    n = rows[0].shape[0]
    fig, axes = plt.subplots(4, n, figsize=(n * 1.1, 4 * 1.2))
    for r, (imgs, label) in enumerate(zip(rows, labels, strict=True)):
        for c in range(n):
            ax = axes[r, c]
            ax.imshow(imgs[c])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
        axes[r, 0].set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9)
    fig.suptitle(
        f"Reconstructions, {variant.replace('hard_a', 'hard sharing α = ')}, seed 0",
        x=0.01,
        ha="left",
        fontsize=11,
        fontweight="bold",
        color=INK,
    )
    fig.tight_layout()
    return _save(fig, "reconstructions.png", out)


def shift(images: data.Images, out: Path, n: int = 8) -> Path:
    """What changes between the training and the AID images."""
    rng = np.random.default_rng(0)
    euro_f = images.x_id[images.y_id == 0]
    euro_r = images.x_id[images.y_id == 1]
    aid_f = images.x_ood[images.group_ood == "Forest"]
    aid_r = images.x_ood[images.y_ood == 1]
    tiles = [
        ("EuroSAT forest", euro_f),
        ("AID forest", aid_f),
        ("EuroSAT residential", euro_r),
        ("AID residential", aid_r),
    ]
    fig, axes = plt.subplots(
        4, n + 1, figsize=((n + 1) * 1.05, 4 * 1.15), gridspec_kw={"width_ratios": [1] * n + [1.6]}
    )
    for r, (label, x) in enumerate(tiles):
        pick = rng.choice(len(x), size=n, replace=False)
        for c, i in enumerate(pick):
            ax = axes[r, c]
            ax.imshow(x[i])
            ax.axis("off")
        axes[r, 0].text(-8, 32, label, ha="right", va="center", fontsize=9, color=INK)
        ax = axes[r, n]
        mean_rgb = x.reshape(-1, 3).mean(0)
        texture = analysis.edge_strength(x)
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0.05, 0.25), 0.3, 0.5, color=mean_rgb / 255))
        ax.text(
            0.42,
            0.62,
            f"mean RGB\n{mean_rgb.round().astype(int).tolist()}",
            fontsize=7.5,
            va="center",
        )
        ax.text(0.42, 0.18, f"edge strength {texture:.1f}", fontsize=7.5, va="center", color=SLATE)
        ax.set_xlim(0, 1.6)
        ax.set_ylim(0, 1)
    st = analysis.shift_stats(images)
    fig.suptitle(
        f"AID forests are far more textured than the EuroSAT forests seen in training: edge"
        f" strength {st['aid_forest']:.1f} vs {st['eurosat_forest']:.1f}"
        f" (EuroSAT residential {st['eurosat_residential']:.1f})",
        x=0.01,
        ha="left",
        fontsize=11,
        fontweight="bold",
        color=INK,
    )
    fig.tight_layout()
    return _save(fig, "shift.png", out)


def effect_phrase(d: dict) -> str:
    """Wording that follows the confidence interval, so a re-run cannot leave a stale claim."""
    if d["lo"] <= 0 <= d["hi"]:
        return f"no detectable effect ({d['mean'] * 100:+.1f} pp, 95% CI {d['lo'] * 100:+.1f} to {d['hi'] * 100:+.1f})"
    verb = "gains" if d["mean"] > 0 else "loses"
    return f"{verb} {abs(d['mean']) * 100:.1f} pp (95% CI {d['lo'] * 100:+.1f} to {d['hi'] * 100:+.1f})"


def hero_title(summary: list[dict], comp: dict) -> str:
    s = {line["variant"]: line for line in summary}
    val = min(line["val_acc"]["min"] for line in summary if line["variant"] != "soft")
    return (
        f"All models ≥ {val:.1%} in distribution, {s['cnn']['ood_acc']['mean']:.0%}–"
        f"{s['hard_a0.6']['ood_acc']['mean']:.0%} on aerial imagery.\n"
        f"Reconstruction loss, architecture held fixed: {effect_phrase(comp['aux_loss'])}"
    )


def build_all(results: Path = RESULTS, out: Path = FIGURES) -> list[Path]:
    matplotlib.use("Agg")  # files only, no window
    _style()
    rows = analysis.runs(results)
    history = analysis.read_csv(results / "history.csv")
    score_rows = analysis.read_csv(results / "ood_scores.csv")
    paths = [
        hero(rows, out, hero_title(analysis.summary(rows), analysis.comparisons(rows))),
        alpha_sweep(rows, out),
        per_group(rows, out),
        epochs(history, out),
        scores(score_rows, out),
        reconstructions(results / "reconstructions.npz", out),
        shift(data.load(), out),
    ]
    for p in paths:
        print(f"wrote {p.relative_to(out.parents[1])}")
    return paths
