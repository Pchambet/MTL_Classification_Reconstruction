"""Self-contained HTML report (``site/index.html``) built from ``results/``.

Charts are Plotly (loaded from jsDelivr) so readers can hover the per-seed values;
static figures are the PNGs of ``docs/figures/``, copied next to the page.
"""

from __future__ import annotations

import html
import json
import shutil
from pathlib import Path
from string import Template

from mtl_eurosat import analysis, data, figures
from mtl_eurosat.config import AMBER, FIGURES, RESULTS, SITE, SLATE, TEAL

PLOTLY = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"
TEMPLATE = Path(__file__).with_name("report_template.html")
LABEL = {k: v.replace("\n", " ") for k, v in figures.SHORT.items()}


def _pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%"


def _ci(v: dict, digits: int = 1) -> str:
    return f"{_pct(v['mean'], digits)} [{_pct(v['lo'], digits)}, {_pct(v['hi'], digits)}]"


def _p(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}" if p < 0.01 else f"p = {p:.2f}"


def _pp(d: dict) -> str:
    """Paired difference in percentage points with its CI."""
    return f"{d['mean'] * 100:+.1f} pp [{d['lo'] * 100:+.1f}, {d['hi'] * 100:+.1f}]"


# What each paired comparison of ``analysis.comparisons`` measures, in prose.
COMPARISON = {
    "aux_loss": "reconstruction loss, architecture fixed (AID accuracy)",
    "aux_loss_forest": "reconstruction loss, architecture fixed (AID-forest accuracy)",
    "aux_loss_auroc": "reconstruction loss, architecture fixed (AUROC)",
    "mtl_vs_cnn": "hard-sharing MTL vs CNN (AID accuracy)",
    "mtl_vs_cnn_auroc": "hard-sharing MTL vs CNN (AUROC)",
    "arch": "larger architecture alone (AID accuracy)",
    "soft_masked_vs_cnn": "soft sharing vs CNN (AID accuracy)",
    "soft_masked_vs_cnn_auroc": "soft sharing vs CNN (AUROC)",
    "cnn_masked_vs_cnn": "masked-input CNN, masked scoring, vs CNN (AID accuracy)",
    "cnn_masked_vs_cnn_auroc": "masked-input CNN, masked scoring, vs CNN (AUROC)",
    "soft_masked_vs_cnn_masked": "soft sharing vs masked-input CNN (AID accuracy)",
    "soft_masked_vs_cnn_masked_auroc": "soft sharing vs masked-input CNN (AUROC)",
    "cnn_masked_clean_vs_cnn": "masked-input CNN, clean scoring, vs CNN (AID accuracy)",
    "cnn_masked_clean_vs_cnn_forest": "masked-input CNN, clean scoring, vs CNN (AID-forest accuracy)",
}


def _dauc(d: dict) -> str:
    """Paired AUROC difference with its CI."""
    return f"{d['mean']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]"


def seed_traces(rows: list[dict]) -> list[dict]:
    names = [n for n in figures.SHORT if any(r["variant"] == n for r in rows)]
    traces = []
    for key, colour, label in (("val_acc", SLATE, "EuroSAT validation"), ("ood_acc", TEAL, "AID")):
        xs, ys, text = [], [], []
        for n in names:
            for seed, v in zip(
                analysis.column(rows, n, "seed"), analysis.column(rows, n, key), strict=True
            ):
                xs.append(LABEL[n])
                ys.append(v)
                text.append(f"seed {int(seed)}")
        traces.append(
            {
                "type": "box",
                "name": label,
                "x": xs,
                "y": ys,
                "text": text,
                "boxpoints": "all",
                "jitter": 0.5,
                "pointpos": 0,
                "marker": {"color": colour, "size": 6, "opacity": 0.7},
                "line": {"color": colour, "width": 1.5},
                "fillcolor": "rgba(0,0,0,0)",
                "hovertemplate": "%{x}<br>%{text}: %{y:.1%}<extra>" + label + "</extra>",
            }
        )
    return traces


def alpha_traces(rows: list[dict]) -> list[dict]:
    s = {line["variant"]: line for line in analysis.summary(rows)}
    names = sorted((n for n in s if n.startswith("hard_a")), reverse=True)
    alphas = [float(n.split("_a")[1]) for n in names]
    traces = []
    for key, colour, label in (
        ("ood_acc", TEAL, "AID, all images"),
        ("ood_acc_Forest", AMBER, "AID forest only"),
    ):
        mean = [s[n][key]["mean"] for n in names]
        traces.append(
            {
                "type": "scatter",
                "mode": "lines+markers",
                "name": label,
                "x": alphas,
                "y": mean,
                "error_y": {
                    "type": "data",
                    "symmetric": False,
                    "array": [s[n][key]["hi"] - m for n, m in zip(names, mean, strict=True)],
                    "arrayminus": [m - s[n][key]["lo"] for n, m in zip(names, mean, strict=True)],
                    "thickness": 1,
                    "width": 3,
                },
                "line": {"color": colour, "width": 2},
                "marker": {"color": colour, "size": 7},
                "hovertemplate": "α = %{x}<br>%{y:.1%}<extra>" + label + "</extra>",
            }
        )
        traces.append(
            {
                "type": "scatter",
                "mode": "lines",
                "name": f"single-task CNN ({label})",
                "x": [min(alphas), max(alphas)],
                "y": [s["cnn"][key]["mean"]] * 2,
                "line": {"color": colour, "width": 1, "dash": "dash"},
                "hovertemplate": "single-task CNN: %{y:.1%}<extra></extra>",
            }
        )
    return traces


def epoch_traces(history: list[dict]) -> list[dict]:
    traces = []
    for name, colour in (("cnn", SLATE), ("hard_a0.6", TEAL)):
        rows = [h for h in history if h["variant"] == name]
        for i, seed in enumerate(sorted({h["seed"] for h in rows})):
            r = sorted((h for h in rows if h["seed"] == seed), key=lambda h: h["epoch"])
            traces.append(
                {
                    "type": "scatter",
                    "mode": "lines",
                    "name": LABEL[name],
                    "legendgroup": name,
                    "showlegend": i == 0,
                    "x": [h["epoch"] for h in r],
                    "y": [h["ood_acc"] for h in r],
                    "line": {"color": colour, "width": 1.2},
                    "opacity": 0.6,
                    "hovertemplate": f"{LABEL[name]}, seed {int(seed)}"
                    + "<br>epoch %{x}: %{y:.1%}<extra></extra>",
                }
            )
    return traces


def table(header: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in header)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in row) + "</tr>" for row in rows
    )
    return f'<div class="table"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def results_table(summary: list[dict]) -> str:
    rows = []
    for line in summary:
        # alpha = 1 trains no decoder, so its reconstruction error means nothing.
        rec = None if line["variant"] == "hard_a1.0" else line.get("val_recon_mse")
        rows.append(
            [
                line["label"],
                f"{line['n_params']:,}",
                str(line["n_seeds"]),
                _pct(line["val_acc"]["mean"], 2),
                _ci(line["ood_acc"]),
                f"{line['ood_auroc']['mean']:.3f}",
                _pct(line["ood_acc_Forest"]["mean"]),
                f"{rec['mean']:.4f}" if rec else "–",
            ]
        )
    return table(
        [
            "variant",
            "parameters",
            "seeds",
            "EuroSAT val acc",
            "AID acc [95% CI]",
            "AID AUROC",
            "AID forest acc",
            "val recon MSE",
        ],
        rows,
    )


def key_facts(rows: list[dict], history: list[dict], out: dict, shift: dict) -> dict[str, str]:
    """Every number quoted in prose, computed in one place."""
    s = {line["variant"]: line for line in out["variants"]}
    comp, nb = out["comparisons"], out["notebook_single_seed"]
    span = analysis.as_trained_range(rows)
    tested = [d for d in comp.values() if d]  # comparisons whose variants were run
    bonferroni = 0.05 / len(tested)
    survivors = [COMPARISON[k] for k, d in comp.items() if d and d["p"] < bonferroni]
    soft, cm = s["soft"], s["cnn_masked"]

    def mean(v: str, key: str, digits: int = 1) -> str:
        return _pct(s[v][key]["mean"], digits)

    def rng(v: str, key: str = "ood_acc") -> str:
        return f"{_pct(s[v][key]['min'], 0)}–{_pct(s[v][key]['max'], 0)}"

    return {
        "edge_eu_forest": f"{shift['eurosat_forest']:.1f}",
        "edge_aid_forest": f"{shift['aid_forest']:.1f}",
        "edge_eu_res": f"{shift['eurosat_residential']:.1f}",
        "edge_aid_res": f"{shift['aid_residential']:.1f}",
        "n_seeds": str(s["cnn"]["n_seeds"]),
        "n_seeds_sweep": str(s["hard_a0.8"]["n_seeds"]) if "hard_a0.8" in s else "",
        "val_min": _pct(span["val_min"]),
        "ood_lo": _pct(span["ood_lo"], 0),
        "ood_hi": _pct(span["ood_hi"], 0),
        "cnn_ood": _ci(s["cnn"]["ood_acc"]),
        "cnn_ood_mean": mean("cnn", "ood_acc"),
        "hard06_ood": _ci(s["hard_a0.6"]["ood_acc"]),
        "hard06_ood_mean": mean("hard_a0.6", "ood_acc"),
        "hard10_ood": _ci(s["hard_a1.0"]["ood_acc"]),
        "hard06_range": rng("hard_a0.6"),
        "cnn_range": rng("cnn"),
        "soft_range": rng("soft"),
        "aux": _pp(comp["aux_loss"]),
        "aux_p": _p(comp["aux_loss"]["p"]),
        "aux_forest": _pp(comp["aux_loss_forest"]),
        "aux_auroc": _dauc(comp["aux_loss_auroc"]),
        "mtl_vs_cnn": _pp(comp["mtl_vs_cnn"]),
        "mtl_vs_cnn_p": _p(comp["mtl_vs_cnn"]["p"]),
        "mtl_vs_cnn_auroc": _dauc(comp["mtl_vs_cnn_auroc"]),
        "mtl_vs_cnn_auroc_p": _p(comp["mtl_vs_cnn_auroc"]["p"]),
        "arch": _pp(comp["arch"]),
        "arch_p": _p(comp["arch"]["p"]),
        "cnn_forest": mean("cnn", "ood_acc_Forest", 0),
        "hard06_forest": mean("hard_a0.6", "ood_acc_Forest", 0),
        "cnn_res_min": _pct(
            min(
                s["cnn"][f"ood_acc_{g}"]["mean"] for g in ("DenseResidential", "MediumResidential")
            ),
            0,
        ),
        "hard06_medium": mean("hard_a0.6", "ood_acc_MediumResidential", 0),
        "cnn_auroc": f"{s['cnn']['ood_auroc']['mean']:.2f}",
        "hard06_auroc": f"{s['hard_a0.6']['ood_auroc']['mean']:.2f}",
        "swing_cnn": f"{analysis.epoch_swing(history, 'cnn', 'ood_acc') * 100:.0f}",
        "swing_hard06": f"{analysis.epoch_swing(history, 'hard_a0.6', 'ood_acc') * 100:.0f}",
        "soft_val": mean("soft", "val_acc"),
        "soft_val_masked": mean("soft", "val_acc_masked"),
        "soft_ood": _ci(soft["ood_acc"]),
        "soft_ood_masked": _ci(soft["ood_acc_masked"]),
        "soft_auroc_masked": f"{soft['ood_auroc_masked']['mean']:.2f}",
        "soft_vs_cnn": _pp(comp["soft_masked_vs_cnn"]),
        "cm_ood_masked": _ci(cm["ood_acc_masked"]),
        "cm_auroc_masked": f"{cm['ood_auroc_masked']['mean']:.2f}",
        "cm_val_clean": mean("cnn_masked", "val_acc"),
        "cm_ood_clean": _ci(cm["ood_acc"]),
        "cm_forest_clean": mean("cnn_masked", "ood_acc_Forest", 0),
        "masking": _pp(comp["cnn_masked_vs_cnn"]),
        "masking_p": _p(comp["cnn_masked_vs_cnn"]["p"]),
        "masking_clean": _pp(comp["cnn_masked_clean_vs_cnn"]),
        "masking_clean_p": _p(comp["cnn_masked_clean_vs_cnn"]["p"]),
        "soft_vs_cm": _pp(comp["soft_masked_vs_cnn_masked"]),
        "soft_vs_cm_p": _p(comp["soft_masked_vs_cnn_masked"]["p"]),
        "soft_vs_cm_auroc": _dauc(comp["soft_masked_vs_cnn_masked_auroc"]),
        "soft_vs_cm_auroc_p": _p(comp["soft_masked_vs_cnn_masked_auroc"]["p"]),
        "params_ratio": f"{soft['n_params'] / cm['n_params']:.0f}",
        "n_comparisons": str(len(tested)),
        "bonferroni": f"{bonferroni:.4f}",
        "n_bonferroni": str(len(survivors)),
        "bonferroni_list": "; ".join(survivors) or "none",
        "params_cnn": f"{s['cnn']['n_params'] / 1000:.0f}k",
        "params_hard": f"{s['hard_a0.6']['n_params'] / 1000:.0f}k",
        "params_soft": f"{soft['n_params'] / 1000:.0f}k",
        "nb_cnn": _pct(nb["baseline"]),
        "nb_mtl": _pct(nb["mtl"]),
        "nb_soft": _pct(nb["softshare"]),
    }


def build(results: Path = RESULTS) -> None:
    figures.build_all(results)
    out = analysis.write_summary(results)
    rows = analysis.runs(results)
    history = analysis.read_csv(results / "history.csv")
    facts = key_facts(rows, history, out, analysis.shift_stats(data.load()))
    (results / "facts.json").write_text(json.dumps(facts, indent=2, ensure_ascii=False) + "\n")
    page = Template(TEMPLATE.read_text()).substitute(
        plotly=PLOTLY,
        seeds=json.dumps(seed_traces(rows)),
        alpha=json.dumps(alpha_traces(rows)),
        epochs=json.dumps(epoch_traces(history)),
        results_table=results_table(out["variants"]),
        **facts,
    )
    SITE.mkdir(parents=True, exist_ok=True)
    (SITE / "index.html").write_text(page)
    shutil.copytree(FIGURES, SITE / "figures", dirs_exist_ok=True)
    print(f"wrote {SITE / 'index.html'}")
    print(f"wrote {results / 'facts.json'} (every number quoted in the report and README)")
