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
        rec = line.get("val_recon_mse")
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


def key_facts(summary: list[dict], comp: dict, nb: dict, shift: dict) -> dict[str, str]:
    """Every number quoted in prose, computed in one place."""
    s = {line["variant"]: line for line in summary}
    soft = s["soft"]
    return {
        "edge_eu_forest": f"{shift['eurosat_forest']:.1f}",
        "edge_aid_forest": f"{shift['aid_forest']:.1f}",
        "edge_eu_res": f"{shift['eurosat_residential']:.1f}",
        "edge_aid_res": f"{shift['aid_residential']:.1f}",
        "n_seeds": str(s["cnn"]["n_seeds"]),
        "n_seeds_sweep": str(s["hard_a0.8"]["n_seeds"]) if "hard_a0.8" in s else "",
        "val_min": _pct(
            min(line["val_acc"]["min"] for line in summary if line["variant"] != "soft"), 1
        ),
        "cnn_ood": _ci(s["cnn"]["ood_acc"]),
        "cnn_ood_mean": _pct(s["cnn"]["ood_acc"]["mean"], 0),
        "hard06_ood": _ci(s["hard_a0.6"]["ood_acc"]),
        "hard06_ood_mean": _pct(s["hard_a0.6"]["ood_acc"]["mean"], 0),
        "hard10_ood": _ci(s["hard_a1.0"]["ood_acc"]),
        "hard06_range": f"{_pct(s['hard_a0.6']['ood_acc']['min'], 0)}–{_pct(s['hard_a0.6']['ood_acc']['max'], 0)}",
        "cnn_range": f"{_pct(s['cnn']['ood_acc']['min'], 0)}–{_pct(s['cnn']['ood_acc']['max'], 0)}",
        "aux": _pp(comp["aux_loss"]),
        "aux_p": _p(comp["aux_loss"]["p"]),
        "aux_forest": _pp(comp["aux_loss_forest"]),
        "mtl_vs_cnn": _pp(comp["mtl_vs_cnn"]),
        "mtl_vs_cnn_p": _p(comp["mtl_vs_cnn"]["p"]),
        "arch": _pp(comp["arch"]),
        "cnn_forest": _pct(s["cnn"]["ood_acc_Forest"]["mean"], 0),
        "hard06_forest": _pct(s["hard_a0.6"]["ood_acc_Forest"]["mean"], 0),
        "cnn_auroc": f"{s['cnn']['ood_auroc']['mean']:.2f}",
        "hard06_auroc": f"{s['hard_a0.6']['ood_auroc']['mean']:.2f}",
        "soft_val": _pct(soft["val_acc"]["mean"], 1),
        "soft_val_masked": _pct(soft["val_acc_masked"]["mean"], 1),
        "soft_ood": _ci(soft["ood_acc"]),
        "soft_ood_masked": _ci(soft["ood_acc_masked"]),
        "nb_cnn": _pct(nb["baseline"], 1),
        "nb_mtl": _pct(nb["mtl"], 1),
        "nb_soft": _pct(nb["softshare"], 1),
    }


def build(results: Path = RESULTS) -> None:
    figures.build_all(results)
    out = analysis.write_summary(results)
    rows = analysis.runs(results)
    history = analysis.read_csv(results / "history.csv")
    facts = key_facts(
        out["variants"],
        out["comparisons"],
        out["notebook_single_seed"],
        analysis.shift_stats(data.load()),
    )
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
    print(json.dumps(facts, indent=2, ensure_ascii=False))
