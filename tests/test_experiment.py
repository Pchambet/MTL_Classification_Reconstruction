import json

import numpy as np
import pytest

from mtl_eurosat import analysis, data, experiment


def _fake_images() -> data.Images:
    z = np.zeros((4, 64, 64, 3), dtype=np.uint8)
    return data.Images(
        x_id=z,
        y_id=np.array([0, 0, 1, 1]),
        x_ood=z,
        y_ood=np.array([0, 0, 1, 1]),
        group_ood=np.array(["Forest", "Forest", "DenseResidential", "MediumResidential"]),
    )


def _run(variant: str, seed: int, ood_acc: float) -> dict:
    summary = {
        "variant": variant,
        "seed": seed,
        "n_params": 10,
        "val_acc": 1.0,
        "val_recon_mse": None if variant == "cnn" else 0.01,
        "ood_acc": ood_acc,
        "ood_auroc": 0.8,
        "ood_acc_Forest": 2 * ood_acc - 1,
        "ood_acc_DenseResidential": 1.0,
        "ood_acc_MediumResidential": 1.0,
    }
    if variant == "soft":
        summary |= {"val_acc_masked": 1.0, "ood_acc_masked": 0.7, "ood_auroc_masked": 0.9}
    history = [{"epoch": e, "val_acc": 1.0, "ood_acc": ood_acc} for e in (1, 2)]
    return {"summary": summary, "history": history, "p_ood": [0.6, 0.4, 0.9, 0.8]}


@pytest.fixture
def results(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    # hard_a0.6 sits 4, 5 and 6 points above hard_a1.0 on the three seeds
    for seed in range(3):
        base = 0.65 + 0.02 * seed
        for variant, acc in (
            ("cnn", base),
            ("hard_a1.0", base),
            ("hard_a0.6", base + 0.04 + 0.01 * seed),
        ):
            (runs / f"{variant}_s{seed}.json").write_text(json.dumps(_run(variant, seed, acc)))
        (runs / f"soft_s{seed}.json").write_text(json.dumps(_run("soft", seed, 0.55)))
    out = tmp_path / "results"
    experiment.collect(runs, out, images=_fake_images())
    return out


def test_collect_writes_one_row_per_run_in_grid_order(results):
    rows = analysis.runs(results)
    assert len(rows) == 12
    assert [r["variant"] for r in rows[:3]] == ["cnn"] * 3
    assert np.isnan(rows[0]["val_recon_mse"])  # no decoder
    scores = analysis.read_csv(results / "ood_scores.csv")
    assert len(scores) == 12 * 4 and scores[0]["group"] == "Forest"


def test_paired_comparison_recovers_the_planted_effect(results):
    summary = analysis.write_summary(results)
    aux = summary["comparisons"]["aux_loss"]
    assert aux["mean"] == pytest.approx(0.05)
    assert aux["n"] == 3
    soft = next(v for v in summary["variants"] if v["variant"] == "soft")
    assert soft["ood_acc_masked"]["mean"] == pytest.approx(0.7)
    assert "val_recon_mse" not in summary["variants"][0]  # cnn: metric does not apply


def test_epoch_swing_is_the_median_within_run_range():
    history = [
        {"variant": "cnn", "seed": seed, "epoch": e, "ood_acc": acc}
        for seed, accs in ((0, [0.1, 0.6, 0.7, 0.5]), (1, [0.9, 0.6, 0.6, 0.6]), (2, [0, 0, 1, 0]))
        for e, acc in enumerate(accs, start=1)
    ]
    # epochs 3-4 only: ranges 0.2, 0.0 and 1.0, median 0.2; epochs 1-2 are ignored
    assert analysis.epoch_swing(history, "cnn", "ood_acc") == pytest.approx(0.2)
    assert np.isnan(analysis.epoch_swing(history, "soft", "ood_acc"))


def test_last_epoch_takes_each_seeds_final_epoch():
    # what the notebooks restored: the last epoch trained, whatever its validation loss
    history = [
        {"variant": "cnn", "seed": seed, "epoch": e, "ood_acc": acc}
        for seed, accs in ((1, [0.9, 0.6, 0.5]), (0, [0.1, 0.7]))
        for e, acc in enumerate(accs, start=1)
    ]
    assert analysis.last_epoch(history, "cnn", "ood_acc").tolist() == [0.7, 0.5]
    assert analysis.last_epoch(history, "soft", "ood_acc").size == 0


def test_as_trained_scores_masked_models_on_masked_inputs(results):
    rows = analysis.runs(results)
    # soft is 0.55 on clean AID images and 0.7 on masked ones, the inputs it was trained on
    assert analysis.as_trained(rows, "soft", "ood_acc").tolist() == [0.7] * 3
    r = analysis.as_trained_range(rows)
    assert r["ood_lo"] == pytest.approx(0.67)  # cnn and hard_a1.0: mean of 0.65, 0.67, 0.69
    assert r["ood_hi"] == pytest.approx(0.72)  # hard_a0.6: mean of 0.69, 0.72, 0.75
    assert r["val_min"] == 1.0
