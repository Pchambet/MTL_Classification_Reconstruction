import numpy as np
import pytest

from mtl_eurosat import metrics


def test_accuracy_and_confusion_hand_case():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.6, 0.4, 0.9, 0.2])
    assert metrics.accuracy(y, p) == pytest.approx(3 / 5)
    assert metrics.confusion(y, p).tolist() == [[2, 1], [1, 1]]


def test_auroc_extremes_and_ties():
    y = np.array([0, 0, 1, 1])
    assert metrics.auroc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert metrics.auroc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert metrics.auroc(y, np.full(4, 0.5)) == 0.5
    # one positive below one negative out of four pairs
    assert metrics.auroc(y, np.array([0.1, 0.5, 0.4, 0.9])) == pytest.approx(0.75)


def test_auroc_is_threshold_free():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.6, 0.7, 0.8, 0.9])  # every image called residential at 0.5
    assert metrics.accuracy(y, p) == 0.5
    assert metrics.auroc(y, p) == 1.0


def test_mean_ci_matches_t_interval():
    m, lo, hi = metrics.mean_ci(np.array([1.0, 2.0, 3.0]))
    # sd = 1, se = 1/sqrt(3), t(0.975, 2) = 4.3027
    assert m == 2.0
    assert hi - m == pytest.approx(4.302653 / np.sqrt(3), rel=1e-5)
    assert m - lo == pytest.approx(hi - m)


def test_paired_difference_removes_shared_noise():
    rng = np.random.default_rng(0)
    shared = rng.normal(0, 10, size=20)  # large seed-to-seed variance common to both
    a, b = shared + 1.0 + rng.normal(0, 0.1, 20), shared
    d = metrics.paired_difference(a, b)
    assert d["lo"] < 1.0 < d["hi"]
    assert d["p"] < 1e-6
