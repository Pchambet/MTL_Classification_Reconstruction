"""Classification metrics and seed-level uncertainty.

Each metric takes true labels and the predicted probability of class 1
(residential), so threshold-free metrics (AUROC) and thresholded ones share inputs.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def accuracy(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> float:
    return float(np.mean((np.asarray(p) >= threshold) == np.asarray(y)))


def confusion(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """2x2 counts, rows = true class, columns = predicted class."""
    pred = (np.asarray(p) >= threshold).astype(int)
    cm = np.zeros((2, 2), dtype=int)
    np.add.at(cm, (np.asarray(y, dtype=int), pred), 1)
    return cm


def auroc(y: np.ndarray, p: np.ndarray) -> float:
    """Area under the ROC curve via the Mann-Whitney U statistic (ties count one half).

    It answers "does the model rank residential above forest?" independently of the
    0.5 threshold, which separates a ranking failure from a calibration shift.
    """
    y = np.asarray(y)
    pos, neg = np.asarray(p)[y == 1], np.asarray(p)[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = stats.rankdata(np.concatenate([pos, neg]))
    u = ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


def mean_ci(values: np.ndarray, level: float = 0.95) -> tuple[float, float, float]:
    """Mean and Student-t confidence interval across seeds."""
    v = np.asarray(values, dtype=float)
    m = float(v.mean())
    if len(v) < 2:
        return m, float("nan"), float("nan")
    half = float(stats.t.ppf((1 + level) / 2, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))
    return m, m - half, m + half


def paired_difference(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Mean of ``a - b`` over seeds, its 95% CI and the paired t-test p-value.

    Seeds are paired: the same seed gives both variants the same train/validation
    split, so pairing removes the split-to-split variance from the comparison.
    """
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    m, lo, hi = mean_ci(d)
    if np.ptp(d) == 0:  # identical differences: the t statistic is 0/0 or infinite
        return {"mean": m, "lo": lo, "hi": hi, "p": 1.0 if d[0] == 0 else 0.0}
    return {"mean": m, "lo": lo, "hi": hi, "p": float(stats.ttest_rel(a, b).pvalue)}
