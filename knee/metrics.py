"""Per-target and macro-averaged ROC AUC, the official competition metric.

Unknown labels (NaN) are ignored per target. A target that is single-class among
the labelled rows is reported as NaN and excluded from the macro average.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def per_target_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> list[float]:
    """Return one ROC AUC per column, using NaN where AUC is undefined."""
    if y_true.ndim != 2 or y_prob.ndim != 2:
        raise ValueError("y_true and y_prob must be 2-D arrays")
    if y_true.shape != y_prob.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_prob.shape}")

    scores: list[float] = []
    for j in range(y_true.shape[1]):
        col = np.asarray(y_true[:, j], dtype=np.float64)
        pred = np.asarray(y_prob[:, j], dtype=np.float64)
        known = np.isfinite(col) & np.isfinite(pred)
        if known.sum() < 2:
            scores.append(float("nan"))
            continue
        labelled = col[known]
        if labelled.min() == labelled.max():
            scores.append(float("nan"))
            continue
        scores.append(float(roc_auc_score(labelled, pred[known])))
    return scores


def macro_roc_auc(
    y_true: np.ndarray, y_prob: np.ndarray
) -> tuple[float, list[float]]:
    """Macro-average of per-target ROC AUC, skipping undefined targets.

    Returns:
        (macro_auc, per_target_auc)
    """
    per_target = per_target_roc_auc(y_true, y_prob)
    valid = [v for v in per_target if v == v]
    macro = float(np.mean(valid)) if valid else float("nan")
    return macro, per_target
