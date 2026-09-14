"""Per-target and macro ROC AUC, including unknown labels."""

from __future__ import annotations

import numpy as np
import pytest

from knee.metrics import macro_roc_auc, per_target_roc_auc
from knee.train import macro_roc_auc as train_macro_roc_auc


def test_macro_roc_auc_perfect():
    y_true = np.array([[0, 1], [1, 0], [1, 1], [0, 0]], dtype=np.float32)
    macro, per_target = macro_roc_auc(y_true, y_true.copy())
    assert macro == 1.0
    assert per_target == [1.0, 1.0]


def test_macro_skips_single_class_and_unknowns():
    y_true = np.array(
        [
            [0, 1, np.nan],
            [1, 1, 0],
            [0, 1, 1],
            [1, 1, np.nan],
        ],
        dtype=np.float32,
    )
    y_prob = np.array(
        [
            [0.1, 0.9, 0.2],
            [0.9, 0.8, 0.1],
            [0.2, 0.7, 0.9],
            [0.8, 0.6, 0.3],
        ],
        dtype=np.float32,
    )
    scores = per_target_roc_auc(y_true, y_prob)
    assert scores[0] == pytest.approx(1.0)
    assert np.isnan(scores[1])  # target is all-positive among known rows
    assert scores[2] == pytest.approx(1.0)
    macro, per_target = macro_roc_auc(y_true, y_prob)
    assert macro == pytest.approx(1.0)
    assert per_target[1] != per_target[1]


def test_train_module_reexports_the_same_metric():
    y_true = np.array([[0, 1], [1, 0]], dtype=np.float32)
    y_prob = np.array([[0.1, 0.9], [0.8, 0.2]], dtype=np.float32)
    assert train_macro_roc_auc(y_true, y_prob) == macro_roc_auc(y_true, y_prob)
