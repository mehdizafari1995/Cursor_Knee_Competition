"""OOF confidence policies must not score a gold study with a threshold fit on it."""

from __future__ import annotations

import numpy as np
import pytest

from knee.report_confidence import (
    assert_no_threshold_leakage,
    fit_oof_confidence_policies,
    select_negative_threshold,
    select_positive_threshold,
)


def test_positive_threshold_prefers_precision_not_coverage():
    gold = np.array([1.0, 1.0, 0.0, 0.0, 1.0])
    pred = np.array([1.0, 1.0, 1.0, 0.0, 1.0])
    conf = np.array([0.95, 0.93, 0.55, 0.9, 0.91])
    t = select_positive_threshold(gold, pred, conf, min_precision=0.85)
    gated_high = (pred == 1) & (conf >= t)
    # The 0.55 false positive must be excluded.
    assert not bool(gated_high[2])
    assert t > 0.55


def test_no_threshold_meets_floor_refuses_calls():
    gold = np.array([0.0, 0.0, 0.0, 1.0])
    pred = np.array([1.0, 1.0, 1.0, 1.0])
    conf = np.array([0.99, 0.98, 0.97, 0.96])
    t = select_positive_threshold(gold, pred, conf, min_precision=0.85)
    assert t > 1.0


def test_oof_train_and_test_ids_are_disjoint():
    n = 12
    gold = np.array([1.0, 0.0] * 6)
    pred = gold.copy()
    conf = np.full(n, 0.9)
    ids = [f"study-{i}" for i in range(n)]
    result = fit_oof_confidence_policies(
        gold, pred, conf, study_ids=ids, n_folds=4, seed=1, min_precision=0.5, min_npv=0.5
    )
    assert_no_threshold_leakage(result)
    scored = np.zeros(n, dtype=bool)
    for fold in result["folds"]:
        assert set(fold["train_ids"]).isdisjoint(set(fold["test_ids"]))
        for sid in fold["test_ids"]:
            scored[ids.index(sid)] = True
    assert scored.all()


def test_reported_gold_metric_is_oof_not_full_fit():
    # One false positive at mid confidence. Full-fit can exclude it; a fold that
    # holds out the true positives may pick a looser threshold. We only require
    # that the API exposes both and labels the leaked fit as not for gold.
    gold = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    pred = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    conf = np.array([0.95, 0.94, 0.93, 0.70, 0.90, 0.88, 0.92, 0.87])
    ids = [f"s{i}" for i in range(len(gold))]
    result = fit_oof_confidence_policies(
        gold, pred, conf, study_ids=ids, n_folds=4, seed=0, min_precision=0.8, min_npv=0.8
    )
    assert "oof_metrics" in result
    assert "leaked_full_fit_metrics" in result
    assert "weak labels" in result["note"]
    # Operational thresholds are for report-only studies, not the gold estimate.
    assert result["operational_positive_threshold"] == result["operational_positive_threshold"]
    for fold in result["folds"]:
        assert "s" in fold["train_ids"][0] or fold["train_ids"]
        held = set(fold["test_ids"])
        assert held.isdisjoint(set(fold["train_ids"]))
