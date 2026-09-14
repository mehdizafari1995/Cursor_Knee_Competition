"""Abstention-aware coverage metrics. Conditional sensitivity is not full-set."""

from __future__ import annotations

import numpy as np
import pytest

from knee.config import OFFICIAL_TARGET_COLUMNS
from knee.coverage_metrics import (
    apply_confidence_gates,
    assert_not_full_set_sensitivity_claim,
    bootstrap_coverage_intervals,
    coverage_confusion,
    coverage_error_curve,
    from_published_counts,
)
from knee.report_v1_aggregates import V1_GOLD_COUNTS, V1_N_GOLD


def test_abstention_is_not_a_false_negative():
    gold = np.array([1.0, 1.0, 0.0, 0.0])
    pred = np.array([1.0, np.nan, 0.0, np.nan])
    stats = coverage_confusion(gold, pred)
    assert stats["tp"] == 1
    assert stats["fn"] == 0
    assert stats["n_abstained_positives"] == 1
    assert stats["n_abstained_negatives"] == 1
    assert stats["positive_capture"] == pytest.approx(0.5)
    assert stats["negative_capture"] == pytest.approx(0.5)
    assert stats["conditional_sensitivity"] == pytest.approx(1.0)
    assert stats["coverage"] == pytest.approx(0.5)
    assert stats["unknown_imputed_as_negative"] is False


def test_positive_capture_is_not_conditional_sensitivity():
    gold = np.array([1.0, 1.0, 1.0, 0.0])
    pred = np.array([1.0, np.nan, np.nan, 0.0])
    stats = coverage_confusion(gold, pred)
    assert stats["conditional_sensitivity"] == pytest.approx(1.0)
    assert stats["positive_capture"] == pytest.approx(1.0 / 3.0)
    assert stats["conditional_sensitivity"] != stats["positive_capture"]


def test_error_rate_counts_only_called_mistakes():
    gold = np.array([1.0, 0.0, 1.0])
    pred = np.array([0.0, 1.0, np.nan])
    stats = coverage_confusion(gold, pred)
    assert stats["fp"] + stats["fn"] == 2
    assert stats["error_rate_all_gold"] == pytest.approx(2 / 3)
    assert stats["selective_accuracy"] == pytest.approx(0.0)


def test_bootstrap_includes_required_metrics():
    gold = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    pred = np.array([1.0, 0.0, np.nan, 1.0, 1.0, 0.0])
    ci = bootstrap_coverage_intervals(gold, pred, n_resamples=40, seed=0, ci_level=0.95)
    for name in ("coverage", "precision", "npv", "positive_capture", "negative_capture"):
        assert name in ci
        assert "low" in ci[name] and "high" in ci[name]


def test_coverage_error_curve_does_not_force_calls():
    gold = np.array([1.0, 0.0, 1.0, 0.0])
    pred = np.array([1.0, 0.0, 1.0, 0.0])
    conf = np.array([0.95, 0.4, 0.7, 0.9])
    curve = coverage_error_curve(gold, pred, conf, thresholds=(0.9, 0.5), polarity="joint")
    high = curve[0]
    low = curve[1]
    assert high["threshold"] == 0.9
    assert high["coverage"] <= low["coverage"] + 1e-9


def test_confidence_gates_are_polarity_specific():
    pred = np.array([1.0, 1.0, 0.0, 0.0])
    conf = np.array([0.9, 0.5, 0.9, 0.5])
    gated = apply_confidence_gates(pred, conf, positive_threshold=0.8, negative_threshold=0.8)
    assert gated[0] == 1.0
    assert np.isnan(gated[1])
    assert gated[2] == 0.0
    assert np.isnan(gated[3])


def test_v1_published_counts_rebuild_macro_capture():
    captures = []
    for official in OFFICIAL_TARGET_COLUMNS:
        row = V1_GOLD_COUNTS[official]
        stats = from_published_counts(
            tp=int(row["tp"]),
            tn=int(row["tn"]),
            fp=int(row["fp"]),
            fn=int(row["fn"]),
            n_gold_positive=int(row["n_gold_positive"]),
            n_gold_known=V1_N_GOLD,
        )
        captures.append(stats["positive_capture"])
        assert stats["fn"] + stats["tp"] + stats["n_abstained_positives"] == row["n_gold_positive"]
    assert float(np.mean(captures)) == pytest.approx(0.422, abs=0.005)


def test_v2_summary_does_not_call_conditional_sensitivity_full_set():
    from pathlib import Path

    text = Path("reports/report_labeler_v2_summary.md").read_text(encoding="utf-8")
    assert_not_full_set_sensitivity_claim(text)
    assert "conditional" in text.lower()
    assert "positive_capture" in text or "positive-capture" in text
    assert "not" in text.lower() and "full-set" in text.lower()
