"""Strict-agreement consensus: disagreement and model-only stay unknown."""

from __future__ import annotations

from knee.report_labels import LABEL_NEGATIVE, LABEL_POSITIVE, LABEL_UNKNOWN
from knee.second_reader_consensus import strict_agreement


def test_positive_agreement():
    call = strict_agreement(LABEL_POSITIVE, "positive")
    assert call.label == LABEL_POSITIVE
    assert call.reason == "strict_positive_agreement"


def test_negative_agreement():
    call = strict_agreement(LABEL_NEGATIVE, "negative")
    assert call.label == LABEL_NEGATIVE


def test_disagreement_is_unknown():
    call = strict_agreement(LABEL_POSITIVE, "negative")
    assert call.label == LABEL_UNKNOWN
    assert call.reason == "disagreement"


def test_model_only_is_not_a_training_label():
    call = strict_agreement(LABEL_UNKNOWN, "positive")
    assert call.label == LABEL_UNKNOWN
    assert call.reason == "model_only_not_training_label"


def test_uncertain_is_unknown_unless_global_normal_negative():
    call = strict_agreement(LABEL_POSITIVE, "uncertain")
    assert call.label == LABEL_UNKNOWN
    allowed = strict_agreement(
        LABEL_NEGATIVE,
        "not_mentioned",
        deterministic_evidence="global_normal",
        global_normal=True,
    )
    assert allowed.label == LABEL_NEGATIVE
    assert allowed.reason == "verified_global_normal_negative"


def test_malformed_reader_is_unknown():
    call = strict_agreement(LABEL_POSITIVE, "positive", reader_malformed=True)
    assert call.label == LABEL_UNKNOWN
    assert call.reason == "reader_malformed"
