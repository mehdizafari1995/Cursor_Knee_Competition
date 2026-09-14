"""Negation, uncertainty, history, postoperative, and sentence splitting."""

from __future__ import annotations

from knee.report_negation import (
    classify_modifiers,
    split_sentences,
)


def test_splits_sentences_and_section_headers():
    text = "HISTORY: prior ACL tear.\nFINDINGS: The ACL is intact.\nIMPRESSION: no acute injury."
    sentences = split_sentences(text)
    sections = {s.section for s in sentences}
    assert "history" in sections
    assert "findings" in sections
    assert "impression" in sections


def test_negation_before_and_after_a_finding():
    before = classify_modifiers("No evidence of ACL tear.", match_start=15, match_end=18, section="findings")
    assert before.negated is True
    assert before.negation_position == "before"
    after = classify_modifiers("The ACL is intact.", match_start=4, match_end=7, section="findings")
    assert after.negated is True
    assert after.normal_structure is True


def test_uncertainty_history_and_postoperative_are_distinct():
    unc = classify_modifiers("Possible ACL tear.", match_start=9, match_end=12, section="findings")
    assert unc.uncertain is True
    assert unc.negated is False
    hist = classify_modifiers("HISTORY of ACL tear.", match_start=11, match_end=14, section="history")
    assert hist.historical is True
    post = classify_modifiers("ACL reconstruction with intact graft.", match_start=0, match_end=3, section="findings")
    assert post.postoperative is True
