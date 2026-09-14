"""Competition submission columns and row order vs sample_submission.csv."""

from __future__ import annotations

import pandas as pd
import pytest

from knee.competition import SAMPLE_SUBMISSION_COLUMNS
from knee.config import OFFICIAL_TARGET_COLUMNS, STUDY_ID_COLUMN, TARGET_KEYS
from knee.submission import (
    SubmissionError,
    align_to_sample,
    load_sample_submission,
    validate_submission,
    write_competition_submission,
)
from tests.competition_fixtures import make_mini_competition_root


def test_sample_submission_header_is_official_spelling(tmp_path):
    root = make_mini_competition_root(tmp_path / "mini")
    sample = load_sample_submission(root / "sample_submission.csv")
    assert list(sample.columns) == list(SAMPLE_SUBMISSION_COLUMNS)
    assert "Baker's" in sample.columns
    assert "baker_cyst" not in sample.columns
    assert list(sample.columns)[1:] == list(OFFICIAL_TARGET_COLUMNS)


def test_row_order_must_match_sample(tmp_path):
    root = make_mini_competition_root(tmp_path / "mini")
    sample = load_sample_submission(root / "sample_submission.csv")
    preds = sample.copy()
    preds[OFFICIAL_TARGET_COLUMNS] = 0.3
    aligned = align_to_sample(preds, sample)
    validate_submission(aligned, sample)

    reversed_preds = preds.iloc[::-1].reset_index(drop=True)
    aligned_rev = align_to_sample(reversed_preds, sample)
    assert aligned_rev[STUDY_ID_COLUMN].tolist() == sample[STUDY_ID_COLUMN].tolist()

    broken = aligned.copy()
    broken = broken.iloc[::-1].reset_index(drop=True)
    with pytest.raises(SubmissionError, match="row order"):
        validate_submission(broken, sample)


def test_internal_keys_are_rewritten_to_official_names(tmp_path):
    root = make_mini_competition_root(tmp_path / "mini")
    sample = load_sample_submission(root / "sample_submission.csv")
    preds = pd.DataFrame({STUDY_ID_COLUMN: sample[STUDY_ID_COLUMN]})
    for key in TARGET_KEYS:
        preds[key] = 0.4
    out = write_competition_submission(preds, sample, tmp_path / "submission.csv")
    assert list(out.columns) == list(SAMPLE_SUBMISSION_COLUMNS)
    loaded = pd.read_csv(tmp_path / "submission.csv")
    assert list(loaded.columns) == list(SAMPLE_SUBMISSION_COLUMNS)


def test_missing_study_is_rejected(tmp_path):
    root = make_mini_competition_root(tmp_path / "mini")
    sample = load_sample_submission(root / "sample_submission.csv")
    preds = sample.iloc[:1].copy()
    with pytest.raises(SubmissionError, match="missing"):
        align_to_sample(preds, sample)
