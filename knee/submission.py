"""Competition submission writer and validator.

Validates column names (verbatim official spelling) and row order against
``sample_submission.csv``. Internal TARGET_KEYS are accepted on input and
rewritten to official names.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .competition import SAMPLE_SUBMISSION_COLUMNS
from .config import OFFICIAL_TARGET_COLUMNS, STUDY_ID_COLUMN
from .labels import remap_prediction_columns


class SubmissionError(ValueError):
    """The submission does not match the official sample_submission contract."""


def load_sample_submission(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    expected = list(SAMPLE_SUBMISSION_COLUMNS)
    columns = list(frame.columns)
    if columns != expected:
        raise SubmissionError(
            "sample_submission.csv columns do not match the official header: "
            f"got {columns}, expected {expected}"
        )
    if frame.empty:
        raise SubmissionError("sample_submission.csv has no rows")
    if frame[STUDY_ID_COLUMN].duplicated().any():
        raise SubmissionError("sample_submission.csv has duplicate StudyInstanceUID values")
    return frame


def validate_submission(frame: pd.DataFrame, sample: pd.DataFrame) -> None:
    """Require identical columns and identical StudyInstanceUID row order."""
    expected_cols = list(sample.columns)
    got_cols = list(frame.columns)
    if got_cols != expected_cols:
        raise SubmissionError(
            f"submission columns {got_cols} do not match sample {expected_cols}"
        )
    expected_ids = sample[STUDY_ID_COLUMN].astype(str).tolist()
    got_ids = frame[STUDY_ID_COLUMN].astype(str).tolist()
    if got_ids != expected_ids:
        raise SubmissionError(
            "submission StudyInstanceUID row order does not match sample_submission.csv"
        )
    for column in OFFICIAL_TARGET_COLUMNS:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any():
            raise SubmissionError(f"submission column {column!r} contains non-numeric values")
        if ((values < 0) | (values > 1)).any():
            raise SubmissionError(
                f"submission column {column!r} has values outside [0, 1]"
            )


def align_to_sample(predictions: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    """Reindex predictions to the sample row order and official columns."""
    mapped = remap_prediction_columns(predictions).copy()
    mapped[STUDY_ID_COLUMN] = mapped[STUDY_ID_COLUMN].astype(str)
    sample_ids = sample[STUDY_ID_COLUMN].astype(str)
    indexed = mapped.set_index(STUDY_ID_COLUMN)
    missing = [sid for sid in sample_ids if sid not in indexed.index]
    extra = [sid for sid in indexed.index if sid not in set(sample_ids)]
    if missing:
        raise SubmissionError(
            f"predictions are missing {len(missing)} StudyInstanceUID values from sample_submission.csv"
        )
    if extra:
        raise SubmissionError(
            f"predictions contain {len(extra)} StudyInstanceUID values not in sample_submission.csv"
        )
    ordered = indexed.loc[sample_ids.tolist()].reset_index()
    return ordered.loc[:, list(SAMPLE_SUBMISSION_COLUMNS)]


def write_competition_submission(
    predictions: pd.DataFrame,
    sample: pd.DataFrame,
    path: Path,
) -> pd.DataFrame:
    aligned = align_to_sample(predictions, sample)
    validate_submission(aligned, sample)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    aligned.to_csv(path, index=False)
    return aligned
