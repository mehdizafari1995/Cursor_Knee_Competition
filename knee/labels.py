"""Study-level label matrices using official Kaggle column spelling.

Unknown labels stay as NaN so the masked loss and the ROC AUC metric can ignore
them. This module never imputes 0 for a missing finding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    NUM_TARGETS,
    OFFICIAL_TARGET_COLUMNS,
    OFFICIAL_TO_TARGET_KEY,
    STUDY_ID_COLUMN,
    TARGET_KEY_TO_OFFICIAL,
    TARGET_KEYS,
)


def official_columns_in(frame: pd.DataFrame) -> tuple[str, ...]:
    """Return the official target columns that are actually present, in official order."""
    return tuple(col for col in OFFICIAL_TARGET_COLUMNS if col in frame.columns)


def missing_official_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    return tuple(col for col in OFFICIAL_TARGET_COLUMNS if col not in frame.columns)


def _coerce_binary_or_nan(series: pd.Series) -> np.ndarray:
    """Map a label column to {0, 1, nan}. Unrecognised values become nan."""
    values = series.copy()
    if values.dtype == object:
        stripped = values.astype(str).str.strip()
        blank = stripped.eq("") | stripped.str.lower().isin(
            {"nan", "none", "na", "n/a", "null", "-"}
        )
        values = pd.to_numeric(stripped.where(~blank, np.nan), errors="coerce")
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    out = np.full(numeric.shape, np.nan, dtype=np.float32)
    known = np.isfinite(numeric)
    zeros = known & (numeric == 0)
    ones = known & (numeric == 1)
    out[zeros] = 0.0
    out[ones] = 1.0
    # Anything else that parsed (e.g. 0.5, 2) is treated as unknown, not coerced.
    return out


def extract_label_matrix(
    frame: pd.DataFrame,
    *,
    require_all: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (labels, mask) with shape (n_rows, 12) in official / TARGET_KEYS order.

    Columns may use official Kaggle names or internal TARGET_KEYS. Official names
    take precedence when both spellings exist. Missing columns are entirely NaN.
    """
    n = len(frame)
    labels = np.full((n, NUM_TARGETS), np.nan, dtype=np.float32)
    for i, official in enumerate(OFFICIAL_TARGET_COLUMNS):
        key = OFFICIAL_TO_TARGET_KEY[official]
        if official in frame.columns:
            labels[:, i] = _coerce_binary_or_nan(frame[official])
        elif key in frame.columns:
            labels[:, i] = _coerce_binary_or_nan(frame[key])
        elif require_all:
            raise KeyError(
                f"label column {official!r} (internal {key!r}) is missing from the frame"
            )
    mask = np.isfinite(labels)
    return labels, mask.astype(np.float32)


def prevalence(labels: np.ndarray) -> dict[str, dict[str, float | int | None]]:
    """Positive / negative / unknown counts per official target. No imputation."""
    summary: dict[str, dict[str, float | int | None]] = {}
    for i, official in enumerate(OFFICIAL_TARGET_COLUMNS):
        col = labels[:, i]
        known = np.isfinite(col)
        n_known = int(known.sum())
        n_pos = int((col[known] == 1).sum()) if n_known else 0
        n_neg = int((col[known] == 0).sum()) if n_known else 0
        n_unknown = int((~known).sum())
        rate = (n_pos / n_known) if n_known else None
        summary[official] = {
            "target_key": TARGET_KEYS[i],
            "n_known": n_known,
            "n_positive": n_pos,
            "n_negative": n_neg,
            "n_unknown": n_unknown,
            "positive_rate_among_known": rate,
        }
    return summary


def remap_prediction_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a frame with StudyInstanceUID + official target columns.

    Accepts predictions keyed by official names or TARGET_KEYS.
    """
    out = pd.DataFrame()
    if STUDY_ID_COLUMN in frame.columns:
        out[STUDY_ID_COLUMN] = frame[STUDY_ID_COLUMN].astype(str)
    elif "study_id" in frame.columns:
        out[STUDY_ID_COLUMN] = frame["study_id"].astype(str)
    else:
        raise KeyError(
            f"predictions are missing {STUDY_ID_COLUMN!r} (or the alias 'study_id')"
        )
    for key, official in TARGET_KEY_TO_OFFICIAL.items():
        if official in frame.columns:
            out[official] = pd.to_numeric(frame[official], errors="coerce")
        elif key in frame.columns:
            out[official] = pd.to_numeric(frame[key], errors="coerce")
        else:
            raise KeyError(
                f"predictions are missing {official!r} (internal {key!r})"
            )
    return out
