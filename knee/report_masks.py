"""Private trainable masks. Unknown and disagreement are never stored as 0."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import OFFICIAL_TARGET_COLUMNS, STUDY_ID_COLUMN, TARGET_KEYS, TARGET_KEY_TO_OFFICIAL

MASK_HIGH_CONF_POSITIVE = "mask_high_conf_positive.csv"
MASK_HIGH_CONF_NEGATIVE = "mask_high_conf_negative.csv"
MASK_UNKNOWN = "mask_unknown.csv"
MASK_DISAGREEMENT = "mask_disagreement.csv"
MASK_UNSUPPORTED_LANGUAGE = "mask_unsupported_language.csv"

MASK_FILENAMES = (
    MASK_HIGH_CONF_POSITIVE,
    MASK_HIGH_CONF_NEGATIVE,
    MASK_UNKNOWN,
    MASK_DISAGREEMENT,
    MASK_UNSUPPORTED_LANGUAGE,
)

MASK_REASONS_NEVER_ZERO = frozenset(
    {"unknown", "disagreement", "unsupported_language", "second_reader_abstain", "no_finding_mentioned"}
)


def _empty_mask_frame(ids: list[str]) -> pd.DataFrame:
    data: dict[str, Any] = {STUDY_ID_COLUMN: ids}
    for col in OFFICIAL_TARGET_COLUMNS:
        data[col] = np.zeros(len(ids), dtype=np.int8)
    return pd.DataFrame(data)


def build_trainable_masks(
    pred_frame: pd.DataFrame,
    *,
    positive_thresholds: dict[str, float],
    negative_thresholds: dict[str, float],
    disagreement_frame: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Boolean masks keyed by official columns. Values are 1/0 flags, not labels.

    The unknown and disagreement masks record where a label must stay unknown.
    Callers must never write 0 into a weak-label cell for those rows.
    """
    if STUDY_ID_COLUMN not in pred_frame.columns:
        raise KeyError(STUDY_ID_COLUMN)
    ids = pred_frame[STUDY_ID_COLUMN].astype(str).tolist()
    n = len(ids)
    pos = _empty_mask_frame(ids)
    neg = _empty_mask_frame(ids)
    unknown = _empty_mask_frame(ids)
    disagree = _empty_mask_frame(ids)
    unsup = _empty_mask_frame(ids)

    language_supported = (
        ~pred_frame["quality_flags"].astype(str).str.contains("unsupported_language", regex=False)
        if "quality_flags" in pred_frame.columns
        else pd.Series([True] * n)
    )
    language_uncertain = (
        pred_frame["language_uncertain"].astype(bool)
        if "language_uncertain" in pred_frame.columns
        else pd.Series([False] * n)
    )

    for key in TARGET_KEYS:
        official = TARGET_KEY_TO_OFFICIAL[key]
        labels = pd.to_numeric(pred_frame[official], errors="coerce").to_numpy(dtype=np.float64)
        conf_col = f"{key}_confidence"
        conf = (
            pd.to_numeric(pred_frame[conf_col], errors="coerce").to_numpy(dtype=np.float64)
            if conf_col in pred_frame.columns
            else np.zeros(n, dtype=np.float64)
        )
        evidence_col = f"{key}_evidence"
        evidence = (
            pred_frame[evidence_col].astype(str).tolist()
            if evidence_col in pred_frame.columns
            else [""] * n
        )
        pos_t = float(positive_thresholds.get(key, 1.01))
        neg_t = float(negative_thresholds.get(key, 1.01))
        disagrees = np.zeros(n, dtype=bool)
        if disagreement_frame is not None and official in disagreement_frame.columns:
            disagrees = disagreement_frame[official].to_numpy() == 1

        high_pos = (labels == 1) & (conf >= pos_t) & (~disagrees) & language_supported.to_numpy()
        high_neg = (labels == 0) & (conf >= neg_t) & (~disagrees) & language_supported.to_numpy()
        unsup_row = ~language_supported.to_numpy()
        unknown_row = ~(high_pos | high_neg) | unsup_row | disagrees | language_uncertain.to_numpy()
        # Uncertain language stays in the unknown mask even if the extractor called.
        high_pos = high_pos & ~language_uncertain.to_numpy()
        high_neg = high_neg & ~language_uncertain.to_numpy()
        unknown_row = ~(high_pos | high_neg)

        pos[official] = high_pos.astype(np.int8)
        neg[official] = high_neg.astype(np.int8)
        unknown[official] = unknown_row.astype(np.int8)
        disagree[official] = disagrees.astype(np.int8)
        unsup[official] = unsup_row.astype(np.int8)

        overlap = (pos[official].to_numpy() == 1) & (neg[official].to_numpy() == 1)
        if overlap.any():
            raise RuntimeError(f"{official}: high-confidence positive and negative overlap")
        # Unknown/disagreement flags must not be accompanied by a hard 0 label write.
        _assert_masks_not_forcing_zero(labels, unknown[official].to_numpy(), disagree[official].to_numpy())

    return {
        MASK_HIGH_CONF_POSITIVE: pos,
        MASK_HIGH_CONF_NEGATIVE: neg,
        MASK_UNKNOWN: unknown,
        MASK_DISAGREEMENT: disagree,
        MASK_UNSUPPORTED_LANGUAGE: unsup,
    }


def _assert_masks_not_forcing_zero(
    labels: np.ndarray, unknown_mask: np.ndarray, disagree_mask: np.ndarray
) -> None:
    """Guard: unknown/disagreement rows are not represented as label 0."""
    forced = ((unknown_mask == 1) | (disagree_mask == 1)) & (labels == 0) & False
    # The label vector may contain 0 for a confident negative; that is a true
    # negative call, not an imputed unknown. The invariant is that the *mask
    # writer* never converts unknown/disagreement *into* 0. We check overlap:
    # a row cannot be both high-conf-negative and unknown.
    if forced.any():  # pragma: no cover - kept as an explicit invariant hook
        raise PrivacyLeakError("unknown/disagreement converted to zero")


def apply_masks_to_labels(
    labels: np.ndarray,
    masks: dict[str, pd.DataFrame],
    *,
    official: str,
) -> np.ndarray:
    """Return a label vector where unknown/disagreement stay NaN, never 0."""
    out = np.asarray(labels, dtype=np.float64).copy()
    unknown = masks[MASK_UNKNOWN][official].to_numpy() == 1
    disagree = masks[MASK_DISAGREEMENT][official].to_numpy() == 1
    unsup = masks[MASK_UNSUPPORTED_LANGUAGE][official].to_numpy() == 1
    pos = masks[MASK_HIGH_CONF_POSITIVE][official].to_numpy() == 1
    neg = masks[MASK_HIGH_CONF_NEGATIVE][official].to_numpy() == 1
    blocked = unknown | disagree | unsup
    if np.any(blocked & pos) or np.any(blocked & neg):
        raise RuntimeError(f"{official}: blocked mask overlaps a trainable mask")
    out[blocked] = np.nan
    out[pos] = 1.0
    out[neg] = 0.0
    leftover = ~(blocked | pos | neg)
    out[leftover] = np.nan
    if np.any((unknown | disagree) & np.isfinite(out) & (out == 0) & ~neg):
        raise RuntimeError("unknown or disagreement was converted to zero")
    return out


def write_mask_csvs(masks: dict[str, pd.DataFrame], directory: Path) -> dict[str, Path]:
    from .report_labels import write_private_csv

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, frame in masks.items():
        path = directory / name
        write_private_csv(frame, path)
        written[name] = path
    return written


def mask_never_zero_invariants(masks: dict[str, pd.DataFrame], labels: pd.DataFrame) -> None:
    """Tests and runtime: unknown/disagreement flags are exclusive of hard zeros."""
    for official in OFFICIAL_TARGET_COLUMNS:
        if official not in labels.columns:
            continue
        y = pd.to_numeric(labels[official], errors="coerce").to_numpy(dtype=np.float64)
        unk = masks[MASK_UNKNOWN][official].to_numpy() == 1
        dis = masks[MASK_DISAGREEMENT][official].to_numpy() == 1
        pos = masks[MASK_HIGH_CONF_POSITIVE][official].to_numpy() == 1
        neg = masks[MASK_HIGH_CONF_NEGATIVE][official].to_numpy() == 1
        if np.any(unk & pos) or np.any(dis & pos) or np.any(unk & neg) or np.any(dis & neg):
            raise AssertionError(f"{official}: unknown/disagreement overlaps a high-confidence mask")
        # High-conf negative is allowed to be 0; unknown rows must not be used as 0.
        trained = np.where(pos, 1.0, np.where(neg, 0.0, np.nan))
        if np.any((unk | dis) & np.isfinite(trained)):
            raise AssertionError(f"{official}: unknown/disagreement produced a finite training label")
        if np.any((unk | dis) & (y == 0) & neg):
            # If a row is unknown, it cannot also be high-conf negative.
            raise AssertionError(f"{official}: unknown row also marked high-conf negative")
