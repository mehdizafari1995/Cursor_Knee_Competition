"""Trainable masks never convert unknown or disagreement to zero."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from knee.config import OFFICIAL_TARGET_COLUMNS, STUDY_ID_COLUMN, TARGET_KEYS
from knee.report_masks import (
    MASK_DISAGREEMENT,
    MASK_HIGH_CONF_NEGATIVE,
    MASK_HIGH_CONF_POSITIVE,
    MASK_UNKNOWN,
    MASK_UNSUPPORTED_LANGUAGE,
    apply_masks_to_labels,
    build_trainable_masks,
    mask_never_zero_invariants,
    summarize_masks,
    write_mask_csvs,
)
from knee.report_second_reader import (
    SECOND_READER_FORBIDDEN,
    combine_conservative,
    run_second_reader,
)
import inspect
import json


def _frame():
    n = 4
    data = {STUDY_ID_COLUMN: [f"id{i}" for i in range(n)]}
    for official in OFFICIAL_TARGET_COLUMNS:
        data[official] = [1.0, 0.0, np.nan, 1.0]
    for key in TARGET_KEYS:
        data[f"{key}_confidence"] = [0.95, 0.95, 0.0, 0.40]
        data[f"{key}_evidence"] = ["explicit_positive", "negated_finding", "insufficient_evidence", "explicit_positive"]
    data["quality_flags"] = ["", "", "unsupported_language", ""]
    data["language_uncertain"] = [False, False, False, False]
    return pd.DataFrame(data)


def test_unknown_and_disagreement_stay_nan_not_zero():
    frame = _frame()
    pos_t = {key: 0.85 for key in TARGET_KEYS}
    neg_t = {key: 0.85 for key in TARGET_KEYS}
    disagree = pd.DataFrame({STUDY_ID_COLUMN: frame[STUDY_ID_COLUMN]})
    for official in OFFICIAL_TARGET_COLUMNS:
        disagree[official] = [0, 0, 0, 1]
    masks = build_trainable_masks(
        frame, positive_thresholds=pos_t, negative_thresholds=neg_t, disagreement_frame=disagree
    )
    mask_never_zero_invariants(masks, frame)
    official = OFFICIAL_TARGET_COLUMNS[0]
    y = apply_masks_to_labels(frame[official].to_numpy(dtype=np.float64), masks, official=official)
    assert y[0] == 1.0
    assert y[1] == 0.0
    assert np.isnan(y[2])  # unsupported language
    assert np.isnan(y[3])  # disagreement
    assert not np.any((masks[MASK_UNKNOWN][official].to_numpy() == 1) & np.isfinite(y) & (y == 0) & (masks[MASK_HIGH_CONF_NEGATIVE][official].to_numpy() != 1))
    assert int(masks[MASK_UNSUPPORTED_LANGUAGE][official].iloc[2]) == 1
    assert int(masks[MASK_DISAGREEMENT][official].iloc[3]) == 1
    assert int(masks[MASK_UNSUPPORTED_LANGUAGE][official].iloc[3]) == 0
    assert int(masks[MASK_DISAGREEMENT][official].iloc[2]) == 0


def test_all_zero_disagreement_and_unsupported_masks_can_share_a_hash(tmp_path):
    """Identical SHA-256 is expected when both independent writers are all-zero."""
    n = 5
    data = {STUDY_ID_COLUMN: [f"id{i}" for i in range(n)]}
    for official in OFFICIAL_TARGET_COLUMNS:
        data[official] = [np.nan] * n
    for key in TARGET_KEYS:
        data[f"{key}_confidence"] = [0.0] * n
        data[f"{key}_evidence"] = ["insufficient_evidence"] * n
    data["quality_flags"] = [""] * n
    data["language_uncertain"] = [False] * n
    frame = pd.DataFrame(data)
    pos_t = {key: 0.85 for key in TARGET_KEYS}
    neg_t = {key: 0.85 for key in TARGET_KEYS}
    masks = build_trainable_masks(frame, positive_thresholds=pos_t, negative_thresholds=neg_t)
    assert masks[MASK_DISAGREEMENT] is not masks[MASK_UNSUPPORTED_LANGUAGE]
    summary = summarize_masks(masks)
    assert summary["files"][MASK_DISAGREEMENT]["all_zero"] is True
    assert summary["files"][MASK_UNSUPPORTED_LANGUAGE]["all_zero"] is True
    assert summary["disagreement_and_unsupported_target_cells_identical"] is True
    written = write_mask_csvs(masks, tmp_path)
    disagree_bytes = written[MASK_DISAGREEMENT].read_bytes()
    unsup_bytes = written[MASK_UNSUPPORTED_LANGUAGE].read_bytes()
    assert disagree_bytes == unsup_bytes
    import hashlib

    assert hashlib.sha256(disagree_bytes).hexdigest() == hashlib.sha256(unsup_bytes).hexdigest()


def test_unsupported_flag_makes_mask_hashes_diverge(tmp_path):
    frame = _frame()
    pos_t = {key: 0.85 for key in TARGET_KEYS}
    neg_t = {key: 0.85 for key in TARGET_KEYS}
    masks = build_trainable_masks(frame, positive_thresholds=pos_t, negative_thresholds=neg_t)
    summary = summarize_masks(masks)
    assert summary["files"][MASK_DISAGREEMENT]["n_true_cells"] == 0
    assert summary["files"][MASK_UNSUPPORTED_LANGUAGE]["n_true_cells"] == len(OFFICIAL_TARGET_COLUMNS)
    assert summary["disagreement_and_unsupported_target_cells_identical"] is False
    blob = json.dumps(summary)
    assert STUDY_ID_COLUMN not in blob or "id0" not in blob
    for i in range(4):
        assert f"id{i}" not in blob
    written = write_mask_csvs(masks, tmp_path)
    assert written[MASK_DISAGREEMENT].read_bytes() != written[MASK_UNSUPPORTED_LANGUAGE].read_bytes()


def test_disagreement_frame_does_not_copy_unsupported_mask():
    frame = _frame()
    pos_t = {key: 0.85 for key in TARGET_KEYS}
    neg_t = {key: 0.85 for key in TARGET_KEYS}
    disagree = pd.DataFrame({STUDY_ID_COLUMN: frame[STUDY_ID_COLUMN]})
    for official in OFFICIAL_TARGET_COLUMNS:
        disagree[official] = [1, 0, 0, 0]
    masks = build_trainable_masks(
        frame, positive_thresholds=pos_t, negative_thresholds=neg_t, disagreement_frame=disagree
    )
    official = OFFICIAL_TARGET_COLUMNS[0]
    assert int(masks[MASK_DISAGREEMENT][official].iloc[0]) == 1
    assert int(masks[MASK_UNSUPPORTED_LANGUAGE][official].iloc[0]) == 0
    assert int(masks[MASK_UNSUPPORTED_LANGUAGE][official].iloc[2]) == 1
    assert int(masks[MASK_DISAGREEMENT][official].iloc[2]) == 0


def test_high_conf_positive_and_negative_are_exclusive():
    frame = _frame()
    pos_t = {key: 0.85 for key in TARGET_KEYS}
    neg_t = {key: 0.85 for key in TARGET_KEYS}
    masks = build_trainable_masks(frame, positive_thresholds=pos_t, negative_thresholds=neg_t)
    official = "ACL"
    assert not np.any(
        (masks[MASK_HIGH_CONF_POSITIVE][official].to_numpy() == 1)
        & (masks[MASK_HIGH_CONF_NEGATIVE][official].to_numpy() == 1)
    )


def test_second_reader_signature_has_no_gold_and_does_not_download():
    params = set(inspect.signature(run_second_reader).parameters)
    assert not (params & SECOND_READER_FORBIDDEN)
    result = run_second_reader("FINDINGS: ACL tear.", config={"second_reader": {"enabled": False, "download_weights": False}})
    assert result.enabled is False
    assert result.weights_sha256 is None
    with pytest.raises(RuntimeError):
        run_second_reader("x", config={"second_reader": {"enabled": False, "download_weights": True}})


def test_conservative_combiner_disagreement_is_unknown():
    from knee.report_second_reader import SecondReaderTarget, LABEL_POSITIVE, LABEL_NEGATIVE, LABEL_UNKNOWN

    reader = SecondReaderTarget(
        target_key="acl_injury",
        label=LABEL_NEGATIVE,
        uncertainty=0.2,
        evidence_class="model_call",
        supported_language=True,
    )
    label, reason, _band = combine_conservative(
        LABEL_POSITIVE, "explicit_positive", reader, language_supported=True, global_normal=False
    )
    assert label == LABEL_UNKNOWN
    assert reason == "disagreement"
    label2, reason2, _ = combine_conservative(
        LABEL_UNKNOWN, "insufficient_evidence", reader, language_supported=False, global_normal=False
    )
    assert label2 == LABEL_UNKNOWN
    assert reason2 == "unsupported_language"


def test_production_labeler_omits_disagreement_frame():
    from pathlib import Path

    text = Path("knee/report_labels.py").read_text(encoding="utf-8")
    assert "disagreement_frame=" not in text
    assert "build_trainable_masks(" in text
