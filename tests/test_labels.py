"""Official label columns, unknown-label masking, TARGET_KEYS mapping."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from knee.config import (
    NUM_TARGETS,
    OFFICIAL_TARGET_COLUMNS,
    TARGET_KEY_TO_OFFICIAL,
    TARGET_KEYS,
)
from knee.labels import extract_label_matrix, prevalence, remap_prediction_columns
from knee.loss import masked_bce_with_logits


def test_official_columns_match_target_keys_in_order_not_spelling():
    assert len(TARGET_KEYS) == len(OFFICIAL_TARGET_COLUMNS) == NUM_TARGETS == 12
    assert list(TARGET_KEYS) != list(OFFICIAL_TARGET_COLUMNS)
    assert TARGET_KEY_TO_OFFICIAL[TARGET_KEYS[0]] == "ACL"
    assert TARGET_KEY_TO_OFFICIAL[TARGET_KEYS[9]] == "Baker's"
    assert TARGET_KEY_TO_OFFICIAL[TARGET_KEYS[-1]] == "Fracture"


def test_extract_label_matrix_official_spelling_and_unknown():
    frame = pd.DataFrame(
        {
            "StudyInstanceUID": ["a", "b"],
            "ACL": [1, 0],
            "MCL": [0, 1],
            "Medial Meniscus": [1, 1],
            "Lateral Meniscus": [0, 0],
            "Medial OA": [1, 0],
            "Lateral OA": [0, 0],
            "PF OA": [1, 1],
            "Effusion": [0, 1],
            "Synovitis": [1, 0],
            "Baker's": [np.nan, 1],
            "Contusion": [0, 0],
            "Fracture": [1, 0],
        }
    )
    labels, mask = extract_label_matrix(frame)
    assert labels.shape == (2, 12)
    assert mask[0, 9] == 0.0  # Baker's unknown in row 0
    assert np.isnan(labels[0, 9])
    assert labels[1, 9] == 1.0
    stats = prevalence(labels)
    assert stats["Baker's"]["n_unknown"] == 1
    assert stats["ACL"]["n_positive"] == 1


def test_extract_accepts_internal_keys_when_official_absent():
    row = {"StudyInstanceUID": "x"}
    row.update({key: 0 for key in TARGET_KEYS})
    row["acl_injury"] = 1
    frame = pd.DataFrame([row])
    labels, mask = extract_label_matrix(frame)
    assert labels[0, 0] == 1.0
    assert mask.all()


def test_unrecognised_label_values_become_unknown_not_zero():
    frame = pd.DataFrame([{col: 0 for col in OFFICIAL_TARGET_COLUMNS}])
    frame["ACL"] = 2
    labels, mask = extract_label_matrix(frame)
    assert np.isnan(labels[0, 0])
    assert mask[0, 0] == 0.0


def test_masked_bce_ignores_nan_targets():
    logits = torch.zeros(2, 12)
    targets = torch.zeros(2, 12)
    targets[0, 0] = float("nan")
    loss = masked_bce_with_logits(logits, targets)
    assert torch.isfinite(loss)
    import torch.nn.functional as F

    expected = F.binary_cross_entropy_with_logits(torch.zeros(23), torch.zeros(23))
    assert loss.item() == pytest.approx(expected.item(), rel=1e-5)


def test_remap_prediction_columns_from_internal_keys():
    frame = pd.DataFrame({"study_id": ["s1"]})
    for key in TARGET_KEYS:
        frame[key] = 0.25
    mapped = remap_prediction_columns(frame)
    assert list(mapped.columns) == ["StudyInstanceUID", *OFFICIAL_TARGET_COLUMNS]
    assert mapped.loc[0, "Baker's"] == 0.25
