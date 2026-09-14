"""Patient/study grouped folds: leakage, coverage, determinism."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from knee.config import NUM_TARGETS, OFFICIAL_TARGET_COLUMNS, STUDY_ID_COLUMN
from knee.folds import (
    FoldError,
    assert_no_group_leakage,
    assign_grouped_folds,
    folds_from_study_table,
    save_folds,
)


def _balanced_frame(n_patients: int = 20, studies_per_patient: int = 2, seed: int = 0):
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_patients):
        pid = f"P{p:03d}"
        for s in range(studies_per_patient):
            row = {
                STUDY_ID_COLUMN: f"study-{p:03d}-{s}",
                "PatientID": pid,
            }
            for col in OFFICIAL_TARGET_COLUMNS:
                row[col] = int(rng.random() < 0.4)
            rows.append(row)
    # Guarantee every target has plenty of positive patients.
    frame = pd.DataFrame(rows)
    for i, col in enumerate(OFFICIAL_TARGET_COLUMNS):
        pos_patients = frame.loc[frame[col] == 1, "PatientID"].unique()
        if len(pos_patients) < 8:
            extra = frame["PatientID"].unique()[:8]
            frame.loc[frame["PatientID"].isin(extra), col] = 1
    return frame


def test_no_patient_crosses_folds():
    frame = _balanced_frame()
    table, metadata = folds_from_study_table(frame, n_folds=5, seed=42)
    assert_no_group_leakage(table)
    grouped = table.groupby("group_id")["fold"].nunique()
    assert (grouped == 1).all()
    assert metadata["group_source"] == "PatientID"
    assert set(table["fold"]) == {0, 1, 2, 3, 4}
    # Two studies per patient stay together.
    studies_per_group = table.groupby("group_id").size()
    assert set(studies_per_group) == {2}


def test_every_fold_has_positives_when_possible():
    frame = _balanced_frame()
    table, metadata = folds_from_study_table(frame, n_folds=5, seed=1)
    merged = frame.merge(table, on=STUDY_ID_COLUMN)
    for fold in range(5):
        subset = merged[merged["fold"] == fold]
        for col in OFFICIAL_TARGET_COLUMNS:
            assert (subset[col] == 1).any(), f"fold {fold} missing positives for {col}"
    assert metadata["impossible_full_coverage_targets"] == []


def test_rare_target_is_reported_as_impossible_not_fabricated():
    frame = _balanced_frame()
    # Only two patients are positive for Fracture — cannot cover 5 folds.
    frame["Fracture"] = 0
    rare = frame["PatientID"].unique()[:2]
    frame.loc[frame["PatientID"].isin(rare), "Fracture"] = 1
    table, metadata = folds_from_study_table(frame, n_folds=5, seed=0)
    assert "Fracture" in metadata["impossible_full_coverage_targets"]
    assert_no_group_leakage(table)


def test_fold_assignment_is_deterministic(tmp_path):
    frame = _balanced_frame(seed=3)
    a, meta_a = folds_from_study_table(frame, n_folds=5, seed=99)
    b, meta_b = folds_from_study_table(frame, n_folds=5, seed=99)
    pd.testing.assert_frame_equal(a, b)
    assert meta_a["seed"] == meta_b["seed"] == 99
    save_folds(a, tmp_path / "folds.csv", meta_a)
    loaded = pd.read_csv(tmp_path / "folds.csv")
    assert list(loaded.columns) == [STUDY_ID_COLUMN, "group_id", "fold"]


def test_shuffled_input_rows_do_not_change_assignment():
    frame = _balanced_frame(seed=4)
    table, _ = folds_from_study_table(frame, n_folds=5, seed=7)
    shuffled = frame.sample(frac=1.0, random_state=1).reset_index(drop=True)
    again, _ = folds_from_study_table(shuffled, n_folds=5, seed=7)
    left = table.set_index(STUDY_ID_COLUMN).sort_index()
    right = again.set_index(STUDY_ID_COLUMN).sort_index()
    pd.testing.assert_frame_equal(left, right)


def test_falls_back_to_study_when_patient_id_incomplete():
    frame = _balanced_frame(n_patients=10)
    frame.loc[0, "PatientID"] = np.nan
    table, metadata = folds_from_study_table(frame, n_folds=5, seed=0)
    assert metadata["group_source"] == STUDY_ID_COLUMN
    assert table["group_id"].tolist() == table[STUDY_ID_COLUMN].tolist()


def test_detects_group_leakage():
    bad = pd.DataFrame(
        {
            "group_id": ["P1", "P1", "P2"],
            "fold": [0, 1, 0],
        }
    )
    with pytest.raises(FoldError):
        assert_no_group_leakage(bad)


def test_assign_grouped_folds_rejects_wrong_label_width():
    with pytest.raises(FoldError):
        assign_grouped_folds(
            np.array(["s1", "s2", "s3"]),
            np.array(["p1", "p2", "p3"]),
            np.zeros((3, 3), dtype=np.float32),
            n_folds=2,
            seed=0,
        )
    assert NUM_TARGETS == 12
