"""Multilabel, group-aware cross-validation with deterministic fold assignments.

Grouping prefers a patient identifier when one is present. When the released
tables have no patient column, ``StudyInstanceUID`` is the finest grouping key
available; that limitation is recorded rather than papered over.

Every validation fold is required to contain at least one positive example of
each target whenever that is mathematically possible (group-level positives
>= n_folds). Rare targets with fewer distinct positive groups than folds are
reported, not silently treated as balanced.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import NUM_TARGETS, OFFICIAL_TARGET_COLUMNS, STUDY_ID_COLUMN
from .labels import extract_label_matrix

FOLD_COLUMNS = (STUDY_ID_COLUMN, "group_id", "fold")


class FoldError(ValueError):
    """Fold construction failed a leakage or coverage invariant."""


def choose_group_ids(
    study_ids: np.ndarray,
    patient_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, str]:
    """Return (group_id, source_name).

    ``patient_ids`` is used only when every study has a non-empty value. A
    partially populated patient column is *not* used, because missing IDs would
    be treated as distinct patients and hide leakage.
    """
    study_ids = np.asarray(study_ids, dtype=object)
    if patient_ids is None:
        return study_ids, STUDY_ID_COLUMN
    patients = np.asarray(patient_ids, dtype=object)
    if len(patients) != len(study_ids):
        raise FoldError("patient_ids and study_ids have different lengths")
    cleaned = np.array(
        ["" if v is None or (isinstance(v, float) and not np.isfinite(v)) else str(v).strip()
         for v in patients],
        dtype=object,
    )
    if np.all(cleaned != ""):
        return cleaned, "PatientID"
    return study_ids, STUDY_ID_COLUMN


def _group_label_matrix(
    group_ids: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse study labels to one row per group with nanmax (unknown stays unknown)."""
    groups, inverse = np.unique(group_ids, return_inverse=True)
    grouped = np.full((len(groups), labels.shape[1]), np.nan, dtype=np.float32)
    for i, group_index in enumerate(inverse):
        row = labels[i]
        current = grouped[group_index]
        known = np.isfinite(row)
        both = np.isfinite(current) & known
        current[both] = np.maximum(current[both], row[both])
        only_new = known & ~np.isfinite(current)
        current[only_new] = row[only_new]
        grouped[group_index] = current
    return groups, grouped


def _positive_coverage(fold_ids: np.ndarray, labels: np.ndarray, n_folds: int) -> np.ndarray:
    """Shape (n_folds, n_targets): number of *studies* with a known positive in that fold."""
    coverage = np.zeros((n_folds, labels.shape[1]), dtype=np.int64)
    known_pos = np.isfinite(labels) & (labels == 1)
    for fold in range(n_folds):
        coverage[fold] = known_pos[fold_ids == fold].sum(axis=0)
    return coverage


def assign_grouped_folds(
    study_ids: np.ndarray,
    group_ids: np.ndarray,
    labels: np.ndarray,
    *,
    n_folds: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, dict]:
    """Greedy multilabel iterative stratification over groups.

    Returns:
        fold_ids: int array, one entry per study, values in ``range(n_folds)``
        metadata: coverage, grouping key notes, and impossible-target list
    """
    study_ids = np.asarray(study_ids, dtype=object)
    group_ids = np.asarray(group_ids, dtype=object)
    labels = np.asarray(labels, dtype=np.float32)
    n = len(study_ids)
    if not (len(group_ids) == n == len(labels)):
        raise FoldError("study_ids, group_ids and labels must have the same length")
    if labels.ndim != 2 or labels.shape[1] != NUM_TARGETS:
        raise FoldError(f"labels must have shape (n, {NUM_TARGETS})")
    if n_folds < 2:
        raise FoldError("n_folds must be >= 2")
    if n < n_folds:
        raise FoldError(f"need at least {n_folds} studies to fill {n_folds} folds")

    groups, grouped_labels = _group_label_matrix(group_ids, labels)
    rng = np.random.RandomState(seed)
    order = np.arange(len(groups))
    # Sort by id first so the subsequent permutation is independent of input row order.
    order = order[np.argsort(groups)]
    order = rng.permutation(order)

    fold_pos = np.zeros((n_folds, NUM_TARGETS), dtype=np.float64)
    fold_n = np.zeros(n_folds, dtype=np.int64)
    group_to_fold: dict[str, int] = {}

    for gi in order:
        y = grouped_labels[gi]
        present = np.isfinite(y) & (y == 1)
        scores = fold_n.astype(np.float64).copy()
        if present.any():
            scores = scores + fold_pos[:, present].sum(axis=1)
        fold = int(np.argmin(scores))
        group_to_fold[str(groups[gi])] = fold
        fold_n[fold] += 1
        fold_pos[fold, present] += 1

    group_to_fold = _repair_positive_coverage(
        groups, grouped_labels, group_to_fold, n_folds
    )

    fold_ids = np.array([group_to_fold[str(g)] for g in group_ids], dtype=np.int64)
    _assert_no_group_leakage(group_ids, fold_ids)

    coverage = _positive_coverage(fold_ids, labels, n_folds)
    group_pos = (np.isfinite(grouped_labels) & (grouped_labels == 1)).sum(axis=0)
    impossible = [
        OFFICIAL_TARGET_COLUMNS[t]
        for t in range(NUM_TARGETS)
        if int(group_pos[t]) < n_folds
    ]
    uncovered = [
        {
            "fold": int(f),
            "target": OFFICIAL_TARGET_COLUMNS[t],
            "positives_in_fold": int(coverage[f, t]),
            "mathematically_possible": OFFICIAL_TARGET_COLUMNS[t] not in impossible,
        }
        for f in range(n_folds)
        for t in range(NUM_TARGETS)
        if coverage[f, t] == 0
    ]
    leftover = [
        row for row in uncovered if row["mathematically_possible"]
    ]
    if leftover:
        raise FoldError(
            "validation folds still lack positives for targets that have enough "
            f"distinct positive groups: {leftover}"
        )

    metadata = {
        "n_folds": n_folds,
        "seed": seed,
        "n_studies": int(n),
        "n_groups": int(len(groups)),
        "impossible_full_coverage_targets": impossible,
        "uncovered_fold_targets": uncovered,
        "positives_per_fold": {
            OFFICIAL_TARGET_COLUMNS[t]: [int(coverage[f, t]) for f in range(n_folds)]
            for t in range(NUM_TARGETS)
        },
    }
    return fold_ids, metadata


def _repair_positive_coverage(
    groups: np.ndarray,
    grouped_labels: np.ndarray,
    group_to_fold: dict[str, int],
    n_folds: int,
) -> dict[str, int]:
    """Move a positive group into an empty fold when that target has enough groups."""
    assignment = dict(group_to_fold)
    for t in range(grouped_labels.shape[1]):
        pos_groups = [
            str(g)
            for g, row in zip(groups, grouped_labels)
            if np.isfinite(row[t]) and row[t] == 1
        ]
        if len(pos_groups) < n_folds:
            continue
        for fold in range(n_folds):
            has_pos = any(assignment[g] == fold for g in pos_groups)
            if has_pos:
                continue
            donor = None
            for g in pos_groups:
                src = assignment[g]
                src_count = sum(1 for other in pos_groups if assignment[other] == src)
                if src_count >= 2:
                    donor = g
                    break
            if donor is None:
                continue
            assignment[donor] = fold
    return assignment


def _assert_no_group_leakage(group_ids: np.ndarray, fold_ids: np.ndarray) -> None:
    mapping: dict[str, int] = {}
    for group, fold in zip(group_ids, fold_ids):
        key = str(group)
        if key in mapping and mapping[key] != int(fold):
            raise FoldError(f"group {key!r} was assigned to multiple folds")
        mapping[key] = int(fold)


def assert_no_group_leakage(frame: pd.DataFrame) -> None:
    """Public invariant used by tests: no group_id appears in two folds."""
    if not {"group_id", "fold"} <= set(frame.columns):
        raise FoldError("fold table must contain group_id and fold")
    grouped = frame.groupby("group_id")["fold"].nunique()
    offenders = grouped[grouped > 1]
    if len(offenders):
        raise FoldError(
            f"{len(offenders)} groups appear in more than one fold"
        )


def folds_from_study_table(
    frame: pd.DataFrame,
    *,
    n_folds: int = 5,
    seed: int = 42,
    patient_column: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Build a fold table from a study-level frame (train.csv or equivalent)."""
    if STUDY_ID_COLUMN not in frame.columns:
        raise FoldError(f"{STUDY_ID_COLUMN} is required")
    study_ids = frame[STUDY_ID_COLUMN].astype(str).to_numpy()
    patient_ids = None
    source = STUDY_ID_COLUMN
    column = patient_column
    if column is None:
        for candidate in ("PatientID", "patient_id", "patientId"):
            if candidate in frame.columns:
                column = candidate
                break
    if column is not None:
        patient_ids = frame[column].to_numpy()
    group_ids, source = choose_group_ids(study_ids, patient_ids)
    labels, _ = extract_label_matrix(frame, require_all=False)
    fold_ids, metadata = assign_grouped_folds(
        study_ids, group_ids, labels, n_folds=n_folds, seed=seed
    )
    metadata["group_source"] = source
    table = pd.DataFrame(
        {
            STUDY_ID_COLUMN: study_ids,
            "group_id": group_ids,
            "fold": fold_ids,
        }
    )
    assert_no_group_leakage(table)
    return table, metadata


def save_folds(table: pd.DataFrame, path: Path, metadata: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    if metadata is not None:
        sidecar = path.with_suffix(".meta.json")
        import json

        sidecar.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def load_folds(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, dtype={STUDY_ID_COLUMN: str, "group_id": str})
    assert_no_group_leakage(table)
    return table
