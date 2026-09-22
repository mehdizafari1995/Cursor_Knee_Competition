"""Coverage-aware evaluation for the gold-only second-reader pilot."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import OFFICIAL_TARGET_COLUMNS, STUDY_ID_COLUMN, TARGET_KEYS, TARGET_KEY_TO_OFFICIAL
from .coverage_metrics import (
    BOOTSTRAP_METRIC_NAMES,
    FULL_METRIC_NAMES,
    bootstrap_coverage_intervals,
    coverage_confusion,
    json_safe_metrics,
    macro_mean,
)
from .labels import extract_label_matrix
from .report_labels import LABEL_NEGATIVE, LABEL_POSITIVE, label_to_float
from .second_reader_schema import official_name

PILOT_BOOTSTRAP_METRICS = BOOTSTRAP_METRIC_NAMES + ("selective_error",)


def ternary_matrix_from_frame(frame: pd.DataFrame, *, prefix: str = "") -> np.ndarray:
    """Official columns if present, else ``{prefix}{target_key}`` float columns."""
    if all(col in frame.columns for col in OFFICIAL_TARGET_COLUMNS):
        labels, _ = extract_label_matrix(frame, require_all=False)
        return labels
    columns = []
    for key in TARGET_KEYS:
        name = f"{prefix}{key}" if prefix else TARGET_KEY_TO_OFFICIAL[key]
        if name not in frame.columns:
            raise KeyError(f"missing prediction column {name}")
        columns.append(pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float64))
    return np.column_stack(columns)


def labels_dict_to_floats(labels: dict[str, str]) -> dict[str, float]:
    return {official_name(key): label_to_float(labels.get(key, "unknown")) for key in TARGET_KEYS}


def evaluate_system_against_gold(
    gold_frame: pd.DataFrame,
    pred_frame: pd.DataFrame,
    *,
    n_resamples: int = 1000,
    seed: int = 42,
    ci_level: float = 0.95,
    system_name: str,
    malformed_by_study: Iterable[bool] | None = None,
) -> dict[str, Any]:
    """Coverage-aware metrics. Abstentions are not false negatives."""
    gold_ids = gold_frame[STUDY_ID_COLUMN].astype(str)
    pred = pred_frame.copy()
    pred[STUDY_ID_COLUMN] = pred[STUDY_ID_COLUMN].astype(str)
    missing = [sid for sid in gold_ids if sid not in set(pred[STUDY_ID_COLUMN])]
    if missing:
        raise ValueError(f"{system_name} missing {len(missing)} gold StudyInstanceUID values")
    aligned = pred.set_index(STUDY_ID_COLUMN).loc[list(gold_ids)].reset_index()
    gold_labels, _ = extract_label_matrix(gold_frame, require_all=True)
    pred_labels = ternary_matrix_from_frame(aligned)
    malformed = (
        np.asarray(list(malformed_by_study), dtype=bool)
        if malformed_by_study is not None
        else np.zeros(len(gold_ids), dtype=bool)
    )
    if malformed.shape[0] != len(gold_ids):
        raise ValueError("malformed_by_study length must match gold studies")

    per_target: dict[str, dict] = {}
    for i, key in enumerate(TARGET_KEYS):
        official = OFFICIAL_TARGET_COLUMNS[i]
        stats = coverage_confusion(gold_labels[:, i], pred_labels[:, i])
        stats["ci"] = bootstrap_coverage_intervals(
            gold_labels[:, i],
            pred_labels[:, i],
            n_resamples=n_resamples,
            seed=seed + i,
            ci_level=ci_level,
            metric_names=PILOT_BOOTSTRAP_METRICS,
        )
        stats["target_key"] = key
        stats["official_column"] = official
        stats["system"] = system_name
        stats["n_malformed_studies"] = int(malformed.sum())
        stats["malformed_output_rate"] = float(malformed.mean()) if len(malformed) else float("nan")
        per_target[official] = json_safe_metrics(stats)

    macro = json_safe_metrics(macro_mean(per_target, FULL_METRIC_NAMES))
    return {
        "system": system_name,
        "n_studies": int(len(gold_ids)),
        "malformed_output_rate": float(malformed.mean()) if len(malformed) else 0.0,
        "n_malformed_studies": int(malformed.sum()),
        "unknown_imputed_as_negative": False,
        "conditional_sensitivity_is_not_full_set_sensitivity": True,
        "per_target": per_target,
        "macro": macro,
    }
