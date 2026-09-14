"""Three-level real-data audit used by the internet-off Kaggle notebook.

Levels:

* ``full_metadata`` — CSV and series-table statistics; no DICOM files opened.
* ``one_header_per_study`` — one representative header per training study,
  pixels not loaded.
* ``sampled_dicom`` — all series and a few pixel arrays for a documented,
  deterministic sample only.

This module never trains a model, never reports AUC, never writes under
``/kaggle/input``, and never exports radiology-report text.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .audit import (
    STATUS_AUDITED,
    STATUS_DATA_NOT_AVAILABLE,
    _audit_labels,
    _audit_reports,
    _audit_series_table,
    _drop_frame,
    _leakage_checks,
    _profile_csv_id,
    _read_csv,
    _structural_diff,
    _target_reconciliation,
    data_not_available_result,
)
from .competition import (
    DOCUMENTED_DIRECTORIES,
    DOCUMENTED_FILES,
    DOCUMENTED_SERIES_COLUMNS,
    SAMPLE_SUBMISSION_COLUMNS,
)
from .config import (
    NUM_TARGETS,
    OFFICIAL_TARGET_COLUMNS,
    REPORT_COLUMN,
    SERIES_ID_COLUMN,
    STUDY_ID_COLUMN,
    TARGET_KEYS,
)
from .dicom_io import (
    DicomReadError,
    header_summary,
    infer_plane_from_orientation,
    list_dicom_paths,
    read_header,
)
from .kaggle_runtime import (
    DATA_SOURCE_REAL_KAGGLE,
    DATA_SOURCE_SOFTWARE_FIXTURE,
    KaggleLayout,
    PreflightError,
    PreflightResult,
    ReadOnlyInputGuard,
    assert_not_kaggle_input,
    count_child_dirs,
    preflight,
    scientific_evaluation_for,
)
from .labels import extract_label_matrix

PROVENANCE_FULL_METADATA = "full_metadata"
PROVENANCE_ONE_HEADER = "one_header_per_study"
PROVENANCE_SAMPLED = "sampled_dicom"
PROVENANCE_VALUES = (
    PROVENANCE_FULL_METADATA,
    PROVENANCE_ONE_HEADER,
    PROVENANCE_SAMPLED,
)

OUTPUT_FILENAMES = (
    "real_data_audit.md",
    "real_data_audit.json",
    "target_prevalence.csv",
    "series_summary.csv",
    "dicom_header_summary.csv",
    "dicom_qc_sample.csv",
    "audit_manifest.json",
)

FORBIDDEN_TEXT_COLUMNS = {
    REPORT_COLUMN,
    "report",
    "report_text",
    "ReportText",
    "clinical_report",
}

DEFAULT_SAMPLE_SIZE = 24
DEFAULT_PIXELS_PER_SERIES = 1
SAMPLE_SIZE_ENV = "RSNA_AUDIT_SAMPLE_SIZE"


class ReportTextLeakError(RuntimeError):
    """An audit artefact contained radiology-report text and was rejected."""


def jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        if isinstance(obj, float) and not np.isfinite(obj):
            return None
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        val = float(obj)
        return val if np.isfinite(val) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, pd.DataFrame):
        raise TypeError("DataFrames must not be serialized into the audit JSON")
    return str(obj)


def configured_sample_size(explicit: int | None = None) -> int:
    if explicit is not None:
        return max(int(explicit), 0)
    raw = os.environ.get(SAMPLE_SIZE_ENV)
    if raw:
        return max(int(raw), 0)
    return DEFAULT_SAMPLE_SIZE


@dataclass
class KaggleAuditConfig:
    sample_size: int = DEFAULT_SAMPLE_SIZE
    max_pixel_arrays_per_series: int = DEFAULT_PIXELS_PER_SERIES
    data_source: str | None = None
    extract_source: bool = False


def collect_report_snippets(*frames: pd.DataFrame) -> list[str]:
    snippets: list[str] = []
    for frame in frames:
        if frame is None or not isinstance(frame, pd.DataFrame):
            continue
        for column in frame.columns:
            if str(column) not in FORBIDDEN_TEXT_COLUMNS:
                continue
            for value in frame[column].tolist():
                if value is None or (isinstance(value, float) and not np.isfinite(value)):
                    continue
                text = str(value).strip()
                if text:
                    snippets.append(text)
    # Longest first so a short token that is a subset still matches.
    unique = sorted(set(snippets), key=len, reverse=True)
    return unique


def _blob_from_outputs(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def assert_no_report_text(paths: list[Path], snippets: list[str]) -> None:
    if not snippets:
        return
    hits: list[str] = []
    for path in paths:
        if not path.is_file():
            continue
        blob = _blob_from_outputs(path)
        for snippet in snippets:
            if snippet and snippet in blob:
                hits.append(f"{path.name}: {snippet[:80]}")
    if hits:
        raise ReportTextLeakError(
            "audit outputs contain clinical report text, which is forbidden: "
            + "; ".join(hits[:5])
        )


def _expected_series_dir(root: Path, split: str, study: str, series: str) -> Path:
    return root / f"{split}_series" / str(study) / str(series)


def _split_directory_counts(root: Path, split: str) -> dict:
    split_dir = root / f"{split}_series"
    return {
        "split": split,
        "directory_present": split_dir.is_dir(),
        "n_study_dirs": count_child_dirs(split_dir, 1),
        "n_series_dirs": count_child_dirs(split_dir, 2),
        "provenance": PROVENANCE_FULL_METADATA,
        "note": (
            "Study/series directory counts only. DICOM instance files were not "
            "listed or opened at this level."
        ),
    }


def _verify_columns(info: dict, expected: tuple[str, ...] | None) -> dict:
    summary = _drop_frame(info)
    if expected is None or not info.get("readable"):
        return summary
    columns = list(info.get("columns") or [])
    summary["expected_columns"] = list(expected)
    summary["missing_expected_columns"] = [c for c in expected if c not in columns]
    summary["column_order_matches_expected"] = columns[: len(expected)] == list(expected)
    summary["provenance"] = PROVENANCE_FULL_METADATA
    return summary


def select_detailed_sample(
    train: pd.DataFrame,
    series: pd.DataFrame | None,
    *,
    sample_size: int,
) -> list[dict]:
    """Deterministic sample: per-target positives, negatives, sequence diversity."""
    if sample_size < 1 or train.empty or STUDY_ID_COLUMN not in train.columns:
        return []
    frame = train.copy()
    frame[STUDY_ID_COLUMN] = frame[STUDY_ID_COLUMN].astype(str)
    frame = frame.sort_values(STUDY_ID_COLUMN, kind="mergesort").reset_index(drop=True)
    labels, _mask = extract_label_matrix(frame, require_all=False)

    selected: list[dict] = []
    seen: set[str] = set()

    def add(idx: int, reason: str) -> bool:
        if len(selected) >= sample_size:
            return False
        uid = str(frame.iloc[idx][STUDY_ID_COLUMN])
        if uid in seen:
            return False
        selected.append(
            {
                STUDY_ID_COLUMN: uid,
                "sample_reason": reason,
                "provenance": PROVENANCE_SAMPLED,
            }
        )
        seen.add(uid)
        return True

    for i, official in enumerate(OFFICIAL_TARGET_COLUMNS):
        positives = np.flatnonzero(np.isfinite(labels[:, i]) & (labels[:, i] == 1))
        for idx in positives:
            if add(int(idx), f"positive:{official}"):
                break

    known_pos = np.isfinite(labels) & (labels == 1)
    negatives = np.flatnonzero(~known_pos.any(axis=1))
    for idx in negatives:
        if add(int(idx), "negative"):
            break

    if series is not None and not series.empty and STUDY_ID_COLUMN in series.columns:
        series_frame = series.copy()
        series_frame[STUDY_ID_COLUMN] = series_frame[STUDY_ID_COLUMN].astype(str)
        plane_col = "Anatomical_Plane" if "Anatomical_Plane" in series_frame.columns else None
        seen_planes: set[str] = set()
        if plane_col:
            for row in selected:
                planes = series_frame.loc[
                    series_frame[STUDY_ID_COLUMN] == row[STUDY_ID_COLUMN], plane_col
                ].astype(str)
                seen_planes.update(planes.tolist())
            for uid in sorted(series_frame[STUDY_ID_COLUMN].unique()):
                if len(selected) >= sample_size:
                    break
                planes = set(
                    series_frame.loc[series_frame[STUDY_ID_COLUMN] == uid, plane_col].astype(str)
                )
                if planes - seen_planes:
                    matches = frame.index[frame[STUDY_ID_COLUMN] == uid].tolist()
                    if matches and add(int(matches[0]), "sequence_diversity"):
                        seen_planes.update(planes)

    for idx in range(len(frame)):
        if len(selected) >= sample_size:
            break
        add(idx, "fill")
    return selected


def _pixel_qc(path: Path) -> dict:
    try:
        ds = read_header(path, stop_before_pixels=False)
        array = np.asarray(ds.pixel_array)
    except Exception as exc:  # noqa: BLE001 - sample QC must survive corrupt files
        return {"readable_pixels": False, "error": str(exc), "path_suffix": path.suffix}
    if array.ndim == 3:
        array = array[0]
    finite = np.isfinite(array.astype(np.float64))
    values = array.astype(np.float64)
    return {
        "readable_pixels": True,
        "pixel_ndim": int(array.ndim),
        "pixel_shape": "x".join(str(d) for d in array.shape),
        "pixel_dtype": str(array.dtype),
        "pixel_min": float(values.min()) if array.size else None,
        "pixel_max": float(values.max()) if array.size else None,
        "pixel_mean": float(values.mean()) if array.size else None,
        "fraction_zero": float((array == 0).mean()) if array.size else None,
        "fraction_finite": float(finite.mean()) if array.size else None,
        "all_zero": bool(array.size and np.all(array == 0)),
    }


def audit_full_metadata(root: Path, files: dict[str, dict]) -> dict:
    """Level A: CSVs and directory counts. Opens no DICOM files."""
    train = files["train.csv"]
    test = files["test.csv"]
    train_series = files["train_series.csv"]
    test_series = files["test_series.csv"]
    sample = files["sample_submission.csv"]

    sample_header_ok = False
    if sample.get("readable"):
        sample_header_ok = list(sample["columns"]) == list(SAMPLE_SUBMISSION_COLUMNS)

    report_availability = {"train": None, "test": None}
    if train.get("readable"):
        frame = train["frame"]
        has_report = REPORT_COLUMN in frame.columns
        n_reports = int(len(frame)) if has_report else 0
        n_empty = 0
        if has_report:
            series = frame[REPORT_COLUMN]
            n_empty = int(series.isna().sum() + (series.astype(str).str.strip() == "").sum())
        report_availability["train"] = {
            "column_present": has_report,
            "n_rows": int(len(frame)),
            "n_non_empty": n_reports - n_empty if has_report else 0,
            "n_empty_or_missing": n_empty if has_report else None,
            "provenance": PROVENANCE_FULL_METADATA,
            "note": "Counts only. Report text is not exported.",
        }
    if test.get("readable"):
        frame = test["frame"]
        report_availability["test"] = {
            "column_present": REPORT_COLUMN in frame.columns,
            "provenance": PROVENANCE_FULL_METADATA,
            "note": "The official data page states Report is not provided at test time.",
        }

    labels = _audit_labels(train)
    labels["provenance"] = PROVENANCE_FULL_METADATA

    return {
        "provenance": PROVENANCE_FULL_METADATA,
        "documented_files_present": {
            name: info.get("present", False) for name, info in files.items()
        },
        "documented_directories_present": {
            name: (root / name).is_dir() for name in DOCUMENTED_DIRECTORIES
        },
        "csv": {
            name: {
                **_verify_columns(
                    {**_drop_frame(info), **_profile_csv_id(info, STUDY_ID_COLUMN)},
                    {
                        "train.csv": None,
                        "test.csv": (STUDY_ID_COLUMN,),
                        "train_series.csv": DOCUMENTED_SERIES_COLUMNS,
                        "test_series.csv": DOCUMENTED_SERIES_COLUMNS,
                        "sample_submission.csv": SAMPLE_SUBMISSION_COLUMNS,
                    }[name],
                ),
            }
            for name, info in files.items()
        },
        "sample_submission_header_matches_official": sample_header_ok,
        "labels": labels,
        "reports_availability": report_availability,
        "report_language_cues": _audit_reports(train, test),
        "train_series": {
            **_audit_series_table(train_series, DOCUMENTED_SERIES_COLUMNS),
            "provenance": PROVENANCE_FULL_METADATA,
        },
        "test_series": {
            **_audit_series_table(test_series, DOCUMENTED_SERIES_COLUMNS),
            "provenance": PROVENANCE_FULL_METADATA,
        },
        "train_test_structural_differences": _structural_diff(train, test),
        "leakage_checks": _leakage_checks(train, test, sample),
        "directory_counts": {
            "train": _split_directory_counts(root, "train"),
            "test": _split_directory_counts(root, "test"),
        },
        "note": (
            "Level A does not open DICOM files and does not walk the instance "
            "tree (about 819,640 files on the full competition set)."
        ),
    }


def audit_one_header_per_study(
    root: Path,
    train_info: dict,
    train_series_info: dict,
) -> dict:
    """Level B: one representative header per training study, no pixel arrays."""
    result: dict[str, Any] = {
        "provenance": PROVENANCE_ONE_HEADER,
        "n_train_studies_in_csv": None,
        "n_headers_attempted": 0,
        "n_headers_read": 0,
        "n_missing_series_dirs": 0,
        "n_empty_series_dirs": 0,
        "n_unreadable": 0,
        "n_with_patient_id": 0,
        "n_unique_patient_ids": 0,
        "n_study_uid_matches_csv": 0,
        "n_series_uid_matches_csv": 0,
        "patient_id_completeness_rate": None,
        "patient_id_sufficiently_complete": False,
        "patient_grouped_folds_created": False,
        "patient_grouped_folds_allowed": False,
        "recommendation": (
            "Do not create patient-grouped folds until PatientID is verified "
            "as present and complete on real data."
        ),
        "modality_counts": {},
        "manufacturer_counts": {},
        "manufacturer_model_counts": {},
        "series_description_counts": {},
        "plane_from_orientation_counts": {},
        "spacing_counts": {},
        "thickness_counts": {},
        "dimension_counts": {},
        "magnetic_field_counts": {},
        "phi_tag_keywords_observed": [],
        "note": (
            "Exactly one DICOM header is read per training study "
            "(stop_before_pixels=True). Identifier values are not exported."
        ),
    }
    if not train_info.get("readable") or not train_series_info.get("readable"):
        result["error"] = "train.csv or train_series.csv unreadable"
        return result

    train = train_info["frame"]
    series = train_series_info["frame"]
    result["n_train_studies_in_csv"] = int(train[STUDY_ID_COLUMN].nunique()) if STUDY_ID_COLUMN in train.columns else None
    if not {STUDY_ID_COLUMN, SERIES_ID_COLUMN} <= set(series.columns):
        result["error"] = "train_series.csv missing StudyInstanceUID or SeriesInstanceUID"
        return result

    subset = series[[STUDY_ID_COLUMN, SERIES_ID_COLUMN]].copy()
    subset[STUDY_ID_COLUMN] = subset[STUDY_ID_COLUMN].astype(str)
    subset[SERIES_ID_COLUMN] = subset[SERIES_ID_COLUMN].astype(str)
    subset = subset.sort_values([STUDY_ID_COLUMN, SERIES_ID_COLUMN], kind="mergesort")
    subset = subset.drop_duplicates(STUDY_ID_COLUMN, keep="first")

    modality: Counter[str] = Counter()
    manufacturer: Counter[str] = Counter()
    model: Counter[str] = Counter()
    description: Counter[str] = Counter()
    plane: Counter[str] = Counter()
    spacing: Counter[str] = Counter()
    thickness: Counter[str] = Counter()
    dims: Counter[str] = Counter()
    field: Counter[str] = Counter()
    phi: set[str] = set()
    patient_ids: set[str] = set()

    for _, row in subset.iterrows():
        result["n_headers_attempted"] += 1
        study = str(row[STUDY_ID_COLUMN])
        series_uid = str(row[SERIES_ID_COLUMN])
        series_dir = _expected_series_dir(root, "train", study, series_uid)
        if not series_dir.is_dir():
            result["n_missing_series_dirs"] += 1
            continue
        paths = list_dicom_paths(series_dir)
        if not paths:
            result["n_empty_series_dirs"] += 1
            continue
        try:
            ds = read_header(paths[0], stop_before_pixels=True)
        except DicomReadError:
            result["n_unreadable"] += 1
            continue
        result["n_headers_read"] += 1
        meta = header_summary(ds)
        header_study = str(getattr(ds, "StudyInstanceUID", "") or "")
        header_series = str(getattr(ds, "SeriesInstanceUID", "") or "")
        if header_study == study:
            result["n_study_uid_matches_csv"] += 1
        if header_series == series_uid:
            result["n_series_uid_matches_csv"] += 1
        patient_value = str(getattr(ds, "PatientID", "") or "").strip()
        if patient_value:
            result["n_with_patient_id"] += 1
            patient_ids.add(patient_value)
        modality[meta["modality"] or "<empty>"] += 1
        manufacturer[meta.get("manufacturer") or "<empty>"] += 1
        model[meta.get("manufacturer_model_name") or "<empty>"] += 1
        description[meta["series_description"] or "<empty>"] += 1
        plane[str(infer_plane_from_orientation(meta["image_orientation_patient"]))] += 1
        spacing[str(meta["pixel_spacing"])] += 1
        thickness[str(meta["slice_thickness"])] += 1
        dims[f"{meta['rows']}x{meta['columns']}"] += 1
        field[str(meta.get("magnetic_field_strength"))] += 1
        phi.update(meta.get("phi_tag_keywords_present") or [])

    n_read = int(result["n_headers_read"])
    result["n_unique_patient_ids"] = len(patient_ids)
    result["patient_id_completeness_rate"] = (
        float(result["n_with_patient_id"]) / n_read if n_read else None
    )
    csv_n = result["n_train_studies_in_csv"]
    result["patient_id_sufficiently_complete"] = bool(
        n_read
        and csv_n is not None
        and n_read == csv_n
        and result["n_with_patient_id"] == n_read
        and result["patient_id_completeness_rate"] == 1.0
    )
    result["modality_counts"] = dict(modality)
    result["manufacturer_counts"] = dict(manufacturer)
    result["manufacturer_model_counts"] = dict(model)
    result["series_description_counts"] = dict(description)
    result["plane_from_orientation_counts"] = dict(plane)
    result["spacing_counts"] = dict(spacing)
    result["thickness_counts"] = dict(thickness)
    result["dimension_counts"] = dict(dims)
    result["magnetic_field_counts"] = dict(field)
    result["phi_tag_keywords_observed"] = sorted(phi)
    return result


def audit_sampled_dicom(
    root: Path,
    train_info: dict,
    train_series_info: dict,
    *,
    sample_size: int,
    max_pixel_arrays_per_series: int = DEFAULT_PIXELS_PER_SERIES,
) -> dict:
    """Level C: all series + selected pixel arrays for a documented sample only."""
    empty = {
        "provenance": PROVENANCE_SAMPLED,
        "sample_size_requested": sample_size,
        "n_studies_sampled": 0,
        "n_series_inspected": 0,
        "n_instances_headers_read": 0,
        "n_pixel_arrays_read": 0,
        "is_full_dataset_statistic": False,
        "sample": [],
        "qc_rows": [],
        "note": (
            "Sampled findings are not full-dataset statistics. Pixel arrays are "
            "inspected only for this documented sample."
        ),
    }
    if not train_info.get("readable"):
        empty["error"] = "train.csv unreadable"
        return empty
    train = train_info["frame"]
    series = train_series_info["frame"] if train_series_info.get("readable") else pd.DataFrame()
    sample = select_detailed_sample(train, series, sample_size=sample_size)
    empty["sample"] = sample
    empty["n_studies_sampled"] = len(sample)
    if not sample:
        return empty

    qc_rows: list[dict] = []
    n_series = 0
    n_headers = 0
    n_pixels = 0
    sample_uids = [row[STUDY_ID_COLUMN] for row in sample]
    reason_by_uid = {row[STUDY_ID_COLUMN]: row["sample_reason"] for row in sample}

    if series is None or series.empty or STUDY_ID_COLUMN not in series.columns:
        empty["qc_rows"] = qc_rows
        return empty

    series_frame = series.copy()
    series_frame[STUDY_ID_COLUMN] = series_frame[STUDY_ID_COLUMN].astype(str)
    series_frame[SERIES_ID_COLUMN] = series_frame[SERIES_ID_COLUMN].astype(str)
    subset = series_frame[series_frame[STUDY_ID_COLUMN].isin(sample_uids)]
    subset = subset.sort_values([STUDY_ID_COLUMN, SERIES_ID_COLUMN], kind="mergesort")

    for _, row in subset.iterrows():
        study = str(row[STUDY_ID_COLUMN])
        series_uid = str(row[SERIES_ID_COLUMN])
        series_dir = _expected_series_dir(root, "train", study, series_uid)
        n_series += 1
        record: dict[str, Any] = {
            STUDY_ID_COLUMN: study,
            SERIES_ID_COLUMN: series_uid,
            "sample_reason": reason_by_uid.get(study, ""),
            "series_dir_present": series_dir.is_dir(),
            "provenance": PROVENANCE_SAMPLED,
            "is_full_dataset_statistic": False,
        }
        if "Anatomical_Plane" in row:
            record["anatomical_plane_csv"] = str(row["Anatomical_Plane"])
        if not series_dir.is_dir():
            record["n_instances"] = 0
            qc_rows.append(record)
            continue
        paths = list_dicom_paths(series_dir)
        record["n_instances"] = len(paths)
        descriptions: Counter[str] = Counter()
        for path in paths:
            try:
                ds = read_header(path, stop_before_pixels=True)
            except DicomReadError as exc:
                record.setdefault("header_errors", 0)
                record["header_errors"] += 1
                record["last_header_error"] = str(exc)
                continue
            n_headers += 1
            meta = header_summary(ds)
            descriptions[meta["series_description"] or "<empty>"] += 1
            record.setdefault("modality", meta["modality"])
            record.setdefault("manufacturer", meta.get("manufacturer"))
            record.setdefault(
                "inferred_plane",
                infer_plane_from_orientation(meta["image_orientation_patient"]),
            )
        if descriptions:
            record["series_description"] = descriptions.most_common(1)[0][0]
        pixel_budget = max(int(max_pixel_arrays_per_series), 0)
        if paths and pixel_budget:
            # Middle slice only by default; never every instance in the study.
            chosen = [paths[len(paths) // 2]][:pixel_budget]
            pixel_stats = []
            for path in chosen:
                pixel_stats.append(_pixel_qc(path))
                n_pixels += 1
            if pixel_stats:
                record.update(pixel_stats[0])
                record["n_pixel_arrays_read"] = len(pixel_stats)
        qc_rows.append(record)

    empty["n_series_inspected"] = n_series
    empty["n_instances_headers_read"] = n_headers
    empty["n_pixel_arrays_read"] = n_pixels
    empty["qc_rows"] = qc_rows
    empty["sample_reasons"] = sorted({row["sample_reason"] for row in sample})
    return empty


def _prevalence_frame(labels: dict) -> pd.DataFrame:
    rows = []
    prevalence = labels.get("prevalence") or {}
    for official, stats in prevalence.items():
        rows.append(
            {
                "target_key": stats.get("target_key"),
                "official_column": official,
                "n_known": stats.get("n_known"),
                "n_positive": stats.get("n_positive"),
                "n_negative": stats.get("n_negative"),
                "n_unknown": stats.get("n_unknown"),
                "positive_rate_among_known": stats.get("positive_rate_among_known"),
                "provenance": PROVENANCE_FULL_METADATA,
            }
        )
    if not rows:
        for key, official in zip(TARGET_KEYS, OFFICIAL_TARGET_COLUMNS):
            rows.append(
                {
                    "target_key": key,
                    "official_column": official,
                    "n_known": None,
                    "n_positive": None,
                    "n_negative": None,
                    "n_unknown": None,
                    "positive_rate_among_known": None,
                    "provenance": PROVENANCE_FULL_METADATA,
                    "status": STATUS_DATA_NOT_AVAILABLE,
                }
            )
    return pd.DataFrame(rows)


def _series_summary_frame(level_a: dict) -> pd.DataFrame:
    rows = []
    for split_key, table_key in (("train", "train_series"), ("test", "test_series")):
        table = level_a.get(table_key) or {}
        dirs = (level_a.get("directory_counts") or {}).get(split_key) or {}
        plane_counts = table.get("plane_counts") or {}
        if plane_counts:
            for plane, count in plane_counts.items():
                rows.append(
                    {
                        "split": split_key,
                        "metric": "anatomical_plane",
                        "key": plane,
                        "value": count,
                        "n_csv_rows": table.get("n_rows"),
                        "n_unique_studies": table.get("n_unique_studies"),
                        "n_unique_series": table.get("n_unique_series"),
                        "n_study_dirs": dirs.get("n_study_dirs"),
                        "n_series_dirs": dirs.get("n_series_dirs"),
                        "provenance": PROVENANCE_FULL_METADATA,
                    }
                )
        else:
            rows.append(
                {
                    "split": split_key,
                    "metric": "series_table",
                    "key": "n_rows",
                    "value": table.get("n_rows"),
                    "n_csv_rows": table.get("n_rows"),
                    "n_unique_studies": table.get("n_unique_studies"),
                    "n_unique_series": table.get("n_unique_series"),
                    "n_study_dirs": dirs.get("n_study_dirs"),
                    "n_series_dirs": dirs.get("n_series_dirs"),
                    "provenance": PROVENANCE_FULL_METADATA,
                }
            )
    if not rows:
        rows.append(
            {
                "split": "",
                "metric": "status",
                "key": STATUS_DATA_NOT_AVAILABLE,
                "value": None,
                "provenance": PROVENANCE_FULL_METADATA,
            }
        )
    return pd.DataFrame(rows)


def _header_summary_frame(level_b: dict) -> pd.DataFrame:
    rows = []
    scalars = [
        "n_train_studies_in_csv",
        "n_headers_attempted",
        "n_headers_read",
        "n_missing_series_dirs",
        "n_empty_series_dirs",
        "n_unreadable",
        "n_with_patient_id",
        "n_unique_patient_ids",
        "n_study_uid_matches_csv",
        "n_series_uid_matches_csv",
        "patient_id_completeness_rate",
        "patient_id_sufficiently_complete",
        "patient_grouped_folds_created",
        "patient_grouped_folds_allowed",
    ]
    for key in scalars:
        rows.append(
            {
                "metric": key,
                "key": "",
                "value": level_b.get(key),
                "provenance": PROVENANCE_ONE_HEADER,
            }
        )
    for metric in (
        "modality_counts",
        "manufacturer_counts",
        "manufacturer_model_counts",
        "series_description_counts",
        "plane_from_orientation_counts",
        "spacing_counts",
        "thickness_counts",
        "dimension_counts",
        "magnetic_field_counts",
    ):
        for name, count in (level_b.get(metric) or {}).items():
            rows.append(
                {
                    "metric": metric,
                    "key": name,
                    "value": count,
                    "provenance": PROVENANCE_ONE_HEADER,
                }
            )
    return pd.DataFrame(rows)


def _qc_sample_frame(level_c: dict) -> pd.DataFrame:
    rows = level_c.get("qc_rows") or []
    if not rows:
        return pd.DataFrame(
            [
                {
                    STUDY_ID_COLUMN: "",
                    "sample_reason": "",
                    "provenance": PROVENANCE_SAMPLED,
                    "is_full_dataset_statistic": False,
                    "status": "no_sample",
                }
            ]
        )
    frame = pd.DataFrame(rows)
    frame["provenance"] = PROVENANCE_SAMPLED
    frame["is_full_dataset_statistic"] = False
    drop = [col for col in frame.columns if str(col) in FORBIDDEN_TEXT_COLUMNS]
    if drop:
        frame = frame.drop(columns=drop)
    return frame


def _not_available_result(
    preflight_result: PreflightResult,
    *,
    reason: str | None = None,
) -> dict:
    payload = data_not_available_result(
        explicit_root=preflight_result.data_root,
        reason=reason or preflight_result.message,
    )
    payload["data_source"] = preflight_result.data_source
    payload["scientific_evaluation"] = False
    payload["preflight"] = preflight_result.to_dict()
    payload["internet_used"] = False
    payload["gpu_required"] = False
    payload["gpu_used"] = False
    payload["training_run"] = False
    payload["folds_created"] = False
    payload["status"] = STATUS_DATA_NOT_AVAILABLE
    return payload


def render_kaggle_audit_markdown(result: dict) -> str:
    lines: list[str] = []
    status = result.get("status", "UNKNOWN")
    lines.append("# Kaggle real-data audit")
    lines.append("")
    lines.append(f"**Status: {status}**")
    lines.append("")
    lines.append(f"- DATA_SOURCE: `{result.get('data_source')}`")
    lines.append(f"- scientific_evaluation: `{result.get('scientific_evaluation')}`")
    lines.append(f"- internet_used: `{result.get('internet_used', False)}`")
    lines.append(f"- gpu_required: `{result.get('gpu_required', False)}`")
    lines.append(f"- gpu_used: `{result.get('gpu_used', False)}`")
    lines.append(f"- training_run: `{result.get('training_run', False)}`")
    lines.append(f"- folds_created: `{result.get('folds_created', False)}`")
    lines.append("")
    lines.append(
        "This notebook does **not** train a model and does **not** report "
        "smoke-test AUC. Sampled DICOM findings are labelled "
        f"`{PROVENANCE_SAMPLED}` and are not full-dataset statistics."
    )
    lines.append("")
    lines.append("## Preflight")
    lines.append("")
    pre = result.get("preflight") or {}
    lines.append(pre.get("message") or result.get("reason") or "")
    lines.append("")
    lines.append("Detected paths (top-level only; the DICOM tree is not printed):")
    lines.append("")
    layout = pre.get("layout") or {}
    for key in ("input_root", "working_root", "competition_dir", "output_dir", "src_extract_dir"):
        if layout.get(key):
            lines.append(f"- `{key}`: `{layout[key]}`")
    if pre.get("top_level_input_datasets"):
        lines.append(
            "- Kaggle input datasets: "
            + ", ".join(f"`{n}`" for n in pre["top_level_input_datasets"])
        )
    if pre.get("top_level_competition"):
        lines.append(
            "- Competition top-level: "
            + ", ".join(f"`{n}`" for n in pre["top_level_competition"])
        )
    lines.append("")

    if status == STATUS_DATA_NOT_AVAILABLE:
        lines.append("## Statistics")
        lines.append("")
        lines.append(
            "`statistics: null` — not computed. Required inputs were missing. "
            "No study counts, prevalences, or DICOM summaries were fabricated."
        )
        lines.append("")
        return "\n".join(lines)

    lines.append("## Level A — full CSV and metadata-table audit")
    lines.append("")
    lines.append(f"Provenance: `{PROVENANCE_FULL_METADATA}`")
    lines.append("")
    level_a = result.get("level_a") or {}
    labels = level_a.get("labels") or {}
    lines.append(f"- Labels available: {labels.get('labels_available')}")
    lines.append(
        "- Official columns present: "
        + ", ".join(f"`{c}`" for c in labels.get("official_label_columns_present") or [])
    )
    if labels.get("prevalence"):
        lines.append("")
        lines.append("| Official column | known | positive | negative | unknown | rate among known | provenance |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for col, stats in labels["prevalence"].items():
            rate = stats["positive_rate_among_known"]
            rate_s = "n/a" if rate is None else f"{rate:.4f}"
            lines.append(
                f"| `{col}` | {stats['n_known']} | {stats['n_positive']} | "
                f"{stats['n_negative']} | {stats['n_unknown']} | {rate_s} | "
                f"`{PROVENANCE_FULL_METADATA}` |"
            )
    reports = level_a.get("reports_availability") or {}
    train_rep = reports.get("train") or {}
    lines.append("")
    lines.append(
        f"- Train report column present: {train_rep.get('column_present')}; "
        f"non-empty reports: {train_rep.get('n_non_empty')} "
        f"(text not exported; provenance `{PROVENANCE_FULL_METADATA}`)"
    )
    dirs = level_a.get("directory_counts") or {}
    for split, info in dirs.items():
        lines.append(
            f"- `{split}` study dirs: {info.get('n_study_dirs')}; "
            f"series dirs: {info.get('n_series_dirs')} "
            f"(provenance `{PROVENANCE_FULL_METADATA}`)"
        )
    lines.append("")
    lines.append(level_a.get("note") or "")
    lines.append("")

    lines.append("## Level B — one header per training study")
    lines.append("")
    lines.append(f"Provenance: `{PROVENANCE_ONE_HEADER}`")
    lines.append("")
    level_b = result.get("level_b") or {}
    lines.append(f"- Headers read: {level_b.get('n_headers_read')} / attempted {level_b.get('n_headers_attempted')}")
    lines.append(f"- PatientID present: {level_b.get('n_with_patient_id')}")
    lines.append(f"- Unique PatientID values (count only): {level_b.get('n_unique_patient_ids')}")
    lines.append(f"- PatientID completeness rate: {level_b.get('patient_id_completeness_rate')}")
    lines.append(
        f"- PatientID sufficiently complete: {level_b.get('patient_id_sufficiently_complete')}"
    )
    lines.append(f"- Patient-grouped folds created: {level_b.get('patient_grouped_folds_created')}")
    lines.append(f"- Recommendation: {level_b.get('recommendation')}")
    lines.append(f"- Modality counts: `{level_b.get('modality_counts')}`")
    lines.append(f"- Manufacturer counts: `{level_b.get('manufacturer_counts')}`")
    lines.append(
        f"- Series description counts: `{level_b.get('series_description_counts')}`"
    )
    lines.append(
        f"- Plane from orientation: `{level_b.get('plane_from_orientation_counts')}`"
    )
    lines.append("")

    lines.append("## Level C — deterministic detailed DICOM sample")
    lines.append("")
    lines.append(f"Provenance: `{PROVENANCE_SAMPLED}`")
    lines.append("")
    level_c = result.get("level_c") or {}
    lines.append(
        f"- Sample size requested: {level_c.get('sample_size_requested')}; "
        f"studies sampled: {level_c.get('n_studies_sampled')}"
    )
    lines.append(f"- Series inspected: {level_c.get('n_series_inspected')}")
    lines.append(f"- Instance headers read (sample only): {level_c.get('n_instances_headers_read')}")
    lines.append(f"- Pixel arrays read (sample only): {level_c.get('n_pixel_arrays_read')}")
    lines.append(f"- Full-dataset statistic: `{level_c.get('is_full_dataset_statistic')}`")
    lines.append("")
    lines.append("Sampled studies (documented sample, not a prevalence table):")
    lines.append("")
    lines.append("| StudyInstanceUID | sample_reason | provenance |")
    lines.append("| --- | --- | --- |")
    for row in level_c.get("sample") or []:
        lines.append(
            f"| `{row.get(STUDY_ID_COLUMN)}` | `{row.get('sample_reason')}` | "
            f"`{PROVENANCE_SAMPLED}` |"
        )
    lines.append("")
    lines.append(level_c.get("note") or "")
    lines.append("")

    lines.append("## Target reconciliation")
    lines.append("")
    recon = result.get("target_reconciliation") or _target_reconciliation()
    lines.append(recon["policy"])
    lines.append("")
    lines.append("| Slot | Internal `TARGET_KEYS` | Official Kaggle column |")
    lines.append("| --- | --- | --- |")
    for row in recon["mapping"]:
        lines.append(
            f"| {row['slot']} | `{row['target_key']}` | `{row['official_column']}` |"
        )
    lines.append("")
    return "\n".join(lines)


def _write_outputs(result: dict, output_dir: Path, snippets: list[str]) -> dict[str, str]:
    assert_not_kaggle_input(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    level_a = result.get("level_a") or {}
    level_b = result.get("level_b") or {}
    level_c = result.get("level_c") or {}

    paths = {name: output_dir / name for name in OUTPUT_FILENAMES}
    _prevalence_frame((level_a.get("labels") or {})).to_csv(
        paths["target_prevalence.csv"], index=False
    )
    _series_summary_frame(level_a).to_csv(paths["series_summary.csv"], index=False)
    _header_summary_frame(level_b).to_csv(paths["dicom_header_summary.csv"], index=False)
    _qc_sample_frame(level_c).to_csv(paths["dicom_qc_sample.csv"], index=False)

    serializable = jsonable({k: v for k, v in result.items() if k != "frames"})
    paths["real_data_audit.json"].write_text(
        json.dumps(serializable, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    paths["real_data_audit.md"].write_text(
        render_kaggle_audit_markdown(result), encoding="utf-8"
    )

    written = []
    for name, path in paths.items():
        if name == "audit_manifest.json":
            continue
        provenance = {
            "target_prevalence.csv": PROVENANCE_FULL_METADATA,
            "series_summary.csv": PROVENANCE_FULL_METADATA,
            "dicom_header_summary.csv": PROVENANCE_ONE_HEADER,
            "dicom_qc_sample.csv": PROVENANCE_SAMPLED,
            "real_data_audit.md": "mixed",
            "real_data_audit.json": "mixed",
        }[name]
        written.append(
            {
                "file": name,
                "path": str(path),
                "bytes": path.stat().st_size if path.is_file() else 0,
                "provenance": provenance,
            }
        )
    manifest = {
        "data_source": result.get("data_source"),
        "scientific_evaluation": result.get("scientific_evaluation"),
        "internet_used": False,
        "gpu_required": False,
        "gpu_used": False,
        "training_run": False,
        "folds_created": False,
        "patient_grouped_folds_allowed": False,
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "output_dir": str(output_dir),
        "outputs": written,
        "provenance_values": list(PROVENANCE_VALUES),
        "sample_size": (result.get("level_c") or {}).get("sample_size_requested"),
        "status": result.get("status"),
    }
    paths["audit_manifest.json"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    result["manifest"] = manifest
    assert_no_report_text(list(paths.values()), snippets)
    return {name: str(path) for name, path in paths.items()}


def run_notebook_audit(
    *,
    data_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    layout: KaggleLayout | None = None,
    data_source: str | None = None,
    sample_size: int | None = None,
    max_pixel_arrays_per_series: int = DEFAULT_PIXELS_PER_SERIES,
    extract_source: bool = False,
    source_root: str | Path | None = None,
) -> dict:
    """Entry point used by the Kaggle notebook and by software-fixture tests."""
    pre = preflight(
        layout=layout,
        data_root=data_root,
        source_root=source_root,
        output_dir=output_dir,
        data_source=data_source,
        extract_source=extract_source,
    )
    out_dir = Path(output_dir) if output_dir is not None else pre.output_dir
    assert_not_kaggle_input(out_dir, pre.layout)
    size = configured_sample_size(sample_size)
    source = data_source or pre.data_source
    scientific = scientific_evaluation_for(source)

    if not pre.ok or pre.data_root is None:
        result = _not_available_result(pre)
        result["data_source"] = source
        result["level_a"] = {"provenance": PROVENANCE_FULL_METADATA, "status": STATUS_DATA_NOT_AVAILABLE}
        result["level_b"] = {"provenance": PROVENANCE_ONE_HEADER, "status": STATUS_DATA_NOT_AVAILABLE}
        result["level_c"] = {
            "provenance": PROVENANCE_SAMPLED,
            "status": STATUS_DATA_NOT_AVAILABLE,
            "sample_size_requested": size,
            "is_full_dataset_statistic": False,
        }
        result["target_reconciliation"] = _target_reconciliation()
        result["output_files"] = _write_outputs(result, out_dir, snippets=[])
        return result

    root = Path(pre.data_root)
    guard = ReadOnlyInputGuard([root])
    files = {name: _read_csv(root / name) for name in DOCUMENTED_FILES}
    snippets = collect_report_snippets(
        *(info["frame"] for info in files.values() if info.get("readable"))
    )

    level_a = audit_full_metadata(root, files)
    level_b = audit_one_header_per_study(root, files["train.csv"], files["train_series.csv"])
    level_c = audit_sampled_dicom(
        root,
        files["train.csv"],
        files["train_series.csv"],
        sample_size=size,
        max_pixel_arrays_per_series=max_pixel_arrays_per_series,
    )

    result = {
        "status": STATUS_AUDITED,
        "data_source": source,
        "scientific_evaluation": scientific,
        "internet_used": False,
        "gpu_required": False,
        "gpu_used": False,
        "training_run": False,
        "folds_created": False,
        "patient_grouped_folds_allowed": False,
        "data_root": str(root),
        "output_dir": str(out_dir),
        "preflight": pre.to_dict(),
        "level_a": jsonable(level_a),
        "level_b": jsonable(level_b),
        "level_c": jsonable(level_c),
        "target_reconciliation": _target_reconciliation(),
        "num_targets": NUM_TARGETS,
        "note": (
            "THIS RUN USED A SOFTWARE FIXTURE, NOT THE KAGGLE COMPETITION DATA."
            if source == DATA_SOURCE_SOFTWARE_FIXTURE
            else "Real Kaggle mount audit. Sampled DICOM rows are not full-dataset statistics."
        ),
    }
    if source == DATA_SOURCE_SOFTWARE_FIXTURE:
        result["scientific_evaluation"] = False
    if source == DATA_SOURCE_REAL_KAGGLE:
        result["scientific_evaluation"] = True

    result["output_files"] = _write_outputs(result, out_dir, snippets)
    guard.assert_unchanged()
    return result


def run_or_stop(**kwargs) -> dict:
    """Notebook helper: run the audit and raise a clear stop if inputs are missing."""
    result = run_notebook_audit(**kwargs)
    if result.get("status") == STATUS_DATA_NOT_AVAILABLE:
        message = (result.get("preflight") or {}).get("message") or result.get("reason")
        raise PreflightError(message or "AUDIT STOPPED: required inputs missing")
    return result
