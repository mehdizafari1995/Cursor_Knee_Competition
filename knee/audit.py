"""Read-only competition-data audit.

When the real files cannot be opened the result status is DATA_NOT_AVAILABLE and
no counts, prevalences, or DICOM summaries are fabricated. Software-only
fixtures may call the same functions from tests; those runs are labelled
``software_fixture`` and must never be copied into the committed scientific
audit as if they were competition data.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .competition import (
    DOCUMENTED_DIRECTORIES,
    DOCUMENTED_FILES,
    DOCUMENTED_LABEL_COLUMN_CANDIDATES,
    DOCUMENTED_PLANES,
    DOCUMENTED_SERIES_COLUMNS,
    DOCUMENTED_TEST_CSV_FIELDS,
    DOCUMENTED_TRAIN_CSV_FIELDS,
    KAGGLE_DATA_PAGE,
    KAGGLE_OVERVIEW_PAGE,
    SAMPLE_SUBMISSION_COLUMNS,
)
from .config import (
    OFFICIAL_TARGET_COLUMNS,
    PROJECT_ROOT,
    REPORT_COLUMN,
    SERIES_ID_COLUMN,
    STUDY_ID_COLUMN,
    TARGET_KEY_TO_OFFICIAL,
    TARGET_KEYS,
)
from .dicom_io import (
    DicomReadError,
    header_summary,
    infer_plane_from_orientation,
    list_dicom_paths,
    read_header,
)
from .labels import extract_label_matrix, missing_official_columns, prevalence
from .language import summarise_reports
from .paths import DataNotAvailable, describe_search, get_data_root, is_competition_root

STATUS_DATA_NOT_AVAILABLE = "DATA_NOT_AVAILABLE"
STATUS_AUDITED = "AUDITED"

PHI_TAG_KEYWORDS = (
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "PatientAddress",
    "OtherPatientIDs",
    "OtherPatientNames",
)


def _read_csv(path: Path) -> dict:
    if not path.is_file():
        return {"present": False, "path": str(path)}
    try:
        frame = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        return {
            "present": True,
            "path": str(path),
            "readable": False,
            "error": str(exc),
        }
    dtypes = {col: str(dtype) for col, dtype in frame.dtypes.items()}
    return {
        "present": True,
        "path": str(path),
        "readable": True,
        "n_rows": int(len(frame)),
        "n_columns": int(frame.shape[1]),
        "columns": list(frame.columns),
        "dtypes": dtypes,
        "duplicate_column_names": sorted(
            name for name, count in Counter(frame.columns).items() if count > 1
        ),
        "frame": frame,
    }


def _drop_frame(info: dict) -> dict:
    out = dict(info)
    out.pop("frame", None)
    return out


def _profile_csv_id(info: dict, column: str) -> dict:
    if not info.get("readable"):
        return {}
    frame: pd.DataFrame = info["frame"]
    if column not in frame.columns:
        return {"column_present": False, "column": column}
    series = frame[column].astype(str)
    return {
        "column_present": True,
        "column": column,
        "n_unique": int(series.nunique(dropna=False)),
        "n_duplicate_rows": int(series.duplicated().sum()),
        "n_missing": int(frame[column].isna().sum()),
    }


def _inventory(root: Path, max_files: int | None = None) -> dict:
    counts: Counter[str] = Counter()
    n_files = 0
    n_dirs = 0
    bytes_total = 0
    truncated = False
    for path in root.rglob("*"):
        if path.is_dir():
            n_dirs += 1
            continue
        n_files += 1
        counts[path.suffix.lower() or "<none>"] += 1
        try:
            bytes_total += path.stat().st_size
        except OSError:
            pass
        if max_files is not None and n_files >= max_files:
            truncated = True
            break
    return {
        "n_files": n_files,
        "n_directories": n_dirs,
        "bytes": bytes_total,
        "suffix_counts": dict(counts),
        "inventory_truncated": truncated,
    }


def _audit_labels(train_info: dict) -> dict:
    if not train_info.get("readable"):
        return {"status": "train_csv_unreadable"}
    frame: pd.DataFrame = train_info["frame"]
    present = [col for col in OFFICIAL_TARGET_COLUMNS if col in frame.columns]
    missing = list(missing_official_columns(frame))
    internal_present = [key for key in TARGET_KEYS if key in frame.columns]
    result = {
        "official_label_columns_present": present,
        "official_label_columns_missing": missing,
        "internal_target_key_columns_present": internal_present,
        "note": (
            "Official Kaggle names are required for reconciliation. Internal "
            "TARGET_KEYS are recorded if they appear, but they are not the "
            "competition header."
        ),
    }
    if not present and not internal_present:
        result["labels_available"] = False
        result["prevalence"] = None
        return result
    labels, mask = extract_label_matrix(frame, require_all=False)
    result["labels_available"] = True
    result["n_studies"] = int(labels.shape[0])
    result["n_fully_labelled_studies"] = int(mask.all(axis=1).sum())
    result["n_studies_with_any_label"] = int(mask.any(axis=1).sum())
    result["prevalence"] = prevalence(labels)
    cooccur = {}
    known_pos = (labels == 1) & np.isfinite(labels)
    for i, a in enumerate(OFFICIAL_TARGET_COLUMNS):
        for j, b in enumerate(OFFICIAL_TARGET_COLUMNS):
            if j <= i:
                continue
            both = int((known_pos[:, i] & known_pos[:, j]).sum())
            if both:
                cooccur[f"{a} + {b}"] = both
    result["positive_cooccurrence_counts"] = cooccur
    return result


def _audit_reports(train_info: dict, test_info: dict) -> dict:
    out: dict = {"train": None, "test": None}
    if train_info.get("readable"):
        frame: pd.DataFrame = train_info["frame"]
        if REPORT_COLUMN in frame.columns:
            reports = frame[REPORT_COLUMN].tolist()
            out["train"] = summarise_reports(reports)
            out["train"]["column_present"] = True
        else:
            out["train"] = {"column_present": False}
    if test_info.get("readable"):
        frame = test_info["frame"]
        out["test"] = {
            "column_present": REPORT_COLUMN in frame.columns,
            "note": (
                "The official data page states the Report field is not provided "
                "at the testing stage. Observation below is from the local file."
            ),
        }
        if REPORT_COLUMN in frame.columns:
            out["test"].update(summarise_reports(frame[REPORT_COLUMN].tolist()))
    return out


def _audit_series_table(info: dict, expected_columns: tuple[str, ...]) -> dict:
    summary = _drop_frame(info)
    if not info.get("readable"):
        return summary
    frame: pd.DataFrame = info["frame"]
    summary["expected_columns"] = list(expected_columns)
    summary["missing_expected_columns"] = [
        col for col in expected_columns if col not in frame.columns
    ]
    summary["unexpected_columns"] = [
        col for col in frame.columns if col not in expected_columns
    ]
    if STUDY_ID_COLUMN in frame.columns:
        summary["n_unique_studies"] = int(frame[STUDY_ID_COLUMN].nunique(dropna=False))
    if SERIES_ID_COLUMN in frame.columns:
        summary["n_unique_series"] = int(frame[SERIES_ID_COLUMN].nunique(dropna=False))
        summary["duplicate_series_uids"] = int(
            frame[SERIES_ID_COLUMN].astype(str).duplicated().sum()
        )
    if "Anatomical_Plane" in frame.columns:
        planes = frame["Anatomical_Plane"].astype(str)
        summary["plane_counts"] = {k: int(v) for k, v in planes.value_counts(dropna=False).items()}
        summary["unexpected_plane_values"] = sorted(
            set(planes.dropna().unique()) - set(DOCUMENTED_PLANES) - {"nan"}
        )
    for flag in ("Fluid_Sensitive", "Fat_Suppression"):
        if flag in frame.columns:
            summary[f"{flag}_value_counts"] = {
                str(k): int(v)
                for k, v in frame[flag].value_counts(dropna=False).items()
            }
    if {"Fluid_Sensitive", "Fat_Suppression"} <= set(frame.columns):
        equal = frame["Fluid_Sensitive"].eq(frame["Fat_Suppression"])
        comparable = frame["Fluid_Sensitive"].notna() & frame["Fat_Suppression"].notna()
        summary["fluid_equals_fat_suppression_among_non_null"] = {
            "n_equal": int(equal[comparable].sum()),
            "n_comparable": int(comparable.sum()),
        }
    return summary


def _expected_series_dir(root: Path, split: str, study: str, series: str) -> Path:
    return root / f"{split}_series" / study / series


def _audit_dicom_tree(
    root: Path,
    split: str,
    series_info: dict,
    *,
    max_series: int | None = None,
) -> dict:
    split_dir = root / f"{split}_series"
    result: dict = {
        "split": split,
        "directory_present": split_dir.is_dir(),
        "n_series_csv_rows": None,
        "n_series_dirs_on_disk": None,
        "n_missing_series_dirs": 0,
        "n_empty_series_dirs": 0,
        "n_unreadable_instances": 0,
        "n_zero_byte_instances": 0,
        "n_instances_scanned": 0,
        "n_series_scanned": 0,
        "dimension_counts": {},
        "spacing_counts": {},
        "thickness_counts": {},
        "plane_from_orientation_counts": {},
        "series_description_counts": {},
        "phi_tag_keywords_observed": [],
        "patient_id_tag_present_series": 0,
        "slice_order_keys_used": Counter(),
        "truncated": False,
    }
    disk_series = 0
    if split_dir.is_dir():
        disk_series = sum(1 for _p in split_dir.glob("*/*") if _p.is_dir())
    result["n_series_dirs_on_disk"] = disk_series

    if not series_info.get("readable"):
        return result
    frame: pd.DataFrame = series_info["frame"]
    result["n_series_csv_rows"] = int(len(frame))
    required = {STUDY_ID_COLUMN, SERIES_ID_COLUMN}
    if not required <= set(frame.columns):
        result["error"] = f"series table missing {sorted(required - set(frame.columns))}"
        return result

    dim_counter: Counter[str] = Counter()
    spacing_counter: Counter[str] = Counter()
    thickness_counter: Counter[str] = Counter()
    plane_counter: Counter[str] = Counter()
    desc_counter: Counter[str] = Counter()
    phi_keywords: set[str] = set()
    order_counter: Counter[str] = Counter()
    missing_dirs = 0
    empty_dirs = 0
    unreadable = 0
    zero_byte = 0
    instances = 0
    scanned_series = 0
    patient_id_series = 0

    subset = frame[[STUDY_ID_COLUMN, SERIES_ID_COLUMN]]
    if max_series is not None:
        result["truncated"] = len(subset) > max_series
        subset = subset.iloc[:max_series]

    for _, data in subset.iterrows():
        study = str(data[STUDY_ID_COLUMN])
        series = str(data[SERIES_ID_COLUMN])
        series_dir = _expected_series_dir(root, split, study, series)
        if not series_dir.is_dir():
            missing_dirs += 1
            continue
        paths = list_dicom_paths(series_dir)
        if not paths:
            empty_dirs += 1
            continue
        scanned_series += 1
        keys = []
        saw_patient_id = False
        for path in paths:
            instances += 1
            try:
                if path.stat().st_size == 0:
                    zero_byte += 1
                    continue
            except OSError:
                unreadable += 1
                continue
            try:
                ds = read_header(path, stop_before_pixels=True)
            except DicomReadError:
                unreadable += 1
                continue
            meta = header_summary(ds)
            dim_counter[f"{meta['rows']}x{meta['columns']}"] += 1
            spacing_counter[str(meta["pixel_spacing"])] += 1
            thickness_counter[str(meta["slice_thickness"])] += 1
            plane = infer_plane_from_orientation(meta["image_orientation_patient"])
            plane_counter[str(plane)] += 1
            desc = meta["series_description"] or "<empty>"
            desc_counter[desc] += 1
            phi_keywords.update(meta["phi_tag_keywords_present"])
            if meta["has_patient_id_tag"]:
                saw_patient_id = True
            if meta["has_image_position_patient"]:
                keys.append("image_position")
            elif meta["instance_number"] is not None:
                keys.append("instance_number")
            else:
                keys.append("filename")
        if saw_patient_id:
            patient_id_series += 1
        if keys:
            order_counter[keys[0]] += 1

    result.update(
        {
            "n_missing_series_dirs": missing_dirs,
            "n_empty_series_dirs": empty_dirs,
            "n_unreadable_instances": unreadable,
            "n_zero_byte_instances": zero_byte,
            "n_instances_scanned": instances,
            "n_series_scanned": scanned_series,
            "dimension_counts": dict(dim_counter),
            "spacing_counts": dict(spacing_counter),
            "thickness_counts": dict(thickness_counter),
            "plane_from_orientation_counts": dict(plane_counter),
            "series_description_counts": dict(desc_counter),
            "phi_tag_keywords_observed": sorted(phi_keywords),
            "patient_id_tag_present_series": patient_id_series,
            "slice_order_keys_used": dict(order_counter),
        }
    )
    return result


def _leakage_checks(train_info: dict, test_info: dict, sample_info: dict) -> list[dict]:
    checks: list[dict] = []

    def ids(info: dict, column: str = STUDY_ID_COLUMN) -> set[str] | None:
        if not info.get("readable") or column not in info["frame"].columns:
            return None
        return set(info["frame"][column].astype(str))

    train_ids = ids(train_info)
    test_ids = ids(test_info)
    sample_ids = ids(sample_info)
    if train_ids is not None and test_ids is not None:
        overlap = sorted(train_ids & test_ids)
        checks.append(
            {
                "check": "study_id_overlap_train_test",
                "method": "set intersection of StudyInstanceUID in train.csv and test.csv",
                "n_overlap": len(overlap),
                "verdict": "overlap_detected" if overlap else "no_overlap_detected_by_this_check",
            }
        )
    if test_ids is not None and sample_ids is not None:
        missing = sorted(test_ids - sample_ids)
        extra = sorted(sample_ids - test_ids)
        same_order = False
        if test_info.get("readable") and sample_info.get("readable"):
            same_order = (
                test_info["frame"][STUDY_ID_COLUMN].astype(str).tolist()
                == sample_info["frame"][STUDY_ID_COLUMN].astype(str).tolist()
            )
        checks.append(
            {
                "check": "test_vs_sample_submission_ids",
                "method": "set compare + row-order compare of StudyInstanceUID",
                "n_test_missing_from_sample": len(missing),
                "n_sample_not_in_test": len(extra),
                "same_row_order": same_order,
                "verdict": (
                    "match"
                    if not missing and not extra
                    else "mismatch"
                ),
            }
        )
    if train_info.get("readable"):
        frame = train_info["frame"]
        dup = int(frame[STUDY_ID_COLUMN].astype(str).duplicated().sum()) if STUDY_ID_COLUMN in frame.columns else None
        checks.append(
            {
                "check": "duplicate_study_ids_in_train",
                "method": "duplicated() on StudyInstanceUID",
                "n_duplicate_rows": dup,
                "verdict": "duplicates_detected" if dup else "no_duplicates_detected_by_this_check",
            }
        )
    return checks


def _structural_diff(train_info: dict, test_info: dict) -> dict:
    if not (train_info.get("readable") and test_info.get("readable")):
        return {"status": "not_compared", "reason": "train and/or test CSV unreadable"}
    train_cols = list(train_info["columns"])
    test_cols = list(test_info["columns"])
    return {
        "train_columns": train_cols,
        "test_columns": test_cols,
        "only_in_train": [c for c in train_cols if c not in test_cols],
        "only_in_test": [c for c in test_cols if c not in train_cols],
        "documented_train_fields_missing": [
            c for c in DOCUMENTED_TRAIN_CSV_FIELDS if c not in train_cols
        ],
        "documented_test_fields_missing": [
            c for c in DOCUMENTED_TEST_CSV_FIELDS if c not in test_cols
        ],
        "report_in_train": REPORT_COLUMN in train_cols,
        "report_in_test": REPORT_COLUMN in test_cols,
        "label_columns_in_train": [
            c for c in DOCUMENTED_LABEL_COLUMN_CANDIDATES if c in train_cols
        ],
        "label_columns_in_test": [
            c for c in DOCUMENTED_LABEL_COLUMN_CANDIDATES if c in test_cols
        ],
    }


def _target_reconciliation() -> dict:
    return {
        "policy": (
            "TARGET_KEYS in knee/config.py is the project's internal source of "
            "truth and was not renamed. Official Kaggle sample_submission.csv "
            "headers differ in spelling; the mismatch is escalated here."
        ),
        "internal_target_keys": list(TARGET_KEYS),
        "official_sample_submission_columns": list(OFFICIAL_TARGET_COLUMNS),
        "mapping": [
            {"slot": i + 1, "target_key": key, "official_column": official}
            for i, (key, official) in enumerate(TARGET_KEY_TO_OFFICIAL.items())
        ],
        "order_matches": True,
        "spelling_matches": list(TARGET_KEYS) == list(OFFICIAL_TARGET_COLUMNS),
        "id_column_internal_synthetic_pipeline": "study_id",
        "id_column_official": STUDY_ID_COLUMN,
        "sources": [KAGGLE_OVERVIEW_PAGE, KAGGLE_DATA_PAGE],
    }


def _official_schema_documentation() -> dict:
    return {
        "evidence_class": "official_documentation_not_local_observation",
        "sources": [KAGGLE_DATA_PAGE, KAGGLE_OVERVIEW_PAGE],
        "documented_files": list(DOCUMENTED_FILES),
        "documented_directories": list(DOCUMENTED_DIRECTORIES),
        "documented_dicom_layout": "{split}_series/{StudyInstanceUID}/{SeriesInstanceUID}/*.dcm",
        "documented_train_csv_fields_named_on_data_page": list(DOCUMENTED_TRAIN_CSV_FIELDS),
        "documented_test_csv_fields_named_on_data_page": list(DOCUMENTED_TEST_CSV_FIELDS),
        "documented_series_columns": list(DOCUMENTED_SERIES_COLUMNS),
        "documented_planes": list(DOCUMENTED_PLANES),
        "documented_sample_submission_columns": list(SAMPLE_SUBMISSION_COLUMNS),
        "label_column_assumption": (
            "The data page lists twelve binary findings with the official "
            "column spelling. It does not explicitly bullet those names under "
            "the train.csv field list (which names StudyInstanceUID and Report). "
            "Whether the twelve columns are physically present in train.csv is "
            "an observation that can only be made from the local files."
        ),
        "patient_identifier_assumption": (
            "The documented CSV schemas do not include PatientID. If a patient "
            "identifier exists it would have to come from DICOM tags. Grouped "
            "splits therefore fall back to StudyInstanceUID until a complete "
            "patient identifier is observed."
        ),
        "organizer_statements_not_verified_locally": [
            "The data page states there are about 1300 studies in the (hidden) test set.",
            "The data page states series typically contain 20–45 slices (median 30).",
        ],
    }


def data_not_available_result(
    *,
    explicit_root: str | Path | None = None,
    reason: str | None = None,
) -> dict:
    searched = describe_search(explicit_root)
    return {
        "status": STATUS_DATA_NOT_AVAILABLE,
        "scientific_evaluation": False,
        "source": "local_filesystem",
        "reason": reason or "no documented competition files were found at any candidate data root",
        "searched_roots": searched,
        "statistics": None,
        "verified_observations": [
            "No local directory among the searched candidates contained "
            "sample_submission.csv, train.csv, train_series.csv, test.csv, or "
            "test_series.csv.",
            f"Candidate roots searched: {', '.join(row['path'] for row in searched)}.",
        ],
        "assumptions": [
            "The on-disk layout, when the dataset is mounted, matches the "
            "official Kaggle data page (see official_schema_documentation).",
            "A patient-level identifier may exist in DICOM PatientID even though "
            "it is not documented as a CSV column.",
        ],
        "official_schema_documentation": _official_schema_documentation(),
        "target_reconciliation": _target_reconciliation(),
        "checks_not_run": [
            "study / patient / series / instance counts",
            "DICOM dimensions, spacing, orientation, slice order",
            "sequence descriptions",
            "missing and corrupt files",
            "label availability and prevalence",
            "multilingual report availability",
            "duplicate patients or studies",
            "leakage checks against local identifiers",
            "train/test structural comparison of local files",
        ],
        "unresolved_questions": [
            "Where will the competition dataset be mounted, and will "
            "RSNA_DATA_ROOT be set to that directory?",
            "Does local train.csv contain the twelve official label columns, "
            "only Report text, or both?",
            "Is DICOM PatientID populated and stable enough to use as a grouping key?",
        ],
    }


def audit_data_root(
    root: Path,
    *,
    max_series: int | None = None,
    source: str = "local_competition_data",
) -> dict:
    """Audit a directory that already looks like a competition root."""
    if not is_competition_root(root):
        return data_not_available_result(
            explicit_root=root,
            reason=f"{root} exists but contains none of the documented competition files",
        )

    files = {name: _read_csv(root / name) for name in DOCUMENTED_FILES}
    dirs = {name: (root / name).is_dir() for name in DOCUMENTED_DIRECTORIES}

    train = files["train.csv"]
    test = files["test.csv"]
    train_series = files["train_series.csv"]
    test_series = files["test_series.csv"]
    sample = files["sample_submission.csv"]

    if sample.get("readable"):
        sample_cols = list(sample["columns"])
        sample_header_ok = sample_cols == list(SAMPLE_SUBMISSION_COLUMNS)
    else:
        sample_header_ok = False

    result = {
        "status": STATUS_AUDITED,
        "scientific_evaluation": True,
        "source": source,
        "data_root": str(root),
        "inventory": _inventory(root),
        "documented_files_present": {
            name: info.get("present", False) for name, info in files.items()
        },
        "documented_directories_present": dirs,
        "csv": {
            name: {**_drop_frame(info), **_profile_csv_id(info, STUDY_ID_COLUMN)}
            for name, info in files.items()
        },
        "sample_submission_header_matches_official": sample_header_ok,
        "labels": _audit_labels(train),
        "reports": _audit_reports(train, test),
        "train_series": _audit_series_table(train_series, DOCUMENTED_SERIES_COLUMNS),
        "test_series": _audit_series_table(test_series, DOCUMENTED_SERIES_COLUMNS),
        "dicom_train": _audit_dicom_tree(root, "train", train_series, max_series=max_series),
        "dicom_test": _audit_dicom_tree(root, "test", test_series, max_series=max_series),
        "leakage_checks": _leakage_checks(train, test, sample),
        "train_test_structural_differences": _structural_diff(train, test),
        "target_reconciliation": _target_reconciliation(),
        "official_schema_documentation": _official_schema_documentation(),
        "verified_observations": [],
        "assumptions": [
            "Series folders follow {split}_series/{StudyInstanceUID}/{SeriesInstanceUID}/ as documented.",
            "Pixel-data decoding is not required for the header audit (stop_before_pixels=True).",
        ],
        "unresolved_questions": [],
    }

    observations = [
        f"Audited data root: {root}",
        f"Source label: {source}",
    ]
    for name, present in result["documented_files_present"].items():
        observations.append(f"{name}: {'present' if present else 'absent'}")
    result["verified_observations"] = observations

    if source == "software_fixture":
        result["scientific_evaluation"] = False
        result["verified_observations"].insert(
            0,
            "THIS RUN USED A SOFTWARE FIXTURE, NOT THE KAGGLE COMPETITION DATA.",
        )
    return result


def run_audit(
    *,
    data_root: str | Path | None = None,
    max_series: int | None = None,
    source: str | None = None,
) -> dict:
    """Resolve the data root and audit it, or return DATA_NOT_AVAILABLE."""
    try:
        root = get_data_root(data_root)
    except DataNotAvailable as exc:
        return data_not_available_result(explicit_root=data_root, reason=str(exc))
    label = source
    if label is None:
        label = "software_fixture" if "pytest" in str(root) or "tmp" in str(root).lower() else "local_competition_data"
        # Do not guess fixture vs real from the path string for the committed
        # audit: default to local_competition_data when the caller did not say.
        if source is None:
            label = "local_competition_data"
    return audit_data_root(root, max_series=max_series, source=label)


def render_audit_markdown(result: dict) -> str:
    """Render an audit result. Never fills in statistics when they were not computed."""
    lines: list[str] = []
    status = result.get("status", "UNKNOWN")
    lines.append("# Competition data audit")
    lines.append("")
    lines.append(f"**Status: {status}**")
    lines.append("")
    lines.append(
        "This document separates **verified observations** (from local files or "
        "from the absence of local files) from **assumptions** and from "
        "**official documentation** that was not re-observed on disk."
    )
    lines.append("")
    if status == STATUS_DATA_NOT_AVAILABLE:
        lines.append(
            "Real competition files could not be accessed. No study counts, "
            "label prevalences, DICOM histograms, or other audit statistics "
            "were generated."
        )
        lines.append("")

    lines.append("## Verified observations")
    lines.append("")
    for item in result.get("verified_observations") or []:
        lines.append(f"- {item}")
    if result.get("reason"):
        lines.append(f"- Resolver reason: `{result['reason']}`")
    if result.get("searched_roots"):
        lines.append("")
        lines.append("Searched data-root candidates:")
        lines.append("")
        lines.append("| Path | Status |")
        lines.append("| --- | --- |")
        for row in result["searched_roots"]:
            lines.append(f"| `{row['path']}` | {row['status']} |")
    lines.append("")

    lines.append("## Assumptions")
    lines.append("")
    for item in result.get("assumptions") or []:
        lines.append(f"- {item}")
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
    lines.append(
        f"Order matches: **{recon['order_matches']}**. "
        f"Spelling matches: **{recon['spelling_matches']}**."
    )
    lines.append("")
    lines.append(
        "Competition submissions written by `scripts/make_submission.py` use the "
        "official spelling. `TARGET_KEYS` are unchanged so the ontology registry "
        "does not drift."
    )
    lines.append("")

    lines.append("## Official documented schema (not a local file observation)")
    lines.append("")
    schema = result.get("official_schema_documentation") or _official_schema_documentation()
    lines.append(f"Evidence class: `{schema['evidence_class']}`")
    lines.append("")
    for url in schema["sources"]:
        lines.append(f"- {url}")
    lines.append("")
    lines.append("Documented files: " + ", ".join(f"`{n}`" for n in schema["documented_files"]))
    lines.append("")
    lines.append(
        "Documented DICOM layout: `" + schema["documented_dicom_layout"] + "`"
    )
    lines.append("")
    lines.append(schema["label_column_assumption"])
    lines.append("")
    lines.append(schema["patient_identifier_assumption"])
    lines.append("")
    lines.append("Organizer statements that were **not** verified locally:")
    lines.append("")
    for item in schema["organizer_statements_not_verified_locally"]:
        lines.append(f"- {item}")
    lines.append("")

    if status == STATUS_DATA_NOT_AVAILABLE:
        lines.append("## Statistics")
        lines.append("")
        lines.append(
            "`statistics: null` — not computed. A table of zeros would be a "
            "synthetic audit statistic and is forbidden when the files are missing."
        )
        lines.append("")
        lines.append("## Checks not run")
        lines.append("")
        for item in result.get("checks_not_run") or []:
            lines.append(f"- {item}")
        lines.append("")
        lines.append("## Unresolved questions")
        lines.append("")
        for item in result.get("unresolved_questions") or []:
            lines.append(f"- {item}")
        lines.append("")
        lines.append("## How to re-run")
        lines.append("")
        lines.append("```bash")
        lines.append("export RSNA_DATA_ROOT=/path/to/rsna-knee-abnormality-detection")
        lines.append("python scripts/audit_competition_data.py")
        lines.append("```")
        lines.append("")
        return "\n".join(lines)

    lines.append("## File inventory")
    lines.append("")
    inv = result.get("inventory") or {}
    lines.append(
        f"- Files: {inv.get('n_files')}; directories: {inv.get('n_directories')}; "
        f"bytes: {inv.get('bytes')}"
    )
    lines.append(f"- Suffix counts: `{inv.get('suffix_counts')}`")
    lines.append("")

    lines.append("## CSV schemas")
    lines.append("")
    for name, info in (result.get("csv") or {}).items():
        lines.append(f"### `{name}`")
        lines.append("")
        if not info.get("present"):
            lines.append("Absent.")
            lines.append("")
            continue
        if not info.get("readable"):
            lines.append(f"Unreadable: {info.get('error')}")
            lines.append("")
            continue
        lines.append(
            f"- Rows: {info.get('n_rows')}; columns: {info.get('n_columns')}"
        )
        lines.append("- Columns: " + ", ".join(f"`{c}`" for c in info.get("columns") or []))
        lines.append(f"- Dtypes: `{info.get('dtypes')}`")
        if "n_unique" in info:
            lines.append(
                f"- `{info.get('column')}` unique={info.get('n_unique')}, "
                f"duplicate_rows={info.get('n_duplicate_rows')}, "
                f"missing={info.get('n_missing')}"
            )
        lines.append("")

    lines.append("## Labels")
    lines.append("")
    labels = result.get("labels") or {}
    lines.append(f"- Labels available: {labels.get('labels_available')}")
    lines.append(
        "- Official columns present: "
        + ", ".join(f"`{c}`" for c in labels.get("official_label_columns_present") or [])
    )
    lines.append(
        "- Official columns missing: "
        + ", ".join(f"`{c}`" for c in labels.get("official_label_columns_missing") or [])
        or "- Official columns missing: (none)"
    )
    if labels.get("prevalence"):
        lines.append("")
        lines.append("| Official column | known | positive | negative | unknown | rate among known |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for col, stats in labels["prevalence"].items():
            rate = stats["positive_rate_among_known"]
            rate_s = "n/a" if rate is None else f"{rate:.4f}"
            lines.append(
                f"| `{col}` | {stats['n_known']} | {stats['n_positive']} | "
                f"{stats['n_negative']} | {stats['n_unknown']} | {rate_s} |"
            )
    lines.append("")

    lines.append("## Reports (heuristic language cues)")
    lines.append("")
    lines.append(f"```json")
    import json

    lines.append(json.dumps(result.get("reports"), indent=2, sort_keys=True))
    lines.append("```")
    lines.append("")

    lines.append("## Series tables")
    lines.append("")
    for key in ("train_series", "test_series"):
        lines.append(f"### `{key}`")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(result.get(key), indent=2, sort_keys=True, default=str))
        lines.append("```")
        lines.append("")

    lines.append("## DICOM")
    lines.append("")
    for key in ("dicom_train", "dicom_test"):
        lines.append(f"### `{key}`")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(result.get(key), indent=2, sort_keys=True, default=str))
        lines.append("```")
        lines.append("")

    lines.append("## Leakage checks")
    lines.append("")
    lines.append(
        "Absence of evidence is not evidence of absence: a check that finds no "
        "overlap is reported as `no_overlap_detected_by_this_check`."
    )
    lines.append("")
    for check in result.get("leakage_checks") or []:
        lines.append(
            f"- `{check.get('check')}` via {check.get('method')}: **{check.get('verdict')}** "
            f"({ {k: v for k, v in check.items() if k not in {'check', 'method', 'verdict'}} })"
        )
    lines.append("")

    lines.append("## Train / test structural differences")
    lines.append("")
    lines.append("```json")
    lines.append(
        json.dumps(
            result.get("train_test_structural_differences"),
            indent=2,
            sort_keys=True,
            default=str,
        )
    )
    lines.append("```")
    lines.append("")

    lines.append("## Unresolved questions")
    lines.append("")
    questions = result.get("unresolved_questions") or ["(none recorded)"]
    for item in questions:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def write_audit_report(result: dict, path: Path | None = None) -> Path:
    path = Path(path) if path else PROJECT_ROOT / "reports" / "data_audit.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_audit_markdown(result), encoding="utf-8")
    return path
