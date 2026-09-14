"""Kaggle three-level audit: provenance, sampling, report-text exclusion, fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from knee.audit import STATUS_AUDITED, STATUS_DATA_NOT_AVAILABLE
from knee.config import NUM_TARGETS, OFFICIAL_TARGET_COLUMNS, REPORT_COLUMN
from knee.kaggle_audit import (
    OUTPUT_FILENAMES,
    PROVENANCE_FULL_METADATA,
    PROVENANCE_ONE_HEADER,
    PROVENANCE_SAMPLED,
    PROVENANCE_VALUES,
    ReportTextLeakError,
    assert_no_report_text,
    run_notebook_audit,
    select_detailed_sample,
)
from knee.kaggle_runtime import KaggleLayout, PreflightError
from tests.competition_fixtures import (
    UNIQUE_REPORT_TOKENS,
    make_kaggle_audit_fixture,
)


PROVENANCE_SET = set(PROVENANCE_VALUES)


def _run_fixture(tmp_path: Path, **kwargs) -> dict:
    root = make_kaggle_audit_fixture(tmp_path / "mini")
    out = tmp_path / "rsna_audit_outputs"
    result = run_notebook_audit(
        data_root=root,
        output_dir=out,
        data_source="software_fixture",
        sample_size=kwargs.pop("sample_size", 13),
        **kwargs,
    )
    return result


def test_notebook_logic_software_fixture_not_scientific(tmp_path):
    result = _run_fixture(tmp_path)
    assert result["status"] == STATUS_AUDITED
    assert result["data_source"] == "software_fixture"
    assert result["scientific_evaluation"] is False
    assert result["internet_used"] is False
    assert result["gpu_required"] is False
    assert result["gpu_used"] is False
    assert result["training_run"] is False
    assert result["folds_created"] is False
    assert result["patient_grouped_folds_allowed"] is False
    blob = json.dumps(result, default=str).lower()
    assert result.get("metrics") is None
    assert "smoke_auc" not in blob
    assert "macro_roc_auc" not in blob


def test_all_required_outputs_exist_with_provenance(tmp_path):
    result = _run_fixture(tmp_path)
    out = Path(result["output_dir"])
    for name in OUTPUT_FILENAMES:
        assert (out / name).is_file(), name
    manifest = json.loads((out / "audit_manifest.json").read_text())
    assert manifest["scientific_evaluation"] is False
    assert manifest["data_source"] == "software_fixture"
    assert manifest["training_run"] is False
    assert manifest["folds_created"] is False
    by_file = {row["file"]: row["provenance"] for row in manifest["outputs"]}
    assert by_file["target_prevalence.csv"] == PROVENANCE_FULL_METADATA
    assert by_file["series_summary.csv"] == PROVENANCE_FULL_METADATA
    assert by_file["dicom_header_summary.csv"] == PROVENANCE_ONE_HEADER
    assert by_file["dicom_qc_sample.csv"] == PROVENANCE_SAMPLED

    prev = pd.read_csv(out / "target_prevalence.csv")
    assert set(prev["provenance"].unique()) <= PROVENANCE_SET
    assert list(prev["official_column"]) == list(OFFICIAL_TARGET_COLUMNS)
    assert REPORT_COLUMN not in prev.columns

    series = pd.read_csv(out / "series_summary.csv")
    assert set(series["provenance"].unique()) == {PROVENANCE_FULL_METADATA}

    headers = pd.read_csv(out / "dicom_header_summary.csv")
    assert set(headers["provenance"].unique()) == {PROVENANCE_ONE_HEADER}

    qc = pd.read_csv(out / "dicom_qc_sample.csv")
    assert set(qc["provenance"].unique()) == {PROVENANCE_SAMPLED}
    assert (~qc["is_full_dataset_statistic"].astype(bool)).all()
    assert REPORT_COLUMN not in qc.columns

    assert result["level_a"]["provenance"] == PROVENANCE_FULL_METADATA
    assert result["level_b"]["provenance"] == PROVENANCE_ONE_HEADER
    assert result["level_c"]["provenance"] == PROVENANCE_SAMPLED
    assert result["level_c"]["is_full_dataset_statistic"] is False


def test_level_a_covers_twelve_targets_and_csv_schema(tmp_path):
    result = _run_fixture(tmp_path)
    labels = result["level_a"]["labels"]
    assert labels["labels_available"] is True
    assert labels["official_label_columns_present"] == list(OFFICIAL_TARGET_COLUMNS)
    assert len(labels["prevalence"]) == NUM_TARGETS
    for official in OFFICIAL_TARGET_COLUMNS:
        assert labels["prevalence"][official]["n_positive"] >= 1
    train_reports = result["level_a"]["reports_availability"]["train"]
    assert train_reports["column_present"] is True
    assert train_reports["n_non_empty"] >= 1
    assert result["level_a"]["sample_submission_header_matches_official"] is True
    dirs = result["level_a"]["directory_counts"]["train"]
    assert dirs["n_study_dirs"] == 14
    assert dirs["n_series_dirs"] == 15


def test_patient_id_incomplete_blocks_grouped_folds(tmp_path):
    result = _run_fixture(tmp_path)
    level_b = result["level_b"]
    assert level_b["n_headers_read"] == 14
    assert level_b["n_with_patient_id"] == 13
    assert level_b["patient_id_sufficiently_complete"] is False
    assert level_b["patient_grouped_folds_created"] is False
    assert "do not create patient-grouped folds" in level_b["recommendation"].lower()
    assert level_b["manufacturer_counts"].get("SIEMENS", 0) >= 1
    assert "MR" in level_b["modality_counts"]


def test_deterministic_sampling_covers_targets_and_negatives(tmp_path):
    root = make_kaggle_audit_fixture(tmp_path / "mini")
    train = pd.read_csv(root / "train.csv")
    series = pd.read_csv(root / "train_series.csv")
    a = select_detailed_sample(train, series, sample_size=13)
    b = select_detailed_sample(train, series, sample_size=13)
    assert a == b
    reasons = [row["sample_reason"] for row in a]
    for official in OFFICIAL_TARGET_COLUMNS:
        assert f"positive:{official}" in reasons
    assert "negative" in reasons
    assert all(row["provenance"] == PROVENANCE_SAMPLED for row in a)

    small = select_detailed_sample(train, series, sample_size=5)
    assert len(small) == 5
    assert small == select_detailed_sample(train, series, sample_size=5)
    assert small[0]["sample_reason"] == f"positive:{OFFICIAL_TARGET_COLUMNS[0]}"


def test_level_c_does_not_read_every_dicom(tmp_path, monkeypatch):
    root = make_kaggle_audit_fixture(tmp_path / "mini")
    all_dcm = list(root.rglob("*.dcm"))
    assert len(all_dcm) > 20
    calls: list[tuple[Path, bool]] = []
    from knee import kaggle_audit as module

    real = module.read_header

    def wrapped(path, *, stop_before_pixels=True):
        calls.append((Path(path), stop_before_pixels))
        return real(path, stop_before_pixels=stop_before_pixels)

    monkeypatch.setattr(module, "read_header", wrapped)
    result = run_notebook_audit(
        data_root=root,
        output_dir=tmp_path / "out",
        data_source="software_fixture",
        sample_size=3,
        max_pixel_arrays_per_series=1,
    )
    read_paths = {path.resolve() for path, _ in calls}
    assert len(read_paths) < len(all_dcm)
    pixel_reads = [path for path, stop in calls if stop is False]
    assert pixel_reads, "Level C should inspect selected pixel arrays"
    assert len(pixel_reads) == result["level_c"]["n_pixel_arrays_read"]
    assert result["level_c"]["n_studies_sampled"] == 3


def test_report_text_is_excluded_from_all_outputs(tmp_path):
    result = _run_fixture(tmp_path)
    out = Path(result["output_dir"])
    snippets = list(UNIQUE_REPORT_TOKENS) + [
        "UNIQUE_FIXTURE_REPORT_TOKEN_NEGATIVE_12",
        "UNIQUE_FIXTURE_REPORT_TOKEN_UNKNOWN_13",
    ]
    paths = [out / name for name in OUTPUT_FILENAMES]
    assert_no_report_text(paths, snippets)
    blob = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    for snippet in snippets:
        assert snippet not in blob
    assert REPORT_COLUMN in json.dumps(result["level_a"]["csv"]["train.csv"]["columns"])
    md = (out / "real_data_audit.md").read_text(encoding="utf-8")
    assert "not exported" in md.lower()


def test_assert_no_report_text_raises_on_leak(tmp_path):
    path = tmp_path / "leaky.md"
    path.write_text("see UNIQUE_FIXTURE_REPORT_TOKEN_ACL_00 in this dump\n")
    with pytest.raises(ReportTextLeakError):
        assert_no_report_text([path], ["UNIQUE_FIXTURE_REPORT_TOKEN_ACL_00"])


def test_graceful_missing_data_failure(tmp_path):
    out = tmp_path / "rsna_audit_outputs"
    result = run_notebook_audit(
        data_root=tmp_path / "does-not-exist",
        output_dir=out,
        data_source="software_fixture",
    )
    assert result["status"] == STATUS_DATA_NOT_AVAILABLE
    assert result["scientific_evaluation"] is False
    assert result["statistics"] is None
    assert "n_studies" not in result
    md = (out / "real_data_audit.md").read_text(encoding="utf-8")
    assert "**Status: DATA_NOT_AVAILABLE**" in md
    assert "not computed" in md.lower()
    assert "AUDIT STOPPED" in (result.get("preflight") or {}).get("message", "")
    manifest = json.loads((out / "audit_manifest.json").read_text())
    assert manifest["status"] == STATUS_DATA_NOT_AVAILABLE
    prev = pd.read_csv(out / "target_prevalence.csv")
    assert prev["n_positive"].isna().all()


def test_run_or_stop_raises_clearly(tmp_path):
    from knee.kaggle_audit import run_or_stop

    with pytest.raises(PreflightError, match="AUDIT STOPPED"):
        run_or_stop(data_root=tmp_path / "missing", output_dir=tmp_path / "out")


def test_read_only_inputs_not_modified_by_audit(tmp_path):
    root = make_kaggle_audit_fixture(tmp_path / "mini")
    before = {
        path: (path.stat().st_mtime_ns, path.stat().st_size)
        for path in root.rglob("*")
        if path.is_file()
    }
    run_notebook_audit(
        data_root=root,
        output_dir=tmp_path / "out",
        data_source="software_fixture",
        sample_size=4,
    )
    after = {
        path: (path.stat().st_mtime_ns, path.stat().st_size)
        for path in root.rglob("*")
        if path.is_file()
    }
    assert before == after
    assert not (root / "real_data_audit.md").exists()


def test_kaggle_mount_label_on_simulated_layout(tmp_path):
    layout = KaggleLayout(
        input_root=tmp_path / "kaggle" / "input",
        working_root=tmp_path / "kaggle" / "working",
    )
    make_kaggle_audit_fixture(layout.competition_dir)
    layout.working_root.mkdir(parents=True, exist_ok=True)
    result = run_notebook_audit(
        layout=layout,
        data_source="software_fixture",
        sample_size=4,
    )
    assert result["scientific_evaluation"] is False
    assert result["preflight"]["layout"]["competition_dir"] == str(layout.competition_dir)
    assert result["output_dir"] == str(layout.output_dir)


def test_notebook_file_is_offline_and_tokenless():
    nb_path = Path(__file__).resolve().parents[1] / "notebooks" / "01_kaggle_real_data_audit.ipynb"
    payload = json.loads(nb_path.read_text(encoding="utf-8"))
    assert payload["nbformat"] == 4
    source = "\n".join("".join(cell.get("source") or []) for cell in payload["cells"])
    lowered = source.lower()
    assert "kaggle.json" not in lowered
    assert "kaggle_username" not in lowered
    assert "kaggle_key" not in lowered
    assert "kaggle.api" not in lowered
    assert "pip install" not in lowered
    assert "!pip" not in lowered
    assert "wget" not in lowered
    assert "requests.get" not in lowered
    assert "urllib.request" not in lowered
    assert "train_baseline" not in lowered
    assert "internet_used=false" in lowered
    assert "rsna_audit_outputs" in lowered
    assert "full_metadata" in lowered
    assert "_find_source_in_tree" in source
    assert "_debug_input_layout" in source
    assert "SOURCE_SEARCH_MAX_DEPTH" in source
    assert "/kaggle/input/rsna-knee-abnormality-detection" in source
    assert "competitions" in source
    meta = payload.get("metadata") or {}
    kaggle_meta = meta.get("kaggle") or {}
    assert kaggle_meta.get("isInternetEnabled") is False
    assert kaggle_meta.get("gpu") is False
