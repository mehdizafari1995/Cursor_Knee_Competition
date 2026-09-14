"""Data-root resolution and competition-data audit (software fixtures only)."""

from __future__ import annotations

from knee.audit import STATUS_AUDITED, STATUS_DATA_NOT_AVAILABLE, run_audit
from knee.config import DATA_ROOT_ENV, OFFICIAL_TARGET_COLUMNS, TARGET_KEYS
from knee.paths import DataNotAvailable, candidate_data_roots, get_data_root
from tests.competition_fixtures import make_mini_competition_root


def test_data_root_env_not_hard_coded():
    assert DATA_ROOT_ENV == "RSNA_DATA_ROOT"
    for path in candidate_data_roots():
        assert "Users/" not in str(path)
        assert "C:\\" not in str(path)


def test_get_data_root_raises_when_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("RSNA_DATA_ROOT", raising=False)
    monkeypatch.delenv("KNEE_DATA_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    try:
        get_data_root(tmp_path / "nope")
        assert False, "expected DataNotAvailable"
    except DataNotAvailable as exc:
        assert exc.status == STATUS_DATA_NOT_AVAILABLE
        assert "DATA_NOT_AVAILABLE" in str(exc)


def test_run_audit_data_not_available_has_null_statistics(monkeypatch, tmp_path):
    monkeypatch.delenv("RSNA_DATA_ROOT", raising=False)
    monkeypatch.delenv("KNEE_DATA_ROOT", raising=False)
    result = run_audit(data_root=tmp_path / "missing-dataset")
    assert result["status"] == STATUS_DATA_NOT_AVAILABLE
    assert result["statistics"] is None
    assert result["scientific_evaluation"] is False
    text = " ".join(result["verified_observations"]).lower()
    assert "sample_submission.csv" in text or "documented" in text
    # Guard against accidentally publishing fixture counts as real stats.
    assert "n_studies" not in result


def test_audit_software_fixture_uses_official_columns(tmp_path):
    root = make_mini_competition_root(tmp_path / "mini")
    result = run_audit(data_root=root, source="software_fixture")
    assert result["status"] == STATUS_AUDITED
    assert result["scientific_evaluation"] is False
    assert result["sample_submission_header_matches_official"] is True
    labels = result["labels"]
    assert labels["labels_available"] is True
    assert labels["official_label_columns_present"] == list(OFFICIAL_TARGET_COLUMNS)
    baker = labels["prevalence"]["Baker's"]
    assert baker["n_unknown"] == 1
    assert result["dicom_train"]["n_series_scanned"] == 4
    assert result["dicom_train"]["n_unreadable_instances"] == 0
    assert result["reports"]["train"]["column_present"] is True
    assert result["reports"]["test"]["column_present"] is False
    overlap = next(
        c for c in result["leakage_checks"] if c["check"] == "study_id_overlap_train_test"
    )
    assert overlap["n_overlap"] == 0
    # Internal TARGET_KEYS must not be silently substituted for official names.
    assert list(TARGET_KEYS) != list(OFFICIAL_TARGET_COLUMNS)
    assert result["target_reconciliation"]["spelling_matches"] is False
    assert result["target_reconciliation"]["order_matches"] is True


def test_audit_detects_missing_series_dir(tmp_path):
    root = make_mini_competition_root(tmp_path / "mini")
    import pandas as pd

    series = pd.read_csv(root / "train_series.csv")
    series.loc[len(series)] = {
        "StudyInstanceUID": "1.2.840.10008.1.missing",
        "SeriesInstanceUID": "1.2.840.missing.series",
        "Fluid_Sensitive": 1,
        "Fat_Suppression": 1,
        "Anatomical_Plane": "Axial",
    }
    series.to_csv(root / "train_series.csv", index=False)
    result = run_audit(data_root=root, source="software_fixture")
    assert result["dicom_train"]["n_missing_series_dirs"] == 1
