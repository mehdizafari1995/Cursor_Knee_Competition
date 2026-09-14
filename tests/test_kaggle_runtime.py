"""Kaggle path detection, preflight, source extraction, and read-only inputs."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from knee.kaggle_runtime import (
    KAGGLE_COMPETITION_SLUG,
    KaggleLayout,
    PreflightError,
    ReadOnlyInputError,
    ReadOnlyInputGuard,
    assert_not_kaggle_input,
    detect_kaggle_paths,
    extract_source_to_working,
    find_source_root,
    infer_data_source,
    is_kaggle_input_path,
    list_top_level,
    preflight,
)
from tests.competition_fixtures import make_kaggle_audit_fixture


def _layout(tmp_path: Path) -> KaggleLayout:
    return KaggleLayout(
        input_root=tmp_path / "kaggle" / "input",
        working_root=tmp_path / "kaggle" / "working",
        competition_slug=KAGGLE_COMPETITION_SLUG,
    )


def test_detect_kaggle_paths_default_mount():
    detected = detect_kaggle_paths()
    assert detected["competition_dir"] == "/kaggle/input/rsna-knee-abnormality-detection"
    assert detected["output_dir"] == "/kaggle/working/rsna_audit_outputs"
    assert detected["src_extract_dir"] == "/kaggle/working/rsna_knee_src"
    assert detected["internet_used"] is False
    assert detected["gpu_required"] is False


def test_detect_kaggle_paths_on_simulated_mount(tmp_path):
    layout = _layout(tmp_path)
    root = make_kaggle_audit_fixture(layout.competition_dir)
    (layout.working_root).mkdir(parents=True)
    detected = detect_kaggle_paths(layout)
    assert detected["competition_dir_exists"] is True
    assert detected["is_competition_root"] is True
    assert "train.csv" in detected["top_level_competition"]
    assert "train_series" in detected["top_level_competition"]
    assert detected["kaggle_runtime"] is True
    # Must not dump the DICOM tree.
    assert not any(name.endswith(".dcm") for name in detected["top_level_competition"])
    assert root == layout.competition_dir


def test_list_top_level_does_not_recurse(tmp_path):
    layout = _layout(tmp_path)
    make_kaggle_audit_fixture(layout.competition_dir)
    names = list_top_level(layout.competition_dir)
    assert "train.csv" in names
    assert all("/" not in name for name in names)
    assert not any(name.endswith(".dcm") for name in names)


def test_infer_data_source_real_kaggle_mount(tmp_path):
    layout = _layout(tmp_path)
    make_kaggle_audit_fixture(layout.competition_dir)
    assert infer_data_source(layout.competition_dir, layout=layout) == "real_kaggle_mount"


def test_preflight_ok_and_missing(tmp_path):
    layout = _layout(tmp_path)
    layout.working_root.mkdir(parents=True)
    missing = preflight(layout=layout)
    assert missing.ok is False
    assert missing.status == "DATA_NOT_AVAILABLE"
    assert missing.scientific_evaluation is False
    assert "AUDIT STOPPED" in missing.message
    assert missing.internet_used is False
    assert missing.gpu_required is False

    make_kaggle_audit_fixture(layout.competition_dir)
    ok = preflight(layout=layout, data_source="software_fixture")
    assert ok.ok is True
    assert ok.data_root == layout.competition_dir
    assert ok.documented_files_present["train.csv"] is True
    assert ok.data_source == "software_fixture"
    assert ok.scientific_evaluation is False


def test_assert_not_kaggle_input(tmp_path):
    layout = _layout(tmp_path)
    layout.input_root.mkdir(parents=True)
    layout.working_root.mkdir(parents=True)
    protected = layout.input_root / "rsna-knee-abnormality-detection" / "train.csv"
    protected.parent.mkdir(parents=True)
    protected.write_text("x\n")
    assert is_kaggle_input_path(protected, layout)
    with pytest.raises(ReadOnlyInputError, match="refusing to write"):
        assert_not_kaggle_input(protected, layout)
    allowed = layout.working_root / "rsna_audit_outputs" / "real_data_audit.md"
    assert assert_not_kaggle_input(allowed, layout) == allowed


def test_read_only_guard_detects_mutation(tmp_path):
    layout = _layout(tmp_path)
    root = make_kaggle_audit_fixture(layout.competition_dir)
    csv_path = root / "train.csv"
    guard = ReadOnlyInputGuard([root], recursive=False)
    original = csv_path.read_text(encoding="utf-8")
    csv_path.write_text(original + "# mutated\n", encoding="utf-8")
    with pytest.raises(ReadOnlyInputError, match="changed"):
        guard.assert_unchanged()


def test_extract_source_zip_to_working_not_input(tmp_path):
    layout = _layout(tmp_path)
    layout.input_root.mkdir(parents=True)
    layout.working_root.mkdir(parents=True)
    source_ds = layout.input_root / "rsna-knee-src"
    source_ds.mkdir()
    zip_path = source_ds / "rsna_knee_source.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("knee/config.py", "TARGET_KEYS = ()\n")
        zf.writestr("knee/kaggle_audit.py", "def run_notebook_audit():\n    pass\n")
        zf.writestr("scripts/build_kaggle_bundle.py", "print(1)\n")
    extracted = extract_source_to_working(layout)
    assert extracted == layout.src_extract_dir or extracted.is_dir()
    assert (layout.src_extract_dir / "knee" / "config.py").is_file()
    assert not (layout.competition_dir / "knee").exists()
    # Input zip mtime/size unchanged.
    assert zip_path.is_file()


def test_extract_rejects_zip_slip(tmp_path):
    layout = _layout(tmp_path)
    layout.input_root.mkdir(parents=True)
    layout.working_root.mkdir(parents=True)
    source_ds = layout.input_root / "evil-src"
    source_ds.mkdir()
    zip_path = source_ds / "evil.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("knee/config.py", "ok\n")
        zf.writestr("knee/kaggle_audit.py", "ok\n")
        zf.writestr("../escaped.py", "nope\n")
    with pytest.raises(PreflightError, match="escapes"):
        extract_source_to_working(layout)


def test_find_source_root_prefers_repo_checkout():
    root = find_source_root(KaggleLayout(), search_repo=True)
    assert root is not None
    assert (root / "knee" / "config.py").is_file()


def test_find_source_root_double_nested_github_archive(tmp_path):
    layout = _layout(tmp_path)
    layout.input_root.mkdir(parents=True)
    layout.working_root.mkdir(parents=True)
    nested = (
        layout.input_root
        / "cursor-knee-source"
        / "Cursor_Knee_Competition-main"
        / "Cursor_Knee_Competition-main"
    )
    (nested / "knee").mkdir(parents=True)
    (nested / "knee" / "config.py").write_text("TARGET_KEYS = ()\n")
    (nested / "knee" / "kaggle_audit.py").write_text("def run_notebook_audit():\n    pass\n")
    found = find_source_root(layout, search_repo=False)
    assert found == nested
