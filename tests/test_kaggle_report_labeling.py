"""Kaggle report-labeling notebook: fixture-only, privacy, no scientific claims."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from knee.config import REPORT_COLUMN
from knee.kaggle_audit import assert_no_report_text
from knee.kaggle_runtime import KaggleLayout
from knee.report_labels import (
    KAGGLE_REPORT_OUTPUT_DIRNAME,
    SANITIZED_SUMMARY_FILENAME,
    assert_no_identifiers,
    collect_identifiers,
    run_notebook_report_labeling,
)
from tests.competition_fixtures import (
    REPORT_LABEL_GOLD_UIDS,
    REPORT_LABEL_LEAK_TOKEN,
    REPORT_LABEL_PATIENTS,
    REPORT_LABEL_WEAK_UIDS,
    make_report_labeling_fixture,
)


def test_notebook_logic_fixture_is_not_scientific(tmp_path):
    root = make_report_labeling_fixture(tmp_path / "mini")
    out = tmp_path / "rsna_report_label_outputs"
    result = run_notebook_report_labeling(
        data_root=root,
        output_dir=out,
        data_source="software_fixture",
    )
    assert result["scientific_evaluation"] is False
    assert result["claim_scientific_performance"] is False
    assert result["internet_used"] is False
    assert result["gpu_used"] is False
    assert result["training_run"] is False
    assert result["folds_created"] is False
    assert result["n_gold_studies"] == 12
    assert result["n_report_only_studies"] == 4
    summary = Path(out) / "sanitized" / SANITIZED_SUMMARY_FILENAME
    text = summary.read_text(encoding="utf-8")
    assert "SOFTWARE FIXTURE" in text
    assert "must not be quoted" in text.lower() or "software verification" in text.lower()
    assert REPORT_LABEL_LEAK_TOKEN not in text
    for uid in REPORT_LABEL_GOLD_UIDS + REPORT_LABEL_WEAK_UIDS:
        assert uid not in text
    for patient in REPORT_LABEL_PATIENTS:
        assert patient not in text


def test_sanitized_outputs_exclude_report_text_and_identifiers(tmp_path):
    root = make_report_labeling_fixture(tmp_path / "mini")
    train = pd.read_csv(root / "train.csv")
    out = tmp_path / "out"
    result = run_notebook_report_labeling(
        data_root=root,
        output_dir=out,
        data_source="software_fixture",
    )
    sanitized = list((Path(out) / "sanitized").glob("*"))
    snippets = [str(v) for v in train[REPORT_COLUMN].tolist()]
    identifiers = collect_identifiers(train)
    assert_no_report_text(sanitized, snippets)
    assert_no_identifiers(sanitized, identifiers)
    private_weak = Path(out) / "private" / "weak_labels.csv"
    assert private_weak.is_file()
    weak = pd.read_csv(private_weak)
    assert REPORT_COLUMN not in weak.columns
    assert "extractor_version" in weak.columns
    assert "language_code" in weak.columns
    assert "quality_flags" in weak.columns
    blob = json.dumps(result, default=str)
    assert REPORT_LABEL_LEAK_TOKEN not in blob
    eval_dirs = list((Path(out) / "eval_archive").rglob("gold_metrics.json"))
    assert eval_dirs, "versioned evaluation archive missing"


def test_missing_data_is_not_scientific(tmp_path):
    result = run_notebook_report_labeling(
        data_root=tmp_path / "missing",
        output_dir=tmp_path / "out",
        data_source="software_fixture",
    )
    assert result["scientific_evaluation"] is False
    assert result["gold_eval"] is None
    summary = Path(tmp_path / "out" / "sanitized" / SANITIZED_SUMMARY_FILENAME)
    assert summary.is_file()
    assert "SOFTWARE FIXTURE" in summary.read_text(encoding="utf-8")


def test_kaggle_layout_uses_report_output_dirname(tmp_path):
    layout = KaggleLayout(
        input_root=tmp_path / "kaggle" / "input",
        working_root=tmp_path / "kaggle" / "working",
        output_dirname=KAGGLE_REPORT_OUTPUT_DIRNAME,
    )
    make_report_labeling_fixture(layout.competition_dir)
    layout.working_root.mkdir(parents=True)
    result = run_notebook_report_labeling(layout=layout, data_source="software_fixture")
    assert result["scientific_evaluation"] is False
    assert KAGGLE_REPORT_OUTPUT_DIRNAME in result["output_dir"]


def test_notebook_file_is_offline_and_tokenless():
    nb_path = Path(__file__).resolve().parents[1] / "notebooks" / "02_kaggle_report_labeling.ipynb"
    payload = json.loads(nb_path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source") or []) for cell in payload["cells"])
    lowered = source.lower()
    assert "kaggle.json" not in lowered
    assert "kaggle_username" not in lowered
    assert "kaggle_key" not in lowered
    assert "kaggle.api" not in lowered
    assert "pip install" not in lowered
    assert "!pip" not in lowered
    assert "openai" not in lowered
    assert "anthropic" not in lowered
    assert "translate" not in lowered
    assert "train_baseline" not in lowered
    assert "internet_used=false" in lowered
    assert "rsna_report_label_outputs" in lowered
    assert "_walk_input_for_source" in source
    assert "/kaggle/input/rsna-knee-abnormality-detection" in source
    assert "scientific_evaluation" in lowered
    meta = payload.get("metadata") or {}
    kaggle_meta = meta.get("kaggle") or {}
    assert kaggle_meta.get("isInternetEnabled") is False
    assert kaggle_meta.get("gpu") is False


REPO_ROOT = Path(__file__).resolve().parents[1]
_UID_LIKE = re.compile(r"\b1\.2\.840\.\d+")
_PRIVATE_ROW_FILES = ("weak_labels.csv", "gold_predictions.csv", "language_assignments.csv")


def test_fixture_summary_includes_language_and_error_sections(tmp_path):
    root = make_report_labeling_fixture(tmp_path / "mini")
    out = tmp_path / "rsna_report_label_outputs"
    run_notebook_report_labeling(
        data_root=root,
        output_dir=out,
        data_source="software_fixture",
    )
    text = (out / "sanitized" / SANITIZED_SUMMARY_FILENAME).read_text(encoding="utf-8")
    assert "Language-level gold performance" in text
    assert "Aggregate error categories" in text
    assert REPORT_LABEL_LEAK_TOKEN not in text


def test_committed_summary_is_scientific_and_sanitized():
    path = REPO_ROOT / "reports" / "report_labeling_summary.md"
    text = path.read_text(encoding="utf-8")
    lowered = text.lower()
    assert "scientific_evaluation: `true`" in lowered
    assert "data_source: `real_kaggle_mount`" in lowered
    assert "software fixture" not in lowered
    assert "kaggle.json" not in lowered
    assert "kaggle_key" not in lowered
    assert REPORT_LABEL_LEAK_TOKEN not in text
    assert _UID_LIKE.search(text) is None
    for uid in REPORT_LABEL_GOLD_UIDS + REPORT_LABEL_WEAK_UIDS:
        assert uid not in text
    for patient in REPORT_LABEL_PATIENTS:
        assert patient not in text
    assert "Language-level gold performance" in text
    assert "Aggregate error categories" in text
    assert "n_gold_studies: 58" in text
    assert "n_report_only_studies: 4349" in text
    assert "`NOT_READY_FOR_HARD_WEAK_LABEL_TRAINING`" in text
    assert "`READY_FOR_SELECTIVE_LABEL_REFINEMENT`" in text
    assert "`READY_FOR_HARD_WEAK_LABEL_TRAINING`" not in text
    assert "NOT_READY_FOR_SELECTIVE_LABEL_REFINEMENT" not in text


def test_committed_phase4_reports_have_no_identifiers_or_report_tokens():
    for relative in (
        "reports/report_labeling_summary.md",
        "reports/report_labeling_protocol.md",
        "reports/kaggle_report_labeling_run.md",
        "reports/kaggle_notebook_instructions.md",
        "reports/weak_label_training_readiness.md",
    ):
        text = (REPO_ROOT / relative).read_text(encoding="utf-8")
        assert REPORT_LABEL_LEAK_TOKEN not in text
        assert _UID_LIKE.search(text) is None
        assert "kaggle.json" not in text.lower() or relative.endswith(
            "kaggle_notebook_instructions.md"
        )
        for uid in REPORT_LABEL_GOLD_UIDS + REPORT_LABEL_WEAK_UIDS:
            assert uid not in text, relative
        for patient in REPORT_LABEL_PATIENTS:
            assert patient not in text, relative


def test_git_does_not_track_private_row_level_label_files():
    import subprocess

    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
        text=True,
    )
    tracked = [path for path in result.stdout.split("\0") if path]
    offenders = [
        path
        for path in tracked
        if Path(path).name in _PRIVATE_ROW_FILES
    ]
    assert not offenders, f"private row-level files are tracked: {offenders}"
    for name in _PRIVATE_ROW_FILES:
        assert name not in tracked
