"""Kaggle report-labeling notebook: fixture-only, privacy, no scientific claims."""

from __future__ import annotations

import json
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
