"""Phase 4B labeler: language ensemble, privacy, pathology morphology, v2 reports."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from knee.config import OFFICIAL_TARGET_COLUMNS, REPORT_COLUMN, TARGET_KEYS
from knee.report_labels import (
    LABEL_NEGATIVE,
    LABEL_POSITIVE,
    LABEL_UNKNOWN,
    extract_labels_from_report,
    load_extractor_config,
    run_notebook_report_labeling,
)
from knee.report_language import detect_language
from knee.report_ngram_language import detect_language_ngram
from tests.competition_fixtures import (
    REPORT_LABEL_GOLD_UIDS,
    REPORT_LABEL_LEAK_TOKEN,
    REPORT_LABEL_PATIENTS,
    REPORT_LABEL_WEAK_UIDS,
    make_report_labeling_fixture,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
_UID_LIKE = re.compile(r"\b1\.2\.840\.\d+")


def test_anatomy_without_pathology_is_unknown_for_acl():
    result = extract_labels_from_report("FINDINGS: The ACL is visualized in the notch.")
    assert result.predictions["acl_injury"].label == LABEL_UNKNOWN


def test_clause_level_negation_does_not_cross_findings():
    result = extract_labels_from_report("FINDINGS: The ACL is intact but the MCL is torn.")
    assert result.predictions["acl_injury"].label == LABEL_NEGATIVE
    assert result.predictions["mcl_injury"].label == LABEL_POSITIVE


def test_generic_cyst_is_not_baker():
    result = extract_labels_from_report("FINDINGS: Parameniscal cyst at the medial joint line.")
    assert result.predictions["baker_cyst"].label != LABEL_POSITIVE


def test_greek_with_latin_mri_tokens_is_greek_not_forced_uncertain():
    text = "Ευρήματα: ρήξη του προσθίου χιαστού συνδέσμου του γόνατος. MRI PD FS. ACL."
    lang = detect_language(text)
    ngram = detect_language_ngram(text)
    assert "greek" in lang.scripts
    assert lang.language == "el"
    assert ngram.language == "el"
    assert lang.reason != "mixed_scripts"


def test_language_disagreement_stays_uncertain(monkeypatch):
    # Independent detector on a short mixed token salad should not promote certainty
    # when the heuristic is undetermined.
    emptyish = "12345"
    lang = detect_language(emptyish)
    assert lang.uncertain is True


def test_fixture_v2_run_writes_masks_without_leaking_text(tmp_path):
    root = make_report_labeling_fixture(tmp_path / "mini")
    out = tmp_path / "rsna_report_label_outputs_v2"
    result = run_notebook_report_labeling(
        data_root=root,
        output_dir=out,
        data_source="software_fixture",
    )
    assert result["scientific_evaluation"] is False
    assert result["extractor_version"] == "report-labeler-v2.0.0"
    assert result["second_reader_enabled"] is False
    assert result["second_reader_weights_downloaded"] is False
    summary = (out / "sanitized" / "report_labeling_summary.md").read_text(encoding="utf-8")
    assert REPORT_LABEL_LEAK_TOKEN not in summary
    assert "positive_capture" in summary
    assert "conditional_sensitivity" in summary
    from knee.coverage_metrics import assert_not_full_set_sensitivity_claim
    assert_not_full_set_sensitivity_claim(summary)
    assert "TRAINABLE_" in summary or "trainable" in summary.lower()
    private = out / "private"
    for name in (
        "mask_high_conf_positive.csv",
        "mask_high_conf_negative.csv",
        "mask_unknown.csv",
        "mask_disagreement.csv",
        "mask_unsupported_language.csv",
        "language_assignments.csv",
    ):
        assert (private / name).is_file(), name
        blob = (private / name).read_text(encoding="utf-8")
        assert REPORT_COLUMN not in blob.split(",")[0]
        assert REPORT_LABEL_LEAK_TOKEN not in blob
    lang = pd.read_csv(private / "language_assignments.csv")
    assert "heuristic_language" in lang.columns
    assert "ngram_language" in lang.columns
    metrics = json.loads((out / "sanitized" / "gold_metrics.json").read_text())
    acl = (metrics.get("per_target") or {}).get("ACL") or {}
    assert "positive_capture" in acl
    assert "npv" in acl
    assert "oof_confidence_policies" in metrics
    for key, payload in (metrics.get("oof_confidence_policies") or {}).items():
        for fold in payload.get("folds") or []:
            assert "train_ids" not in fold
            assert "test_ids" not in fold


def test_committed_v2_reports_are_sanitized():
    for relative in (
        "reports/report_labeler_v2_summary.md",
        "reports/report_labeler_v2_protocol.md",
        "reports/competition_pretrained_model_rules.md",
    ):
        text = (REPO_ROOT / relative).read_text(encoding="utf-8")
        assert REPORT_LABEL_LEAK_TOKEN not in text
        assert _UID_LIKE.search(text) is None
        for uid in REPORT_LABEL_GOLD_UIDS + REPORT_LABEL_WEAK_UIDS:
            assert uid not in text
        for patient in REPORT_LABEL_PATIENTS:
            assert patient not in text
        assert "kaggle.json" not in text.lower()


def test_v2_notebook_is_offline_and_does_not_train():
    nb_path = REPO_ROOT / "notebooks" / "03_kaggle_report_labeler_v2.ipynb"
    payload = json.loads(nb_path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source") or []) for cell in payload["cells"])
    lowered = source.lower()
    assert "kaggle.json" not in lowered
    assert "pip install" not in lowered
    assert "openai" not in lowered
    assert "train_baseline" not in lowered
    assert "internet_used=false" in lowered
    assert "second_reader_enabled" in lowered
    assert "deterministic extractor" in lowered
    meta = payload.get("metadata") or {}
    kaggle_meta = meta.get("kaggle") or {}
    assert kaggle_meta.get("isInternetEnabled") is False
    assert kaggle_meta.get("gpu") is False


def test_v1_config_remains_frozen():
    v1 = load_extractor_config(REPO_ROOT / "configs" / "report_labeler_v1.yaml")
    v2 = load_extractor_config()
    assert v1["extractor_version"] == "report-labeler-v1.0.0"
    assert v2["extractor_version"] == "report-labeler-v2.0.0"
    assert tuple(v1["targets"]) == TARGET_KEYS
    assert tuple(v2["targets"]) == TARGET_KEYS
