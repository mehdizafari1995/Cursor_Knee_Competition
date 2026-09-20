"""Phase 4C.7 gold-only constrained evaluation: preflight, privacy, frozen hashes."""

from __future__ import annotations

import json
from pathlib import Path

from knee.kaggle_bundle import sha256_file, would_include
from knee.report_labels import (
    HARD_WEAK_LABEL_TRAINING_ALLOWED,
    SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED,
)
from knee.second_reader_constrained import never_coerces_unknown_to_negative
from knee.second_reader_phase46 import FORBIDDEN_SEMANTIC_PASS, prove_processor_order
from knee.second_reader_phase47 import (
    EFFECTIVE_RUNTIME_PREFIX_ALLOWED_TOKENS_FN,
    EXPECTED_GOLD_STUDIES,
    FROZEN_PILOT_V14_SHA256,
    PHASE47,
    PHASE47_KERNEL_SLUG,
    YAML_PREFIX_ALLOWED_TOKENS_FN,
    count_gold_and_report_only,
    effective_runtime_decoding,
    gold_selection_ok,
    run_phase47_gold_constrained,
)
from knee.second_reader_pilot import load_pilot_config
from tests.competition_fixtures import (
    REPORT_LABEL_GOLD_UIDS,
    REPORT_LABEL_LEAK_TOKEN,
    REPORT_LABEL_PATIENTS,
    REPORT_LABEL_WEAK_UIDS,
    make_report_labeling_fixture,
)
from tests.test_second_reader_pilot import fixture_generate_fn

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_gates_remain_closed_and_unknown_is_not_negative():
    assert HARD_WEAK_LABEL_TRAINING_ALLOWED is False
    assert SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED is False
    assert never_coerces_unknown_to_negative() is True
    assert YAML_PREFIX_ALLOWED_TOKENS_FN is True
    assert EFFECTIVE_RUNTIME_PREFIX_ALLOWED_TOKENS_FN is False
    assert FORBIDDEN_SEMANTIC_PASS == "CONSTRAINED_FORMAT_PASS_SEMANTIC_PASS"


def test_effective_runtime_does_not_pass_prefix_fn():
    config = load_pilot_config(REPO_ROOT / "configs" / "second_reader_pilot_v1_4.yaml")
    assert (config.get("decoding") or {}).get("prefix_allowed_tokens_fn") is True
    runtime = effective_runtime_decoding(config)
    assert runtime["yaml_prefix_allowed_tokens_fn"] is True
    assert runtime["effective_runtime_prefix_allowed_tokens_fn"] is False
    assert runtime["authority"] == "source_code"
    assert runtime["inspect_unconstrained"]["ok_unconstrained"] is True
    assert runtime["inspect_constrained"]["ok_constrained"] is True
    assert runtime["inspect_constrained"]["has_prefix_allowed_tokens_fn"] is False
    assert prove_processor_order()["ok"] is True
    assert sha256_file(REPO_ROOT / "configs" / "second_reader_pilot_v1_4.yaml") == (
        FROZEN_PILOT_V14_SHA256
    )


def test_real_kaggle_requires_exactly_58_gold():
    ok, reason = gold_selection_ok(n_gold=58, data_source="real_kaggle_mount")
    assert ok is True
    assert reason is None
    bad, bad_reason = gold_selection_ok(n_gold=12, data_source="real_kaggle_mount")
    assert bad is False
    assert bad_reason == "gold_count_not_58:12"
    fixture_ok, _ = gold_selection_ok(n_gold=12, data_source="software_fixture")
    assert fixture_ok is True


def test_phase47_skips_without_gpu_and_keeps_gold_blocked(tmp_path):
    result = run_phase47_gold_constrained(output_dir=tmp_path)
    assert result["protocol"] == PHASE47
    assert result["gold_rerun_authorized"] is False
    assert result["gold_rerun_executed"] is False
    assert result["notebook_07_executed"] is False
    assert result["n_gold_studies_processed"] == 0
    assert result["n_report_only_studies_processed"] == 0
    assert result["HARD_WEAK_LABEL_TRAINING_ALLOWED"] is False
    assert result["SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED"] is False
    assert result["forbidden_outcome_not_used"] == FORBIDDEN_SEMANTIC_PASS
    assert result["kernel_slug"] == PHASE47_KERNEL_SLUG
    assert result["status"] in {
        "GOLD_EVAL_NOT_EXECUTED",
        "DATA_NOT_AVAILABLE",
        "GATE1_FAIL",
    }
    written = (tmp_path / "sanitized" / "second_reader_phase47_gold.json").read_text(
        encoding="utf-8"
    )
    assert REPORT_LABEL_LEAK_TOKEN not in written
    assert "FINDINGS:" not in written


def test_fixture_gold_only_does_not_process_report_only(tmp_path):
    root = make_report_labeling_fixture(tmp_path / "mini")
    out = tmp_path / "phase47"
    result = run_phase47_gold_constrained(
        data_root=root,
        output_dir=out,
        data_source="software_fixture",
        generate_fn=fixture_generate_fn,
    )
    assert result["scientific_evaluation"] is False
    assert result["claim_scientific_performance"] is False
    assert result["n_gold_studies_processed"] == 12
    assert result["n_report_only_studies_processed"] == 0
    assert result["n_report_only_studies_selected"] == 0
    assert result["gold_rerun_authorized"] is False
    assert result["gold_eval_executed"] is True
    assert result["process_report_only_studies"] is False
    assert result["internet_used"] is False
    assert result["training_run"] is False
    assert result["folds_created"] is False
    assert result["download_weights"] is False
    assert result["code_never_coerces_unknown_to_negative"] is True
    recorder = result["recorder"]
    assert recorder["fixture_not_a_model_result"] is True
    summary = (out / "sanitized" / "second_reader_gold_pilot_summary.md").read_text(
        encoding="utf-8"
    )
    phase47 = (out / "sanitized" / "second_reader_phase47_gold.json").read_text(
        encoding="utf-8"
    )
    for blob in (summary, phase47, json.dumps(result, default=str)):
        assert REPORT_LABEL_LEAK_TOKEN not in blob
        assert "FINDINGS:" not in blob
        for uid in REPORT_LABEL_GOLD_UIDS + REPORT_LABEL_WEAK_UIDS:
            assert uid not in blob
        for patient in REPORT_LABEL_PATIENTS:
            assert patient not in blob


def test_real_kaggle_wrong_gold_count_does_not_call_generate(tmp_path):
    root = make_report_labeling_fixture(tmp_path / "mini")
    train = __import__("pandas").read_csv(root / "train.csv")
    counts = count_gold_and_report_only(train)
    assert counts["n_gold_studies_selected"] != EXPECTED_GOLD_STUDIES
    called = {"n": 0}

    def boom(_report: str) -> str:
        called["n"] += 1
        raise AssertionError("generate must not run before a 58-gold selection")

    out = tmp_path / "mismatch"
    result = run_phase47_gold_constrained(
        data_root=root,
        output_dir=out,
        data_source="real_kaggle_mount",
        generate_fn=boom,
    )
    assert called["n"] == 0
    assert result["status"] == "GOLD_COUNT_MISMATCH"
    assert result["n_gold_studies_processed"] == 0
    assert result["n_report_only_studies_processed"] == 0
    assert result["gold_eval_executed"] is False
    assert result["gold_rerun_authorized"] is False
    assert "gold_count_not_58" in str(result.get("skip_reason"))


def test_phase47_notebook_is_offline_gold_only_and_does_not_run_notebook_07():
    nb_path = REPO_ROOT / "notebooks" / "09_kaggle_second_reader_gold_constrained_v14.ipynb"
    payload = json.loads(nb_path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source") or []) for cell in payload["cells"])
    lowered = source.lower()
    assert "kaggle.json" not in lowered
    assert "pypi.org" not in lowered
    assert "huggingface.co" not in lowered
    assert "!pip" not in lowered
    assert "install_from_local_wheels" in source
    assert "run_phase47_gold_constrained" in source
    assert "second_reader_pilot_v1_4.yaml" in source
    assert "gold_rerun_authorized=false" in lowered
    assert "process_report_only=false" in lowered
    assert "notebook_07_executed=false" in lowered
    assert PHASE47_KERNEL_SLUG in source
    assert "rsna-knee-second-reader-gold-pilot" in source
    assert "rsna-knee-second-reader-constrained-canary" in source
    assert "prefix_allowed_tokens_fn" in source
    assert "effective_runtime_prefix_allowed_tokens_fn=false" in lowered
    assert "CONSTRAINED_FORMAT_PASS_SEMANTIC_PASS" in source
    assert "vendor" in source and "third_party" in source
    assert "n_gold_studies_processed" in source
    meta = payload.get("metadata") or {}
    assert (meta.get("kaggle") or {}).get("isInternetEnabled") is False
    assert (meta.get("kaggle") or {}).get("gpu") is True
    nb07 = json.loads(
        (REPO_ROOT / "notebooks" / "07_kaggle_second_reader_gold_constrained.ipynb").read_text(
            encoding="utf-8"
        )
    )
    src07 = "\n".join("".join(cell.get("source") or []) for cell in nb07["cells"])
    assert "PREPARED_GOLD_RERUN_BLOCKED" in src07


def test_phase47_report_and_bundle_allowlist():
    path = REPO_ROOT / "reports" / "second_reader_phase47_gold_constrained_v14.md"
    text = path.read_text(encoding="utf-8")
    assert "gold_rerun_authorized: false" in text
    assert REPORT_LABEL_LEAK_TOKEN not in text
    assert "FINDINGS:" not in text
    assert would_include("knee/second_reader_phase47.py")
    assert would_include("notebooks/09_kaggle_second_reader_gold_constrained_v14.ipynb")
    assert would_include("scripts/run_second_reader_phase47.py")
    generate_src = (REPO_ROOT / "knee" / "report_second_reader.py").read_text(encoding="utf-8")
    assert 'generate_kwargs["prefix_allowed_tokens_fn"]' not in generate_src
    assert sha256_file(REPO_ROOT / "configs" / "second_reader_pilot_v1_4.yaml") == (
        FROZEN_PILOT_V14_SHA256
    )
