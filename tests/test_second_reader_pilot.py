"""Gold-only second-reader pilot: privacy, malformed output, no report-only scale."""

from __future__ import annotations

import json
import re
from pathlib import Path

from knee.config import REPORT_COLUMN, TARGET_KEYS
from knee.kaggle_audit import assert_no_report_text
from knee.report_labels import (
    SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED,
    assert_second_reader_report_only_scale_forbidden,
    collect_identifiers,
)
from knee.report_second_reader import (
    CANDIDATE_MODELS,
    assert_no_weight_download,
    run_second_reader,
)
from knee.second_reader_acceptance import STATUS_FAIL, STATUS_PASS, STATUS_RESTRICTED, decide_pilot_status
from knee.second_reader_parser import parse_second_reader_output
from knee.second_reader_pilot import (
    SANITIZED_FREEZE_FILENAME,
    SANITIZED_SUMMARY_FILENAME,
    run_second_reader_gold_pilot,
)
from knee.second_reader_schema import SCHEMA_VERSION
from tests.competition_fixtures import (
    REPORT_LABEL_GOLD_UIDS,
    REPORT_LABEL_LEAK_TOKEN,
    REPORT_LABEL_PATIENTS,
    REPORT_LABEL_WEAK_UIDS,
    make_report_labeling_fixture,
)
from tests.test_second_reader_parser import valid_payload

REPO_ROOT = Path(__file__).resolve().parents[1]
_UID_LIKE = re.compile(r"\b1\.2\.840\.\d+")


def _record(status, evidence, uncertainty="none", confidence="high"):
    return {
        "status": status,
        "confidence": confidence,
        "evidence_category": evidence,
        "uncertainty_or_negation_category": uncertainty,
    }


def fixture_generate_fn(report: str) -> str:
    """Local fixture double. Not a scientific model. Never used on Kaggle gold."""
    mentioned = {key: _record("not_mentioned", "not_mentioned", "not_mentioned", "low") for key in TARGET_KEYS}
    if "_ACL." in report or "_ES." in report:
        mentioned["acl_injury"] = _record("positive", "explicit_positive")
        mentioned["fracture"] = _record("negative", "fracture_excluded", "negated")
        if "_ACL." in report:
            mentioned["mcl_injury"] = _record("negative", "explicit_negative", "negated")
    elif "_NEG." in report:
        mentioned["acl_injury"] = _record("negative", "explicit_negative", "negated")
        mentioned["mcl_injury"] = _record("negative", "explicit_negative", "negated")
    elif "_UNC." in report:
        mentioned["acl_injury"] = _record("uncertain", "uncertain_finding", "uncertain_hedged", "low")
    elif "_HX." in report:
        mentioned["acl_injury"] = _record(
            "uncertain", "postoperative_or_reconstructed", "postoperative", "medium"
        )
    elif "_MM." in report:
        mentioned["medial_meniscus_injury"] = _record("positive", "explicit_positive")
        mentioned["lateral_meniscus_injury"] = _record("negative", "explicit_negative", "negated")
    elif "_OA." in report:
        for key in ("medial_compartment_oa", "lateral_compartment_oa", "patellofemoral_oa"):
            mentioned[key] = _record("positive", "compartment_specific_oa")
    elif "_EFF." in report:
        mentioned["joint_effusion"] = _record("positive", "effusion_not_synovitis")
        mentioned["synovitis"] = _record("negative", "explicit_negative", "negated")
        mentioned["baker_cyst"] = _record("negative", "explicit_negative", "negated")
    elif "_SYN." in report:
        mentioned["synovitis"] = _record("positive", "synovitis_not_effusion")
        mentioned["joint_effusion"] = _record("negative", "explicit_negative", "negated")
    elif "_BAKER." in report:
        mentioned["baker_cyst"] = _record("positive", "baker_cyst")
        mentioned["joint_effusion"] = _record("negative", "explicit_negative", "negated")
    elif "_BONE." in report:
        mentioned["bone_contusion"] = _record("positive", "bone_contusion")
        mentioned["fracture"] = _record("negative", "fracture_excluded", "negated")
    elif "_NORM." in report:
        for key in TARGET_KEYS:
            mentioned[key] = _record("negative", "explicit_negative", "negated")
    payload = {"schema_version": SCHEMA_VERSION, "targets": mentioned}
    return json.dumps(payload)


def test_candidates_are_instruction_models_with_licenses():
    assert len(CANDIDATE_MODELS) == 3
    ids = {row["id"] for row in CANDIDATE_MODELS}
    assert "microsoft/Phi-3.5-mini-instruct" in ids
    assert "google/gemma-2-2b-it" in ids
    assert "Qwen/Qwen2.5-7B-Instruct" in ids
    for row in CANDIDATE_MODELS:
        assert row["license"]
        assert row["public"] is True
        assert row["structured_json"] is True
        assert row["parameter_count"] > 0
        assert row["expected_gpu_memory_gb"] <= 16


def test_frozen_phi35_shard_checksums_are_full_sha256():
    import yaml
    from knee.report_second_reader import digest_map_from_config

    config = yaml.safe_load((REPO_ROOT / "configs" / "second_reader_pilot.yaml").read_text())
    digests = digest_map_from_config(config)
    shard1 = digests["model-00001-of-00002.safetensors"]
    shard2 = digests["model-00002-of-00002.safetensors"]
    assert len(shard1) == 64
    assert len(shard2) == 64
    assert shard1 == "c5214cdb995ed3dd716add8d9efbfe016b76bb2f1c4c1e6c1c6a95497d7a8837"
    assert shard2 == "41246eed2b75b66526339c5d32d6f7acdefe0bd24180f97c74303f4656877344"
    assert len(digests["config.json"]) == 64
    assert len(digests["tokenizer.model"]) == 64


def test_download_weights_is_refused():
    try:
        assert_no_weight_download({"download_weights": True})
        raised = False
    except RuntimeError:
        raised = True
    assert raised


def test_second_reader_has_no_gold_parameter():
    import inspect
    from knee.report_second_reader import SECOND_READER_FORBIDDEN, run_second_reader as fn

    assert not (set(inspect.signature(fn).parameters) & SECOND_READER_FORBIDDEN)


def test_scale_gate_remains_closed():
    assert SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED is False
    try:
        assert_second_reader_report_only_scale_forbidden()
        raised = False
    except Exception as exc:
        raised = True
        assert "NOT_READY_TO_SCALE_SECOND_READER" in str(exc)
    assert raised


def test_fixture_pilot_is_gold_only_and_private(tmp_path):
    root = make_report_labeling_fixture(tmp_path / "mini")
    out = tmp_path / "rsna_second_reader_gold_pilot"
    result = run_second_reader_gold_pilot(
        data_root=root,
        output_dir=out,
        data_source="software_fixture",
        generate_fn=fixture_generate_fn,
    )
    assert result["scientific_evaluation"] is False
    assert result["claim_scientific_performance"] is False
    assert result["n_gold_studies"] == 12
    assert result["n_report_only_studies_processed"] == 0
    assert result["n_report_only_studies_refused"] == 4
    assert result["internet_used"] is False
    assert result["download_weights"] is False
    assert result["training_run"] is False
    assert result["folds_created"] is False
    assert result["inference_backend"] == "fixture_generate_fn"
    freeze = json.loads((out / "sanitized" / SANITIZED_FREEZE_FILENAME).read_text())
    assert freeze["metrics_computed_after_freeze"] is True
    assert freeze["gold_only"] is True
    assert freeze["download_weights"] is False
    summary = (out / "sanitized" / SANITIZED_SUMMARY_FILENAME).read_text(encoding="utf-8")
    assert "SOFTWARE FIXTURE" in summary
    assert REPORT_LABEL_LEAK_TOKEN not in summary
    assert "positive_capture" in summary
    assert "PILOT_" in summary
    assert "NOT_READY_TO_SCALE_SECOND_READER_TO_REPORT_ONLY" in summary
    from knee.coverage_metrics import assert_not_full_set_sensitivity_claim

    assert_not_full_set_sensitivity_claim(summary)
    import pandas as pd

    train = pd.read_csv(root / "train.csv")
    snippets = [str(v) for v in train[REPORT_COLUMN].tolist()]
    sanitized = list((out / "sanitized").glob("*"))
    assert_no_report_text(sanitized, snippets)
    from knee.report_labels import assert_no_identifiers

    assert_no_identifiers(sanitized, collect_identifiers(train))
    private_reader = out / "private" / "gold_second_reader_predictions.csv"
    assert private_reader.is_file()
    blob = private_reader.read_text(encoding="utf-8")
    assert REPORT_COLUMN not in blob.splitlines()[0]
    assert REPORT_LABEL_LEAK_TOKEN not in blob
    for uid in REPORT_LABEL_WEAK_UIDS:
        assert uid not in blob
    reader = pd.read_csv(private_reader)
    assert len(reader) == 12
    comparisons = json.loads((out / "sanitized" / "second_reader_gold_metrics.json").read_text())
    assert set(comparisons) == {"deterministic_v2", "second_reader", "strict_agreement"}
    acl = comparisons["second_reader"]["per_target"]["ACL"]
    assert "coverage" in acl
    assert "npv" in acl
    assert "positive_capture" in acl
    assert "n_abstentions" in acl


def test_malformed_output_is_counted_and_abstains(tmp_path):
    root = make_report_labeling_fixture(tmp_path / "mini")

    def broken(report: str) -> str:
        if "_UNC." in report:
            return "not json at all"
        return fixture_generate_fn(report)

    result = run_second_reader_gold_pilot(
        data_root=root,
        output_dir=tmp_path / "out",
        data_source="software_fixture",
        generate_fn=broken,
    )
    assert result["malformed_output_rate"] > 0
    assert result["scientific_evaluation"] is False


def test_missing_data_is_not_scientific(tmp_path):
    result = run_second_reader_gold_pilot(
        data_root=tmp_path / "missing",
        output_dir=tmp_path / "out",
        data_source="software_fixture",
        generate_fn=fixture_generate_fn,
    )
    assert result["scientific_evaluation"] is False
    text = (tmp_path / "out" / "sanitized" / SANITIZED_SUMMARY_FILENAME).read_text()
    assert "SOFTWARE FIXTURE" in text or "PILOT STOPPED" in text


def test_clinical_distinctions_are_encoded_in_schema_payload():
    payload = valid_payload(
        acl_injury={"status": "uncertain", "evidence_category": "reconstruction_not_acute", "uncertainty_or_negation_category": "postoperative"},
        medial_meniscus_injury={"status": "uncertain", "evidence_category": "degeneration_not_meniscal_injury", "uncertainty_or_negation_category": "none"},
        medial_compartment_oa={"status": "positive", "evidence_category": "compartment_specific_oa", "uncertainty_or_negation_category": "none", "confidence": "high"},
        joint_effusion={"status": "positive", "evidence_category": "effusion_not_synovitis", "uncertainty_or_negation_category": "none", "confidence": "high"},
        baker_cyst={"status": "uncertain", "evidence_category": "other_popliteal_finding", "uncertainty_or_negation_category": "none"},
        bone_contusion={"status": "uncertain", "evidence_category": "marrow_signal_not_contusion", "uncertainty_or_negation_category": "uncertain_hedged"},
        fracture={"status": "uncertain", "evidence_category": "suspected_fracture", "uncertainty_or_negation_category": "suspected_not_confirmed"},
    )
    parsed = parse_second_reader_output(json.dumps(payload))
    assert parsed.ok
    assert parsed.targets["acl_injury"].evidence_category == "reconstruction_not_acute"
    assert parsed.targets["medial_meniscus_injury"].evidence_category == "degeneration_not_meniscal_injury"
    assert parsed.targets["joint_effusion"].evidence_category == "effusion_not_synovitis"
    assert parsed.targets["baker_cyst"].evidence_category == "other_popliteal_finding"
    assert parsed.targets["bone_contusion"].evidence_category == "marrow_signal_not_contusion"
    assert parsed.targets["fracture"].evidence_category == "suspected_fracture"


def test_acceptance_prioritizes_precision_and_npv():
    passed = decide_pilot_status(
        official="ACL",
        precision=0.95,
        npv=0.92,
        coverage=0.10,
        n_gold_positive=20,
        n_gold_negative=20,
        n_abstentions=40,
        malformed_output_rate=0.0,
        precision_ci_low=0.80,
        npv_ci_low=0.80,
    )
    assert passed["status"] == STATUS_PASS
    assert passed["coverage_used_as_pass_gate"] is False
    restricted = decide_pilot_status(
        official="ACL",
        precision=0.95,
        npv=0.40,
        coverage=0.50,
        n_gold_positive=20,
        n_gold_negative=20,
        n_abstentions=5,
        malformed_output_rate=0.0,
        precision_ci_low=0.80,
        npv_ci_low=0.20,
    )
    assert restricted["status"] == STATUS_RESTRICTED
    failed = decide_pilot_status(
        official="Baker's",
        precision=0.40,
        npv=0.40,
        coverage=0.90,
        n_gold_positive=12,
        n_gold_negative=20,
        n_abstentions=0,
        malformed_output_rate=0.0,
    )
    assert failed["status"] == STATUS_FAIL
    small = decide_pilot_status(
        official="MCL",
        precision=0.95,
        npv=0.95,
        coverage=0.50,
        n_gold_positive=9,
        n_gold_negative=20,
        n_abstentions=5,
        malformed_output_rate=0.0,
        precision_ci_low=0.80,
        npv_ci_low=0.80,
    )
    assert small["status"] == STATUS_RESTRICTED


def test_notebook_is_offline_gpu_and_gold_only():
    nb_path = REPO_ROOT / "notebooks" / "04_kaggle_second_reader_gold_pilot.ipynb"
    payload = json.loads(nb_path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source") or []) for cell in payload["cells"])
    lowered = source.lower()
    assert "kaggle.json" not in lowered
    assert "pip install" not in lowered
    assert "!pip" not in lowered
    assert "openai" not in lowered
    assert "anthropic" not in lowered
    assert "train_baseline" not in lowered
    assert "create_folds" not in lowered
    assert "internet_used=false" in lowered
    assert "process_report_only=false" in lowered
    assert "run_second_reader_gold_pilot" in source
    assert "_walk_input_for_source" in source
    assert "literature/ontology.yaml" in source
    assert "second_reader_pilot_v1_1.yaml" in source
    assert "HARD_WEAK_LABEL_TRAINING_ALLOWED" in source
    assert "SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED" in source
    assert "assert_hard_weak_label_training_forbidden" not in source
    assert "assert_second_reader_report_only_scale_forbidden" not in source
    meta = payload.get("metadata") or {}
    kaggle_meta = meta.get("kaggle") or {}
    assert kaggle_meta.get("isInternetEnabled") is False
    assert kaggle_meta.get("gpu") is True


def test_committed_pilot_reports_are_sanitized():
    for relative in (
        "reports/second_reader_model_compliance.md",
        "reports/second_reader_pilot_protocol.md",
        "reports/second_reader_gold_pilot_comparison.md",
        "reports/kaggle_second_reader_pilot_run.md",
        "reports/second_reader_v7_execution_fail.md",
        "reports/second_reader_synthetic_canary.md",
        "reports/second_reader_termination.md",
        "reports/second_reader_synthetic_canary_v12.md",
        "reports/second_reader_synthetic_canary_v13.md",
        "reports/second_reader_constrained_canary_v14.md",
        "reports/second_reader_phase45_intervention_audit.md",
        "reports/second_reader_phase46_intervention_replay.md",
        "reports/second_reader_phase47_gold_constrained_v14.md",
        "reports/second_reader_phase47_acceptance_audit.md",
        "reports/phase4c_second_reader_parked.md",
        "reports/phase4d_competition_readiness.md",
        "reports/gold58_image_oof_protocol.md",
        "reports/gold58_image_oof_run.md",
        "reports/gold58_image_oof_v2_protocol.md",
        "reports/gold58_image_oof_v2_run.md",
        "reports/phase4e_submission_rehearsal_protocol.md",
    ):
        text = (REPO_ROOT / relative).read_text(encoding="utf-8")
        assert REPORT_LABEL_LEAK_TOKEN not in text
        assert _UID_LIKE.search(text) is None
        for uid in REPORT_LABEL_GOLD_UIDS + REPORT_LABEL_WEAK_UIDS:
            assert uid not in text
        for patient in REPORT_LABEL_PATIENTS:
            assert patient not in text
        assert "kaggle.json" not in text.lower()
    comparison = (REPO_ROOT / "reports" / "second_reader_gold_pilot_comparison.md").read_text()
    assert "scientific_evaluation: `true`" in comparison.lower() or "real_data_execution: `true`" in comparison.lower()
    assert "PILOT_EXECUTION_FAIL" in comparison
    assert "scientific_performance_evaluable: `false`" in comparison.lower()
    assert "malformed_output_rate: 1.0" in comparison
    assert "n_gold_studies: 58" in comparison
    assert "n_report_only_studies_processed: 0" in comparison
    assert "4,349" in comparison or "4349" in comparison
    assert "Tesla T4" in comparison
    from knee.kaggle_bundle import sha256_file
    from knee.config import PROJECT_ROOT
    for rel, digest in (
        ("configs/second_reader_pilot.yaml", "82c257f2e8cf14cb5476e1a949ed9acbe0693308e9fdf2a8470a9ab58d5a42a5"),
        ("configs/second_reader_prompt.yaml", "e5bbfd9ce21bc8c5b6c25ca19177e3ef96b5cfcd9ccde5da61f5e2f1f27645d5"),
        ("configs/second_reader_schema.json", "639c41cc4f9ba1621cc1dd45404ca82ab0a6595c3e7494dca2ef2be1888f9b20"),
    ):
        assert sha256_file(PROJECT_ROOT / rel) == digest
        assert digest in comparison


def test_disabled_second_reader_still_skips_without_weights():
    result = run_second_reader("FINDINGS: ACL tear.", config={"second_reader": {"enabled": False, "download_weights": False}})
    assert result.enabled is False
    assert result.inference_backend == "skipped"


def test_freeze_manifest_hashes_ontology_when_cwd_is_elsewhere(tmp_path, monkeypatch):
    from knee.kaggle_bundle import sha256_file
    from knee.report_second_reader import load_prompt_bundle
    from knee.second_reader_pilot import freeze_manifest, load_pilot_config

    monkeypatch.chdir(tmp_path)
    config = load_pilot_config()
    prompt = load_prompt_bundle(config.get("prompt_path"))
    assert prompt.get("prompt_version")
    freeze = freeze_manifest(config, prompt=prompt)
    assert freeze["metrics_computed"] is False
    assert freeze["file_sha256"]["ontology"] == sha256_file(REPO_ROOT / "literature" / "ontology.yaml")
    assert freeze["file_sha256"]["prompt"] == sha256_file(REPO_ROOT / "configs" / "second_reader_prompt.yaml")


def test_cuda_usable_is_false_on_cpu_fixture_hosts():
    from knee.report_second_reader import _cuda_usable

    assert _cuda_usable() is False
