"""Phase 4C.7 acceptance audit: fail-closed CIs, execution flags, no inference."""

from __future__ import annotations

import json
import math
from pathlib import Path

from knee.kaggle_bundle import would_include
from knee.second_reader_acceptance import (
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_RESTRICTED,
    decide_pilot_status,
    execution_flag_mismatch_reasons,
    execution_flags_for_counts,
)
from knee.second_reader_phase47 import FROZEN_PILOT_V14_SHA256
from knee.second_reader_phase47_acceptance_audit import (
    STATUS_COMPLETE,
    STATUS_INCOMPLETE,
    audit_metrics_payload,
    render_audit_markdown,
    run_phase47_acceptance_audit,
)
from tests.competition_fixtures import (
    REPORT_LABEL_GOLD_UIDS,
    REPORT_LABEL_LEAK_TOKEN,
    REPORT_LABEL_PATIENTS,
    REPORT_LABEL_WEAK_UIDS,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_METRICS = REPO_ROOT / "tests" / "fixtures" / "phase47_acceptance_audit_synthetic_metrics.json"
FIXTURE_ACCEPTANCE = (
    REPO_ROOT / "tests" / "fixtures" / "phase47_acceptance_audit_synthetic_acceptance.json"
)


def test_missing_ci_does_not_pass_when_point_estimates_are_high():
    missing = decide_pilot_status(
        official="ACL",
        precision=0.95,
        npv=0.92,
        coverage=0.10,
        n_gold_positive=20,
        n_gold_negative=20,
        n_abstentions=40,
        malformed_output_rate=0.0,
    )
    assert missing["status"] != STATUS_PASS
    assert "precision_ci_low_nonfinite" in missing["reasons"]
    assert "npv_ci_low_nonfinite" in missing["reasons"]


def test_nan_and_nonfinite_ci_fail_the_configured_gate():
    nan_ci = decide_pilot_status(
        official="ACL",
        precision=0.95,
        npv=0.92,
        coverage=0.10,
        n_gold_positive=20,
        n_gold_negative=20,
        n_abstentions=40,
        malformed_output_rate=0.0,
        precision_ci_low=float("nan"),
        npv_ci_low=float("nan"),
    )
    assert nan_ci["status"] != STATUS_PASS
    assert "precision_ci_low_nonfinite" in nan_ci["reasons"]
    inf_ci = decide_pilot_status(
        official="ACL",
        precision=0.95,
        npv=0.92,
        coverage=0.10,
        n_gold_positive=20,
        n_gold_negative=20,
        n_abstentions=40,
        malformed_output_rate=0.0,
        precision_ci_low=float("inf"),
        npv_ci_low=0.80,
    )
    assert inf_ci["status"] != STATUS_PASS
    assert "precision_ci_low_nonfinite" in inf_ci["reasons"]
    below = decide_pilot_status(
        official="ACL",
        precision=0.95,
        npv=0.92,
        coverage=0.10,
        n_gold_positive=20,
        n_gold_negative=20,
        n_abstentions=40,
        malformed_output_rate=0.0,
        precision_ci_low=0.60,
        npv_ci_low=0.80,
    )
    assert below["status"] == STATUS_RESTRICTED
    assert any(reason.startswith("precision_ci_low_below_min") for reason in below["reasons"])


def test_tiny_denominator_is_reported_and_is_not_a_yaml_pass_gate():
    passed = decide_pilot_status(
        official="Medial Meniscus",
        precision=1.0,
        npv=1.0,
        coverage=0.17,
        n_gold_positive=26,
        n_gold_negative=32,
        n_abstentions=48,
        malformed_output_rate=0.0,
        precision_ci_low=1.0,
        npv_ci_low=1.0,
        n_positive_decision_denominator=8,
        n_negative_decision_denominator=2,
    )
    assert passed["status"] == STATUS_PASS
    assert passed["tiny_negative_decision_denominator"] is True
    assert "tiny_negative_decision_denominator=2" in passed["reasons"]
    mcl = decide_pilot_status(
        official="MCL",
        precision=1.0,
        npv=1.0,
        coverage=0.02,
        n_gold_positive=9,
        n_gold_negative=49,
        n_abstentions=57,
        malformed_output_rate=0.0,
        precision_ci_low=1.0,
        npv_ci_low=1.0,
        n_positive_decision_denominator=1,
        n_negative_decision_denominator=0,
    )
    assert mcl["status"] == STATUS_RESTRICTED
    assert "tiny_positive_decision_denominator=1" in mcl["reasons"]
    assert "zero_negative_decision_denominator" in mcl["reasons"]


def test_execution_flags_follow_gold_counts_fail_closed():
    done = execution_flags_for_counts(n_gold_studies_processed=58)
    assert done["gold_eval_executed"] is True
    assert done["gold_rerun_executed"] is True
    assert done["gold_rerun_authorized"] is False
    skipped = execution_flags_for_counts(n_gold_studies_processed=0)
    assert skipped["gold_eval_executed"] is False
    assert skipped["gold_rerun_executed"] is False
    bad = {
        "n_gold_studies_processed": 58,
        "gold_eval_executed": True,
        "gold_rerun_executed": False,
        "gold_rerun_authorized": False,
    }
    reasons = execution_flag_mismatch_reasons(bad)
    assert "execution_flag_mismatch:gold_rerun_executed_false_with_n_gold" in reasons
    authorized = dict(done)
    authorized["gold_rerun_authorized"] = True
    assert "execution_flag_mismatch:gold_rerun_authorized_true" in execution_flag_mismatch_reasons(
        authorized
    )


def test_deterministic_recompute_from_synthetic_fixture():
    first = run_phase47_acceptance_audit(
        metrics_path=FIXTURE_METRICS,
        acceptance_path=FIXTURE_ACCEPTANCE,
        n_gold_studies_processed=58,
    )
    second = run_phase47_acceptance_audit(
        metrics_path=FIXTURE_METRICS,
        acceptance_path=FIXTURE_ACCEPTANCE,
        n_gold_studies_processed=58,
    )
    assert first["status"] == STATUS_COMPLETE
    assert first["model_generate_called"] is False
    assert first["inference_run"] is False
    primary = first["systems"]["strict_agreement"]["per_target"]
    assert primary["ACL"]["audited_status"] == STATUS_PASS
    assert primary["Medial Meniscus"]["audited_status"] == STATUS_PASS
    assert primary["Medial Meniscus"]["tiny_negative_decision_denominator"] is True
    assert primary["Lateral Meniscus"]["audited_status"] != STATUS_PASS
    assert "precision_ci_low_nonfinite" in primary["Lateral Meniscus"]["reasons"]
    assert primary["Lateral Meniscus"]["stored_status"] == STATUS_PASS
    assert primary["Lateral Meniscus"]["status_changed"] is True
    assert primary["MCL"]["tiny_positive_decision_denominator"] is True
    assert primary["MCL"]["audited_status"] == STATUS_RESTRICTED
    drop = {
        key: value
        for key, value in first.items()
        if key not in {"utc", "git_head_sha"}
    }
    drop2 = {
        key: value
        for key, value in second.items()
        if key not in {"utc", "git_head_sha"}
    }
    assert json.dumps(drop, sort_keys=True, default=str) == json.dumps(
        drop2, sort_keys=True, default=str
    )


def test_missing_metrics_is_incomplete_and_does_not_preserve_pass():
    result = run_phase47_acceptance_audit(
        metrics_path=REPO_ROOT / "tests" / "fixtures" / "does_not_exist.json"
    )
    assert result["status"] == STATUS_INCOMPLETE
    assert result["pass_claims_preserved_by_assumption"] is False
    assert result["gold_eval_executed"] is False
    assert result["n_gold_studies_processed"] == 0


def test_audit_module_never_calls_generate():
    src = (REPO_ROOT / "knee" / "second_reader_phase47_acceptance_audit.py").read_text(
        encoding="utf-8"
    )
    assert "run_second_reader_gold_pilot" not in src
    assert "run_phase47_gold_constrained" not in src
    assert "generate_fn" not in src


def test_real_phase47_metrics_recompute_when_present():
    result = run_phase47_acceptance_audit()
    if result["status"] == STATUS_INCOMPLETE:
        assert result["reason"] == "missing_phase47_gold_metrics"
        return
    assert result["status"] == STATUS_COMPLETE
    assert result["n_gold_studies_processed"] == 58
    assert result["gold_eval_executed"] is True
    assert result["gold_rerun_executed"] is True
    assert result["gold_rerun_authorized"] is False
    assert result["n_report_only_studies_processed"] == 0
    proof = result["medial_meniscus_ci_proof"]
    assert proof["precision_ci_finite_ge_0_70"] is True
    assert proof["npv_ci_finite_ge_0_70"] is True
    assert proof["still_pilot_pass_after_fail_closed_ci"] is True
    assert proof["tiny_negative_decision_denominator"] is True
    assert proof["n_negative_decision_denominator"] == 2
    mcl = result["mcl_tiny_denominator"]
    assert mcl["n_positive_decision_denominator"] == 1
    assert mcl["point_estimate_not_sufficient_evidence"] is True
    primary = result["systems"]["strict_agreement"]["per_target"]
    assert primary["Medial Meniscus"]["audited_status"] == STATUS_PASS
    assert primary["MCL"]["audited_status"] == STATUS_RESTRICTED
    for official, row in primary.items():
        if row["stored_status"] is not None:
            # Fail-closed CI must not invent a PASS that stored metrics cannot support.
            if row["audited_status"] == STATUS_PASS:
                assert math.isfinite(float(row["precision_ci_low"]))
                assert math.isfinite(float(row["npv_ci_low"]))
                assert float(row["precision_ci_low"]) >= 0.70
                assert float(row["npv_ci_low"]) >= 0.70
        assert official


def test_audit_report_and_bundle_allowlist():
    path = REPO_ROOT / "reports" / "second_reader_phase47_acceptance_audit.md"
    text = path.read_text(encoding="utf-8")
    assert "gold_rerun_authorized: `false`" in text or "gold_rerun_authorized: false" in text
    assert "ACCEPTANCE_AUDIT" in text
    assert REPORT_LABEL_LEAK_TOKEN not in text
    assert "FINDINGS:" not in text
    for uid in REPORT_LABEL_GOLD_UIDS + REPORT_LABEL_WEAK_UIDS:
        assert uid not in text
    for patient in REPORT_LABEL_PATIENTS:
        assert patient not in text
    md = render_audit_markdown(
        audit_metrics_payload(
            json.loads(FIXTURE_METRICS.read_text(encoding="utf-8")),
            old_acceptance=json.loads(FIXTURE_ACCEPTANCE.read_text(encoding="utf-8")),
            n_gold_studies_processed=58,
            metrics_path=str(FIXTURE_METRICS),
        )
    )
    assert "StudyInstanceUID" not in md
    assert would_include("knee/second_reader_phase47_acceptance_audit.py")
    assert would_include("scripts/run_second_reader_phase47_acceptance_audit.py")
    from knee.kaggle_bundle import sha256_file

    assert sha256_file(REPO_ROOT / "configs" / "second_reader_pilot_v1_4.yaml") == (
        FROZEN_PILOT_V14_SHA256
    )
