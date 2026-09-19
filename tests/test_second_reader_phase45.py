"""Phase 4C.5: paired-run identity, malformed replay, intervention accounting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from knee.kaggle_bundle import sha256_file, would_include
from knee.second_reader_canary import fixture_canary_generate_fn
from knee.second_reader_constrained import (
    ConstraintInterventionRecorder,
    mask_scores_to_allowed,
    run_constrained_canary,
)
from knee.second_reader_intervention_audit import (
    FOCUS_EXAMPLE_IDS,
    FROZEN_GPU_V3,
    OUTCOME_FORMAT_SEMANTIC_GATE_NOT_ESTABLISHED,
    OUTCOME_INTERVENTION_DIAGNOSTIC_FAIL,
    OUTCOME_SCHEMA_ENFORCEMENT_MISMATCH,
    PHASE45_OUTCOMES,
    assert_cloned_inputs_identical,
    canonical_shared_generation_config,
    clone_tensor_payload,
    decide_phase45_outcome,
    first_sequence_divergence,
    frozen_gpu_intervention_audit,
    malformed_json_variants,
    paired_run_identity,
    reconcile_intervention_with_sequences,
    replay_malformed_variants,
    run_phase45_audit,
    sanitized_focus_example_hashes,
    semantic_gate_audit,
    sha256_int_sequence,
)
from tests.competition_fixtures import REPORT_LABEL_LEAK_TOKEN

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_cloned_inputs_are_equal_and_not_aliased():
    original = {
        "input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 1]], dtype=torch.long),
    }
    cloned = clone_tensor_payload(original)
    assert_cloned_inputs_identical(original, cloned)
    cloned["input_ids"][0, 0] = 99
    assert int(original["input_ids"][0, 0]) == 1


def test_paired_identity_fails_if_non_schema_settings_differ():
    shared = canonical_shared_generation_config()
    left = dict(shared)
    left["schema_constrained"] = False
    right = dict(shared)
    right["schema_constrained"] = True
    right["max_new_tokens"] = 512
    payload = paired_run_identity(
        example_id="CANARY_SYNTHETIC_07",
        synthetic_input_sha256="abc",
        rendered_prompt_sha256="def",
        input_ids=[1, 2, 3],
        attention_mask=[1, 1, 1],
        unconstrained_config=left,
        constrained_config=right,
    )
    assert payload["ok"] is False
    assert "max_new_tokens" in payload["differ_except_schema_enforcement"]
    identical = paired_run_identity(
        example_id="CANARY_SYNTHETIC_07",
        synthetic_input_sha256="abc",
        rendered_prompt_sha256="def",
        input_ids=[1, 2, 3],
        attention_mask=[1, 1, 1],
        unconstrained_config=left,
        constrained_config={**shared, "schema_constrained": True},
    )
    assert identical["ok"] is True
    assert identical["input_ids_sha256"] == sha256_int_sequence([1, 2, 3])


def test_focus_example_hashes_cover_all_fourteen_without_report_text():
    rows = sanitized_focus_example_hashes()
    assert len(rows) == 14
    assert {row["example_id"] for row in rows if row["focus"]} == set(FOCUS_EXAMPLE_IDS)
    blob = json.dumps(rows)
    assert REPORT_LABEL_LEAK_TOKEN not in blob
    assert "FINDINGS:" not in blob
    assert "BEFUND:" not in blob


def test_zero_interventions_require_identical_tokens_and_termination():
    ok = reconcile_intervention_with_sequences(
        unconstrained_ids=[10, 11, 12],
        constrained_ids=[10, 11, 12],
        unconstrained_termination="json_close",
        constrained_termination="json_close",
        n_top1_allowed=3,
        n_top1_rejected=0,
        n_generated_steps=3,
        n_structural=0,
        n_clinical=0,
    )
    assert ok["ok"] is True
    fail = reconcile_intervention_with_sequences(
        unconstrained_ids=[10, 11, 12],
        constrained_ids=[10, 11, 99],
        unconstrained_termination="json_close",
        constrained_termination="json_close",
        n_top1_allowed=3,
        n_top1_rejected=0,
        n_generated_steps=3,
        n_structural=0,
        n_clinical=0,
    )
    assert fail["ok"] is False
    assert fail["status"] == OUTCOME_INTERVENTION_DIAGNOSTIC_FAIL
    assert first_sequence_divergence([10, 11, 12], [10, 11, 99])["position"] == 2


def test_conservation_assertions():
    bad = reconcile_intervention_with_sequences(
        unconstrained_ids=[1],
        constrained_ids=[1],
        unconstrained_termination="eos",
        constrained_termination="eos",
        n_top1_allowed=2,
        n_top1_rejected=0,
        n_generated_steps=3,
        n_structural=0,
        n_clinical=0,
    )
    assert bad["reason"] == "allowed_plus_blocked_ne_inspected_steps"
    subtypes = reconcile_intervention_with_sequences(
        unconstrained_ids=[1, 2],
        constrained_ids=[1, 9],
        unconstrained_termination="eos",
        constrained_termination="eos",
        n_top1_allowed=1,
        n_top1_rejected=1,
        n_generated_steps=2,
        n_structural=0,
        n_clinical=0,
    )
    assert subtypes["reason"] == "subtype_counts_ne_total_interventions"


def test_mask_changes_argmax_and_recorder_counts_block():
    scores = torch.tensor([[3.0, 2.0, 1.0, 0.0]])
    masked = mask_scores_to_allowed(scores, [1, 2])
    assert int(masked[0].argmax()) == 1
    assert int(scores[0].argmax()) == 0

    class _Tok:
        def decode(self, ids, skip_special_tokens=True):
            return ""

    recorder = ConstraintInterventionRecorder(lambda _bid, _ids: [1, 2], _Tok(), prompt_length=1)
    input_ids = torch.tensor([[7]])
    out = recorder._observe(input_ids, scores.clone())
    assert int(out[0].argmax()) == 1
    summary = recorder.summarize([1])
    assert summary["n_generated_tokens"] == 1
    assert summary["n_top1_allowed"] == 0
    assert summary["n_top1_rejected"] == 1
    assert summary["n_structural_interventions"] + summary["n_clinical_enum_interventions"] == 1


def test_malformed_variants_are_parser_invalid_without_report_text():
    from knee.second_reader_parser import parse_second_reader_output

    variants = malformed_json_variants()
    assert {row["example_id"] for row in variants} >= set(FOCUS_EXAMPLE_IDS)
    for row in variants:
        parsed = parse_second_reader_output(row["text"], protocol="v1.4")
        assert parsed.malformed
        assert "FINDINGS:" not in row["text"]
        assert REPORT_LABEL_LEAK_TOKEN not in row["text"]


def test_lmfe_replay_rejects_invalid_and_forbids_early_eos():
    result = replay_malformed_variants()
    if result.get("skipped"):
        pytest.skip(str(result.get("reason")))
    assert result["ok"] is True
    assert result["schema_enforcement_mismatch"] is False
    assert result["status"] != OUTCOME_SCHEMA_ENFORCEMENT_MISMATCH
    assert result["valid_skeleton_accepted"] is True
    assert result["eos_forbidden_until_complete"] is True
    forbidden_modes = {row["mode"] for row in result["variants"] if row["first_forbidden"]}
    assert "invalid_enum" in forbidden_modes or any(
        not row["can_end"] for row in result["variants"] if row["mode"] == "invalid_enum"
    )


def test_semantic_gate_is_not_predeclared():
    audit = semantic_gate_audit()
    assert audit["semantic_gate_status"] == "NOT_PREDECLARED"
    assert audit["predeclared_threshold"] is None
    assert audit["predeclared"] is False
    assert audit["internal_degrade_floor_commit"] == "ff4709e476268866213e10af2c762dd02b26e9e0"
    assert audit["do_not_create_threshold_after_0.815"] is True
    assert audit["reports_available"] is True


def test_semantic_gate_audit_missing_reports_stays_not_predeclared(monkeypatch):
    from pathlib import Path

    import knee.second_reader_intervention_audit as audit

    real_read = Path.read_text

    def fake_read(self, *args, **kwargs):
        if "reports" in self.parts:
            raise FileNotFoundError(self)
        return real_read(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fake_read)
    payload = semantic_gate_audit()
    assert payload["semantic_gate_status"] == "NOT_PREDECLARED"
    assert payload["predeclared_threshold"] is None
    assert payload["predeclared"] is False
    assert payload["reports_available"] is False
    assert payload["do_not_create_threshold_after_0.815"] is True


def test_frozen_gpu_v3_is_intervention_diagnostic_fail():
    payload = frozen_gpu_intervention_audit()
    assert payload["status"] == OUTCOME_INTERVENTION_DIAGNOSTIC_FAIL
    assert payload["ok"] is False
    assert payload["zero_interventions"] is True
    assert payload["format_valid_differed"] is True
    assert payload["token_identity_proven"] is False
    assert FROZEN_GPU_V3["synthetic_status_accuracy"] == 0.815
    assert payload["n_gold_studies_processed"] == 0


def test_phase45_audit_outcome_and_privacy(tmp_path):
    result = run_phase45_audit()
    assert result["status"] in PHASE45_OUTCOMES
    assert result["status"] == OUTCOME_INTERVENTION_DIAGNOSTIC_FAIL
    assert result["gold_rerun_authorized"] is False
    assert result["gpu_replay_executed"] is False
    assert result["identity"]["n_examples"] == 14
    assert result["identity"]["n_focus"] == 3
    assert result["semantic_gate"]["semantic_gate_status"] == "NOT_PREDECLARED"
    blob = json.dumps(result)
    assert REPORT_LABEL_LEAK_TOKEN not in blob
    assert "FINDINGS:" not in blob
    fixture = run_constrained_canary(
        output_dir=tmp_path / "fixture",
        generate_fn=fixture_canary_generate_fn,
        n_repeats=2,
    )
    assert fixture["status"] == OUTCOME_FORMAT_SEMANTIC_GATE_NOT_ESTABLISHED
    assert fixture["semantic_gate_status"] == "NOT_PREDECLARED"
    assert fixture["n_gold_studies_processed"] == 0


def test_decide_outcome_priority():
    assert (
        decide_phase45_outcome(
            schema_mismatch=True,
            intervention_ok=False,
            format_ok=False,
            semantic_predeclared=False,
            semantic_fail=True,
            gpu_replay_executed=False,
        )
        == OUTCOME_SCHEMA_ENFORCEMENT_MISMATCH
    )
    assert (
        decide_phase45_outcome(
            schema_mismatch=False,
            intervention_ok=True,
            format_ok=True,
            semantic_predeclared=False,
            semantic_fail=False,
            gpu_replay_executed=True,
        )
        == OUTCOME_FORMAT_SEMANTIC_GATE_NOT_ESTABLISHED
    )


def test_phase45_report_and_bundle_allowlist():
    path = REPO_ROOT / "reports" / "second_reader_phase45_intervention_audit.md"
    text = path.read_text(encoding="utf-8")
    assert "INTERVENTION_DIAGNOSTIC_FAIL" in text
    assert "NOT_PREDECLARED" in text
    assert "n_gold_studies_processed: 0" in text
    assert REPORT_LABEL_LEAK_TOKEN not in text
    assert would_include("knee/second_reader_intervention_audit.py")
    assert sha256_file(REPO_ROOT / "configs" / "second_reader_pilot_v1_4.yaml") == (
        "992af7a4b4483278445a6f3fb363c818611e79005618dff063460d0aed3bd7f7"
    )
    generate_src = (REPO_ROOT / "knee" / "report_second_reader.py").read_text(encoding="utf-8")
    assert "generate_kwargs[\"prefix_allowed_tokens_fn\"]" not in generate_src
    assert "schema_constraint_logits_processor" in generate_src
