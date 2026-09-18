"""Phase 4C.6: Gate 1 processor order, skip-without-GPU, intervention accounting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from knee.kaggle_bundle import sha256_file, would_include
from knee.report_second_reader import inspect_generate_kwargs
from knee.second_reader_constrained import ConstraintInterventionRecorder, mask_scores_to_allowed
from knee.second_reader_intervention_audit import (
    FOCUS_EXAMPLE_IDS,
    OUTCOME_FORMAT_SEMANTIC_GATE_NOT_ESTABLISHED,
    OUTCOME_FORMAT_SEMANTIC_PASS,
    OUTCOME_INTERVENTION_DIAGNOSTIC_FAIL,
)
from knee.second_reader_phase46 import (
    FORBIDDEN_SEMANTIC_PASS,
    PHASE46,
    PHASE46_KERNEL_SLUG,
    PHASE46_OUTCOMES,
    FakeSchemaMaskProcessor,
    decide_phase46_outcome,
    evaluate_paired_sequences,
    gate1_file_hashes,
    prove_processor_order,
    run_phase46_replay,
    sanitized_example_row,
)
from tests.competition_fixtures import REPORT_LABEL_LEAK_TOKEN

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_inspect_generate_kwargs_rejects_prefix_fn_and_requires_one_mask():
    unconstrained = inspect_generate_kwargs({"max_new_tokens": 2048, "do_sample": False})
    assert unconstrained["ok_unconstrained"] is True
    assert unconstrained["ok_constrained"] is False
    constrained = inspect_generate_kwargs(
        {"max_new_tokens": 2048, "logits_processor": [FakeSchemaMaskProcessor()]}
    )
    assert constrained["ok_constrained"] is True
    assert constrained["n_schema_mask_processors"] == 1
    bad = inspect_generate_kwargs(
        {
            "max_new_tokens": 2048,
            "logits_processor": [FakeSchemaMaskProcessor()],
            "prefix_allowed_tokens_fn": lambda *_a, **_k: [],
        }
    )
    assert bad["has_prefix_allowed_tokens_fn"] is True
    assert bad["ok_constrained"] is False


def test_recorder_processor_is_marked_as_schema_mask():
    class _Tok:
        def decode(self, ids, skip_special_tokens=True):
            return ""

    proc = ConstraintInterventionRecorder(lambda *_a: [1], _Tok(), prompt_length=1).as_logits_processor()
    assert getattr(proc, "schema_constraint_mask", False) is True


def test_prove_processor_order_and_file_hashes():
    proof = prove_processor_order()
    assert proof["ok"] is True
    assert proof["generate_receives_prefix_allowed_tokens_fn"] is False
    assert proof["exactly_one_schema_mask_processor_when_constrained"] is True
    hashes = gate1_file_hashes()
    assert hashes["ok"] is True
    assert hashes["mismatches"] == []
    assert hashes["observed"]["pilot_v14_sha256"] == (
        "992af7a4b4483278445a6f3fb363c818611e79005618dff063460d0aed3bd7f7"
    )


def test_evaluate_paired_sequences_zero_intervention_identity():
    intervention = {
        "n_generated_tokens": 3,
        "n_top1_allowed": 3,
        "n_top1_rejected": 0,
        "n_structural_interventions": 0,
        "n_clinical_enum_interventions": 0,
        "n_target_key_interventions": 0,
        "n_property_name_interventions": 0,
        "n_json_structure_interventions": 0,
        "n_other_interventions": 0,
    }
    ok = evaluate_paired_sequences(
        unconstrained_ids=[10, 11, 12],
        constrained_ids=[10, 11, 12],
        unconstrained_termination="json_close",
        constrained_termination="json_close",
        intervention=intervention,
        constrained_repeat_ids=[10, 11, 12],
        constrained_repeat_termination="json_close",
        constrained_json_sha256="abc",
        constrained_repeat_json_sha256="abc",
        recorder_n_calls=3,
    )
    assert ok["ok"] is True
    fail = evaluate_paired_sequences(
        unconstrained_ids=[10, 11, 12],
        constrained_ids=[10, 11, 99],
        unconstrained_termination="json_close",
        constrained_termination="json_close",
        intervention=intervention,
        constrained_repeat_ids=[10, 11, 99],
        constrained_repeat_termination="json_close",
        recorder_n_calls=3,
    )
    assert fail["ok"] is False
    assert fail["status"] == OUTCOME_INTERVENTION_DIAGNOSTIC_FAIL


def test_evaluate_paired_sequences_divergence_requires_blocked_top1():
    intervention = {
        "n_generated_tokens": 3,
        "n_top1_allowed": 2,
        "n_top1_rejected": 1,
        "n_structural_interventions": 1,
        "n_clinical_enum_interventions": 0,
        "n_target_key_interventions": 0,
        "n_property_name_interventions": 0,
        "n_json_structure_interventions": 1,
        "n_other_interventions": 0,
    }
    ok = evaluate_paired_sequences(
        unconstrained_ids=[10, 11, 12],
        constrained_ids=[10, 11, 99],
        unconstrained_termination="json_close",
        constrained_termination="json_close",
        intervention=intervention,
        constrained_repeat_ids=[10, 11, 99],
        constrained_repeat_termination="json_close",
        constrained_json_sha256="abc",
        constrained_repeat_json_sha256="abc",
        recorder_n_calls=3,
    )
    assert ok["ok"] is True


def test_decide_outcome_never_emits_semantic_pass():
    status = decide_phase46_outcome(
        gpu_replay_executed=True,
        skip_reason=None,
        gate1_ok=True,
        gate2_ok=True,
        gate3_format_ok=True,
        intervention_ok=True,
        semantic_ok=True,
    )
    assert status == OUTCOME_FORMAT_SEMANTIC_GATE_NOT_ESTABLISHED
    assert status != FORBIDDEN_SEMANTIC_PASS
    assert status != OUTCOME_FORMAT_SEMANTIC_PASS
    skipped = decide_phase46_outcome(
        gpu_replay_executed=False,
        skip_reason="cuda_required",
        gate1_ok=True,
        gate2_ok=None,
        gate3_format_ok=None,
        intervention_ok=None,
        semantic_ok=None,
    )
    assert skipped == OUTCOME_INTERVENTION_DIAGNOSTIC_FAIL


def test_phase46_replay_skips_without_gpu_and_keeps_gold_blocked(tmp_path):
    result = run_phase46_replay(output_dir=tmp_path)
    assert result["protocol"] == PHASE46
    assert result["status"] in PHASE46_OUTCOMES
    assert result["status"] != FORBIDDEN_SEMANTIC_PASS
    assert result["gpu_replay_executed"] is False
    assert result["skip_reason"] in {"transformers_unavailable", "cuda_required", "weights_not_attached"}
    assert result["gold_rerun_authorized"] is False
    assert result["gold_rerun_executed"] is False
    assert result["n_gold_studies_processed"] == 0
    assert result["n_report_only_studies_processed"] == 0
    assert result["gate1"]["processor_order"]["ok"] is True
    assert result["kernel_slug"] == PHASE46_KERNEL_SLUG
    blob = json.dumps(result)
    assert REPORT_LABEL_LEAK_TOKEN not in blob
    assert "FINDINGS:" not in blob
    written = (tmp_path / "sanitized" / "second_reader_phase46_replay.json").read_text(encoding="utf-8")
    assert "CONSTRAINED_FORMAT_PASS_SEMANTIC_PASS" not in written or result["forbidden_outcome_not_used"] == (
        "CONSTRAINED_FORMAT_PASS_SEMANTIC_PASS"
    )
    assert "n_gold_studies_processed" in written


def test_sanitized_example_row_has_no_report_text():
    class _Parsed:
        ok = True
        schema_version = "second-reader-schema-v1.0.0"
        malformed_stage = ""
        enum_violations = ()
        extra_properties = ()
        generation = type("G", (), {"prompt_included_in_decoded_output": False})()
        targets = {}

    row = sanitized_example_row(
        example_id="CANARY_SYNTHETIC_07",
        parsed=_Parsed(),
        token_ids=[1, 2, 3],
        termination="json_close",
    )
    blob = json.dumps(row)
    assert "FINDINGS:" not in blob
    assert REPORT_LABEL_LEAK_TOKEN not in blob
    assert row["raw_completions_stored"] is False
    assert row["example_id"] == "CANARY_SYNTHETIC_07"


def test_mask_still_observes_unconstrained_top1():
    scores = torch.tensor([[4.0, 1.0, 0.0]])
    masked = mask_scores_to_allowed(scores, [1])
    assert int(scores[0].argmax()) == 0
    assert int(masked[0].argmax()) == 1


def test_phase46_notebook_is_offline_synthetic_and_does_not_run_gold():
    nb_path = REPO_ROOT / "notebooks" / "08_kaggle_second_reader_phase46_replay.ipynb"
    payload = json.loads(nb_path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source") or []) for cell in payload["cells"])
    lowered = source.lower()
    assert "kaggle.json" not in lowered
    assert "pypi.org" not in lowered
    assert "huggingface.co" not in lowered
    assert "!pip" not in lowered
    assert "install_from_local_wheels" in source
    assert "run_phase46_replay" in source
    assert "gold_rerun_authorized=false" in lowered
    assert "process_report_only=false" in lowered
    assert "run_second_reader_gold_pilot" not in source
    assert "notebooks/07" not in source or "do not execute" in lowered
    assert PHASE46_KERNEL_SLUG in source
    assert "rsna-knee-second-reader-constrained-canary" in source
    assert "prefix_allowed_tokens_fn" in source
    assert "CONSTRAINED_FORMAT_PASS_SEMANTIC_PASS" in source
    assert "CONSTRAINED_FORMAT_PASS_SEMANTIC_GATE_NOT_ESTABLISHED" in source
    assert all(eid in source for eid in FOCUS_EXAMPLE_IDS)
    meta = payload.get("metadata") or {}
    assert (meta.get("kaggle") or {}).get("isInternetEnabled") is False
    assert (meta.get("kaggle") or {}).get("gpu") is True


def test_phase46_report_and_bundle_allowlist():
    path = REPO_ROOT / "reports" / "second_reader_phase46_intervention_replay.md"
    text = path.read_text(encoding="utf-8")
    assert "n_gold_studies_processed: 0" in text
    assert "gold_rerun_authorized: false" in text
    assert REPORT_LABEL_LEAK_TOKEN not in text
    assert "FINDINGS:" not in text
    assert would_include("knee/second_reader_phase46.py")
    assert would_include("notebooks/08_kaggle_second_reader_phase46_replay.ipynb")
    assert would_include("scripts/run_second_reader_phase46.py")
    generate_src = (REPO_ROOT / "knee" / "report_second_reader.py").read_text(encoding="utf-8")
    assert 'generate_kwargs["prefix_allowed_tokens_fn"]' not in generate_src
    assert sha256_file(REPO_ROOT / "configs" / "second_reader_pilot_v1_4.yaml") == (
        "992af7a4b4483278445a6f3fb363c818611e79005618dff063460d0aed3bd7f7"
    )


def test_phase46_does_not_overwrite_v14_config_hash():
    assert FORBIDDEN_SEMANTIC_PASS == "CONSTRAINED_FORMAT_PASS_SEMANTIC_PASS"
    text = (REPO_ROOT / "knee" / "second_reader_phase46.py").read_text(encoding="utf-8")
    assert "CONSTRAINED_FORMAT_PASS_SEMANTIC_PASS" in text
    assert "never" in text.lower() or "forbidden" in text.lower()
