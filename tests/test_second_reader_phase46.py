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
    capture_isolated_recorder_stats,
    decide_phase46_outcome,
    evaluate_paired_sequences,
    gate1_file_hashes,
    prove_processor_order,
    run_phase46_replay,
    sanitized_example_row,
)
from tests.competition_fixtures import REPORT_LABEL_LEAK_TOKEN

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _reset_constraint_module_globals():
    import knee.second_reader_constrained as constrained_mod

    constrained_mod.reset_constraint_stats()
    yield
    constrained_mod.reset_constraint_stats()


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
    assert "from .second_reader_constrained import CONSTRAINT_STAT_LOG, LAST_CONSTRAINT_STATS" not in text
    assert "from . import second_reader_constrained as constrained_mod" in text


def _gpu_recorder_stats(marker: int, *, n_tokens: int = 3) -> dict:
    return {
        "source": "transformers_logits_processor",
        "fixture_not_a_model_result": False,
        "n_calls": n_tokens,
        "n_generated_tokens": n_tokens,
        "n_top1_allowed": n_tokens,
        "n_top1_rejected": 0,
        "n_structural_interventions": 0,
        "n_clinical_enum_interventions": 0,
        "n_target_key_interventions": 0,
        "n_property_name_interventions": 0,
        "n_json_structure_interventions": 0,
        "n_other_interventions": 0,
        "intervention_rate": 0.0,
        "interventions_by_field": {},
        "capture_marker": marker,
        "raw_completions_stored": False,
        "logits_retained": False,
    }


def test_capture_reads_rebound_module_global_not_stale_import():
    import knee.second_reader_constrained as constrained_mod
    from knee.second_reader_phase46 import capture_isolated_recorder_stats

    constrained_mod.reset_constraint_stats()
    from knee.second_reader_constrained import LAST_CONSTRAINT_STATS as stale_name

    assert stale_name is None
    rebound = _gpu_recorder_stats(7)
    constrained_mod.LAST_CONSTRAINT_STATS = rebound
    constrained_mod.CONSTRAINT_STAT_LOG.append(rebound)
    payload, err = capture_isolated_recorder_stats(constrained_mod, role="constrained")
    assert err is None
    assert payload is not None
    assert payload["capture_marker"] == 7
    assert payload["n_calls"] == 3
    assert stale_name is None
    constrained_mod.reset_constraint_stats()


def test_capture_rejects_missing_stale_and_non_gpu_recorder():
    import knee.second_reader_constrained as constrained_mod
    from knee.second_reader_phase46 import capture_isolated_recorder_stats, reset_constraint_stats

    reset_constraint_stats()
    payload, err = capture_isolated_recorder_stats(constrained_mod, role="constrained")
    assert payload is None
    assert err == "missing_recorder:constrained:LAST_CONSTRAINT_STATS_is_none"

    constrained_mod.LAST_CONSTRAINT_STATS = _gpu_recorder_stats(1)
    payload, err = capture_isolated_recorder_stats(constrained_mod, role="constrained")
    assert payload is None
    assert err == "missing_recorder:constrained:expected_one_summary_got_0"

    constrained_mod.CONSTRAINT_STAT_LOG.append(_gpu_recorder_stats(1))
    constrained_mod.CONSTRAINT_STAT_LOG.append(_gpu_recorder_stats(2))
    constrained_mod.LAST_CONSTRAINT_STATS = _gpu_recorder_stats(2)
    payload, err = capture_isolated_recorder_stats(constrained_mod, role="constrained")
    assert payload is None
    assert "expected_one_summary_got_2" in str(err)

    reset_constraint_stats()
    bad = _gpu_recorder_stats(1)
    bad["source"] = "missing_recorder"
    constrained_mod.LAST_CONSTRAINT_STATS = bad
    constrained_mod.CONSTRAINT_STAT_LOG.append(bad)
    payload, err = capture_isolated_recorder_stats(constrained_mod, role="constrained_repeat")
    assert payload is None
    assert err == "missing_recorder:constrained_repeat:source_not_transformers_logits_processor"
    reset_constraint_stats()


def test_aggregate_n_calls_is_sum_of_per_run_calls_not_row_count():
    from knee.second_reader_constrained import aggregate_constraint_stats

    rows = [_gpu_recorder_stats(1, n_tokens=10), _gpu_recorder_stats(2, n_tokens=7)]
    aggregated = aggregate_constraint_stats(rows)
    assert aggregated["n_calls"] == 17
    assert aggregated["n_generated_tokens"] == 17
    assert aggregated["n_calls"] != len(rows)


def test_run_one_example_pair_captures_rebound_stats_for_both_constrained_calls(monkeypatch):
    import knee.second_reader_constrained as constrained_mod
    import knee.second_reader_phase46 as phase46
    from knee.report_second_reader import load_prompt_bundle
    from knee.second_reader_canary import SYNTHETIC_CANARY_EXAMPLES
    from knee.second_reader_parser import GenerationDiagnostics
    from knee.second_reader_schema import compact_output_skeleton
    from knee.second_reader_phase46 import run_one_example_pair

    example = next(item for item in SYNTHETIC_CANARY_EXAMPLES if item["example_id"] == "CANARY_SYNTHETIC_07")
    prompt = load_prompt_bundle()
    markers: list[int] = []

    def fake_encode(*_args, **_kwargs):
        return {
            "input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1, 1]], dtype=torch.long),
        }

    def fake_generate(
        report,
        *,
        schema_constrained_override=None,
        return_token_ids=True,
        **_kwargs,
    ):
        diag = GenerationDiagnostics(n_generated_tokens=3, stopped_on_json_close=True)
        ids = [10, 11, 12]
        text = compact_output_skeleton()
        if schema_constrained_override:
            marker = len(markers) + 1
            stats = _gpu_recorder_stats(marker)
            constrained_mod.LAST_CONSTRAINT_STATS = stats
            constrained_mod.CONSTRAINT_STAT_LOG.append(stats)
            markers.append(marker)
        return text, diag, ids

    monkeypatch.setattr(phase46, "_encode_second_reader_inputs", fake_encode)
    monkeypatch.setattr(phase46, "_generate_with_transformers", fake_generate)
    constrained_mod.reset_constraint_stats()
    result = run_one_example_pair(
        example,
        tokenizer=None,
        weights_dir=Path("/tmp/not-a-real-snapshot"),
        cfg={"decoding": {"schema_constrained": True, "max_new_tokens": 2048}},
        prompt=prompt,
    )
    assert result["ok"] is True
    assert markers == [1, 2]
    assert result["constrained_intervention"]["capture_marker"] == 1
    assert result["repeat_intervention"]["capture_marker"] == 2
    assert result["constrained_intervention"]["source"] == "transformers_logits_processor"
    assert result["repeat_intervention"]["fixture_not_a_model_result"] is False
    blob = json.dumps(result)
    assert REPORT_LABEL_LEAK_TOKEN not in blob
    assert "FINDINGS:" not in blob


def test_missing_or_stale_recorder_cannot_pass_gate2(monkeypatch):
    import knee.second_reader_constrained as constrained_mod
    import knee.second_reader_phase46 as phase46
    from knee.report_second_reader import load_prompt_bundle
    from knee.second_reader_canary import SYNTHETIC_CANARY_EXAMPLES
    from knee.second_reader_parser import GenerationDiagnostics
    from knee.second_reader_schema import compact_output_skeleton
    from knee.second_reader_phase46 import run_one_example_pair

    example = next(item for item in SYNTHETIC_CANARY_EXAMPLES if item["example_id"] == "CANARY_SYNTHETIC_12")
    prompt = load_prompt_bundle()

    def fake_encode(*_args, **_kwargs):
        return {
            "input_ids": torch.tensor([[9, 8, 7]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        }

    def fake_generate_missing(report, *, schema_constrained_override=None, **_kwargs):
        diag = GenerationDiagnostics(n_generated_tokens=3, stopped_on_json_close=True)
        return compact_output_skeleton(), diag, [1, 2, 3]

    monkeypatch.setattr(phase46, "_encode_second_reader_inputs", fake_encode)
    monkeypatch.setattr(phase46, "_generate_with_transformers", fake_generate_missing)
    constrained_mod.reset_constraint_stats()
    missing = run_one_example_pair(
        example,
        tokenizer=None,
        weights_dir=Path("/tmp/not-a-real-snapshot"),
        cfg={"decoding": {"schema_constrained": True}},
        prompt=prompt,
    )
    assert missing["ok"] is False
    assert missing["status"] == OUTCOME_INTERVENTION_DIAGNOSTIC_FAIL
    assert missing["reason"] == "missing_recorder:constrained:LAST_CONSTRAINT_STATS_is_none"
    assert "constrained_intervention" not in missing or missing.get("constrained_intervention") is None

    def fake_generate_empty_substitute(report, *, schema_constrained_override=None, **_kwargs):
        diag = GenerationDiagnostics(n_generated_tokens=3, stopped_on_json_close=True)
        if schema_constrained_override:
            from knee.second_reader_constrained import empty_intervention_payload

            empty = empty_intervention_payload(source="missing_recorder", note="stale")
            constrained_mod.LAST_CONSTRAINT_STATS = empty
            constrained_mod.CONSTRAINT_STAT_LOG.append(empty)
        return compact_output_skeleton(), diag, [1, 2, 3]

    monkeypatch.setattr(phase46, "_generate_with_transformers", fake_generate_empty_substitute)
    constrained_mod.reset_constraint_stats()
    stale = run_one_example_pair(
        example,
        tokenizer=None,
        weights_dir=Path("/tmp/not-a-real-snapshot"),
        cfg={"decoding": {"schema_constrained": True}},
        prompt=prompt,
    )
    assert stale["ok"] is False
    assert stale["status"] == OUTCOME_INTERVENTION_DIAGNOSTIC_FAIL
    assert "missing_recorder" in str(stale["reason"])
    assert stale.get("constrained_intervention") is None

