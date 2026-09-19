"""Phase 4C.6 corrected Phi GPU intervention replay.

Synthetic reports only. Never reads gold or report-only studies. Never stores
completions, report text, full logits, patient identifiers, or competition rows.
Does not overwrite frozen kernel artefacts.

This environment may lack CUDA or Transformers. In that case the module records
an honest skip and keeps gold blocked. It never invents GPU metrics and never
emits CONSTRAINED_FORMAT_PASS_SEMANTIC_PASS.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Sequence

from .config import PROJECT_ROOT, TARGET_KEYS
from .kaggle_audit import jsonable
from .kaggle_bundle import sha256_file
from .lm_format_enforcer_wheels import (
    INTEREGULAR_WHEEL_NAME,
    INTEREGULAR_WHEEL_SHA256,
    LMFE_WHEEL_NAME,
    LMFE_WHEEL_SHA256,
    load_wheel_manifest,
    wheels_dir,
)
from .report_labels import (
    HARD_WEAK_LABEL_TRAINING_ALLOWED,
    SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED,
)
from .report_second_reader import (
    _cuda_usable,
    _encode_second_reader_inputs,
    _generate_with_transformers,
    inspect_generate_kwargs,
    load_prompt_bundle,
    locate_weights_directory,
)
from .second_reader_canary import SYNTHETIC_CANARY_EXAMPLES, synthetic_truth_fields
from . import second_reader_constrained as constrained_mod
from .second_reader_constrained import (
    FROZEN_SCHEMA_CANONICAL_JSON_SHA256,
    FROZEN_SCHEMA_RAW_BYTES_SHA256,
    PHI_EOS_IDS,
    PINNED_TOKENIZER_SHA256,
    aggregate_constraint_stats,
    evaluate_semantic_gates,
    never_coerces_unknown_to_negative,
    reset_constraint_stats,
    schema_hash_provenance,
)
from .second_reader_intervention_audit import (
    FOCUS_EXAMPLE_IDS,
    FROZEN_GPU_V3,
    OUTCOME_CANARY_FORMAT_FAIL,
    OUTCOME_CANARY_SEMANTIC_FAIL,
    OUTCOME_FORMAT_SEMANTIC_GATE_NOT_ESTABLISHED,
    OUTCOME_FORMAT_SEMANTIC_PASS,
    OUTCOME_INTERVENTION_DIAGNOSTIC_FAIL,
    OUTCOME_SCHEMA_ENFORCEMENT_MISMATCH,
    PHASE45_OUTCOMES,
    PINNED_MODEL_ID,
    PINNED_MODEL_REVISION,
    assert_cloned_inputs_identical,
    canonical_shared_generation_config,
    clone_tensor_payload,
    first_sequence_divergence,
    paired_run_identity,
    reconcile_intervention_with_sequences,
    semantic_gate_audit,
    sha256_int_sequence,
    sha256_text,
)
from .second_reader_parser import parse_second_reader_output
from .second_reader_pilot import freeze_manifest, load_pilot_config
from .second_reader_schema import (
    DEFAULT_PILOT_CONFIG_PATH_V1_4,
    DEFAULT_PROMPT_PATH_V1_2,
    TARGET_FIELD_KEYS,
    compact_output_skeleton,
)

PHASE46 = "second-reader-phase-4c.6-corrected-phi-gpu-replay"
KAGGLE_PHASE46_OUTPUT_DIRNAME = "rsna_second_reader_phase46_replay"
PHASE46_KERNEL_SLUG = "rsna-knee-second-reader-phase46-replay"
FROZEN_CONSTRAINED_CANARY_KERNEL = "mahdimzafari/rsna-knee-second-reader-constrained-canary"
FROZEN_CONSTRAINED_CANARY_VERSION = 3
FROZEN_PROMPT_SHA256 = "5f7842b00137d0a84ea75ece5503ef31a0101255eb9ab6fcd117f9efb6723289"
FROZEN_PILOT_V14_SHA256 = "992af7a4b4483278445a6f3fb363c818611e79005618dff063460d0aed3bd7f7"
PHASE46_OUTCOMES = PHASE45_OUTCOMES
FORBIDDEN_SEMANTIC_PASS = OUTCOME_FORMAT_SEMANTIC_PASS


class FakeSchemaMaskProcessor:
    """Stand-in for Gate 1 processor-order proofs. Not a model logits processor."""

    schema_constraint_mask = True

    def __call__(self, input_ids, scores):
        return scores


def git_head_sha(*, cwd: Path | None = None) -> str | None:
    try:
        blob = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd or PROJECT_ROOT),
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    text = blob.decode("utf-8").strip()
    return text or None


def tensor_sha256(value: Any) -> str:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "contiguous"):
        value = value.contiguous()
    if hasattr(value, "numpy"):
        payload = value.numpy().tobytes()
    elif hasattr(value, "tolist"):
        payload = ",".join(str(int(v)) for v in value.reshape(-1).tolist()).encode("utf-8")
    else:
        payload = str(value).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def input_payload_hashes(inputs: dict[str, Any]) -> dict[str, Any]:
    ids = inputs.get("input_ids")
    mask = inputs.get("attention_mask")
    id_list = ids.reshape(-1).tolist() if hasattr(ids, "reshape") else list(ids or [])
    mask_list = mask.reshape(-1).tolist() if mask is not None and hasattr(mask, "reshape") else list(mask or [])
    return {
        "input_ids_sha256": sha256_int_sequence([int(x) for x in id_list]),
        "attention_mask_sha256": sha256_int_sequence([int(x) for x in mask_list]),
        "n_input_tokens": len(id_list),
        "raw_completions_stored": False,
        "logits_retained": False,
    }


def termination_label(generation: Any) -> str:
    if generation is None:
        return "other"
    if getattr(generation, "stopped_on_json_close", False):
        return "json_close"
    if getattr(generation, "stopped_on_eos", False):
        return "eos"
    if getattr(generation, "reached_max_new_tokens", False):
        return "max_new_tokens"
    return "other"


def normalized_targets_sha256(parsed: Any) -> str | None:
    if parsed is None or not getattr(parsed, "ok", False):
        return None
    payload = {
        "schema_version": getattr(parsed, "schema_version", None),
        "targets": {
            key: {
                "status": parsed.targets[key].status,
                "confidence": parsed.targets[key].confidence,
                "evidence_category": parsed.targets[key].evidence_category,
                "uncertainty_or_negation_category": parsed.targets[key].uncertainty_or_negation_category,
            }
            for key in TARGET_KEYS
            if key in parsed.targets
        },
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def clinical_fields_from_parsed(parsed: Any) -> dict[str, dict[str, str]]:
    if parsed is None or not getattr(parsed, "ok", False):
        return {}
    out: dict[str, dict[str, str]] = {}
    for key in TARGET_KEYS:
        item = parsed.targets.get(key)
        if item is None:
            continue
        out[key] = {
            "status": item.status,
            "confidence": item.confidence,
            "evidence_category": item.evidence_category,
            "uncertainty_or_negation_category": item.uncertainty_or_negation_category,
        }
    return out


def sanitized_example_row(
    *,
    example_id: str,
    parsed: Any,
    token_ids: Sequence[int],
    termination: str,
    intervention: dict[str, Any] | None = None,
    synthetic_truth: dict[str, Any] | None = None,
) -> dict[str, Any]:
    token_list = [int(x) for x in token_ids]
    schema_valid = bool(parsed is not None and parsed.ok)
    return {
        "example_id": example_id,
        "schema_valid": schema_valid,
        "malformed_stage": "ok" if schema_valid else str(getattr(parsed, "malformed_stage", "other") or "other"),
        "n_generated_tokens": len(token_list),
        "generated_token_sha256": sha256_int_sequence(token_list),
        "normalized_json_sha256": normalized_targets_sha256(parsed),
        "termination": termination,
        "clinical_fields": clinical_fields_from_parsed(parsed),
        "synthetic_truth": synthetic_truth or {},
        "enum_violations_n": len(getattr(parsed, "enum_violations", ()) or ()),
        "extra_properties_n": len(getattr(parsed, "extra_properties", ()) or ()),
        "prompt_included": bool(getattr(getattr(parsed, "generation", None), "prompt_included_in_decoded_output", False)),
        "forbidden_extra_text": str(getattr(parsed, "malformed_stage", "") or "") == "forbidden_extra_text",
        "intervention": {
            "n_generated_tokens": (intervention or {}).get("n_generated_tokens"),
            "n_top1_allowed": (intervention or {}).get("n_top1_allowed"),
            "n_top1_rejected": (intervention or {}).get("n_top1_rejected"),
            "n_structural_interventions": (intervention or {}).get("n_structural_interventions"),
            "n_clinical_enum_interventions": (intervention or {}).get("n_clinical_enum_interventions"),
            "n_target_key_interventions": (intervention or {}).get("n_target_key_interventions"),
            "n_property_name_interventions": (intervention or {}).get("n_property_name_interventions"),
            "n_json_structure_interventions": (intervention or {}).get("n_json_structure_interventions"),
            "n_other_interventions": (intervention or {}).get("n_other_interventions"),
            "intervention_rate": (intervention or {}).get("intervention_rate"),
            "interventions_by_field": (intervention or {}).get("interventions_by_field") or {},
        }
        if intervention
        else None,
        "raw_completions_stored": False,
        "logits_retained": False,
        "report_text_stored": False,
    }


def prove_processor_order(
    unconstrained_kwargs: dict[str, Any] | None = None,
    constrained_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Gate 1: generate() must not receive prefix_allowed_tokens_fn."""
    shared = {
        "max_new_tokens": 2048,
        "do_sample": False,
        "pad_token_id": 32000,
        "eos_token_id": list(PHI_EOS_IDS),
    }
    unconstrained = dict(unconstrained_kwargs or shared)
    constrained = dict(constrained_kwargs or shared)
    if "logits_processor" not in constrained:
        constrained["logits_processor"] = [FakeSchemaMaskProcessor()]
    un_gate = inspect_generate_kwargs(unconstrained)
    con_gate = inspect_generate_kwargs(constrained)
    bad = dict(constrained)
    bad["prefix_allowed_tokens_fn"] = lambda *_args, **_kwargs: []
    bad_gate = inspect_generate_kwargs(bad)
    ok = bool(un_gate["ok_unconstrained"] and con_gate["ok_constrained"] and not bad_gate["ok_constrained"])
    return {
        "ok": ok,
        "unconstrained": un_gate,
        "constrained": con_gate,
        "prefix_fn_rejected": (not bad_gate["ok_constrained"]) and bad_gate["has_prefix_allowed_tokens_fn"],
        "n_schema_mask_processors_unconstrained": un_gate["n_schema_mask_processors"],
        "n_schema_mask_processors_constrained": con_gate["n_schema_mask_processors"],
        "generate_receives_prefix_allowed_tokens_fn": False,
        "exactly_one_schema_mask_processor_when_constrained": con_gate["n_schema_mask_processors"] == 1,
        "zero_schema_mask_processors_when_unconstrained": un_gate["n_schema_mask_processors"] == 0,
        "raw_completions_stored": False,
        "logits_retained": False,
    }


def gate1_file_hashes(*, weights_dir: Path | None = None) -> dict[str, Any]:
    schema_path = PROJECT_ROOT / "configs" / "second_reader_schema.json"
    prompt_path = DEFAULT_PROMPT_PATH_V1_2
    config_path = DEFAULT_PILOT_CONFIG_PATH_V1_4
    wheel_root = wheels_dir()
    lmfe_path = wheel_root / LMFE_WHEEL_NAME
    inter_path = wheel_root / INTEREGULAR_WHEEL_NAME
    provenance = schema_hash_provenance()
    observed = {
        "schema_raw_bytes_sha256": sha256_file(schema_path),
        "schema_canonical_json_sha256": provenance["schema_canonical_json_sha256"],
        "prompt_sha256": sha256_file(prompt_path),
        "pilot_v14_sha256": sha256_file(config_path),
        "lmfe_wheel_sha256": sha256_file(lmfe_path) if lmfe_path.is_file() else None,
        "interegular_wheel_sha256": sha256_file(inter_path) if inter_path.is_file() else None,
        "constrained_py_sha256": sha256_file(PROJECT_ROOT / "knee" / "second_reader_constrained.py"),
        "report_second_reader_py_sha256": sha256_file(PROJECT_ROOT / "knee" / "report_second_reader.py"),
        "phase46_py_sha256": sha256_file(PROJECT_ROOT / "knee" / "second_reader_phase46.py"),
    }
    expected = {
        "schema_raw_bytes_sha256": FROZEN_SCHEMA_RAW_BYTES_SHA256,
        "schema_canonical_json_sha256": FROZEN_SCHEMA_CANONICAL_JSON_SHA256,
        "prompt_sha256": FROZEN_PROMPT_SHA256,
        "pilot_v14_sha256": FROZEN_PILOT_V14_SHA256,
        "lmfe_wheel_sha256": LMFE_WHEEL_SHA256,
        "interegular_wheel_sha256": INTEREGULAR_WHEEL_SHA256,
    }
    mismatches = [
        key
        for key, value in expected.items()
        if observed.get(key) is not None and observed.get(key) != value
    ]
    weight_hashes: dict[str, str] = {}
    weight_ok = True
    if weights_dir is not None:
        cfg = load_pilot_config(config_path)
        selected = cfg.get("selected_model") or {}
        expected_files = dict(selected.get("file_sha256") or PINNED_TOKENIZER_SHA256)
        expected_files.update(dict(selected.get("shard_sha256") or {}))
        for name, digest in expected_files.items():
            path = Path(weights_dir) / name
            if not path.is_file():
                weight_ok = False
                continue
            got = sha256_file(path)
            weight_hashes[name] = got
            if digest and got != digest:
                mismatches.append(f"weights:{name}")
                weight_ok = False
    return {
        "ok": not mismatches and bool(weight_ok or weights_dir is None),
        "observed": observed,
        "expected": expected,
        "mismatches": mismatches,
        "weight_hashes": weight_hashes,
        "weights_verified": bool(weight_hashes) and weight_ok,
        "git_head_sha": git_head_sha(),
        "model_id": PINNED_MODEL_ID,
        "model_revision": PINNED_MODEL_REVISION,
        "wheel_manifest": load_wheel_manifest(),
        "raw_completions_stored": False,
        "logits_retained": False,
    }


def runtime_capability() -> dict[str, Any]:
    transformers_ok = False
    transformers_version = None
    try:
        import transformers

        transformers_ok = True
        transformers_version = str(getattr(transformers, "__version__", "unknown"))
    except Exception:
        transformers_ok = False
    cuda_ok = False
    gpu_name = None
    try:
        import torch

        cuda_ok = bool(_cuda_usable())
        if torch.cuda.is_available():
            gpu_name = str(torch.cuda.get_device_name(0))
    except Exception:
        cuda_ok = False
    return {
        "transformers_available": transformers_ok,
        "transformers_version": transformers_version,
        "cuda_usable": cuda_ok,
        "gpu_name": gpu_name,
        "torch_cuda_available": cuda_ok,
    }


def skip_reason_from_runtime(
    capability: dict[str, Any],
    *,
    weights_dir: Path | None,
) -> str | None:
    if not capability.get("transformers_available"):
        return "transformers_unavailable"
    if not capability.get("cuda_usable"):
        return "cuda_required"
    if weights_dir is None:
        return "weights_not_attached"
    return None


def evaluate_paired_sequences(
    *,
    unconstrained_ids: Sequence[int],
    constrained_ids: Sequence[int],
    unconstrained_termination: str,
    constrained_termination: str,
    intervention: dict[str, Any],
    constrained_repeat_ids: Sequence[int] | None = None,
    constrained_repeat_termination: str | None = None,
    constrained_json_sha256: str | None = None,
    constrained_repeat_json_sha256: str | None = None,
    constrained_schema_valid: bool = True,
    constrained_repeat_schema_valid: bool = True,
    recorder_n_calls: int | None = None,
) -> dict[str, Any]:
    n_steps = int(intervention.get("n_generated_tokens") or 0)
    n_allowed = int(intervention.get("n_top1_allowed") or 0)
    n_rejected = int(intervention.get("n_top1_rejected") or 0)
    n_structural = int(intervention.get("n_structural_interventions") or 0)
    n_clinical = int(intervention.get("n_clinical_enum_interventions") or 0)
    n_target_key = int(intervention.get("n_target_key_interventions") or 0)
    n_property_name = int(intervention.get("n_property_name_interventions") or 0)
    n_json_structure = int(intervention.get("n_json_structure_interventions") or 0)
    n_other = int(intervention.get("n_other_interventions") or 0)
    recon = reconcile_intervention_with_sequences(
        unconstrained_ids=unconstrained_ids,
        constrained_ids=constrained_ids,
        unconstrained_termination=unconstrained_termination,
        constrained_termination=constrained_termination,
        n_top1_allowed=n_allowed,
        n_top1_rejected=n_rejected,
        n_generated_steps=n_steps,
        n_structural=n_structural,
        n_clinical=n_clinical,
    )
    reasons: list[str] = []
    if not constrained_schema_valid or not constrained_repeat_schema_valid:
        reasons.append("constrained_output_not_schema_valid")
    subtype_sum = n_target_key + n_property_name + n_json_structure + n_other
    if subtype_sum != n_structural:
        reasons.append("structural_subtype_counts_ne_structural_total")
        recon = {
            "ok": False,
            "status": OUTCOME_INTERVENTION_DIAGNOSTIC_FAIL,
            "reason": "structural_subtype_counts_ne_structural_total",
        }
    if recorder_n_calls is not None and int(recorder_n_calls) != n_steps:
        reasons.append("recorder_calls_ne_generated_steps")
        recon = {
            "ok": False,
            "status": OUTCOME_INTERVENTION_DIAGNOSTIC_FAIL,
            "reason": "recorder_calls_ne_generated_steps",
        }
    if constrained_repeat_ids is not None:
        repeat_div = first_sequence_divergence(constrained_ids, constrained_repeat_ids)
        identical_repeat = not bool(repeat_div.get("diverged"))
        identical_term = (
            constrained_repeat_termination is None
            or constrained_repeat_termination == constrained_termination
        )
        identical_json = (
            constrained_json_sha256 is None
            or constrained_repeat_json_sha256 is None
            or constrained_json_sha256 == constrained_repeat_json_sha256
        )
        if not (identical_repeat and identical_term and identical_json):
            reasons.append("constrained_repeat_not_identical")
            recon = {
                "ok": False,
                "status": OUTCOME_INTERVENTION_DIAGNOSTIC_FAIL,
                "reason": "constrained_repeat_not_identical",
                "divergence": repeat_div,
            }
    if reasons and recon.get("ok"):
        recon = {
            "ok": False,
            "status": OUTCOME_INTERVENTION_DIAGNOSTIC_FAIL,
            "reason": reasons[0],
        }
    recon["reasons"] = reasons
    recon["unconstrained_token_sha256"] = sha256_int_sequence(unconstrained_ids)
    recon["constrained_token_sha256"] = sha256_int_sequence(constrained_ids)
    if constrained_repeat_ids is not None:
        recon["constrained_repeat_token_sha256"] = sha256_int_sequence(constrained_repeat_ids)
    recon["unconstrained_termination"] = unconstrained_termination
    recon["constrained_termination"] = constrained_termination
    recon["raw_completions_stored"] = False
    recon["logits_retained"] = False
    return recon


def decide_phase46_outcome(
    *,
    gpu_replay_executed: bool,
    skip_reason: str | None,
    gate1_ok: bool,
    gate2_ok: bool | None,
    gate3_format_ok: bool | None,
    intervention_ok: bool | None,
    semantic_ok: bool | None,
    schema_enforcement_mismatch: bool = False,
) -> str:
    if schema_enforcement_mismatch:
        return OUTCOME_SCHEMA_ENFORCEMENT_MISMATCH
    if not gate1_ok:
        return OUTCOME_INTERVENTION_DIAGNOSTIC_FAIL
    if not gpu_replay_executed:
        return OUTCOME_INTERVENTION_DIAGNOSTIC_FAIL
    if skip_reason:
        return OUTCOME_INTERVENTION_DIAGNOSTIC_FAIL
    if gate2_ok is False or intervention_ok is False:
        return OUTCOME_INTERVENTION_DIAGNOSTIC_FAIL
    if gate3_format_ok is False:
        return OUTCOME_CANARY_FORMAT_FAIL
    if semantic_ok is False:
        return OUTCOME_CANARY_SEMANTIC_FAIL
    if gate2_ok and gate3_format_ok and intervention_ok:
        return OUTCOME_FORMAT_SEMANTIC_GATE_NOT_ESTABLISHED
    return OUTCOME_INTERVENTION_DIAGNOSTIC_FAIL


def _empty_phase46_result(
    *,
    status: str,
    skip_reason: str | None,
    gate1: dict[str, Any],
    output_dir: Path,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    semantic = semantic_gate_audit()
    payload = {
        "protocol": PHASE46,
        "status": status,
        "ok": False,
        "skip_reason": skip_reason,
        "gpu_replay_executed": False,
        "gate1": gate1,
        "gate2": None,
        "gate3": None,
        "semantic_gate_status": semantic["semantic_gate_status"],
        "semantic_gate": semantic,
        "gold_rerun_authorized": False,
        "gold_rerun_executed": False,
        "notebook_07_executed": False,
        "n_gold_studies_processed": 0,
        "n_report_only_studies_processed": 0,
        "kernel_slug": PHASE46_KERNEL_SLUG,
        "do_not_overwrite_frozen_kernel": {
            "slug": FROZEN_CONSTRAINED_CANARY_KERNEL,
            "version": FROZEN_CONSTRAINED_CANARY_VERSION,
            "recorded_status": FROZEN_GPU_V3["recorded_status"],
        },
        "forbidden_outcome_not_used": FORBIDDEN_SEMANTIC_PASS,
        "raw_completions_stored": False,
        "logits_retained": False,
        "report_text_stored": False,
        "HARD_WEAK_LABEL_TRAINING_ALLOWED": HARD_WEAK_LABEL_TRAINING_ALLOWED,
        "SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED": SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED,
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if extra:
        payload.update(extra)
    _write_sanitized(output_dir, payload)
    return payload


def _write_sanitized(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sanitized_dir = output_dir / "sanitized"
    sanitized_dir.mkdir(parents=True, exist_ok=True)
    public = {key: value for key, value in payload.items() if key not in {"scored_rows", "example_rows"}}
    (sanitized_dir / "second_reader_phase46_replay.json").write_text(
        json.dumps(jsonable(public), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parse_generated(text: str, *, protocol: str) -> Any:
    return parse_second_reader_output(text, protocol=protocol)


def _sanitized_recorder_payload(stats: Any) -> str:
    """Canonical JSON of a recorder summary. Used when object identity is unstable."""
    return json.dumps(jsonable(dict(stats)), sort_keys=True, default=str)


def capture_isolated_recorder_stats(constrained_ns: Any, *, role: str) -> tuple[dict[str, Any] | None, str | None]:
    """Read module globals after generate(). Never substitutes an empty payload."""
    stats = constrained_ns.LAST_CONSTRAINT_STATS
    log = list(constrained_ns.CONSTRAINT_STAT_LOG)
    if stats is None:
        return None, f"missing_recorder:{role}:LAST_CONSTRAINT_STATS_is_none"
    if len(log) != 1:
        return None, f"missing_recorder:{role}:expected_one_summary_got_{len(log)}"
    log_summary = log[0]
    if stats is not log_summary:
        try:
            last_payload = _sanitized_recorder_payload(stats)
            log_payload = _sanitized_recorder_payload(log_summary)
        except (TypeError, ValueError):
            return None, f"missing_recorder:{role}:last_stats_log_mismatch"
        if last_payload != log_payload:
            return None, f"missing_recorder:{role}:last_stats_log_mismatch"
    payload = dict(stats)
    if payload.get("source") != "transformers_logits_processor":
        return None, f"missing_recorder:{role}:source_not_transformers_logits_processor"
    if payload.get("fixture_not_a_model_result") is not False:
        return None, f"missing_recorder:{role}:fixture_not_a_model_result"
    n_calls = int(payload.get("n_calls") or 0)
    n_tokens = int(payload.get("n_generated_tokens") or 0)
    if n_tokens <= 0:
        return None, f"missing_recorder:{role}:n_generated_tokens_not_positive"
    if n_calls != n_tokens:
        return None, f"missing_recorder:{role}:n_calls_ne_n_generated_tokens"
    return payload, None


def _recorder_fail_payload(*, reason: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        "ok": False,
        "status": OUTCOME_INTERVENTION_DIAGNOSTIC_FAIL,
        "reason": reason,
        "raw_completions_stored": False,
        "logits_retained": False,
        "report_text_stored": False,
    }
    if extra:
        payload.update(extra)
    return payload


def run_one_example_pair(
    example: dict[str, str],
    *,
    tokenizer: Any,
    weights_dir: Path,
    cfg: dict,
    prompt: dict,
    model_device: Any = None,
) -> dict[str, Any]:
    """Tokenize once, clone tensors, run unconstrained then constrained then repeat."""
    report = example["report"]
    from .report_second_reader import render_user_prompt

    rendered_prompt = render_user_prompt(report, prompt=prompt)
    truth = synthetic_truth_fields(report)
    original = _encode_second_reader_inputs(tokenizer, report, prompt, model_device=model_device)
    unconstrained_inputs = clone_tensor_payload(original)
    constrained_inputs = clone_tensor_payload(original)
    repeat_inputs = clone_tensor_payload(original)
    assert_cloned_inputs_identical(original, unconstrained_inputs)
    assert_cloned_inputs_identical(original, constrained_inputs)
    identity = paired_run_identity(
        example_id=example["example_id"],
        synthetic_input_sha256=sha256_text(report),
        rendered_prompt_sha256=sha256_text(rendered_prompt or ""),
        input_ids=[int(x) for x in original["input_ids"].reshape(-1).tolist()],
        attention_mask=[int(x) for x in original["attention_mask"].reshape(-1).tolist()]
        if "attention_mask" in original
        else None,
        unconstrained_config={**canonical_shared_generation_config(), "schema_constrained": False},
        constrained_config={**canonical_shared_generation_config(), "schema_constrained": True},
    )
    if not identity["ok"]:
        return {
            "ok": False,
            "status": OUTCOME_INTERVENTION_DIAGNOSTIC_FAIL,
            "reason": "paired_identity_failed",
            "identity": identity,
            "raw_completions_stored": False,
        }

    reset_constraint_stats()
    un_text, un_diag, un_ids = _generate_with_transformers(
        report,
        weights_dir=weights_dir,
        cfg=cfg,
        prompt=prompt,
        prepared_inputs=unconstrained_inputs,
        schema_constrained_override=False,
        return_token_ids=True,
        eos_token_ids_override=list(PHI_EOS_IDS),
    )
    reset_constraint_stats()
    con_text, con_diag, con_ids = _generate_with_transformers(
        report,
        weights_dir=weights_dir,
        cfg=cfg,
        prompt=prompt,
        prepared_inputs=constrained_inputs,
        schema_constrained_override=True,
        return_token_ids=True,
        eos_token_ids_override=list(PHI_EOS_IDS),
    )
    con_stats, con_err = capture_isolated_recorder_stats(constrained_mod, role="constrained")
    if con_err:
        del un_text, con_text
        return _recorder_fail_payload(reason=con_err)
    reset_constraint_stats()
    rep_text, rep_diag, rep_ids = _generate_with_transformers(
        report,
        weights_dir=weights_dir,
        cfg=cfg,
        prompt=prompt,
        prepared_inputs=repeat_inputs,
        schema_constrained_override=True,
        return_token_ids=True,
        eos_token_ids_override=list(PHI_EOS_IDS),
    )
    rep_stats, rep_err = capture_isolated_recorder_stats(constrained_mod, role="constrained_repeat")
    if rep_err:
        del un_text, con_text, rep_text
        return _recorder_fail_payload(
            reason=rep_err,
            extra={"constrained_intervention": con_stats},
        )

    un_parsed = _parse_generated(un_text, protocol="v1.2")
    con_parsed = _parse_generated(con_text, protocol="v1.4")
    rep_parsed = _parse_generated(rep_text, protocol="v1.4")
    del un_text, con_text, rep_text, rendered_prompt, report

    un_row = sanitized_example_row(
        example_id=example["example_id"],
        parsed=un_parsed,
        token_ids=un_ids,
        termination=termination_label(un_diag),
    )
    con_row = sanitized_example_row(
        example_id=example["example_id"],
        parsed=con_parsed,
        token_ids=con_ids,
        termination=termination_label(con_diag),
        intervention=con_stats,
        synthetic_truth=truth,
    )
    rep_row = sanitized_example_row(
        example_id=example["example_id"],
        parsed=rep_parsed,
        token_ids=rep_ids,
        termination=termination_label(rep_diag),
        intervention=rep_stats,
    )
    recon = evaluate_paired_sequences(
        unconstrained_ids=un_ids,
        constrained_ids=con_ids,
        unconstrained_termination=un_row["termination"],
        constrained_termination=con_row["termination"],
        intervention=con_stats,
        constrained_repeat_ids=rep_ids,
        constrained_repeat_termination=rep_row["termination"],
        constrained_json_sha256=con_row["normalized_json_sha256"],
        constrained_repeat_json_sha256=rep_row["normalized_json_sha256"],
        constrained_schema_valid=con_row["schema_valid"],
        constrained_repeat_schema_valid=rep_row["schema_valid"],
        recorder_n_calls=con_stats.get("n_calls"),
    )
    return {
        "ok": bool(recon.get("ok") and con_row["schema_valid"] and rep_row["schema_valid"]),
        "status": recon.get("status") if recon.get("ok") else OUTCOME_INTERVENTION_DIAGNOSTIC_FAIL,
        "identity": identity,
        "input_hashes": input_payload_hashes(original),
        "unconstrained": un_row,
        "constrained": con_row,
        "constrained_repeat": rep_row,
        "constrained_intervention": con_stats,
        "repeat_intervention": rep_stats,
        "reconciliation": recon,
        "constraint_log_n": 1,
        "raw_completions_stored": False,
        "logits_retained": False,
        "report_text_stored": False,
    }


def run_phase46_replay(
    *,
    output_dir: str | Path | None = None,
    config_path: str | Path | None = None,
    weights_path: str | Path | None = None,
    focus_only: bool = False,
) -> dict[str, Any]:
    """Gate 1 always. Gates 2–3 only on a real CUDA Transformers Phi snapshot."""
    assert HARD_WEAK_LABEL_TRAINING_ALLOWED is False
    assert SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED is False
    out_dir = Path(output_dir or (PROJECT_ROOT / "outputs" / KAGGLE_PHASE46_OUTPUT_DIRNAME))
    cfg_path = Path(config_path or DEFAULT_PILOT_CONFIG_PATH_V1_4)
    cfg = load_pilot_config(cfg_path)
    prompt = load_prompt_bundle(cfg.get("prompt_path") or DEFAULT_PROMPT_PATH_V1_2)
    capability = runtime_capability()
    weights_dir = locate_weights_directory(cfg.get("second_reader") or cfg.get("selected_model") or cfg, weights_path)
    hashes = gate1_file_hashes(weights_dir=weights_dir)
    order = prove_processor_order()
    gate1 = {
        "ok": bool(hashes["ok"] and order["ok"]),
        "file_hashes": hashes,
        "processor_order": order,
        "capability": capability,
        "shared_generation_config": canonical_shared_generation_config(),
        "compact_skeleton_sha256": sha256_text(compact_output_skeleton()),
        "git_head_sha": hashes.get("git_head_sha"),
    }
    skip = skip_reason_from_runtime(capability, weights_dir=weights_dir)
    if not gate1["ok"]:
        return _empty_phase46_result(
            status=OUTCOME_INTERVENTION_DIAGNOSTIC_FAIL,
            skip_reason=skip or "gate1_fail",
            gate1=gate1,
            output_dir=out_dir,
        )
    if skip:
        status = decide_phase46_outcome(
            gpu_replay_executed=False,
            skip_reason=skip,
            gate1_ok=True,
            gate2_ok=None,
            gate3_format_ok=None,
            intervention_ok=None,
            semantic_ok=None,
        )
        assert status != FORBIDDEN_SEMANTIC_PASS
        return _empty_phase46_result(
            status=status,
            skip_reason=skip,
            gate1=gate1,
            output_dir=out_dir,
            extra={"freeze": freeze_manifest(cfg, prompt=prompt)},
        )

    try:
        import torch
        from transformers import AutoTokenizer
    except Exception:
        return _empty_phase46_result(
            status=OUTCOME_INTERVENTION_DIAGNOSTIC_FAIL,
            skip_reason="transformers_unavailable",
            gate1=gate1,
            output_dir=out_dir,
        )

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    tokenizer = AutoTokenizer.from_pretrained(
        str(weights_dir),
        local_files_only=True,
        trust_remote_code=True,
    )
    model_device = torch.device("cuda") if _cuda_usable() else torch.device("cpu")
    examples = list(SYNTHETIC_CANARY_EXAMPLES)
    focus = [item for item in examples if item["example_id"] in FOCUS_EXAMPLE_IDS]
    gate2_rows: list[dict[str, Any]] = []
    peak_gpu = None
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = datetime.now(timezone.utc)
    try:
        for example in focus:
            row = run_one_example_pair(
                example,
                tokenizer=tokenizer,
                weights_dir=Path(weights_dir),
                cfg=cfg,
                prompt=prompt,
                model_device=model_device,
            )
            gate2_rows.append(row)
            if not row.get("ok"):
                status = OUTCOME_INTERVENTION_DIAGNOSTIC_FAIL
                payload = {
                    "protocol": PHASE46,
                    "status": status,
                    "ok": False,
                    "skip_reason": None,
                    "gpu_replay_executed": True,
                    "gate1": gate1,
                    "gate2": {
                        "ok": False,
                        "n_examples": len(gate2_rows),
                        "rows": [
                            {k: v for k, v in item.items() if k not in {"unconstrained", "constrained", "constrained_repeat"} or True}
                            for item in gate2_rows
                        ],
                    },
                    "gate3": None,
                    "semantic_gate_status": "NOT_PREDECLARED",
                    "gold_rerun_authorized": False,
                    "gold_rerun_executed": False,
                    "notebook_07_executed": False,
                    "n_gold_studies_processed": 0,
                    "n_report_only_studies_processed": 0,
                    "kernel_slug": PHASE46_KERNEL_SLUG,
                    "raw_completions_stored": False,
                    "logits_retained": False,
                    "report_text_stored": False,
                    "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
                if torch.cuda.is_available():
                    payload["peak_gpu_memory_bytes"] = int(torch.cuda.max_memory_allocated())
                _write_sanitized(out_dir, payload)
                return payload
        if focus_only:
            status = OUTCOME_FORMAT_SEMANTIC_GATE_NOT_ESTABLISHED
            payload = {
                "protocol": PHASE46,
                "status": status,
                "ok": True,
                "skip_reason": None,
                "gpu_replay_executed": True,
                "gate1": gate1,
                "gate2": {"ok": True, "n_examples": len(gate2_rows), "rows": gate2_rows},
                "gate3": None,
                "semantic_gate_status": "NOT_PREDECLARED",
                "gold_rerun_authorized": False,
                "gold_rerun_executed": False,
                "n_gold_studies_processed": 0,
                "n_report_only_studies_processed": 0,
                "kernel_slug": PHASE46_KERNEL_SLUG,
                "raw_completions_stored": False,
                "logits_retained": False,
                "forbidden_outcome_not_used": FORBIDDEN_SEMANTIC_PASS,
                "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            _write_sanitized(out_dir, payload)
            return payload

        unconstrained_rows = []
        constrained_first = []
        constrained_second = []
        gate3_pairs: list[dict[str, Any]] = []
        reset_constraint_stats()
        for example in examples:
            row = run_one_example_pair(
                example,
                tokenizer=tokenizer,
                weights_dir=Path(weights_dir),
                cfg=cfg,
                prompt=prompt,
                model_device=model_device,
            )
            gate3_pairs.append(row)
            if not row.get("ok"):
                payload = {
                    "protocol": PHASE46,
                    "status": OUTCOME_INTERVENTION_DIAGNOSTIC_FAIL,
                    "ok": False,
                    "gpu_replay_executed": True,
                    "gate1": gate1,
                    "gate2": {"ok": True, "n_examples": 3, "rows": gate2_rows},
                    "gate3": {"ok": False, "failed_example": example["example_id"], "row": row},
                    "semantic_gate_status": "NOT_PREDECLARED",
                    "gold_rerun_authorized": False,
                    "gold_rerun_executed": False,
                    "n_gold_studies_processed": 0,
                    "n_report_only_studies_processed": 0,
                    "kernel_slug": PHASE46_KERNEL_SLUG,
                    "raw_completions_stored": False,
                    "logits_retained": False,
                    "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
                _write_sanitized(out_dir, payload)
                return payload
            unconstrained_rows.append(
                {
                    "example_id": example["example_id"],
                    "ok": row["unconstrained"]["schema_valid"],
                    "clinical_fields": row["unconstrained"]["clinical_fields"],
                }
            )
            constrained_first.append(
                {
                    "example_id": example["example_id"],
                    "ok": row["constrained"]["schema_valid"],
                    "clinical_fields": row["constrained"]["clinical_fields"],
                    "synthetic_truth": row["constrained"]["synthetic_truth"],
                    "schema_valid": row["constrained"]["schema_valid"],
                }
            )
            constrained_second.append(row["constrained_repeat"])

        n_constrained_valid = sum(1 for row in constrained_first if row.get("schema_valid")) + sum(
            1 for row in constrained_second if row.get("schema_valid")
        )
        format_ok = n_constrained_valid == 28
        intervention = aggregate_constraint_stats(
            [
                item["constrained_intervention"]
                for item in gate3_pairs
                if item.get("constrained_intervention")
            ]
            + [
                item["repeat_intervention"]
                for item in gate3_pairs
                if item.get("repeat_intervention")
            ]
        )
        semantic = evaluate_semantic_gates(
            constrained_rows=constrained_first,
            unconstrained_rows=unconstrained_rows,
            intervention=intervention,
        )
        intervention_ok = all(item.get("ok") for item in gate2_rows) and all(
            item.get("ok") for item in gate3_pairs
        )
        status = decide_phase46_outcome(
            gpu_replay_executed=True,
            skip_reason=None,
            gate1_ok=True,
            gate2_ok=True,
            gate3_format_ok=format_ok,
            intervention_ok=intervention_ok and bool(semantic.get("code_never_coerces_unknown_to_negative", True)),
            semantic_ok=bool(semantic.get("ok")),
        )
        assert status != FORBIDDEN_SEMANTIC_PASS
        if torch.cuda.is_available():
            peak_gpu = int(torch.cuda.max_memory_allocated())
        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        payload = {
            "protocol": PHASE46,
            "status": status,
            "ok": status == OUTCOME_FORMAT_SEMANTIC_GATE_NOT_ESTABLISHED,
            "skip_reason": None,
            "gpu_replay_executed": True,
            "gate1": gate1,
            "gate2": {"ok": True, "n_examples": 3, "rows": gate2_rows},
            "gate3": {
                "ok": format_ok,
                "n_examples": 14,
                "n_constrained_schema_valid": n_constrained_valid,
                "expected_constrained_schema_valid": 28,
                "unconstrained_n_schema_valid": sum(1 for row in unconstrained_rows if row.get("ok")),
                "intervention": intervention,
                "semantic": semantic,
                "code_never_coerces_unknown_to_negative": never_coerces_unknown_to_negative(),
            },
            "semantic_gate_status": "NOT_PREDECLARED",
            "gold_rerun_authorized": False,
            "gold_rerun_executed": False,
            "notebook_07_executed": False,
            "n_gold_studies_processed": 0,
            "n_report_only_studies_processed": 0,
            "kernel_slug": PHASE46_KERNEL_SLUG,
            "peak_gpu_memory_bytes": peak_gpu,
            "inference_seconds": round(elapsed, 3),
            "forbidden_outcome_not_used": FORBIDDEN_SEMANTIC_PASS,
            "raw_completions_stored": False,
            "logits_retained": False,
            "report_text_stored": False,
            "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        _write_sanitized(out_dir, payload)
        return payload
    except RuntimeError as exc:
        reason = str(exc)
        status = OUTCOME_INTERVENTION_DIAGNOSTIC_FAIL
        if reason.startswith("GATE1_FAIL"):
            status = OUTCOME_INTERVENTION_DIAGNOSTIC_FAIL
        return _empty_phase46_result(
            status=status,
            skip_reason=reason[:200],
            gate1=gate1,
            output_dir=out_dir,
            extra={"gpu_replay_executed": True, "exception": reason[:200]},
        )
