"""Phase 4C.5 constrained-decoding provenance and intervention audit.

Synthetic reports only. Never reads gold or report-only studies. Never stores
completions, report text, full logits, or patient/competition rows.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any, Iterable, Sequence

from .config import PROJECT_ROOT, TARGET_KEYS
from .kaggle_bundle import sha256_file
from .second_reader_canary import SYNTHETIC_CANARY_EXAMPLES
from .second_reader_constrained import (
    FROZEN_SCHEMA_CANONICAL_JSON_SHA256,
    FROZEN_SCHEMA_RAW_BYTES_SHA256,
    PHI_EOS_IDS,
    PINNED_TOKENIZER_SHA256,
    _field_from_path,
    json_path_from_prefix,
)
from .second_reader_json_stop import PINNED_PHI35_TERMINATION
from .second_reader_parser import parse_second_reader_output
from .second_reader_schema import (
    DEFAULT_PILOT_CONFIG_PATH_V1_4,
    DEFAULT_PROMPT_PATH_V1_2,
    DEFAULT_SCHEMA_PATH,
    SCHEMA_VERSION,
    TARGET_FIELD_KEYS,
    compact_output_skeleton,
)

PHASE45 = "second-reader-phase-4c.5-intervention-audit"
FOCUS_EXAMPLE_IDS = (
    "CANARY_SYNTHETIC_07",
    "CANARY_SYNTHETIC_12",
    "CANARY_SYNTHETIC_13",
)

OUTCOME_FORMAT_SEMANTIC_PASS = "CONSTRAINED_FORMAT_PASS_SEMANTIC_PASS"
OUTCOME_FORMAT_SEMANTIC_GATE_NOT_ESTABLISHED = (
    "CONSTRAINED_FORMAT_PASS_SEMANTIC_GATE_NOT_ESTABLISHED"
)
OUTCOME_INTERVENTION_DIAGNOSTIC_FAIL = "INTERVENTION_DIAGNOSTIC_FAIL"
OUTCOME_SCHEMA_ENFORCEMENT_MISMATCH = "SCHEMA_ENFORCEMENT_MISMATCH"
OUTCOME_CANARY_FORMAT_FAIL = "CANARY_FORMAT_FAIL"
OUTCOME_CANARY_SEMANTIC_FAIL = "CANARY_SEMANTIC_FAIL"

PHASE45_OUTCOMES = (
    OUTCOME_FORMAT_SEMANTIC_PASS,
    OUTCOME_FORMAT_SEMANTIC_GATE_NOT_ESTABLISHED,
    OUTCOME_INTERVENTION_DIAGNOSTIC_FAIL,
    OUTCOME_SCHEMA_ENFORCEMENT_MISMATCH,
    OUTCOME_CANARY_FORMAT_FAIL,
    OUTCOME_CANARY_SEMANTIC_FAIL,
)

# Internal material-degrade floor shipped with v1.4 implementation. Not a
# frozen predeclared canary semantic pass threshold.
INTERNAL_DEGRADE_FLOOR_COMMIT = "ff4709e476268866213e10af2c762dd02b26e9e0"
GPU_CANARY_RECORD_COMMIT = "cf8278edcf3888eb061a6417db7246fb591bc05b"
PINNED_MODEL_ID = "microsoft/Phi-3.5-mini-instruct"
PINNED_MODEL_REVISION = "2fe192450127e6a83f7441aef6e3ca586c338b77"

# Frozen Tesla T4 kernel v3 aggregates. Completions were not retained.
FROZEN_GPU_V3 = {
    "kernel": "mahdimzafari/rsna-knee-second-reader-constrained-canary",
    "version": 3,
    "recorded_status": "CONSTRAINED_CANARY_PASS",
    "unconstrained_format_valid": "11/14",
    "constrained_format_valid": "28/28",
    "n_generated_tokens": 16304,
    "n_top1_allowed": 16304,
    "n_top1_rejected": 0,
    "intervention_rate": 0.0,
    "n_structural_interventions": 0,
    "n_clinical_enum_interventions": 0,
    "synthetic_status_accuracy": 0.815,
    "n_status_correct": 137,
    "n_status_scored": 168,
    "gold_rerun_authorized": False,
    "gold_rerun_executed": False,
    "n_gold_studies_processed": 0,
    "n_report_only_studies_processed": 0,
}


def sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_int_sequence(values: Sequence[int]) -> str:
    packed = ",".join(str(int(v)) for v in values).encode("utf-8")
    return sha256_bytes(packed)


def clone_tensor_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Clone token tensors. Fail closed if a tensor cannot be cloned."""
    cloned: dict[str, Any] = {}
    for key, value in payload.items():
        if hasattr(value, "clone"):
            cloned[key] = value.clone()
        else:
            cloned[key] = deepcopy(value)
    return cloned


def tensors_equal(left: Any, right: Any) -> bool:
    if hasattr(left, "equal") and hasattr(right, "equal"):
        return bool(left.equal(right))
    if hasattr(left, "tolist") and hasattr(right, "tolist"):
        return list(left.tolist()) == list(right.tolist())
    return left == right


def assert_cloned_inputs_identical(original: dict[str, Any], cloned: dict[str, Any]) -> None:
    if set(original) != set(cloned):
        raise ValueError("paired inputs have different keys")
    for key in original:
        if not tensors_equal(original[key], cloned[key]):
            raise ValueError(f"paired input {key} is not identical after clone")
        left_id = id(original[key])
        right_id = id(cloned[key])
        if hasattr(original[key], "clone") and left_id == right_id:
            raise ValueError(f"paired input {key} was not cloned")


def canonical_shared_generation_config() -> dict[str, Any]:
    """Shared deterministic generate() settings. Schema enforcement is excluded."""
    return {
        "max_new_tokens": 2048,
        "do_sample": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "json_object_stop": True,
        "eos_token_ids": list(PHI_EOS_IDS),
        "pad_token_id": int(PINNED_PHI35_TERMINATION["pad_token_id"]),
        "attn_implementation": "eager",
        "dtype_gpu": "float16",
        "deterministic": True,
        "seed": None,
    }


def paired_run_identity(
    *,
    example_id: str,
    synthetic_input_sha256: str,
    rendered_prompt_sha256: str,
    input_ids: Sequence[int],
    attention_mask: Sequence[int] | None,
    unconstrained_config: dict[str, Any],
    constrained_config: dict[str, Any],
) -> dict[str, Any]:
    """Sanitized identity record. Report text and prompts are hashed only."""
    shared = canonical_shared_generation_config()
    unconstrained_extra = sorted(set(unconstrained_config) - set(shared) - {"schema_constrained"})
    constrained_extra = sorted(set(constrained_config) - set(shared) - {"schema_constrained"})
    differ_except_schema: list[str] = []
    for key in sorted(set(unconstrained_config) | set(constrained_config)):
        if key in {"schema_constrained", "logits_processor", "prefix_allowed_tokens_fn", "recorder"}:
            continue
        if unconstrained_config.get(key) != constrained_config.get(key):
            differ_except_schema.append(key)
    ok = not differ_except_schema
    return {
        "example_id": example_id,
        "synthetic_input_sha256": synthetic_input_sha256,
        "rendered_prompt_sha256": rendered_prompt_sha256,
        "input_ids_sha256": sha256_int_sequence(input_ids),
        "attention_mask_sha256": sha256_int_sequence(attention_mask or []),
        "n_input_tokens": len(list(input_ids)),
        "model_id": PINNED_MODEL_ID,
        "model_revision": PINNED_MODEL_REVISION,
        "tokenizer_sha256": dict(PINNED_TOKENIZER_SHA256),
        "schema_version": SCHEMA_VERSION,
        "schema_raw_bytes_sha256": FROZEN_SCHEMA_RAW_BYTES_SHA256,
        "schema_canonical_json_sha256": FROZEN_SCHEMA_CANONICAL_JSON_SHA256,
        "prompt_path": "configs/second_reader_prompt_v1_2.yaml",
        "prompt_sha256": sha256_file(DEFAULT_PROMPT_PATH_V1_2),
        "transport_unconstrained": "second-reader-transport-v1.2",
        "transport_constrained": "second-reader-transport-v1.4",
        "shared_generation_config": shared,
        "unconstrained_schema_constrained": bool(unconstrained_config.get("schema_constrained")),
        "constrained_schema_constrained": bool(constrained_config.get("schema_constrained")),
        "differ_except_schema_enforcement": differ_except_schema,
        "ok": ok,
        "unconstrained_extra_keys": unconstrained_extra,
        "constrained_extra_keys": constrained_extra,
        "raw_completions_stored": False,
        "logits_retained": False,
    }


def token_category(*, token_id: int, json_path: str, eos_ids: Sequence[int] = PHI_EOS_IDS) -> str:
    if int(token_id) in {int(x) for x in eos_ids}:
        return "eos"
    field = _field_from_path(json_path)
    if field in TARGET_FIELD_KEYS:
        return "clinical_enum"
    if field in {"schema_version", "targets", "target_key", "structure"}:
        return "structural"
    return "other"


def first_sequence_divergence(
    unconstrained_ids: Sequence[int],
    constrained_ids: Sequence[int],
) -> dict[str, Any]:
    left = [int(x) for x in unconstrained_ids]
    right = [int(x) for x in constrained_ids]
    limit = min(len(left), len(right))
    for index in range(limit):
        if left[index] != right[index]:
            return {
                "diverged": True,
                "position": index,
                "unconstrained_token_id": left[index],
                "constrained_token_id": right[index],
                "unconstrained_n": len(left),
                "constrained_n": len(right),
            }
    if len(left) != len(right):
        return {
            "diverged": True,
            "position": limit,
            "unconstrained_token_id": left[limit] if limit < len(left) else None,
            "constrained_token_id": right[limit] if limit < len(right) else None,
            "unconstrained_n": len(left),
            "constrained_n": len(right),
            "length_mismatch": True,
        }
    return {
        "diverged": False,
        "position": None,
        "unconstrained_n": len(left),
        "constrained_n": len(right),
    }


def reconcile_intervention_with_sequences(
    *,
    unconstrained_ids: Sequence[int],
    constrained_ids: Sequence[int],
    unconstrained_termination: str,
    constrained_termination: str,
    n_top1_allowed: int,
    n_top1_rejected: int,
    n_generated_steps: int,
    n_structural: int,
    n_clinical: int,
    earlier_state_difference: bool = False,
) -> dict[str, Any]:
    """Zero interventions implies identical tokens and identical termination."""
    if n_top1_allowed + n_top1_rejected != n_generated_steps:
        return {
            "ok": False,
            "status": OUTCOME_INTERVENTION_DIAGNOSTIC_FAIL,
            "reason": "allowed_plus_blocked_ne_inspected_steps",
        }
    if n_structural + n_clinical != n_top1_rejected:
        return {
            "ok": False,
            "status": OUTCOME_INTERVENTION_DIAGNOSTIC_FAIL,
            "reason": "subtype_counts_ne_total_interventions",
        }
    divergence = first_sequence_divergence(unconstrained_ids, constrained_ids)
    identical_tokens = not bool(divergence.get("diverged"))
    identical_termination = unconstrained_termination == constrained_termination
    if n_top1_rejected == 0:
        if earlier_state_difference:
            return {
                "ok": False,
                "status": OUTCOME_INTERVENTION_DIAGNOSTIC_FAIL,
                "reason": "zero_interventions_but_unproven_state_difference",
                "divergence": divergence,
            }
        if not identical_tokens or not identical_termination:
            return {
                "ok": False,
                "status": OUTCOME_INTERVENTION_DIAGNOSTIC_FAIL,
                "reason": "sequences_or_termination_differ_with_zero_interventions",
                "divergence": divergence,
                "unconstrained_termination": unconstrained_termination,
                "constrained_termination": constrained_termination,
            }
        return {
            "ok": True,
            "status": "zero_intervention_token_identical",
            "divergence": divergence,
        }
    if identical_tokens and n_top1_rejected > 0:
        return {
            "ok": False,
            "status": OUTCOME_INTERVENTION_DIAGNOSTIC_FAIL,
            "reason": "interventions_counted_but_token_ids_identical",
            "divergence": divergence,
        }
    if divergence.get("diverged") and n_top1_rejected == 0:
        return {
            "ok": False,
            "status": OUTCOME_INTERVENTION_DIAGNOSTIC_FAIL,
            "reason": "first_divergence_not_counted_as_intervention",
            "divergence": divergence,
        }
    return {
        "ok": True,
        "status": "interventions_reconciled_with_divergence",
        "divergence": divergence,
        "first_divergence_counted": True,
    }


def mask_scores_after_recording(scores: Any, allowed_ids: Iterable[int]) -> Any:
    """Mask logits after unconstrained top-1 was observed. No full logits stored."""
    import torch

    masked = scores.clone()
    vocab = int(masked.shape[-1])
    allow = torch.zeros(vocab, dtype=torch.bool, device=masked.device)
    for tid in allowed_ids:
        idx = int(tid)
        if 0 <= idx < vocab:
            allow[idx] = True
    if not bool(allow.any()):
        return masked
    fill = torch.finfo(masked.dtype).min
    if masked.dim() == 1:
        return masked.masked_fill(~allow, fill)
    return masked.masked_fill(~allow.unsqueeze(0), fill)


def replay_generated_tokens(
    enforcer: Any,
    generated_ids: Sequence[int],
    *,
    prompt_ids: Sequence[int] | None = None,
    decode_fn=None,
    production_schema_invalid: bool | None = None,
) -> dict[str, Any]:
    """Walk generated token IDs through the LMFE prefix tree. No completion stored."""
    seq = [int(x) for x in (prompt_ids or [1])]
    eos_ids = {int(x) for x in PHI_EOS_IDS}
    first_forbidden: dict[str, Any] | None = None
    decoded_prefix = ""
    n_steps = 0
    for position, token_id in enumerate(int(x) for x in generated_ids):
        allowed_obj = enforcer.get_allowed_tokens(seq)
        allowed_list = getattr(allowed_obj, "allowed_tokens", allowed_obj)
        allowed = {int(x) for x in allowed_list}
        n_steps += 1
        path = json_path_from_prefix(decoded_prefix)
        category = token_category(token_id=token_id, json_path=path)
        if token_id not in allowed and first_forbidden is None:
            first_forbidden = {
                "position": position,
                "token_id": token_id,
                "token_category": category,
                "json_path": path,
                "parser_field": _field_from_path(path),
                "unconstrained_top1": True,
                "n_allowed": len(allowed),
                "eos_allowed": bool(eos_ids.issubset(allowed) or any(eid in allowed for eid in eos_ids)),
            }
            break
        seq.append(token_id)
        if decode_fn is not None:
            try:
                decoded_prefix = str(decode_fn(list(generated_ids[: position + 1])) or "")
            except Exception:
                decoded_prefix = ""
    final_allowed_obj = enforcer.get_allowed_tokens(seq)
    final_allowed_list = getattr(final_allowed_obj, "allowed_tokens", final_allowed_obj)
    final_allowed = {int(x) for x in final_allowed_list}
    accepted_fully = first_forbidden is None
    can_end = False
    state = getattr(enforcer, "prefix_states", None) or {}
    last_state = state.get(tuple(seq))
    parser = getattr(last_state, "parser", None)
    if parser is not None and hasattr(parser, "can_end"):
        can_end = bool(parser.can_end())
    schema_mismatch = bool(production_schema_invalid) and accepted_fully and can_end
    return {
        "n_inspected": n_steps,
        "accepted_fully": accepted_fully,
        "can_end": can_end,
        "first_forbidden": first_forbidden,
        "eos_allowed_at_end": bool(eos_ids.issubset(final_allowed) or any(eid in final_allowed for eid in eos_ids)),
        "schema_enforcement_mismatch": schema_mismatch,
        "status": OUTCOME_SCHEMA_ENFORCEMENT_MISMATCH if schema_mismatch else "ok",
        "raw_completions_stored": False,
        "logits_retained": False,
    }


def malformed_json_variants() -> list[dict[str, Any]]:
    """Construct schema-invalid JSON from the compact skeleton. No report text."""
    skeleton = json.loads(compact_output_skeleton())
    variants: list[dict[str, Any]] = []

    invalid_enum = deepcopy(skeleton)
    invalid_enum["targets"]["fracture"]["evidence_category"] = "suspected_not_confirmed"
    variants.append(
        {
            "example_id": "CANARY_SYNTHETIC_12",
            "mode": "invalid_enum",
            "production_schema_invalid": True,
            "text": json.dumps(invalid_enum, separators=(",", ":"), ensure_ascii=True),
        }
    )

    missing_target = deepcopy(skeleton)
    missing_target["targets"].pop("joint_effusion")
    variants.append(
        {
            "example_id": "CANARY_SYNTHETIC_07",
            "mode": "missing_target",
            "production_schema_invalid": True,
            "text": json.dumps(missing_target, separators=(",", ":"), ensure_ascii=True),
        }
    )

    unbalanced = compact_output_skeleton()[:-2]
    variants.append(
        {
            "example_id": "CANARY_SYNTHETIC_13",
            "mode": "unbalanced_braces",
            "production_schema_invalid": True,
            "text": unbalanced,
        }
    )

    extra = compact_output_skeleton() + " trailing"
    variants.append(
        {
            "example_id": "CANARY_SYNTHETIC_13",
            "mode": "forbidden_extra_text",
            "production_schema_invalid": True,
            "text": extra,
        }
    )
    return variants


def replay_malformed_variants(tokenizer_dir: str | None = None) -> dict[str, Any]:
    from .second_reader_constrained import (
        build_token_enforcer_from_tokenizers_json,
        locate_phi_tokenizer_dir,
        verify_phi_tokenizer_files,
    )

    directory = locate_phi_tokenizer_dir(tokenizer_dir)
    if directory is None:
        return {"ok": False, "skipped": True, "reason": "phi_tokenizer_files_missing"}
    try:
        verify_phi_tokenizer_files(directory)
    except (FileNotFoundError, ValueError) as exc:
        return {"ok": False, "skipped": True, "reason": str(exc)}
    tokenizer_json = directory / "tokenizer.json"
    if not tokenizer_json.is_file():
        return {"ok": False, "skipped": True, "reason": "tokenizer.json_missing"}
    try:
        tok, enforcer, _data = build_token_enforcer_from_tokenizers_json(tokenizer_json)
    except ImportError as exc:
        return {"ok": False, "skipped": True, "reason": f"prefix_tree_deps_unavailable:{exc}"}

    def decode_fn(ids: list[int]) -> str:
        return tok.decode(ids, skip_special_tokens=False)

    skeleton = compact_output_skeleton()
    skeleton_ids = list(tok.encode(skeleton).ids)
    valid = replay_generated_tokens(
        enforcer,
        skeleton_ids,
        decode_fn=decode_fn,
        production_schema_invalid=False,
    )
    parsed_valid = parse_second_reader_output(skeleton, protocol="v1.4")
    eos_after_open = replay_generated_tokens(
        enforcer,
        list(tok.encode("{").ids) + [PHI_EOS_IDS[0]],
        decode_fn=decode_fn,
        production_schema_invalid=True,
    )

    rows = []
    mismatch = False
    for variant in malformed_json_variants():
        parsed = parse_second_reader_output(variant["text"], protocol="v1.4")
        ids = list(tok.encode(variant["text"]).ids)
        replay = replay_generated_tokens(
            enforcer,
            ids,
            decode_fn=decode_fn,
            production_schema_invalid=bool(parsed.malformed),
        )
        if replay.get("schema_enforcement_mismatch"):
            mismatch = True
        rows.append(
            {
                "example_id": variant["example_id"],
                "mode": variant["mode"],
                "parser_malformed": bool(parsed.malformed),
                "parser_stage": parsed.malformed_stage,
                "n_tokens": len(ids),
                "text_sha256": sha256_text(variant["text"]),
                "accepted_fully": replay.get("accepted_fully"),
                "can_end": replay.get("can_end"),
                "first_forbidden": replay.get("first_forbidden"),
                "schema_enforcement_mismatch": replay.get("schema_enforcement_mismatch"),
            }
        )

    eos_forbidden_until_complete = bool((eos_after_open.get("first_forbidden") or {}).get("token_category") == "eos") or (
        not eos_after_open.get("accepted_fully")
    )
    ok = (
        bool(parsed_valid.ok)
        and bool(valid.get("accepted_fully"))
        and bool(valid.get("eos_allowed_at_end"))
        and eos_forbidden_until_complete
        and not mismatch
        and all((row["first_forbidden"] is not None) or (not row["can_end"]) for row in rows)
    )
    status = OUTCOME_SCHEMA_ENFORCEMENT_MISMATCH if mismatch else ("ok" if ok else "replay_incomplete")
    return {
        "ok": ok,
        "skipped": False,
        "status": status,
        "valid_skeleton_accepted": bool(valid.get("accepted_fully")),
        "valid_skeleton_eos_allowed": bool(valid.get("eos_allowed_at_end")),
        "eos_forbidden_until_complete": eos_forbidden_until_complete,
        "n_variants": len(rows),
        "variants": rows,
        "schema_enforcement_mismatch": mismatch,
        "n_gold_studies_processed": 0,
        "n_report_only_studies_processed": 0,
        "raw_completions_stored": False,
    }


def _read_text_if_exists(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None


def semantic_gate_audit() -> dict[str, Any]:
    """Locate a predeclared synthetic clinical-accuracy threshold. Do not invent one."""
    protocol_path = PROJECT_ROOT / "reports" / "second_reader_pilot_protocol.md"
    v14_path = PROJECT_ROOT / "reports" / "second_reader_constrained_canary_v14.md"
    protocol = _read_text_if_exists(protocol_path) or ""
    v14_report = _read_text_if_exists(v14_path) or ""
    config_text = DEFAULT_PILOT_CONFIG_PATH_V1_4.read_text(encoding="utf-8")
    schema_text = DEFAULT_SCHEMA_PATH.read_text(encoding="utf-8")
    reports_available = bool(protocol) and bool(v14_report)
    needles = (
        "synthetic_status_accuracy",
        "synthetic clinical-accuracy",
        "canary semantic threshold",
        "predeclared semantic",
    )
    protocol_mentions = [needle for needle in needles if needle in protocol]
    config_has_threshold = "synthetic_status_accuracy" in config_text
    schema_has_threshold = "synthetic_status_accuracy" in schema_text
    v14_claims_predeclared = "Predeclared semantic gate is accuracy < 0.50" in v14_report
    predeclared = False
    return {
        "metric": "synthetic_status_accuracy",
        "predeclared_threshold": None,
        "semantic_gate_status": "NOT_PREDECLARED",
        "internal_degrade_floor": 0.50,
        "internal_degrade_floor_file": "knee/second_reader_constrained.py",
        "internal_degrade_floor_commit": INTERNAL_DEGRADE_FLOOR_COMMIT,
        "internal_floor_existed_before_gpu_record": True,
        "gpu_record_commit": GPU_CANARY_RECORD_COMMIT,
        "gpu_accuracy": FROZEN_GPU_V3["synthetic_status_accuracy"],
        "protocol_mentions": protocol_mentions,
        "config_declares_synthetic_accuracy_threshold": config_has_threshold,
        "schema_declares_synthetic_accuracy_threshold": schema_has_threshold,
        "v14_report_called_floor_predeclared": v14_claims_predeclared,
        "reports_available": reports_available,
        "predeclared": predeclared,
        "note": (
            "The 0.50 value is an internal material_degrade floor added in the "
            "v1.4 implementation commit, not an immutable pre-run canary semantic "
            "pass threshold. Gold acceptance uses precision/NPV gates. Do not "
            "create a threshold after observing accuracy 0.815."
        ),
        "do_not_create_threshold_after_0.815": True,
    }


def frozen_gpu_intervention_audit() -> dict[str, Any]:
    """Apply C assertions to the immutable Tesla T4 v3 aggregates."""
    gpu = FROZEN_GPU_V3
    n_steps = int(gpu["n_generated_tokens"])
    n_allowed = int(gpu["n_top1_allowed"])
    n_rejected = int(gpu["n_top1_rejected"])
    n_structural = int(gpu["n_structural_interventions"])
    n_clinical = int(gpu["n_clinical_enum_interventions"])
    format_differed = gpu["unconstrained_format_valid"] != "14/14"
    conservation_ok = (n_allowed + n_rejected == n_steps) and (n_structural + n_clinical == n_rejected)
    # Completions were not retained. Format-valid counts differ with zero
    # recorded interventions, which is INTERVENTION_DIAGNOSTIC_FAIL.
    status = OUTCOME_INTERVENTION_DIAGNOSTIC_FAIL
    return {
        "ok": False,
        "status": status,
        "conservation_allowed_plus_blocked": n_allowed + n_rejected == n_steps,
        "conservation_subtypes": n_structural + n_clinical == n_rejected,
        "conservation_ok": conservation_ok,
        "format_valid_differed": format_differed,
        "zero_interventions": n_rejected == 0,
        "token_identity_proven": False,
        "completions_retained": False,
        "kernel": gpu["kernel"],
        "kernel_version": gpu["version"],
        "recorded_status_immutable": gpu["recorded_status"],
        "n_gold_studies_processed": 0,
        "n_report_only_studies_processed": 0,
        "note": (
            "Constrained decoding produced 28/28 schema-valid outputs after an "
            "unconstrained capture of 11/14. Recorded intervention_rate was 0.0. "
            "Likely cause: PrefixConstrainedLogitsProcessor ran before the "
            "observer, so unconstrained top-1 was read from already-masked scores. "
            "Completions were not stored, so actual 07/12/13 token prefixes cannot "
            "be replayed from the frozen kernel."
        ),
    }


def sanitized_focus_example_hashes() -> list[dict[str, str]]:
    from .report_second_reader import load_prompt_bundle, render_user_prompt

    prompt = load_prompt_bundle(DEFAULT_PROMPT_PATH_V1_2)
    rows = []
    for example in SYNTHETIC_CANARY_EXAMPLES:
        rendered = render_user_prompt(example["report"], prompt=prompt)
        rows.append(
            {
                "example_id": example["example_id"],
                "focus": example["example_id"] in FOCUS_EXAMPLE_IDS,
                "synthetic_input_sha256": sha256_text(example["report"]),
                "rendered_prompt_sha256": sha256_text(rendered),
                "language": example["language"],
                "coverage": example["coverage"],
            }
        )
    return rows


def decide_phase45_outcome(
    *,
    schema_mismatch: bool,
    intervention_ok: bool,
    format_ok: bool | None,
    semantic_predeclared: bool,
    semantic_fail: bool,
    gpu_replay_executed: bool,
) -> str:
    if schema_mismatch:
        return OUTCOME_SCHEMA_ENFORCEMENT_MISMATCH
    if not intervention_ok:
        return OUTCOME_INTERVENTION_DIAGNOSTIC_FAIL
    if format_ok is False:
        return OUTCOME_CANARY_FORMAT_FAIL
    if semantic_fail:
        return OUTCOME_CANARY_SEMANTIC_FAIL
    if not gpu_replay_executed:
        return OUTCOME_INTERVENTION_DIAGNOSTIC_FAIL
    if not semantic_predeclared:
        return OUTCOME_FORMAT_SEMANTIC_GATE_NOT_ESTABLISHED
    return OUTCOME_FORMAT_SEMANTIC_PASS


def run_phase45_audit() -> dict[str, Any]:
    """CPU audit for A–D. Does not run gold, report-only, or a real Phi GPU canary."""
    identity_rows = []
    shared = canonical_shared_generation_config()
    unconstrained_cfg = dict(shared)
    unconstrained_cfg["schema_constrained"] = False
    constrained_cfg = dict(shared)
    constrained_cfg["schema_constrained"] = True
    hashes = sanitized_focus_example_hashes()
    dummy_ids = list(range(8))
    dummy_mask = [1] * 8
    for row in hashes:
        identity = paired_run_identity(
            example_id=row["example_id"],
            synthetic_input_sha256=row["synthetic_input_sha256"],
            rendered_prompt_sha256=row["rendered_prompt_sha256"],
            input_ids=dummy_ids,
            attention_mask=dummy_mask,
            unconstrained_config=unconstrained_cfg,
            constrained_config=constrained_cfg,
        )
        identity_rows.append(
            {
                "example_id": row["example_id"],
                "focus": row["focus"],
                "synthetic_input_sha256": row["synthetic_input_sha256"],
                "rendered_prompt_sha256": row["rendered_prompt_sha256"],
                "ok": identity["ok"],
                "differ_except_schema_enforcement": identity["differ_except_schema_enforcement"],
            }
        )
    replay = replay_malformed_variants()
    semantic = semantic_gate_audit()
    frozen = frozen_gpu_intervention_audit()
    schema_mismatch = bool(replay.get("schema_enforcement_mismatch"))
    outcome = decide_phase45_outcome(
        schema_mismatch=schema_mismatch,
        intervention_ok=bool(frozen.get("ok")),
        format_ok=None,
        semantic_predeclared=bool(semantic.get("predeclared")),
        semantic_fail=False,
        gpu_replay_executed=False,
    )
    return {
        "phase": PHASE45,
        "status": outcome,
        "gold_rerun_authorized": False,
        "gold_rerun_executed": False,
        "n_gold_studies_processed": 0,
        "n_report_only_studies_processed": 0,
        "gpu_replay_executed": False,
        "identity": {
            "n_examples": len(identity_rows),
            "n_focus": sum(1 for row in identity_rows if row["focus"]),
            "all_shared_config_ok": all(row["ok"] for row in identity_rows),
            "rows": identity_rows,
        },
        "replay": {
            "ok": replay.get("ok"),
            "skipped": replay.get("skipped"),
            "status": replay.get("status"),
            "valid_skeleton_accepted": replay.get("valid_skeleton_accepted"),
            "eos_forbidden_until_complete": replay.get("eos_forbidden_until_complete"),
            "schema_enforcement_mismatch": replay.get("schema_enforcement_mismatch"),
            "variants": replay.get("variants"),
        },
        "semantic_gate": semantic,
        "frozen_gpu": frozen,
        "instrumentation_fix": (
            "SchemaConstraint logits processor now records unconstrained top-1 "
            "then masks. generate() must not also pass prefix_allowed_tokens_fn."
        ),
        "raw_completions_stored": False,
        "logits_retained": False,
        "report_text_stored": False,
    }
