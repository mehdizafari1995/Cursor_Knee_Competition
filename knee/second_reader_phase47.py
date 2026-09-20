"""Phase 4C.7 gold-only constrained second-reader evaluation.

One authorized scientific run of the same 58 expert-labelled studies using
transport v1.4 and the corrected constrained generator. Never processes
report-only studies. Never trains, scales, or submits. Notebook 07 stays a
blocked historical stub.

The frozen v1.4 YAML still records ``prefix_allowed_tokens_fn: true``. That
hash must not change. Effective runtime does not pass the prefix function to
``model.generate``; source code is authoritative.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import PROJECT_ROOT
from .kaggle_audit import jsonable
from .kaggle_bundle import sha256_file
from .kaggle_runtime import (
    DATA_SOURCE_REAL_KAGGLE,
    DATA_SOURCE_SOFTWARE_FIXTURE,
    KaggleLayout,
    preflight,
)
from .report_labels import (
    HARD_WEAK_LABEL_TRAINING_ALLOWED,
    SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED,
    split_gold_and_report_only,
)
from .report_second_reader import (
    GenerateFn,
    assert_second_reader_has_no_gold_parameter,
    inspect_generate_kwargs,
    locate_weights_directory,
)
from . import second_reader_constrained as constrained_mod
from .second_reader_constrained import (
    PHI_EOS_IDS,
    aggregate_constraint_stats,
    never_coerces_unknown_to_negative,
    reset_constraint_stats,
)
from .second_reader_intervention_audit import semantic_gate_audit
from .second_reader_phase46 import (
    FakeSchemaMaskProcessor,
    FORBIDDEN_SEMANTIC_PASS,
    gate1_file_hashes,
    git_head_sha,
    prove_processor_order,
    runtime_capability,
    skip_reason_from_runtime,
)
from .second_reader_acceptance import execution_flag_mismatch_reasons, execution_flags_for_counts
from .second_reader_pilot import load_pilot_config, run_second_reader_gold_pilot
from .second_reader_schema import DEFAULT_PILOT_CONFIG_PATH_V1_4

PHASE47 = "second-reader-phase-4c.7-gold-constrained-v14"
KAGGLE_PHASE47_OUTPUT_DIRNAME = "rsna_second_reader_gold_constrained_v14"
PHASE47_KERNEL_SLUG = "rsna-knee-second-reader-gold-constrained-v14"
FROZEN_GOLD_KERNEL = "mahdimzafari/rsna-knee-second-reader-gold-pilot"
FROZEN_GOLD_KERNEL_VERSION = 7
FROZEN_CONSTRAINED_CANARY_KERNEL = "mahdimzafari/rsna-knee-second-reader-constrained-canary"
FROZEN_CONSTRAINED_CANARY_VERSION = 3
FROZEN_PILOT_V14_SHA256 = "992af7a4b4483278445a6f3fb363c818611e79005618dff063460d0aed3bd7f7"
EXPECTED_GOLD_STUDIES = 58
EXPECTED_REPORT_ONLY_STUDIES = 4349

YAML_PREFIX_ALLOWED_TOKENS_FN = True
EFFECTIVE_RUNTIME_PREFIX_ALLOWED_TOKENS_FN = False


def effective_runtime_decoding(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """YAML may still say prefix_fn true; generate() must not receive it."""
    decoding = (config or {}).get("decoding") or {}
    yaml_flag = bool(decoding.get("prefix_allowed_tokens_fn", YAML_PREFIX_ALLOWED_TOKENS_FN))
    un_kwargs = {
        "max_new_tokens": int(decoding.get("max_new_tokens") or 2048),
        "do_sample": False,
        "eos_token_id": list(PHI_EOS_IDS),
    }
    con_kwargs = dict(un_kwargs)
    con_kwargs["logits_processor"] = [FakeSchemaMaskProcessor()]
    return {
        "yaml_prefix_allowed_tokens_fn": yaml_flag,
        "effective_runtime_prefix_allowed_tokens_fn": EFFECTIVE_RUNTIME_PREFIX_ALLOWED_TOKENS_FN,
        "authority": "source_code",
        "inspect_unconstrained": inspect_generate_kwargs(un_kwargs),
        "inspect_constrained": inspect_generate_kwargs(con_kwargs),
        "eos_token_ids": list(PHI_EOS_IDS),
        "schema_constrained": True,
        "n_schema_mask_processors_constrained": 1,
        "generate_receives_prefix_allowed_tokens_fn": False,
        "frozen_v14_yaml_unchanged": True,
    }


def count_gold_and_report_only(train: pd.DataFrame) -> dict[str, int]:
    gold_frame, report_only = split_gold_and_report_only(train)
    return {
        "n_gold_studies_selected": int(len(gold_frame)),
        "n_report_only_studies_present": int(len(report_only)),
        "n_report_only_studies_selected": 0,
    }


def gold_selection_ok(*, n_gold: int, data_source: str) -> tuple[bool, str | None]:
    """Real Kaggle gold must be exactly 58 before any generate() call."""
    if data_source == DATA_SOURCE_REAL_KAGGLE and int(n_gold) != EXPECTED_GOLD_STUDIES:
        return False, f"gold_count_not_{EXPECTED_GOLD_STUDIES}:{n_gold}"
    return True, None


def _write_sanitized(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sanitized_dir = output_dir / "sanitized"
    sanitized_dir.mkdir(parents=True, exist_ok=True)
    public = {
        key: value
        for key, value in payload.items()
        if key not in {"comparisons", "acceptance_full", "pilot_result"}
    }
    (sanitized_dir / "second_reader_phase47_gold.json").write_text(
        json.dumps(jsonable(public), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _base_payload(
    *,
    status: str,
    skip_reason: str | None,
    gate1: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    semantic = semantic_gate_audit()
    payload: dict[str, Any] = {
        "protocol": PHASE47,
        "status": status,
        "ok": False,
        "skip_reason": skip_reason,
        "gold_eval_executed": False,
        "gate1": gate1,
        "semantic_gate_status": semantic.get("semantic_gate_status") or "NOT_PREDECLARED",
        "semantic_gate": semantic,
        "gold_rerun_authorized": False,
        "gold_rerun_executed": False,
        "notebook_07_executed": False,
        "n_gold_studies_processed": 0,
        "n_gold_studies": 0,
        "n_report_only_studies_processed": 0,
        "process_report_only_studies": False,
        "kernel_slug": PHASE47_KERNEL_SLUG,
        "do_not_overwrite_frozen_kernels": {
            "gold_pilot": {"slug": FROZEN_GOLD_KERNEL, "version": FROZEN_GOLD_KERNEL_VERSION},
            "constrained_canary": {
                "slug": FROZEN_CONSTRAINED_CANARY_KERNEL,
                "version": FROZEN_CONSTRAINED_CANARY_VERSION,
            },
        },
        "forbidden_outcome_not_used": FORBIDDEN_SEMANTIC_PASS,
        "HARD_WEAK_LABEL_TRAINING_ALLOWED": HARD_WEAK_LABEL_TRAINING_ALLOWED,
        "SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED": SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED,
        "yaml_prefix_allowed_tokens_fn": YAML_PREFIX_ALLOWED_TOKENS_FN,
        "effective_runtime_prefix_allowed_tokens_fn": EFFECTIVE_RUNTIME_PREFIX_ALLOWED_TOKENS_FN,
        "effective_runtime_authority": "source_code",
        "code_never_coerces_unknown_to_negative": never_coerces_unknown_to_negative(),
        "raw_completions_stored": False,
        "logits_retained": False,
        "report_text_stored": False,
        "training_run": False,
        "folds_created": False,
        "internet_used": False,
        "download_weights": False,
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_head_sha": git_head_sha(),
    }
    if extra:
        payload.update(extra)
    payload.update(
        execution_flags_for_counts(
            n_gold_studies_processed=int(payload.get("n_gold_studies_processed") or 0)
        )
    )
    payload["notebook_07_executed"] = False
    mismatches = execution_flag_mismatch_reasons(payload)
    payload["execution_flag_invariant_ok"] = not mismatches
    payload["execution_flag_mismatch_reasons"] = mismatches
    if mismatches:
        payload["ok"] = False
        payload["status"] = "EXECUTION_FLAG_INVARIANT_FAIL"
        payload["skip_reason"] = mismatches[0]
    return payload


def _recorder_after_gold(*, scientific: bool, n_gold: int) -> dict[str, Any]:
    """Aggregate CONSTRAINT_STAT_LOG after the gold loop. Module namespace only."""
    log = list(constrained_mod.CONSTRAINT_STAT_LOG)
    last = constrained_mod.LAST_CONSTRAINT_STATS
    aggregated = aggregate_constraint_stats(log)
    reasons: list[str] = []
    if scientific:
        if not log:
            reasons.append("missing_recorder:gold:CONSTRAINT_STAT_LOG_empty")
        elif last is None:
            reasons.append("missing_recorder:gold:LAST_CONSTRAINT_STATS_is_none")
        if log and int(len(log)) != int(n_gold):
            reasons.append(f"missing_recorder:gold:log_len_ne_n_gold:{len(log)}!={n_gold}")
        if aggregated.get("source") != "transformers_logits_processor":
            reasons.append("missing_recorder:gold:source")
        if aggregated.get("fixture_not_a_model_result"):
            reasons.append("missing_recorder:gold:fixture_not_a_model_result")
        n_calls = int(aggregated.get("n_calls") or 0)
        n_tokens = int(aggregated.get("n_generated_tokens") or 0)
        if n_calls <= 0 or n_tokens <= 0:
            reasons.append("missing_recorder:gold:n_calls_or_tokens_zero")
        elif n_calls != n_tokens:
            reasons.append("missing_recorder:gold:n_calls_ne_n_generated_tokens")
    return {
        "ok": not reasons if scientific else True,
        "reasons": reasons,
        "n_log_entries": int(len(log)),
        "last_stats_present": last is not None,
        "aggregated": aggregated,
        "source": aggregated.get("source"),
        "fixture_not_a_model_result": bool(aggregated.get("fixture_not_a_model_result")),
        "n_calls": aggregated.get("n_calls"),
        "n_generated_tokens": aggregated.get("n_generated_tokens"),
        "n_top1_allowed": aggregated.get("n_top1_allowed"),
        "n_top1_rejected": aggregated.get("n_top1_rejected"),
        "n_structural_interventions": aggregated.get("n_structural_interventions"),
        "n_clinical_enum_interventions": aggregated.get("n_clinical_enum_interventions"),
        "n_target_key_interventions": aggregated.get("n_target_key_interventions"),
        "n_property_name_interventions": aggregated.get("n_property_name_interventions"),
        "n_json_structure_interventions": aggregated.get("n_json_structure_interventions"),
        "n_other_interventions": aggregated.get("n_other_interventions"),
        "intervention_rate": aggregated.get("intervention_rate"),
        "raw_completions_stored": False,
        "logits_retained": False,
    }


def run_phase47_gold_constrained(
    *,
    data_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    layout: KaggleLayout | None = None,
    data_source: str | None = None,
    config_path: str | Path | None = None,
    weights_path: str | Path | None = None,
    generate_fn: GenerateFn | None = None,
) -> dict[str, Any]:
    """Score gold studies only. Report-only rows are counted and refused."""
    assert HARD_WEAK_LABEL_TRAINING_ALLOWED is False
    assert SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED is False
    assert_second_reader_has_no_gold_parameter()

    layout = layout or KaggleLayout(output_dirname=KAGGLE_PHASE47_OUTPUT_DIRNAME)
    out_dir = Path(output_dir or (PROJECT_ROOT / "outputs" / KAGGLE_PHASE47_OUTPUT_DIRNAME))
    cfg_path = Path(config_path or DEFAULT_PILOT_CONFIG_PATH_V1_4)
    config = load_pilot_config(cfg_path)
    if config.get("process_report_only_studies"):
        raise RuntimeError("phase 4C.7 must keep process_report_only_studies false")
    if sha256_file(cfg_path) != FROZEN_PILOT_V14_SHA256 and cfg_path.resolve() == DEFAULT_PILOT_CONFIG_PATH_V1_4.resolve():
        raise RuntimeError("frozen v1.4 config hash changed")

    capability = runtime_capability()
    weights_dir = locate_weights_directory(
        config.get("second_reader") or config.get("selected_model") or config,
        weights_path,
    )
    hashes = gate1_file_hashes(weights_dir=weights_dir)
    order = prove_processor_order()
    runtime_dec = effective_runtime_decoding(config)
    gate1 = {
        "ok": bool(hashes["ok"] and order["ok"] and not runtime_dec["generate_receives_prefix_allowed_tokens_fn"]),
        "file_hashes": hashes,
        "processor_order": order,
        "capability": capability,
        "effective_runtime_decoding": runtime_dec,
        "phase47_py_sha256": sha256_file(PROJECT_ROOT / "knee" / "second_reader_phase47.py"),
        "git_head_sha": hashes.get("git_head_sha") or git_head_sha(),
        "yaml_prefix_allowed_tokens_fn": runtime_dec["yaml_prefix_allowed_tokens_fn"],
        "effective_runtime_prefix_allowed_tokens_fn": runtime_dec[
            "effective_runtime_prefix_allowed_tokens_fn"
        ],
    }
    if not gate1["ok"]:
        payload = _base_payload(
            status="GATE1_FAIL",
            skip_reason="gate1_fail",
            gate1=gate1,
        )
        _write_sanitized(out_dir, payload)
        return payload

    if generate_fn is None:
        skip = skip_reason_from_runtime(capability, weights_dir=weights_dir)
        if skip:
            payload = _base_payload(
                status="GOLD_EVAL_NOT_EXECUTED",
                skip_reason=skip,
                gate1=gate1,
                extra={"weights_attached": weights_dir is not None},
            )
            _write_sanitized(out_dir, payload)
            return payload

    pre = preflight(
        layout=layout,
        data_root=data_root,
        output_dir=out_dir,
        data_source=data_source,
    )
    source = data_source or pre.data_source or DATA_SOURCE_SOFTWARE_FIXTURE
    if not pre.ok or pre.data_root is None:
        payload = _base_payload(
            status="DATA_NOT_AVAILABLE",
            skip_reason="competition_data_missing",
            gate1=gate1,
            extra={"data_source": source, "preflight_ok": False},
        )
        _write_sanitized(out_dir, payload)
        return payload

    train = pd.read_csv(Path(pre.data_root) / "train.csv")
    counts = count_gold_and_report_only(train)
    selection_ok, selection_reason = gold_selection_ok(
        n_gold=counts["n_gold_studies_selected"],
        data_source=source,
    )
    if not selection_ok:
        payload = _base_payload(
            status="GOLD_COUNT_MISMATCH",
            skip_reason=selection_reason,
            gate1=gate1,
            extra={
                "data_source": source,
                "n_gold_studies_selected": counts["n_gold_studies_selected"],
                "n_report_only_studies_present": counts["n_report_only_studies_present"],
                "n_report_only_studies_selected": 0,
                "n_report_only_studies_processed": 0,
            },
        )
        _write_sanitized(out_dir, payload)
        return payload

    reset_constraint_stats()
    pilot = run_second_reader_gold_pilot(
        data_root=pre.data_root,
        output_dir=out_dir,
        layout=layout,
        data_source=source,
        config_path=cfg_path,
        weights_path=weights_dir if generate_fn is None else weights_path,
        generate_fn=generate_fn,
    )
    n_gold = int(pilot.get("n_gold_studies") or 0)
    scientific = bool(pilot.get("scientific_evaluation")) and source == DATA_SOURCE_REAL_KAGGLE
    recorder = _recorder_after_gold(scientific=scientific, n_gold=n_gold)
    runtime = dict(pilot.get("runtime") or {})
    payload = _base_payload(
        status=str(pilot.get("execution_status") or pilot.get("status") or "ok"),
        skip_reason=None,
        gate1=gate1,
        extra={
            "ok": bool(
                n_gold == (EXPECTED_GOLD_STUDIES if source == DATA_SOURCE_REAL_KAGGLE else n_gold)
                and int(pilot.get("n_report_only_studies_processed") or 0) == 0
                and recorder.get("ok")
            ),
            "gold_eval_executed": n_gold > 0,
            "data_source": source,
            "scientific_evaluation": bool(pilot.get("scientific_evaluation")),
            "claim_scientific_performance": bool(pilot.get("claim_scientific_performance")),
            "scientific_performance_evaluable": bool(pilot.get("scientific_performance_evaluable")),
            "real_data_execution": bool(pilot.get("real_data_execution")),
            "execution_status": pilot.get("execution_status"),
            "inference_backend": pilot.get("inference_backend"),
            "n_gold_studies": n_gold,
            "n_gold_studies_processed": n_gold,
            "n_valid_second_reader_outputs": pilot.get("n_valid_second_reader_outputs"),
            "n_report_only_studies_processed": int(pilot.get("n_report_only_studies_processed") or 0),
            "n_report_only_studies_refused": pilot.get("n_report_only_studies_refused"),
            "n_gold_studies_selected": counts["n_gold_studies_selected"],
            "n_report_only_studies_present": counts["n_report_only_studies_present"],
            "n_report_only_studies_selected": 0,
            "malformed_output_rate": pilot.get("malformed_output_rate"),
            "n_malformed_studies": runtime.get("n_malformed_studies"),
            "gpu_used": bool(pilot.get("gpu_used")),
            "gpu_name": runtime.get("gpu_name"),
            "cuda_capability": runtime.get("cuda_capability"),
            "wall_seconds": runtime.get("wall_seconds"),
            "mean_seconds_per_study": runtime.get("mean_seconds_per_study"),
            "gpu_memory_bytes": runtime.get("gpu_memory_bytes"),
            "peak_gpu_memory_bytes": runtime.get("gpu_memory_bytes"),
            "private_hashes": pilot.get("private_hashes") or {},
            "acceptance": {
                official: {
                    "status": row.get("status"),
                    "precision": row.get("precision"),
                    "npv": row.get("npv"),
                    "coverage": row.get("coverage"),
                    "n_abstentions": row.get("n_abstentions"),
                    "n_gold_positive": row.get("n_gold_positive"),
                    "n_gold_negative": row.get("n_gold_negative"),
                    "precision_ci_low": row.get("precision_ci_low"),
                    "npv_ci_low": row.get("npv_ci_low"),
                    "small_n": row.get("small_n"),
                    "reasons": row.get("reasons"),
                }
                for official, row in (pilot.get("acceptance") or {}).items()
            },
            "macro": {
                system: (comp.get("macro") or {})
                for system, comp in (pilot.get("comparisons") or {}).items()
            },
            "recorder": recorder,
            "pilot_version": (pilot.get("freeze") or {}).get("pilot_version"),
            "eval_protocol_version": (pilot.get("freeze") or {}).get("eval_protocol_version"),
            "transport_protocol_version": (pilot.get("freeze") or {}).get(
                "transport_protocol_version"
            ),
            "freeze": {
                key: (pilot.get("freeze") or {}).get(key)
                for key in (
                    "pilot_version",
                    "eval_protocol_version",
                    "prompt_version",
                    "schema_version",
                    "model_id",
                    "model_revision",
                    "file_sha256",
                    "config_sha256",
                    "metrics_computed_after_freeze",
                    "gold_only",
                    "process_report_only_studies",
                    "transport_protocol_version",
                    "verified_file_sha256",
                    "weights_directory_name",
                )
            },
            "output_dir": str(out_dir),
        },
    )
    if scientific and not recorder.get("ok"):
        payload["status"] = "INTERVENTION_DIAGNOSTIC_FAIL"
        payload["ok"] = False
        payload["skip_reason"] = (recorder.get("reasons") or ["missing_recorder"])[0]
    payload.update(
        execution_flags_for_counts(
            n_gold_studies_processed=int(payload.get("n_gold_studies_processed") or 0)
        )
    )
    payload["notebook_07_executed"] = False
    mismatches = execution_flag_mismatch_reasons(payload)
    payload["execution_flag_invariant_ok"] = not mismatches
    payload["execution_flag_mismatch_reasons"] = mismatches
    if mismatches:
        payload["ok"] = False
        payload["status"] = "EXECUTION_FLAG_INVARIANT_FAIL"
        payload["skip_reason"] = mismatches[0]
    _write_sanitized(out_dir, payload)
    return payload
