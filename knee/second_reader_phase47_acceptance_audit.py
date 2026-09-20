"""Phase 4C.7 acceptance audit from existing sanitized gold metrics.

Aggregate-only. Never invokes the model, never loads weights, never
reads private row CSVs, and never processes report-only studies. If the
sanitized Phase 4C.7 metrics file is missing, the result is
``ACCEPTANCE_AUDIT_INCOMPLETE`` and no PILOT_PASS claim is preserved by
assumption.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

from .config import OFFICIAL_TARGET_COLUMNS, PROJECT_ROOT, TARGET_KEYS
from .kaggle_audit import jsonable
from .kaggle_bundle import sha256_file
from .report_labels import (
    HARD_WEAK_LABEL_TRAINING_ALLOWED,
    SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED,
)
from .second_reader_acceptance import (
    TINY_DECISION_DENOMINATOR_MAX,
    decide_all_pilot_targets,
    execution_flag_mismatch_reasons,
    execution_flags_for_counts,
)
from .second_reader_constrained import never_coerces_unknown_to_negative
from .second_reader_phase46 import FORBIDDEN_SEMANTIC_PASS, git_head_sha
from .second_reader_phase47 import (
    EXPECTED_GOLD_STUDIES,
    FROZEN_PILOT_V14_SHA256,
    PHASE47,
)
from .second_reader_pilot import load_pilot_config
from .second_reader_schema import DEFAULT_PILOT_CONFIG_PATH_V1_4

AUDIT_PROTOCOL = "second-reader-phase-4c.7-acceptance-audit"
STATUS_COMPLETE = "ACCEPTANCE_AUDIT_COMPLETE"
STATUS_INCOMPLETE = "ACCEPTANCE_AUDIT_INCOMPLETE"
PRIMARY_SYSTEM = "strict_agreement"
EVALUATED_SYSTEMS = ("deterministic_v2", "second_reader", "strict_agreement")
FROZEN_V14_CONFIG_SHA256 = FROZEN_PILOT_V14_SHA256
BOOTSTRAP_METHOD = "study_level_percentile_bootstrap"
BOOTSTRAP_N_RESAMPLES = 1000
BOOTSTRAP_SEED_BASE = 42
BOOTSTRAP_CI_LEVEL = 0.95

PHASE47_METRICS_CANDIDATES = (
    PROJECT_ROOT
    / "outputs"
    / "kaggle_phase47_v1"
    / "sanitized"
    / "rsna_second_reader_gold_constrained_v14"
    / "sanitized"
    / "second_reader_gold_metrics.json",
    PROJECT_ROOT
    / "outputs"
    / "rsna_second_reader_gold_constrained_v14"
    / "sanitized"
    / "second_reader_gold_metrics.json",
)
PHASE47_ACCEPTANCE_CANDIDATES = tuple(
    path.with_name("pilot_acceptance.json") for path in PHASE47_METRICS_CANDIDATES
)
PHASE47_SUMMARY_CANDIDATES = tuple(
    path.with_name("second_reader_phase47_gold.json") for path in PHASE47_METRICS_CANDIDATES
)


def first_existing(paths: tuple[Path, ...]) -> Path | None:
    for path in paths:
        if path.is_file():
            return path
    return None


def _ci_block(stats: dict[str, Any], metric: str) -> dict[str, Any]:
    block = ((stats.get("ci") or {}).get(metric)) or {}
    low = block.get("low")
    high = block.get("high")
    n_valid = block.get("n_valid_resamples")
    low_f = float(low) if isinstance(low, (int, float)) else float("nan")
    high_f = float(high) if isinstance(high, (int, float)) else float("nan")
    return {
        "low": low_f if math.isfinite(low_f) else None,
        "high": high_f if math.isfinite(high_f) else None,
        "n_valid_resamples": int(n_valid) if n_valid is not None else 0,
        "finite": bool(math.isfinite(low_f) and math.isfinite(high_f)),
    }


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def bootstrap_seed_for_official(official: str) -> int:
    return BOOTSTRAP_SEED_BASE + list(OFFICIAL_TARGET_COLUMNS).index(official)


def sanitized_target_aggregates(official: str, stats: dict[str, Any]) -> dict[str, Any]:
    """Counts, rates, and finite CIs only. No study IDs or row predictions."""
    tp = _int_or_none(stats.get("tp")) or 0
    fp = _int_or_none(stats.get("fp")) or 0
    tn = _int_or_none(stats.get("tn")) or 0
    fn = _int_or_none(stats.get("fn")) or 0
    n_pos_denom = _int_or_none(stats.get("n_positive_calls"))
    if n_pos_denom is None:
        n_pos_denom = tp + fp
    n_neg_denom = _int_or_none(stats.get("n_negative_calls"))
    if n_neg_denom is None:
        n_neg_denom = tn + fn
    return {
        "official_column": official,
        "target_key": stats.get("target_key")
        or TARGET_KEYS[list(OFFICIAL_TARGET_COLUMNS).index(official)],
        "n_gold_positive": _int_or_none(stats.get("n_gold_positive")) or 0,
        "n_gold_negative": _int_or_none(stats.get("n_gold_negative")) or 0,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "n_abstentions": _int_or_none(stats.get("n_abstentions")) or 0,
        "n_covered": _int_or_none(stats.get("n_covered")) or 0,
        "n_positive_decision_denominator": n_pos_denom,
        "n_negative_decision_denominator": n_neg_denom,
        "tiny_positive_decision_denominator": bool(
            0 < n_pos_denom <= TINY_DECISION_DENOMINATOR_MAX
        ),
        "tiny_negative_decision_denominator": bool(
            0 < n_neg_denom <= TINY_DECISION_DENOMINATOR_MAX
        ),
        "precision": _float_or_none(stats.get("precision")),
        "npv": _float_or_none(stats.get("npv")),
        "coverage": _float_or_none(stats.get("coverage")),
        "positive_capture": _float_or_none(stats.get("positive_capture")),
        "negative_capture": _float_or_none(stats.get("negative_capture")),
        "selective_error": _float_or_none(stats.get("selective_error")),
        "ci": {
            "precision": _ci_block(stats, "precision"),
            "npv": _ci_block(stats, "npv"),
            "coverage": _ci_block(stats, "coverage"),
            "positive_capture": _ci_block(stats, "positive_capture"),
            "negative_capture": _ci_block(stats, "negative_capture"),
            "selective_error": _ci_block(stats, "selective_error"),
        },
        "bootstrap": {
            "method": BOOTSTRAP_METHOD,
            "n_resamples": BOOTSTRAP_N_RESAMPLES,
            "seed": bootstrap_seed_for_official(official),
            "seed_rule": "42 + official_column_index",
            "ci_level": BOOTSTRAP_CI_LEVEL,
            "undefined_resamples_dropped": True,
        },
    }


def _incomplete(
    *,
    reason: str,
    metrics_path: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "protocol": AUDIT_PROTOCOL,
        "phase47_protocol": PHASE47,
        "status": STATUS_INCOMPLETE,
        "ok": False,
        "reason": reason,
        "metrics_path": metrics_path,
        "pass_claims_preserved_by_assumption": False,
        "inference_run": False,
        "model_generate_called": False,
        "n_gold_studies_processed": 0,
        "n_report_only_studies_processed": 0,
        "HARD_WEAK_LABEL_TRAINING_ALLOWED": HARD_WEAK_LABEL_TRAINING_ALLOWED,
        "SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED": SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED,
        "gold_rerun_authorized": False,
        "gold_eval_executed": False,
        "gold_rerun_executed": False,
        "forbidden_outcome_not_used": FORBIDDEN_SEMANTIC_PASS,
        "code_never_coerces_unknown_to_negative": never_coerces_unknown_to_negative(),
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_head_sha": git_head_sha(),
    }
    if extra:
        payload.update(extra)
    return payload


def audit_metrics_payload(
    metrics: dict[str, Any],
    *,
    old_acceptance: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    n_gold_studies_processed: int | None = None,
    metrics_path: str | None = None,
    metrics_sha256: str | None = None,
) -> dict[str, Any]:
    """Recompute sanitized labels from a metrics JSON object. No generate()."""
    cfg = config if config is not None else load_pilot_config(DEFAULT_PILOT_CONFIG_PATH_V1_4)
    cfg_path = Path(cfg.get("_config_path") or DEFAULT_PILOT_CONFIG_PATH_V1_4)
    cfg_sha = str(cfg.get("_config_sha256") or sha256_file(cfg_path))
    if cfg_sha != FROZEN_V14_CONFIG_SHA256 and cfg_path.resolve() == DEFAULT_PILOT_CONFIG_PATH_V1_4.resolve():
        return _incomplete(reason="frozen_v14_config_hash_changed", metrics_path=metrics_path)

    missing_systems = [name for name in EVALUATED_SYSTEMS if name not in metrics]
    if missing_systems:
        return _incomplete(
            reason="missing_systems:" + ",".join(missing_systems),
            metrics_path=metrics_path,
        )

    n_gold = int(
        n_gold_studies_processed
        if n_gold_studies_processed is not None
        else (metrics.get(PRIMARY_SYSTEM) or {}).get("n_studies")
        or 0
    )
    flags = execution_flags_for_counts(n_gold_studies_processed=n_gold)
    malformed = float((metrics.get(PRIMARY_SYSTEM) or {}).get("malformed_output_rate") or 0.0)
    systems: dict[str, Any] = {}
    label_changes: list[dict[str, Any]] = []
    for system in EVALUATED_SYSTEMS:
        block = metrics[system] or {}
        per_target = block.get("per_target") or {}
        system_malformed = float(block.get("malformed_output_rate") if block.get("malformed_output_rate") is not None else malformed)
        audited = decide_all_pilot_targets(
            per_target,
            malformed_output_rate=system_malformed,
            config=cfg,
        )
        rows: dict[str, Any] = {}
        for official in OFFICIAL_TARGET_COLUMNS:
            aggregates = sanitized_target_aggregates(official, per_target.get(official) or {})
            decision = audited[official]
            old_status = None
            if system == PRIMARY_SYSTEM and old_acceptance:
                old_status = (old_acceptance.get(official) or {}).get("status")
            row = {
                **aggregates,
                "stored_status": old_status if system == PRIMARY_SYSTEM else None,
                "audited_status": decision["status"],
                "status_changed": bool(
                    system == PRIMARY_SYSTEM and old_status is not None and old_status != decision["status"]
                ),
                "precision_ci_low": None if not math.isfinite(float(decision["precision_ci_low"])) else float(decision["precision_ci_low"]),
                "npv_ci_low": None if not math.isfinite(float(decision["npv_ci_low"])) else float(decision["npv_ci_low"]),
                "precision_ci_finite": bool(decision.get("precision_ci_finite")),
                "npv_ci_finite": bool(decision.get("npv_ci_finite")),
                "small_n": bool(decision.get("small_n")),
                "reasons": list(decision.get("reasons") or []),
                "coverage_used_as_pass_gate": False,
            }
            if row["status_changed"]:
                label_changes.append(
                    {
                        "system": system,
                        "official_column": official,
                        "stored_status": old_status,
                        "audited_status": decision["status"],
                    }
                )
            rows[official] = row
        systems[system] = {
            "n_studies": int(block.get("n_studies") or n_gold),
            "malformed_output_rate": system_malformed,
            "per_target": rows,
        }

    primary = systems[PRIMARY_SYSTEM]["per_target"]
    medial = primary.get("Medial Meniscus") or {}
    medial_proof = {
        "official_column": "Medial Meniscus",
        "stored_status": medial.get("stored_status"),
        "audited_status": medial.get("audited_status"),
        "n_gold_positive": medial.get("n_gold_positive"),
        "n_gold_negative": medial.get("n_gold_negative"),
        "tp": medial.get("tp"),
        "fp": medial.get("fp"),
        "tn": medial.get("tn"),
        "fn": medial.get("fn"),
        "n_positive_decision_denominator": medial.get("n_positive_decision_denominator"),
        "n_negative_decision_denominator": medial.get("n_negative_decision_denominator"),
        "tiny_negative_decision_denominator": medial.get("tiny_negative_decision_denominator"),
        "precision": medial.get("precision"),
        "npv": medial.get("npv"),
        "precision_ci_low": (medial.get("ci") or {}).get("precision", {}).get("low"),
        "precision_ci_high": (medial.get("ci") or {}).get("precision", {}).get("high"),
        "precision_ci_n_valid_resamples": (medial.get("ci") or {}).get("precision", {}).get("n_valid_resamples"),
        "npv_ci_low": (medial.get("ci") or {}).get("npv", {}).get("low"),
        "npv_ci_high": (medial.get("ci") or {}).get("npv", {}).get("high"),
        "npv_ci_n_valid_resamples": (medial.get("ci") or {}).get("npv", {}).get("n_valid_resamples"),
        "precision_ci_finite_ge_0_70": bool(
            isinstance(medial.get("precision_ci_low"), (int, float))
            and math.isfinite(float(medial["precision_ci_low"]))
            and float(medial["precision_ci_low"]) >= 0.70
        ),
        "npv_ci_finite_ge_0_70": bool(
            isinstance(medial.get("npv_ci_low"), (int, float))
            and math.isfinite(float(medial["npv_ci_low"]))
            and float(medial["npv_ci_low"]) >= 0.70
        ),
        "still_pilot_pass_after_fail_closed_ci": medial.get("audited_status") == "PILOT_PASS",
        "tiny_npv_denominator_reported_not_a_yaml_gate": True,
    }
    mcl = primary.get("MCL") or {}
    mcl_note = {
        "official_column": "MCL",
        "stored_status": mcl.get("stored_status"),
        "audited_status": mcl.get("audited_status"),
        "n_covered": mcl.get("n_covered"),
        "n_positive_decision_denominator": mcl.get("n_positive_decision_denominator"),
        "n_negative_decision_denominator": mcl.get("n_negative_decision_denominator"),
        "precision": mcl.get("precision"),
        "npv": mcl.get("npv"),
        "point_estimate_not_sufficient_evidence": True,
        "tiny_positive_decision_denominator": mcl.get("tiny_positive_decision_denominator"),
        "npv_ci_finite": mcl.get("npv_ci_finite"),
    }

    mismatches = execution_flag_mismatch_reasons(flags)
    payload = {
        "protocol": AUDIT_PROTOCOL,
        "phase47_protocol": PHASE47,
        "status": STATUS_COMPLETE,
        "ok": not mismatches and n_gold > 0,
        "reason": "",
        "metrics_path": metrics_path,
        "metrics_sha256": metrics_sha256,
        "pass_claims_preserved_by_assumption": False,
        "inference_run": False,
        "model_generate_called": False,
        "primary_system": PRIMARY_SYSTEM,
        "v14_config_sha256": cfg_sha,
        "v14_config_sha256_unchanged": cfg_sha == FROZEN_V14_CONFIG_SHA256,
        "tiny_decision_denominator_max_reporting_only": TINY_DECISION_DENOMINATOR_MAX,
        "bootstrap": {
            "method": BOOTSTRAP_METHOD,
            "n_resamples": BOOTSTRAP_N_RESAMPLES,
            "seed_base": BOOTSTRAP_SEED_BASE,
            "seed_rule": "42 + official_column_index",
            "ci_level": BOOTSTRAP_CI_LEVEL,
            "undefined_resamples_dropped": True,
        },
        "malformed_output_rate": malformed,
        "n_report_only_studies_processed": 0,
        "HARD_WEAK_LABEL_TRAINING_ALLOWED": HARD_WEAK_LABEL_TRAINING_ALLOWED,
        "SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED": SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED,
        "forbidden_outcome_not_used": FORBIDDEN_SEMANTIC_PASS,
        "semantic_gate_status": "NOT_PREDECLARED",
        "code_never_coerces_unknown_to_negative": never_coerces_unknown_to_negative(),
        "systems": systems,
        "label_changes": label_changes,
        "medial_meniscus_ci_proof": medial_proof,
        "mcl_tiny_denominator": mcl_note,
        "execution_flag_mismatch_reasons": mismatches,
        "execution_flag_invariant_ok": not mismatches,
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_head_sha": git_head_sha(),
        **flags,
    }
    if n_gold != EXPECTED_GOLD_STUDIES and metrics_path and "synthetic" not in str(metrics_path):
        payload["gold_count_note"] = f"n_gold_studies_processed={n_gold}"
    if mismatches:
        payload["ok"] = False
        payload["reason"] = mismatches[0]
    return jsonable(payload)


def run_phase47_acceptance_audit(
    *,
    metrics_path: str | Path | None = None,
    acceptance_path: str | Path | None = None,
    n_gold_studies_processed: int | None = None,
) -> dict[str, Any]:
    path = Path(metrics_path) if metrics_path else first_existing(PHASE47_METRICS_CANDIDATES)
    if path is None or not path.is_file():
        return _incomplete(
            reason="missing_phase47_gold_metrics",
            metrics_path=str(metrics_path) if metrics_path else None,
        )
    try:
        metrics = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _incomplete(reason=f"unreadable_metrics:{exc}", metrics_path=str(path))
    if not isinstance(metrics, dict):
        return _incomplete(reason="metrics_not_a_mapping", metrics_path=str(path))

    acc_path = Path(acceptance_path) if acceptance_path else first_existing(PHASE47_ACCEPTANCE_CANDIDATES)
    old_acceptance = None
    if acc_path is not None and acc_path.is_file():
        loaded = json.loads(acc_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            old_acceptance = loaded

    n_gold = n_gold_studies_processed
    if n_gold is None:
        summary_path = first_existing(PHASE47_SUMMARY_CANDIDATES)
        if summary_path is not None:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(summary, dict) and summary.get("n_gold_studies_processed") is not None:
                n_gold = int(summary["n_gold_studies_processed"])
        if n_gold is None:
            n_gold = int((metrics.get(PRIMARY_SYSTEM) or {}).get("n_studies") or 0)

    return audit_metrics_payload(
        metrics,
        old_acceptance=old_acceptance,
        n_gold_studies_processed=n_gold,
        metrics_path=str(path),
        metrics_sha256=sha256_file(path),
    )


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "n/a"
        return f"{value:.{digits}f}"
    return str(value)


def render_audit_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Phase 4C.7 acceptance audit",
        "",
        "Aggregate-only recompute from existing sanitized Phase 4C.7 gold",
        "metrics. No model inference, no new Kaggle kernel, no private",
        "row contents, no report-only processing, and no training. Frozen",
        "kernel `mahdimzafari/rsna-knee-second-reader-gold-constrained-v14`",
        "version 1, its hashes, and its scientific tables remain the original",
        "record. This file is a reporting/implementation correction.",
        "",
        f"- status: `{payload.get('status')}`",
        f"- primary system: `{payload.get('primary_system')}`",
        f"- n_gold_studies_processed: {payload.get('n_gold_studies_processed')}",
        f"- n_report_only_studies_processed: {payload.get('n_report_only_studies_processed')}",
        f"- gold_eval_executed: `{str(payload.get('gold_eval_executed')).lower()}`",
        f"- gold_rerun_executed: `{str(payload.get('gold_rerun_executed')).lower()}`",
        f"- gold_rerun_authorized: `{str(payload.get('gold_rerun_authorized')).lower()}`",
        f"- v1.4 config SHA-256 unchanged: `{payload.get('v14_config_sha256')}`",
        f"- forbidden outcome not used: `{payload.get('forbidden_outcome_not_used')}`",
        f"- semantic_gate_status: `{payload.get('semantic_gate_status')}`",
        "- bootstrap: study-level percentile, n_resamples=1000, seed=42+target_index, ci_level=0.95",
        f"- tiny-denominator reporting max (not a YAML gate): {payload.get('tiny_decision_denominator_max_reporting_only')}",
        "- code never maps uncertain / not_mentioned / unknown to negative",
        "",
        "## Execution-flag correction",
        "",
        "A run that processed 58 gold studies must record execution flags",
        "true. `gold_rerun_authorized` remains false after the run.",
        "",
        f"- execution_flag_invariant_ok: `{str(payload.get('execution_flag_invariant_ok')).lower()}`",
        "",
        "## Medial Meniscus fail-closed CI proof",
        "",
    ]
    proof = payload.get("medial_meniscus_ci_proof") or {}
    lines.extend(
        [
            f"- stored_status: `{proof.get('stored_status')}`",
            f"- audited_status: `{proof.get('audited_status')}`",
            f"- precision={_fmt(proof.get('precision'), 4)} CI low/high="
            f"{_fmt(proof.get('precision_ci_low'), 4)}/{_fmt(proof.get('precision_ci_high'), 4)}"
            f" (n_valid_resamples={proof.get('precision_ci_n_valid_resamples')})",
            f"- NPV={_fmt(proof.get('npv'), 4)} CI low/high="
            f"{_fmt(proof.get('npv_ci_low'), 4)}/{_fmt(proof.get('npv_ci_high'), 4)}"
            f" (n_valid_resamples={proof.get('npv_ci_n_valid_resamples')})",
            f"- precision_ci_finite_ge_0.70: `{str(proof.get('precision_ci_finite_ge_0_70')).lower()}`",
            f"- npv_ci_finite_ge_0.70: `{str(proof.get('npv_ci_finite_ge_0_70')).lower()}`",
            f"- still PILOT_PASS after fail-closed CI: `{str(proof.get('still_pilot_pass_after_fail_closed_ci')).lower()}`",
            f"- positive_decision_denominator={proof.get('n_positive_decision_denominator')}"
            f"; negative_decision_denominator={proof.get('n_negative_decision_denominator')}"
            f"; tiny_negative_denominator=`{str(proof.get('tiny_negative_decision_denominator')).lower()}`"
            " (reported, not a YAML pass gate)",
            "",
            "## MCL tiny denominator",
            "",
            "A single non-abstaining decision with point precision 1.0 is not",
            "sufficient evidence. MCL cannot `PILOT_PASS` under small-n, and",
            "the missing NPV CI is now a fail-closed non-finite CI.",
            "",
        ]
    )
    mcl = payload.get("mcl_tiny_denominator") or {}
    lines.extend(
        [
            f"- stored_status: `{mcl.get('stored_status')}` audited_status: `{mcl.get('audited_status')}`",
            f"- n_covered={mcl.get('n_covered')}; positive_denom={mcl.get('n_positive_decision_denominator')};"
            f" negative_denom={mcl.get('n_negative_decision_denominator')}",
            f"- precision={_fmt(mcl.get('precision'))}; NPV={_fmt(mcl.get('npv'))}",
            "",
        ]
    )
    for system in EVALUATED_SYSTEMS:
        block = (payload.get("systems") or {}).get(system) or {}
        lines.extend(
            [
                f"## {system}",
                "",
                "| Official | stored | audited | gold+/gold- | TP FP TN FN | abst | pos_denom | neg_denom | prec | prec_CI | NPV | NPV_CI | cov | pos_cap | neg_cap | sel_err | tiny | seed |",
                "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | --- | ---: |",
            ]
        )
        for official in OFFICIAL_TARGET_COLUMNS:
            row = (block.get("per_target") or {}).get(official) or {}
            tiny = []
            if row.get("tiny_positive_decision_denominator"):
                tiny.append("pos")
            if row.get("tiny_negative_decision_denominator"):
                tiny.append("neg")
            stored = row.get("stored_status") or "—"
            prec_ci = (row.get("ci") or {}).get("precision") or {}
            npv_ci = (row.get("ci") or {}).get("npv") or {}
            lines.append(
                "| {official} | `{stored}` | `{audited}` | {gp}/{gn} | {tp} {fp} {tn} {fn} | {abst} | {pd} | {nd} | {prec} | {p_lo} | {npv} | {n_lo} | {cov} | {pc} | {nc} | {se} | {tiny} | {seed} |".format(
                    official=official,
                    stored=stored,
                    audited=row.get("audited_status"),
                    gp=row.get("n_gold_positive"),
                    gn=row.get("n_gold_negative"),
                    tp=row.get("tp"),
                    fp=row.get("fp"),
                    tn=row.get("tn"),
                    fn=row.get("fn"),
                    abst=row.get("n_abstentions"),
                    pd=row.get("n_positive_decision_denominator"),
                    nd=row.get("n_negative_decision_denominator"),
                    prec=_fmt(row.get("precision"), 4),
                    p_lo=_fmt(prec_ci.get("low"), 4) + "/" + _fmt(prec_ci.get("high"), 4),
                    npv=_fmt(row.get("npv"), 4),
                    n_lo=_fmt(npv_ci.get("low"), 4) + "/" + _fmt(npv_ci.get("high"), 4),
                    cov=_fmt(row.get("coverage"), 4),
                    pc=_fmt(row.get("positive_capture"), 4),
                    nc=_fmt(row.get("negative_capture"), 4),
                    se=_fmt(row.get("selective_error"), 4),
                    tiny=",".join(tiny) or "—",
                    seed=(row.get("bootstrap") or {}).get("seed"),
                )
            )
        lines.append("")
    changes = payload.get("label_changes") or []
    lines.extend(["## Stored vs audited primary-system labels", ""])
    if not changes:
        lines.append("No primary-system status labels changed after fail-closed CI.")
        lines.append("")
    else:
        lines.append("| Official | stored | audited |")
        lines.append("| --- | --- | --- |")
        for change in changes:
            lines.append(
                f"| {change.get('official_column')} | `{change.get('stored_status')}` | `{change.get('audited_status')}` |"
            )
        lines.append("")
    lines.extend(
        [
            "## Coordinator gates",
            "",
            "- `HARD_WEAK_LABEL_TRAINING_ALLOWED: false`",
            "- `SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED: false`",
            "- `gold_rerun_authorized: false`",
            "- do not process the 4,349 report-only studies",
            "- do not merge PR #10",
            "",
        ]
    )
    return "\n".join(lines) + "\n"
