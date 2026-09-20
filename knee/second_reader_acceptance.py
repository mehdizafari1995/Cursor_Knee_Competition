"""Per-target PILOT_PASS / PILOT_RESTRICTED / PILOT_FAIL decisions."""

from __future__ import annotations

import math
from typing import Any

from .config import OFFICIAL_TARGET_COLUMNS

STATUS_PASS = "PILOT_PASS"
STATUS_RESTRICTED = "PILOT_RESTRICTED"
STATUS_FAIL = "PILOT_FAIL"
STATUS_EXECUTION_FAIL = "PILOT_EXECUTION_FAIL"

# Reporting-only. Not a frozen YAML pass gate. Point estimates on 1–5
# non-abstaining decisions are flagged explicitly.
TINY_DECISION_DENOMINATOR_MAX = 5


def decide_execution_status(
    *,
    n_gold_studies_processed: int,
    n_valid_second_reader_outputs: int,
    n_report_only_studies_processed: int,
    malformed_output_rate: float,
    real_data_execution: bool,
) -> dict[str, Any]:
    """Run-level technical status. Not a clinical-performance conclusion."""
    all_malformed = (
        int(n_gold_studies_processed) > 0
        and int(n_valid_second_reader_outputs) == 0
        and float(malformed_output_rate) >= 1.0
    )
    if all_malformed:
        return {
            "status": STATUS_EXECUTION_FAIL,
            "real_data_execution": bool(real_data_execution),
            "scientific_performance_evaluable": False,
            "reason": "all second-reader outputs malformed",
            "n_gold_studies_processed": int(n_gold_studies_processed),
            "n_valid_second_reader_outputs": int(n_valid_second_reader_outputs),
            "n_report_only_studies_processed": int(n_report_only_studies_processed),
        }
    return {
        "status": "PILOT_EXECUTED",
        "real_data_execution": bool(real_data_execution),
        "scientific_performance_evaluable": bool(real_data_execution),
        "reason": "",
        "n_gold_studies_processed": int(n_gold_studies_processed),
        "n_valid_second_reader_outputs": int(n_valid_second_reader_outputs),
        "n_report_only_studies_processed": int(n_report_only_studies_processed),
    }

SMALL_N_OFFICIAL = {"MCL", "Lateral OA", "Baker's"}


def _finite(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def execution_flags_for_counts(*, n_gold_studies_processed: int) -> dict[str, Any]:
    """Post-run bookkeeping. Authorization is always reset to false.

    A completed gold evaluation (`n_gold_studies_processed > 0`) must record
    both execution flags true. `gold_rerun_authorized` stays false after the
    run so the completed evaluation cannot be mistaken for a new authorization.
    """
    n = int(n_gold_studies_processed)
    executed = n > 0
    return {
        "n_gold_studies_processed": n,
        "gold_eval_executed": executed,
        "gold_rerun_executed": executed,
        "gold_rerun_authorized": False,
    }


def execution_flag_mismatch_reasons(payload: dict[str, Any]) -> list[str]:
    """Fail-closed invariants tying execution flags to gold-study counts."""
    n = int(payload.get("n_gold_studies_processed") or 0)
    eval_executed = bool(payload.get("gold_eval_executed"))
    rerun_executed = bool(payload.get("gold_rerun_executed"))
    authorized = bool(payload.get("gold_rerun_authorized"))
    reasons: list[str] = []
    if n > 0 and not eval_executed:
        reasons.append("execution_flag_mismatch:gold_eval_executed_false_with_n_gold")
    if n > 0 and not rerun_executed:
        reasons.append("execution_flag_mismatch:gold_rerun_executed_false_with_n_gold")
    if n <= 0 and eval_executed:
        reasons.append("execution_flag_mismatch:gold_eval_executed_true_with_n_gold_zero")
    if n <= 0 and rerun_executed:
        reasons.append("execution_flag_mismatch:gold_rerun_executed_true_with_n_gold_zero")
    if authorized:
        reasons.append("execution_flag_mismatch:gold_rerun_authorized_true")
    return reasons


def _ci_gate_ok(
    raw_value: Any,
    *,
    threshold: float | None,
    nonfinite_reason: str,
    below_reason: str,
    reasons: list[str],
) -> tuple[bool, float]:
    """Configured CI lower bounds are requirements, not optional checks."""
    low = _finite(raw_value)
    if threshold is None:
        return True, low
    if not math.isfinite(low):
        reasons.append(nonfinite_reason)
        return False, low
    if low < float(threshold):
        reasons.append(f"{below_reason}={low:.3f}<{float(threshold)}")
        return False, low
    return True, low


def _denom_reasons(n: int | None, *, polarity: str) -> list[str]:
    if n is None:
        return []
    value = int(n)
    if value == 0:
        return [f"zero_{polarity}_decision_denominator"]
    if 0 < value <= TINY_DECISION_DENOMINATOR_MAX:
        return [f"tiny_{polarity}_decision_denominator={value}"]
    return []


def decide_pilot_status(
    *,
    official: str,
    precision: float,
    npv: float,
    coverage: float,
    n_gold_positive: int,
    n_gold_negative: int,
    n_abstentions: int,
    malformed_output_rate: float,
    precision_ci_low: float | None = None,
    npv_ci_low: float | None = None,
    n_positive_decision_denominator: int | None = None,
    n_negative_decision_denominator: int | None = None,
    min_positive_precision: float = 0.85,
    min_npv: float = 0.85,
    min_precision_ci_low: float | None = 0.70,
    min_npv_ci_low: float | None = 0.70,
    max_malformed_rate: float = 0.05,
    small_n_cannot_pass: bool = True,
    small_n_gold_positive_max: int = 11,
) -> dict[str, Any]:
    """Prioritize positive precision and NPV. Coverage is reported, not a pass gate.

    Abstentions stay visible as ``n_abstentions`` and are not folded into
    sensitivity. Conditional sensitivity is not used as a pass criterion.
    Configured CI lower bounds are required: a missing/NaN/non-finite CI
    fails that metric instead of bypassing the threshold.
    Tiny decision denominators are reported and never treated as a YAML gate.
    """
    prec = _finite(precision)
    npv_v = _finite(npv)
    cov = _finite(coverage)
    malformed = _finite(malformed_output_rate)
    reasons: list[str] = []
    small_n = official in SMALL_N_OFFICIAL or int(n_gold_positive) <= int(small_n_gold_positive_max)
    if small_n:
        reasons.append(f"small_n gold_positives={int(n_gold_positive)}")

    pos_point_ok = math.isfinite(prec) and prec >= min_positive_precision
    neg_point_ok = math.isfinite(npv_v) and npv_v >= min_npv
    prec_ci_ok, p_low = _ci_gate_ok(
        precision_ci_low,
        threshold=min_precision_ci_low,
        nonfinite_reason="precision_ci_low_nonfinite",
        below_reason="precision_ci_low_below_min",
        reasons=reasons,
    )
    npv_ci_ok, n_low = _ci_gate_ok(
        npv_ci_low,
        threshold=min_npv_ci_low,
        nonfinite_reason="npv_ci_low_nonfinite",
        below_reason="npv_ci_low_below_min",
        reasons=reasons,
    )
    pos_ok = pos_point_ok and prec_ci_ok
    neg_ok = neg_point_ok and npv_ci_ok
    n_pos_denom = (
        int(n_positive_decision_denominator)
        if n_positive_decision_denominator is not None
        else None
    )
    n_neg_denom = (
        int(n_negative_decision_denominator)
        if n_negative_decision_denominator is not None
        else None
    )
    reasons.extend(_denom_reasons(n_pos_denom, polarity="positive"))
    reasons.extend(_denom_reasons(n_neg_denom, polarity="negative"))
    malformed_ok = (not math.isfinite(malformed)) or malformed <= max_malformed_rate
    if not malformed_ok:
        reasons.append(f"malformed_output_rate={malformed:.3f}>{max_malformed_rate}")

    if not malformed_ok:
        status = STATUS_FAIL
        reasons.append("malformed_output_gate_failed")
    elif pos_ok and neg_ok and not (small_n and small_n_cannot_pass):
        status = STATUS_PASS
        reasons.append("precision_and_npv_pass")
    elif pos_ok or neg_ok:
        status = STATUS_RESTRICTED
        if pos_ok:
            reasons.append("positive_precision_pass_npv_or_n_fail")
        if neg_ok:
            reasons.append("npv_pass_precision_or_n_fail")
        if small_n:
            reasons.append("small_n_cannot_pass")
    else:
        status = STATUS_FAIL
        reasons.append("precision_and_npv_fail")

    return {
        "official_column": official,
        "status": status,
        "precision": prec,
        "npv": npv_v,
        "coverage": cov,
        "n_gold_positive": int(n_gold_positive),
        "n_gold_negative": int(n_gold_negative),
        "n_abstentions": int(n_abstentions),
        "n_positive_decision_denominator": n_pos_denom,
        "n_negative_decision_denominator": n_neg_denom,
        "tiny_positive_decision_denominator": bool(
            n_pos_denom is not None and 0 < n_pos_denom <= TINY_DECISION_DENOMINATOR_MAX
        ),
        "tiny_negative_decision_denominator": bool(
            n_neg_denom is not None and 0 < n_neg_denom <= TINY_DECISION_DENOMINATOR_MAX
        ),
        "malformed_output_rate": malformed,
        "precision_ci_low": p_low,
        "npv_ci_low": n_low,
        "precision_ci_finite": math.isfinite(p_low),
        "npv_ci_finite": math.isfinite(n_low),
        "small_n": small_n,
        "coverage_used_as_pass_gate": False,
        "reasons": reasons,
    }


def decide_all_pilot_targets(
    per_target: dict[str, dict],
    *,
    malformed_output_rate: float,
    config: dict | None = None,
) -> dict[str, dict]:
    acc = (config or {}).get("acceptance") or {}
    out: dict[str, dict] = {}
    for official in OFFICIAL_TARGET_COLUMNS:
        stats = per_target.get(official) or {}
        ci = stats.get("ci") or {}
        tp = stats.get("tp")
        fp = stats.get("fp")
        tn = stats.get("tn")
        fn = stats.get("fn")
        n_pos_denom = stats.get("n_positive_calls")
        if n_pos_denom is None and tp is not None and fp is not None:
            n_pos_denom = int(tp) + int(fp)
        n_neg_denom = stats.get("n_negative_calls")
        if n_neg_denom is None and tn is not None and fn is not None:
            n_neg_denom = int(tn) + int(fn)
        out[official] = decide_pilot_status(
            official=official,
            precision=stats.get("positive_precision", stats.get("precision")),
            npv=stats.get("npv"),
            coverage=stats.get("coverage"),
            n_gold_positive=int(stats.get("n_gold_positive") or 0),
            n_gold_negative=int(stats.get("n_gold_negative") or 0),
            n_abstentions=int(stats.get("n_abstentions") or 0),
            malformed_output_rate=stats.get("malformed_output_rate", malformed_output_rate),
            precision_ci_low=(ci.get("precision") or {}).get("low"),
            npv_ci_low=(ci.get("npv") or {}).get("low"),
            n_positive_decision_denominator=n_pos_denom,
            n_negative_decision_denominator=n_neg_denom,
            min_positive_precision=float(acc.get("min_positive_precision", 0.85)),
            min_npv=float(acc.get("min_npv", 0.85)),
            min_precision_ci_low=float(acc.get("min_precision_ci_low", 0.70)),
            min_npv_ci_low=float(acc.get("min_npv_ci_low", 0.70)),
            max_malformed_rate=float(acc.get("max_malformed_rate", 0.05)),
            small_n_cannot_pass=bool(acc.get("small_n_cannot_pass", True)),
            small_n_gold_positive_max=int(acc.get("small_n_gold_positive_max", 11)),
        )
    return out
