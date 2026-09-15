"""Per-target PILOT_PASS / PILOT_RESTRICTED / PILOT_FAIL decisions."""

from __future__ import annotations

from typing import Any

from .config import OFFICIAL_TARGET_COLUMNS

STATUS_PASS = "PILOT_PASS"
STATUS_RESTRICTED = "PILOT_RESTRICTED"
STATUS_FAIL = "PILOT_FAIL"
STATUS_EXECUTION_FAIL = "PILOT_EXECUTION_FAIL"


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
    return number if number == number else float("nan")


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
    min_positive_precision: float = 0.85,
    min_npv: float = 0.85,
    min_precision_ci_low: float = 0.70,
    min_npv_ci_low: float = 0.70,
    max_malformed_rate: float = 0.05,
    small_n_cannot_pass: bool = True,
    small_n_gold_positive_max: int = 11,
) -> dict[str, Any]:
    """Prioritize positive precision and NPV. Coverage is reported, not a pass gate.

    Abstentions stay visible as ``n_abstentions`` and are not folded into
    sensitivity. Conditional sensitivity is not used as a pass criterion.
    """
    prec = _finite(precision)
    npv_v = _finite(npv)
    cov = _finite(coverage)
    malformed = _finite(malformed_output_rate)
    p_low = _finite(precision_ci_low) if precision_ci_low is not None else float("nan")
    n_low = _finite(npv_ci_low) if npv_ci_low is not None else float("nan")
    reasons: list[str] = []
    small_n = official in SMALL_N_OFFICIAL or int(n_gold_positive) <= int(small_n_gold_positive_max)
    if small_n:
        reasons.append(f"small_n gold_positives={int(n_gold_positive)}")

    pos_ok = prec == prec and prec >= min_positive_precision
    neg_ok = npv_v == npv_v and npv_v >= min_npv
    if p_low == p_low:
        pos_ok = pos_ok and p_low >= min_precision_ci_low
    if n_low == n_low:
        neg_ok = neg_ok and n_low >= min_npv_ci_low
    malformed_ok = malformed != malformed or malformed <= max_malformed_rate
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
        "malformed_output_rate": malformed,
        "precision_ci_low": p_low,
        "npv_ci_low": n_low,
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
            min_positive_precision=float(acc.get("min_positive_precision", 0.85)),
            min_npv=float(acc.get("min_npv", 0.85)),
            min_precision_ci_low=float(acc.get("min_precision_ci_low", 0.70)),
            min_npv_ci_low=float(acc.get("min_npv_ci_low", 0.70)),
            max_malformed_rate=float(acc.get("max_malformed_rate", 0.05)),
            small_n_cannot_pass=bool(acc.get("small_n_cannot_pass", True)),
            small_n_gold_positive_max=int(acc.get("small_n_gold_positive_max", 11)),
        )
    return out
