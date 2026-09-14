"""Per-target trainable-status decisions. No single macro threshold."""

from __future__ import annotations

from typing import Any

from .config import OFFICIAL_TARGET_COLUMNS
from .report_v1_aggregates import INFLATED_PREVALENCE_TARGETS, V1_WEAK_POS_RATE_AMONG_KNOWN

STATUS_HIGH = "TRAINABLE_HIGH_CONFIDENCE"
STATUS_RESTRICTED = "TRAINABLE_WITH_RESTRICTIONS"
STATUS_NOT = "NOT_TRAINABLE"

SMALL_N_OFFICIAL = {"MCL", "Lateral OA", "Baker's"}


def _finite(value: Any, default: float = float("nan")) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if number == number else default


def decide_trainable_status(
    *,
    official: str,
    precision: float,
    npv: float,
    coverage: float,
    positive_capture: float,
    negative_capture: float,
    n_gold_positive: int,
    n_gold_negative: int,
    precision_ci_low: float | None = None,
    npv_ci_low: float | None = None,
    coverage_ci_low: float | None = None,
    weak_pos_rate: float | None = None,
    expert_pos_rate: float | None = None,
    small_n: bool = False,
) -> dict[str, Any]:
    """Issue one status from precision, NPV, coverage, CIs and prevalence.

    Positive weak labels need precision. Negative weak labels need NPV.
    Inflated weak prevalence versus gold blocks high-confidence positives.
    """
    reasons: list[str] = []
    restrictions: list[str] = []
    prec = _finite(precision)
    npv_v = _finite(npv)
    cov = _finite(coverage)
    pos_cap = _finite(positive_capture)
    neg_cap = _finite(negative_capture)
    p_low = _finite(precision_ci_low) if precision_ci_low is not None else float("nan")
    n_low = _finite(npv_ci_low) if npv_ci_low is not None else float("nan")
    c_low = _finite(coverage_ci_low) if coverage_ci_low is not None else float("nan")

    prevalence_ratio = float("nan")
    inflated = official in INFLATED_PREVALENCE_TARGETS
    if weak_pos_rate is not None and expert_pos_rate not in (None, 0):
        prevalence_ratio = float(weak_pos_rate) / float(expert_pos_rate)
        if prevalence_ratio > 1.8:
            inflated = True
            reasons.append(f"weak_prevalence_ratio={prevalence_ratio:.2f}>1.8")

    if small_n or official in SMALL_N_OFFICIAL or n_gold_positive < 12:
        small_n = True
        reasons.append(f"small_n gold_positives={n_gold_positive}")

    pos_usable = prec == prec and prec >= 0.80 and (p_low != p_low or p_low >= 0.70)
    neg_usable = npv_v == npv_v and npv_v >= 0.80 and (n_low != n_low or n_low >= 0.70)
    cov_ok = cov == cov and cov >= 0.20
    capture_ok = (pos_cap == pos_cap and pos_cap >= 0.50) or (
        neg_cap == neg_cap and neg_cap >= 0.50
    )
    ci_ok = p_low == p_low and n_low == n_low and p_low >= 0.80 and n_low >= 0.80

    if inflated:
        restrictions.append("do_not_use_positive_calls_as_hard_labels")
    if small_n:
        restrictions.append("wide_intervals_unstable")
    restrictions.append("never_impute_unknown_or_disagreement_as_zero")
    restrictions.append("use_high_confidence_masks_only")

    high = (
        pos_usable
        and neg_usable
        and cov_ok
        and capture_ok
        and not inflated
        and not small_n
        and n_gold_positive >= 12
        and n_gold_negative >= 12
        and ci_ok
    )
    if high:
        status = STATUS_HIGH
        reasons.append("precision_npv_coverage_and_prevalence_all_pass")
    elif (pos_usable or neg_usable) and cov == cov and cov >= 0.05:
        status = STATUS_RESTRICTED
        if pos_usable and not inflated:
            restrictions.append("positives_allowed_in_high_conf_mask")
        else:
            restrictions.append("positives_not_approved")
        if neg_usable:
            restrictions.append("negatives_allowed_in_high_conf_mask")
        else:
            restrictions.append("negatives_not_approved")
        reasons.append("usable_polarity_exists_but_not_high_confidence")
    else:
        status = STATUS_NOT
        reasons.append("precision_npv_coverage_or_prevalence_fail")

    return {
        "official_column": official,
        "status": status,
        "precision": prec,
        "npv": npv_v,
        "coverage": cov,
        "positive_capture": pos_cap,
        "negative_capture": neg_cap,
        "n_gold_positive": int(n_gold_positive),
        "n_gold_negative": int(n_gold_negative),
        "precision_ci_low": p_low,
        "npv_ci_low": n_low,
        "coverage_ci_low": c_low,
        "prevalence_ratio": prevalence_ratio,
        "inflated_prevalence": inflated,
        "small_n": small_n,
        "reasons": reasons,
        "restrictions": restrictions,
    }


def decide_all_targets(per_target: dict[str, dict], *, weak_prev: dict | None = None) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for official in OFFICIAL_TARGET_COLUMNS:
        stats = per_target.get(official) or {}
        ci = stats.get("ci") or {}
        weak_rate = None
        if weak_prev and official in weak_prev:
            weak_rate = (weak_prev[official] or {}).get("positive_rate_among_known")
        if weak_rate is None:
            weak_rate = V1_WEAK_POS_RATE_AMONG_KNOWN.get(official)
        expert = stats.get("expert_pos_rate")
        if expert is None:
            n_pos = stats.get("n_gold_positive")
            n_known = stats.get("n_gold_known")
            if n_pos is not None and n_known:
                expert = float(n_pos) / float(n_known)
        out[official] = decide_trainable_status(
            official=official,
            precision=stats.get("precision") if stats.get("precision") == stats.get("precision") else stats.get("positive_precision"),
            npv=stats.get("npv") if stats.get("npv") == stats.get("npv") else stats.get("negative_predictive_value"),
            coverage=stats.get("coverage"),
            positive_capture=stats.get("positive_capture"),
            negative_capture=stats.get("negative_capture"),
            n_gold_positive=int(stats.get("n_gold_positive") or 0),
            n_gold_negative=int(stats.get("n_gold_negative") or 0),
            precision_ci_low=(ci.get("precision") or {}).get("low"),
            npv_ci_low=(ci.get("npv") or {}).get("low"),
            coverage_ci_low=(ci.get("coverage") or {}).get("low"),
            weak_pos_rate=weak_rate,
            expert_pos_rate=expert,
            small_n=bool(stats.get("small_n_flag")),
        )
    return out
