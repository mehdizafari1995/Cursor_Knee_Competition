"""Coverage-aware gold evaluation for ternary (positive/negative/unknown) labels.

Unknown extractor outputs are abstentions, never false negatives. Unknown gold
is skipped, never imputed as negative. Conditional sensitivity among called
cases is never described as full-set sensitivity; full-set analogue is
positive_capture = TP / all gold positives.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

METRIC_NAMES_V1_CONDITIONAL = (
    "conditional_sensitivity",
    "conditional_specificity",
    "conditional_precision",
    "conditional_recall",
    "conditional_f1",
    "conditional_balanced_accuracy",
)

BOOTSTRAP_METRIC_NAMES = (
    "coverage",
    "precision",
    "npv",
    "positive_capture",
    "negative_capture",
)

FULL_METRIC_NAMES = BOOTSTRAP_METRIC_NAMES + (
    "error_rate_all_gold",
    "selective_accuracy",
    "selective_error",
    "conditional_sensitivity",
    "conditional_specificity",
)


def _safe_div(n: float, d: float) -> float:
    if d == 0:
        return float("nan")
    return n / d


def coverage_confusion(gold: np.ndarray, pred: np.ndarray) -> dict[str, int | float]:
    """Per-target coverage-aware confusion. Vectors are aligned studies.

    ``pred`` uses 1 / 0 / NaN. NaN is an abstention, not a negative.
    """
    gold = np.asarray(gold, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    gold_known = np.isfinite(gold)
    pred_known = np.isfinite(pred)
    covered = gold_known & pred_known
    g = gold[covered]
    p = pred[covered]
    tp = int(((g == 1) & (p == 1)).sum())
    tn = int(((g == 0) & (p == 0)).sum())
    fp = int(((g == 0) & (p == 1)).sum())
    fn = int(((g == 1) & (p == 0)).sum())
    n_gold_positive = int(((gold == 1) & gold_known).sum())
    n_gold_negative = int(((gold == 0) & gold_known).sum())
    n_gold_known = int(gold_known.sum())
    n_positive_calls = int((gold_known & (pred == 1)).sum())
    n_negative_calls = int((gold_known & (pred == 0)).sum())
    n_abstentions = int((gold_known & ~pred_known).sum())
    n_abstained_positives = int(((gold == 1) & gold_known & ~pred_known).sum())
    n_abstained_negatives = int(((gold == 0) & gold_known & ~pred_known).sum())
    n_covered = int(covered.sum())

    precision = _safe_div(tp, tp + fp)
    npv = _safe_div(tn, tn + fn)
    positive_capture = _safe_div(tp, n_gold_positive)
    negative_capture = _safe_div(tn, n_gold_negative)
    cond_sens = _safe_div(tp, tp + fn)
    cond_spec = _safe_div(tn, tn + fp)
    cond_recall = cond_sens
    cond_f1 = (
        _safe_div(2 * precision * cond_recall, precision + cond_recall)
        if precision == precision and cond_recall == cond_recall
        else float("nan")
    )
    cond_bal = (
        (cond_sens + cond_spec) / 2.0
        if cond_sens == cond_sens and cond_spec == cond_spec
        else float("nan")
    )
    return {
        "n_gold_positive": n_gold_positive,
        "n_gold_negative": n_gold_negative,
        "n_gold_known": n_gold_known,
        "n_gold_unknown": int((~gold_known).sum()),
        "n_positive_calls": n_positive_calls,
        "n_negative_calls": n_negative_calls,
        "n_abstentions": n_abstentions,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "n_abstained_positives": n_abstained_positives,
        "n_abstained_negatives": n_abstained_negatives,
        "n_covered": n_covered,
        "n_pred_unknown_on_gold_known": n_abstentions,
        "coverage": _safe_div(n_covered, n_gold_known) if n_gold_known else float("nan"),
        "positive_capture": positive_capture,
        "negative_capture": negative_capture,
        "error_rate_all_gold": _safe_div(fp + fn, n_gold_known),
        "selective_accuracy": _safe_div(tp + tn, n_covered),
        "selective_error": _safe_div(fp + fn, n_covered),
        "precision": precision,
        "positive_precision": precision,
        "npv": npv,
        "negative_predictive_value": npv,
        "conditional_sensitivity": cond_sens,
        "conditional_specificity": cond_spec,
        "conditional_precision": precision,
        "conditional_recall": cond_recall,
        "conditional_f1": cond_f1,
        "conditional_balanced_accuracy": cond_bal,
        # v1 aliases kept for comparison; they are conditional-on-call.
        "sensitivity": cond_sens,
        "specificity": cond_spec,
        "recall": cond_recall,
        "f1": cond_f1,
        "balanced_accuracy": cond_bal,
        "unknown_imputed_as_negative": False,
        "note": (
            "conditional_sensitivity is TP/(TP+FN) among called cases only. "
            "positive_capture is TP / all gold positives (full gold set). "
            "Do not treat conditional_sensitivity as recall on every gold-positive study."
        ),
    }


def bootstrap_coverage_intervals(
    gold: np.ndarray,
    pred: np.ndarray,
    *,
    n_resamples: int,
    seed: int,
    ci_level: float,
    metric_names: Iterable[str] = BOOTSTRAP_METRIC_NAMES,
) -> dict[str, dict[str, float]]:
    """Study-level percentile bootstrap for coverage-aware metrics."""
    rng = np.random.default_rng(seed)
    gold = np.asarray(gold, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    n = gold.shape[0]
    names = tuple(metric_names)
    samples = {name: [] for name in names}
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        stats = coverage_confusion(gold[idx], pred[idx])
        for name in names:
            value = stats.get(name)
            if value is not None and value == value:
                samples[name].append(float(value))
    alpha = (1.0 - ci_level) / 2.0
    out: dict[str, dict[str, float]] = {}
    for name in names:
        values = np.asarray(samples[name], dtype=np.float64)
        if values.size == 0:
            out[name] = {"low": float("nan"), "high": float("nan"), "n_valid_resamples": 0}
            continue
        out[name] = {
            "low": float(np.quantile(values, alpha)),
            "high": float(np.quantile(values, 1.0 - alpha)),
            "n_valid_resamples": int(values.size),
        }
    return out


def apply_confidence_gates(
    pred: np.ndarray,
    confidence: np.ndarray,
    *,
    positive_threshold: float,
    negative_threshold: float,
) -> np.ndarray:
    """Abstain unless the called polarity meets its own threshold."""
    pred = np.asarray(pred, dtype=np.float64).copy()
    conf = np.asarray(confidence, dtype=np.float64)
    pos_mask = pred == 1
    neg_mask = pred == 0
    pred[pos_mask & ~(conf >= float(positive_threshold))] = np.nan
    pred[neg_mask & ~(conf >= float(negative_threshold))] = np.nan
    return pred


def coverage_error_curve(
    gold: np.ndarray,
    pred: np.ndarray,
    confidence: np.ndarray,
    *,
    thresholds: Iterable[float] | None = None,
    polarity: str = "joint",
) -> list[dict[str, float]]:
    """Risk-coverage curve. Does not force coverage by lowering thresholds.

    ``polarity='joint'`` applies the same numeric cut to whichever polarity
    the extractor already called. Positive and negative policies should be
    selected separately via ``apply_confidence_gates``.
    """
    if thresholds is None:
        thresholds = tuple(np.round(np.linspace(0.35, 0.95, 13), 4))
    gold = np.asarray(gold, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    conf = np.asarray(confidence, dtype=np.float64)
    points: list[dict[str, float]] = []
    for t in thresholds:
        t = float(t)
        gated = pred.copy()
        if polarity == "positive":
            gated[pred == 1] = np.where(conf[pred == 1] >= t, 1.0, np.nan)
            gated[pred != 1] = np.nan
        elif polarity == "negative":
            gated[pred == 0] = np.where(conf[pred == 0] >= t, 0.0, np.nan)
            gated[pred != 0] = np.nan
        else:
            gated = apply_confidence_gates(
                pred, conf, positive_threshold=t, negative_threshold=t
            )
        stats = coverage_confusion(gold, gated)
        points.append(
            {
                "threshold": t,
                "coverage": float(stats["coverage"]) if stats["coverage"] == stats["coverage"] else float("nan"),
                "error_rate_all_gold": float(stats["error_rate_all_gold"])
                if stats["error_rate_all_gold"] == stats["error_rate_all_gold"]
                else float("nan"),
                "selective_error": float(stats["selective_error"])
                if stats["selective_error"] == stats["selective_error"]
                else float("nan"),
                "precision": float(stats["precision"]) if stats["precision"] == stats["precision"] else float("nan"),
                "npv": float(stats["npv"]) if stats["npv"] == stats["npv"] else float("nan"),
                "positive_capture": float(stats["positive_capture"])
                if stats["positive_capture"] == stats["positive_capture"]
                else float("nan"),
                "negative_capture": float(stats["negative_capture"])
                if stats["negative_capture"] == stats["negative_capture"]
                else float("nan"),
                "n_covered": int(stats["n_covered"]),
            }
        )
    return points


def macro_mean(per_target: dict[str, dict], names: Iterable[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in names:
        values = [row.get(name) for row in per_target.values()]
        finite = [float(v) for v in values if v is not None and v == v]
        out[name] = float(np.mean(finite)) if finite else float("nan")
    return out


def assert_not_full_set_sensitivity_claim(text: str) -> None:
    """Fail if a v2 artefact claims conditional sensitivity is full-set recall."""
    lowered = " ".join(text.lower().split())
    # Allow the protocol to *deny* the identification.
    stripped = lowered.replace("not full-set sensitivity", " ")
    stripped = stripped.replace("not full set sensitivity", " ")
    stripped = stripped.replace("is not full-set sensitivity", " ")
    stripped = stripped.replace("never describe conditional sensitivity as full-set", " ")
    forbidden = (
        "is full-set sensitivity",
        "as full-set sensitivity",
        "unconditional sensitivity",
    )
    for phrase in forbidden:
        if phrase in stripped:
            raise AssertionError(f"v2 artefact described conditional sensitivity as {phrase!r}")


def from_published_counts(
    *,
    tp: int,
    tn: int,
    fp: int,
    fn: int,
    n_gold_positive: int,
    n_gold_known: int,
) -> dict[str, int | float]:
    """Rebuild coverage-aware metrics from committed v1 aggregate counts."""
    n_gold_negative = int(n_gold_known - n_gold_positive)
    n_covered = int(tp + tn + fp + fn)
    n_abstentions = int(n_gold_known - n_covered)
    n_abstained_positives = int(n_gold_positive - tp - fn)
    n_abstained_negatives = int(n_gold_negative - tn - fp)
    gold = np.array([1.0] * n_gold_positive + [0.0] * n_gold_negative, dtype=np.float64)
    pred = np.full(n_gold_known, np.nan, dtype=np.float64)
    # Reconstruct a vector with the same confusion; order does not matter for these stats.
    i = 0
    pred[i : i + tp] = 1.0
    i += tp
    i_fn = i
    pred[i_fn : i_fn + fn] = 0.0
    i = i_fn + fn
    # remaining gold positives are abstained
    i = n_gold_positive
    pred[i : i + tn] = 0.0
    i += tn
    pred[i : i + fp] = 1.0
    stats = coverage_confusion(gold, pred)
    # Sanity against the published cells.
    assert int(stats["tp"]) == tp
    assert int(stats["tn"]) == tn
    assert int(stats["fp"]) == fp
    assert int(stats["fn"]) == fn
    assert int(stats["n_abstained_positives"]) == n_abstained_positives
    assert int(stats["n_abstained_negatives"]) == n_abstained_negatives
    assert int(stats["n_abstentions"]) == n_abstentions
    return stats


def json_safe_metrics(stats: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in stats.items():
        if isinstance(value, float) and value != value:
            out[key] = None
        else:
            out[key] = value
    return out
