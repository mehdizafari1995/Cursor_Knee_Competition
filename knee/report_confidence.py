"""Target-specific positive/negative confidence policies with OOF selection.

Positive weak labels favour precision. Negative weak labels favour NPV.
Ambiguous reports stay unknown. Thresholds are never lowered globally just
to raise coverage. Thresholds used to score a gold study are never fit on
that same study.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .config import STUDY_ID_COLUMN, TARGET_KEYS
from .coverage_metrics import (
    apply_confidence_gates,
    coverage_confusion,
    coverage_error_curve,
)

DEFAULT_POS_GRID = tuple(np.round(np.linspace(0.50, 0.95, 10), 4))
DEFAULT_NEG_GRID = tuple(np.round(np.linspace(0.50, 0.95, 10), 4))


@dataclass(frozen=True)
class TargetPolicy:
    target_key: str
    positive_threshold: float
    negative_threshold: float
    positive_objective: str
    negative_objective: str
    n_train: int
    selected_on: str  # "oof_fold" | "full_gold_operational"


def _as_float_array(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def select_positive_threshold(
    gold: np.ndarray,
    pred: np.ndarray,
    confidence: np.ndarray,
    *,
    grid: Iterable[float] = DEFAULT_POS_GRID,
    min_precision: float = 0.85,
) -> float:
    """Highest threshold that meets min_precision. Does not chase coverage."""
    gold = _as_float_array(gold)
    pred = _as_float_array(pred)
    conf = _as_float_array(confidence)
    candidates: list[tuple[float, float, float]] = []
    for t in sorted(grid, reverse=True):
        gated = apply_confidence_gates(
            pred, conf, positive_threshold=float(t), negative_threshold=1.01
        )
        stats = coverage_confusion(gold, gated)
        prec = stats["precision"]
        n_pos = int(stats["n_positive_calls"])
        if n_pos <= 0 or prec != prec:
            continue
        if float(prec) + 1e-12 >= float(min_precision):
            candidates.append((float(t), float(prec), float(n_pos)))
    if not candidates:
        # No threshold met the precision floor: refuse all positive calls.
        return 1.01
    # Highest threshold first (already sorted); among equals, higher precision.
    candidates.sort(key=lambda row: (-row[0], -row[1]))
    return candidates[0][0]


def select_negative_threshold(
    gold: np.ndarray,
    pred: np.ndarray,
    confidence: np.ndarray,
    *,
    grid: Iterable[float] = DEFAULT_NEG_GRID,
    min_npv: float = 0.85,
) -> float:
    """Highest threshold that meets min_npv. Does not chase coverage."""
    gold = _as_float_array(gold)
    pred = _as_float_array(pred)
    conf = _as_float_array(confidence)
    candidates: list[tuple[float, float, float]] = []
    for t in sorted(grid, reverse=True):
        gated = apply_confidence_gates(
            pred, conf, positive_threshold=1.01, negative_threshold=float(t)
        )
        stats = coverage_confusion(gold, gated)
        npv = stats["npv"]
        n_neg = int(stats["n_negative_calls"])
        if n_neg <= 0 or npv != npv:
            continue
        if float(npv) + 1e-12 >= float(min_npv):
            candidates.append((float(t), float(npv), float(n_neg)))
    if not candidates:
        return 1.01
    candidates.sort(key=lambda row: (-row[0], -row[1]))
    return candidates[0][0]


def _fold_indices(n: int, n_folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Deterministic contiguous-strided folds. Works for tiny n (fixture/LOO-like)."""
    n_folds = int(max(2, min(n_folds, n)))
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for k in range(n_folds):
        test_mask = np.zeros(n, dtype=bool)
        test_mask[k::n_folds] = True
        test_idx = order[test_mask]
        train_idx = order[~test_mask]
        if test_idx.size == 0 or train_idx.size == 0:
            continue
        folds.append((train_idx, test_idx))
    return folds


def fit_oof_confidence_policies(
    gold: np.ndarray,
    pred: np.ndarray,
    confidence: np.ndarray,
    *,
    study_ids: list[str] | None = None,
    n_folds: int = 5,
    seed: int = 42,
    min_precision: float = 0.85,
    min_npv: float = 0.85,
    pos_grid: Iterable[float] = DEFAULT_POS_GRID,
    neg_grid: Iterable[float] = DEFAULT_NEG_GRID,
) -> dict[str, Any]:
    """Out-of-fold thresholds. Each study is scored with a policy fit without it."""
    gold = _as_float_array(gold)
    pred = _as_float_array(pred)
    conf = _as_float_array(confidence)
    n = gold.shape[0]
    ids = [str(x) for x in (study_ids if study_ids is not None else range(n))]
    oof_pred = np.full(n, np.nan, dtype=np.float64)
    oof_pos_t = np.full(n, np.nan, dtype=np.float64)
    oof_neg_t = np.full(n, np.nan, dtype=np.float64)
    fold_records: list[dict[str, Any]] = []
    for fold_i, (train_idx, test_idx) in enumerate(_fold_indices(n, n_folds, seed)):
        train_ids = [ids[i] for i in train_idx.tolist()]
        test_ids = [ids[i] for i in test_idx.tolist()]
        if set(train_ids) & set(test_ids):
            raise RuntimeError("threshold leakage: train/test study overlap")
        pos_t = select_positive_threshold(
            gold[train_idx], pred[train_idx], conf[train_idx],
            grid=pos_grid, min_precision=min_precision,
        )
        neg_t = select_negative_threshold(
            gold[train_idx], pred[train_idx], conf[train_idx],
            grid=neg_grid, min_npv=min_npv,
        )
        gated = apply_confidence_gates(
            pred[test_idx], conf[test_idx],
            positive_threshold=pos_t, negative_threshold=neg_t,
        )
        oof_pred[test_idx] = gated
        oof_pos_t[test_idx] = pos_t
        oof_neg_t[test_idx] = neg_t
        fold_records.append(
            {
                "fold": fold_i,
                "train_ids": train_ids,
                "test_ids": test_ids,
                "positive_threshold": float(pos_t),
                "negative_threshold": float(neg_t),
                "n_train": int(train_idx.size),
                "n_test": int(test_idx.size),
            }
        )
    operational_pos = select_positive_threshold(
        gold, pred, conf, grid=pos_grid, min_precision=min_precision
    )
    operational_neg = select_negative_threshold(
        gold, pred, conf, grid=neg_grid, min_npv=min_npv
    )
    leaked = apply_confidence_gates(
        pred, conf, positive_threshold=operational_pos, negative_threshold=operational_neg
    )
    return {
        "oof_pred": oof_pred,
        "oof_metrics": coverage_confusion(gold, oof_pred),
        "leaked_full_fit_pred": leaked,
        "leaked_full_fit_metrics": coverage_confusion(gold, leaked),
        "operational_positive_threshold": float(operational_pos),
        "operational_negative_threshold": float(operational_neg),
        "oof_positive_thresholds": oof_pos_t,
        "oof_negative_thresholds": oof_neg_t,
        "folds": fold_records,
        "n_folds": len(fold_records),
        "seed": int(seed),
        "min_precision": float(min_precision),
        "min_npv": float(min_npv),
        "note": (
            "Reported gold metrics use oof_pred. operational_* thresholds are "
            "fit on all gold studies and may be applied only to report-only "
            "weak labels, never as the gold performance estimate."
        ),
        "risk_coverage_positive": coverage_error_curve(
            gold, pred, conf, polarity="positive"
        ),
        "risk_coverage_negative": coverage_error_curve(
            gold, pred, conf, polarity="negative"
        ),
        "risk_coverage_joint": coverage_error_curve(
            gold, pred, conf, polarity="joint"
        ),
    }


def evaluate_oof_all_targets(
    gold_labels: np.ndarray,
    pred_labels: np.ndarray,
    confidence_by_target: dict[str, np.ndarray],
    study_ids: list[str],
    *,
    config: dict,
) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    """OOF policies for every target. Sanitized fold records omit study ids."""
    policies = policies_from_config(config)
    oof_cfg = ((config.get("confidence_policies") or {}).get("oof") or {})
    n_folds = int(oof_cfg.get("n_folds") or 5)
    seed = int(oof_cfg.get("seed") or 42)
    sanitized: dict[str, Any] = {}
    operational: dict[str, dict[str, float]] = {}
    for i, key in enumerate(TARGET_KEYS):
        pol = policies[key]
        result = fit_oof_confidence_policies(
            gold_labels[:, i],
            pred_labels[:, i],
            confidence_by_target[key],
            study_ids=study_ids,
            n_folds=n_folds,
            seed=seed + i,
            min_precision=pol["min_precision"],
            min_npv=pol["min_npv"],
        )
        assert_no_threshold_leakage(result)
        operational[key] = {
            "positive_threshold": result["operational_positive_threshold"],
            "negative_threshold": result["operational_negative_threshold"],
        }
        sanitized[key] = {
            "oof_metrics": result["oof_metrics"],
            "leaked_full_fit_metrics": result["leaked_full_fit_metrics"],
            "operational_positive_threshold": result["operational_positive_threshold"],
            "operational_negative_threshold": result["operational_negative_threshold"],
            "n_folds": result["n_folds"],
            "seed": result["seed"],
            "min_precision": result["min_precision"],
            "min_npv": result["min_npv"],
            "folds": [
                {
                    "fold": f["fold"],
                    "n_train": f["n_train"],
                    "n_test": f["n_test"],
                    "positive_threshold": f["positive_threshold"],
                    "negative_threshold": f["negative_threshold"],
                }
                for f in result["folds"]
            ],
            "risk_coverage_positive": result["risk_coverage_positive"],
            "risk_coverage_negative": result["risk_coverage_negative"],
            "note": result["note"],
        }
    return sanitized, operational


def policies_from_config(config: dict) -> dict[str, dict[str, float]]:
    block = (config.get("confidence_policies") or {})
    default = block.get("default") or {}
    min_precision = float(default.get("min_positive_precision", 0.85))
    min_npv = float(default.get("min_negative_npv", 0.85))
    per_target = block.get("targets") or {}
    out: dict[str, dict[str, float]] = {}
    for key in TARGET_KEYS:
        row = per_target.get(key) or {}
        out[key] = {
            "min_precision": float(row.get("min_positive_precision", min_precision)),
            "min_npv": float(row.get("min_negative_npv", min_npv)),
            "prior_positive_threshold": float(row.get("positive_threshold", 0.85)),
            "prior_negative_threshold": float(row.get("negative_threshold", 0.80)),
        }
    return out


def assert_no_threshold_leakage(oof_result: dict) -> None:
    for fold in oof_result.get("folds") or []:
        train = set(fold["train_ids"])
        test = set(fold["test_ids"])
        overlap = train & test
        if overlap:
            raise AssertionError(f"threshold leakage: {sorted(overlap)[:8]}")
        if not train or not test:
            raise AssertionError("empty train or test fold in OOF threshold selection")
