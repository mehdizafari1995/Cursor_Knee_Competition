"""Prepared gold-58 three-plane fusion OOF protocol. Not authorized to execute."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

from .config import PROJECT_ROOT
from .gold58_oof import (
    EXPECTED_LABELLED_REAL,
    IMAGENET_SHA256,
    SERIES_RULE_V2,
    Gold58PreflightError,
    normalize_series_description,
)
from .kaggle_bundle import sha256_file
from .phase4d_readiness import SECOND_READER_SCORE_BEARING_ALLOWED
from .phase4f_ssl import (
    FROZEN_V2_CONFIG_SHA256,
    GOLD58_SSL_OOF_EXECUTION_ALLOWED,
    UNLABELED_IMAGE_SSL_EXECUTION_ALLOWED,
)
from .report_labels import (
    HARD_WEAK_LABEL_TRAINING_ALLOWED,
    SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED,
)
from .second_reader_phase46 import FORBIDDEN_SEMANTIC_PASS, git_head_sha

PROTOCOL = "gold58-image-resnet18-3plane-oof-v1"
KERNEL_SLUG = "mahdimzafari/rsna-knee-gold58-3plane-oof-v1"
SERIES_RULE_3PLANE = "deterministic-one-series-per-plane-v1"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "gold58_image_resnet18_3plane_oof_v1.yaml"
PLANES = ("sagittal", "coronal", "axial")
FROZEN_V2_MACRO_AUC = 0.6237496021463534
V2_WALL_SECONDS = 8168.106
V2_PEAK_GPU_BYTES = 404743680
GOLD58_3PLANE_OOF_EXECUTION_ALLOWED = False
STATUS_PREPARED = "GOLD58_3PLANE_PROTOCOL_PREPARED_NOT_LAUNCHED"
STATUS_NOT_AUTHORIZED = "GOLD58_3PLANE_OOF_NOT_AUTHORIZED"
STATUS_SERIES = "SERIES_SELECTION_PREFLIGHT_FAIL"

SSL_ENGINEERING_PATH_PARKED = True


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_3plane_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{cfg_path} must contain a mapping")
    payload["_config_path"] = str(cfg_path)
    payload["_config_sha256"] = sha256_file(cfg_path)
    return payload


def _rank_key(item: dict[str, Any]) -> tuple[int, int, str, str]:
    return (
        0 if item.get("fluid_sensitive") else 1,
        -int(item.get("n_readable") or 0),
        normalize_series_description(item.get("series_description")),
        str(item.get("series_uid") or ""),
    )


def select_one_series_per_plane(
    inventory: list[dict[str, Any]],
    study_ids: list[str],
    *,
    min_planes_required: int = 1,
) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    """One readable series per anatomical plane. Labels are not used."""
    by_study: dict[str, list[dict[str, Any]]] = {}
    n_unreadable = 0
    n_missing_dir = 0
    for item in inventory:
        if item.get("missing_dir"):
            n_missing_dir += 1
        n_unreadable += int(item.get("n_unreadable") or 0)
        if int(item.get("n_readable") or 0) < 1:
            continue
        by_study.setdefault(str(item["study"]), []).append(item)

    lookup: dict[str, dict[str, str]] = {}
    n_zero = 0
    plane_counts = Counter()
    n_planes_hist = Counter()
    for study in study_ids:
        readable = list(by_study.get(str(study)) or [])
        chosen: dict[str, str] = {}
        for plane in PLANES:
            rows = [item for item in readable if str(item.get("plane") or "") == plane]
            if not rows:
                continue
            winner = sorted(rows, key=_rank_key)[0]
            chosen[plane] = str(winner["series_uid"])
            plane_counts[plane] += 1
        n_planes = int(len(chosen))
        n_planes_hist[str(n_planes)] += 1
        if n_planes < int(min_planes_required):
            n_zero += 1
            continue
        lookup[str(study)] = chosen

    aggregates = {
        "series_selection_rule": SERIES_RULE_3PLANE,
        "n_required": int(len(study_ids)),
        "n_selected": int(len(lookup)),
        "n_below_min_planes": int(n_zero),
        "n_unreadable_instances": int(n_unreadable),
        "n_missing_series_dir": int(n_missing_dir),
        "selected_plane_counts": dict(sorted(plane_counts.items())),
        "n_planes_histogram": dict(sorted(n_planes_hist.items(), key=lambda kv: int(kv[0]))),
        "min_planes_required": int(min_planes_required),
        "planes": list(PLANES),
    }
    if n_zero or len(lookup) != len(study_ids):
        raise Gold58PreflightError(
            STATUS_SERIES,
            (
                f"{SERIES_RULE_3PLANE} did not cover every labelled study "
                f"(n_below_min_planes={n_zero}, n_selected={len(lookup)}, "
                f"n_required={len(study_ids)})"
            ),
        )
    return lookup, aggregates


def mean_fuse_encoder_embeddings(plane_embeddings: list[torch.Tensor]) -> torch.Tensor:
    if not plane_embeddings:
        raise Gold58PreflightError(STATUS_SERIES, "no_plane_embeddings_to_fuse")
    stacked = torch.stack(plane_embeddings, dim=0)
    return stacked.mean(dim=0)


def estimate_3plane_resources(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Estimate from frozen v2 wall only. Does not inspect private rows."""
    cfg = cfg or load_3plane_config()
    safety = float(cfg.get("safety_factor") or 1.25)
    reserve = float(cfg.get("shutdown_reserve_seconds") or 3600)
    limit = float(cfg.get("max_wall_seconds") or 39600)
    projected = float(V2_WALL_SECONDS) * 3.0
    guarded = projected * safety
    planned = guarded + reserve
    return {
        "comparator_protocol": "gold58-image-resnet18-grouped-oof-v2",
        "comparator_macro_roc_auc": FROZEN_V2_MACRO_AUC,
        "comparator_wall_seconds": V2_WALL_SECONDS,
        "comparator_peak_gpu_memory_bytes": V2_PEAK_GPU_BYTES,
        "plane_factor": 3.0,
        "projected_seconds": projected,
        "guarded_seconds": guarded,
        "planned_seconds_with_reserve": planned,
        "max_wall_seconds": limit,
        "fits_private_wall": bool(planned <= limit),
        "gpu": "Tesla T4",
        "n_labelled_studies": EXPECTED_LABELLED_REAL,
        "formula": "3 * frozen_v2_wall * safety + shutdown_reserve",
        "execution_authorized": GOLD58_3PLANE_OOF_EXECUTION_ALLOWED,
    }


def render_3plane_protocol_markdown(cfg: dict[str, Any] | None = None) -> str:
    cfg = cfg or load_3plane_config()
    estimate = estimate_3plane_resources(cfg)
    return "\n".join(
        [
            "# Gold-58 image-only three-plane fusion OOF protocol v1",
            "",
            "Prepared after SimSiam SSL was parked. Not launched.",
            "Frozen v2 primary-series OOF remains the comparator.",
            "",
            f"- protocol: `{PROTOCOL}`",
            f"- kernel slug (not launched): `{KERNEL_SLUG}`",
            "- encoder: ImageNet ResNet-18; shared weights across planes",
            f"- ImageNet SHA-256: `{IMAGENET_SHA256}`",
            f"- planes: {', '.join(PLANES)}",
            f"- series rule: `{SERIES_RULE_3PLANE}` (one readable series per plane;",
            "  fluid-sensitive first, then DICOM count, description, SeriesInstanceUID)",
            "- missing plane: average available planes; min_planes_required=1",
            "- fusion: mean of per-plane encoder embeddings, then the existing head",
            "- slices/preprocess/optimizer/folds: copied from frozen v2, not retuned",
            f"- comparator: frozen `{SERIES_RULE_V2}` OOF macro AUC {FROZEN_V2_MACRO_AUC}",
            f"- comparator config SHA-256: `{FROZEN_V2_CONFIG_SHA256}`",
            "- leakage-safe validation: same 5-fold PatientID grouped OOF as v2",
            f"- resource estimate: projected {estimate['projected_seconds']:.1f} s;",
            f"  guarded {estimate['guarded_seconds']:.1f} s; planned with reserve "
            f"{estimate['planned_seconds_with_reserve']:.1f} s; "
            f"fits_private_wall={str(estimate['fits_private_wall']).lower()}",
            "- no report-only rows, no weak labels, no second reader, no hidden test,",
            "  no submission",
            "- GOLD58_3PLANE_OOF_EXECUTION_ALLOWED=false",
            "- UNLABELED_IMAGE_SSL_EXECUTION_ALLOWED=false",
            "- HARD_WEAK_LABEL_TRAINING_ALLOWED=false",
            f"- config SHA-256: `{cfg['_config_sha256']}`",
            f"- forbidden outcome not used: `{FORBIDDEN_SEMANTIC_PASS}`",
            "",
        ]
    )


def run_gold58_3plane_oof(
    *,
    execute_training: bool = False,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Refuse scientific execution. Local protocol tests may inspect selection helpers."""
    cfg = load_3plane_config(config_path)
    estimate = estimate_3plane_resources(cfg)
    payload = {
        "status": STATUS_NOT_AUTHORIZED if execute_training else STATUS_PREPARED,
        "ok": False,
        "skip_reason": "GOLD58_3PLANE_OOF_EXECUTION_ALLOWED=false",
        "protocol": PROTOCOL,
        "kernel_slug": KERNEL_SLUG,
        "GOLD58_3PLANE_OOF_EXECUTION_ALLOWED": GOLD58_3PLANE_OOF_EXECUTION_ALLOWED,
        "UNLABELED_IMAGE_SSL_EXECUTION_ALLOWED": UNLABELED_IMAGE_SSL_EXECUTION_ALLOWED,
        "GOLD58_SSL_OOF_EXECUTION_ALLOWED": GOLD58_SSL_OOF_EXECUTION_ALLOWED,
        "HARD_WEAK_LABEL_TRAINING_ALLOWED": HARD_WEAK_LABEL_TRAINING_ALLOWED,
        "SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED": SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED,
        "SECOND_READER_SCORE_BEARING_ALLOWED": SECOND_READER_SCORE_BEARING_ALLOWED,
        "SSL_ENGINEERING_PATH_PARKED": SSL_ENGINEERING_PATH_PARKED,
        "n_labelled_studies": 0,
        "n_report_only_studies_processed": 0,
        "n_report_text_rows_processed": 0,
        "n_second_reader_rows_processed": 0,
        "training_run": False,
        "OOF_run": False,
        "inference_run": False,
        "competition_submission_written": False,
        "consumes_submission_slot": False,
        "kaggle_kernel_launched": False,
        "resource_estimate": estimate,
        "comparator_macro_roc_auc": FROZEN_V2_MACRO_AUC,
        "frozen_v2_config_sha256": FROZEN_V2_CONFIG_SHA256,
        "config_sha256": cfg["_config_sha256"],
        "forbidden_outcome_not_used": FORBIDDEN_SEMANTIC_PASS,
        "git_head_sha": git_head_sha(),
        "utc": _utc(),
    }
    if execute_training or GOLD58_3PLANE_OOF_EXECUTION_ALLOWED:
        payload["status"] = STATUS_NOT_AUTHORIZED
        payload["ok"] = False
    return payload
