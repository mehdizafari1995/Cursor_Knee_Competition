"""Phase 4D.1 gold-58 image-only grouped OOF trainer.

Supervised image training on the 58 fully labelled competition-training studies.
Fail-closed preflight. Never uses reports, report-only rows, second-reader
outputs, weak 0/1 labels, or hidden-test images. Never weakens
``scripts/train_baseline.py``.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import yaml

from .baseline import load_checkpoint, save_checkpoint
from .baseline_model import BaselineStudyModel, TinySliceEncoder, build_baseline_model
from .config import (
    NUM_TARGETS,
    OFFICIAL_TARGET_COLUMNS,
    PROJECT_ROOT,
    SERIES_ID_COLUMN,
    STUDY_ID_COLUMN,
)
from .dataset import StudySliceDataset, collate_studies
from .dicom_io import list_dicom_paths, read_header
from .folds import FoldError, assert_no_group_leakage, assign_grouped_folds
from .kaggle_audit import jsonable
from .kaggle_bundle import sha256_file
from .kaggle_runtime import (
    DATA_SOURCE_REAL_KAGGLE,
    KaggleLayout,
    infer_data_source,
    is_kaggle_runtime,
    preflight,
)
from .labels import extract_label_matrix, prevalence
from .loss import masked_bce_with_logits
from .metrics import macro_roc_auc
from .phase4d_readiness import SECOND_READER_SCORE_BEARING_ALLOWED
from .report_labels import (
    HARD_WEAK_LABEL_TRAINING_ALLOWED,
    SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED,
    split_gold_and_report_only,
)
from .second_reader_phase46 import FORBIDDEN_SEMANTIC_PASS, git_head_sha
from .competition import SAMPLE_SUBMISSION_COLUMNS as OFFICIAL_SUBMISSION_COLUMNS
from .submission import align_to_sample, validate_submission, write_competition_submission

PROTOCOL = "gold58-image-resnet18-grouped-oof-v1"
PROTOCOL_V2 = "gold58-image-resnet18-grouped-oof-v2"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "gold58_image_resnet18_oof_v1.yaml"
DEFAULT_CONFIG_PATH_V2 = PROJECT_ROOT / "configs" / "gold58_image_resnet18_oof_v2.yaml"
KERNEL_SLUG = "mahdimzafari/rsna-knee-gold58-image-oof-v1"
KERNEL_SLUG_V2 = "mahdimzafari/rsna-knee-gold58-image-oof-v2"
EXPECTED_LABELLED_REAL = 58
IMAGENET_FILENAME = "resnet18-f37072fd.pth"
IMAGENET_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
SERIES_RULE = "sagittal-fluid-sensitive-unique-v1"
SERIES_RULE_V2 = "deterministic-primary-series-v2"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
STATUS_COMPLETE = "GOLD58_OOF_COMPLETE"
STATUS_GROUPING = "GROUPING_PREFLIGHT_FAIL"
STATUS_SERIES = "SERIES_SELECTION_PREFLIGHT_FAIL"
STATUS_WEIGHTS = "IMAGENET_WEIGHTS_PREFLIGHT_FAIL"
STATUS_FOLDS = "FOLD_SUPPORT_PREFLIGHT_FAIL"
STATUS_PREFLIGHT = "PREFLIGHT_FAIL"
ROUND_TRIP_ATOL = 1.0e-5

SKIP_WEIGHT_WALK = {
    "train_series",
    "test_series",
    "competitions",
    "rsna-knee-abnormality-detection",
    ".git",
    "__pycache__",
}


class Gold58PreflightError(RuntimeError):
    """Fail-closed preflight. Training must not start."""

    def __init__(self, status: str, reason: str):
        super().__init__(reason)
        self.status = status
        self.reason = reason


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_gold58_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{cfg_path} must contain a mapping")
    payload["_config_path"] = str(cfg_path)
    payload["_config_sha256"] = sha256_file(cfg_path)
    return payload


def set_deterministic_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _truthy(value: Any) -> bool:
    if isinstance(value, (int, float, np.integer, np.floating)):
        if not np.isfinite(float(value)):
            return False
        return int(value) == 1
    if isinstance(value, bool):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "t"}


def select_labelled_studies(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    labelled, report_only = split_gold_and_report_only(train)
    return labelled.copy(), report_only.copy()


def select_sagittal_fluid_sensitive_series(series: pd.DataFrame) -> pd.DataFrame:
    """Deterministic metadata rule. Does not use labels."""
    required = {STUDY_ID_COLUMN, SERIES_ID_COLUMN, "Anatomical_Plane", "Fluid_Sensitive"}
    missing = sorted(required - set(series.columns))
    if missing:
        raise Gold58PreflightError(
            STATUS_SERIES, f"train_series.csv missing columns: {missing}"
        )
    plane = series["Anatomical_Plane"].astype(str).str.strip().str.lower()
    fluid = series["Fluid_Sensitive"].map(_truthy)
    return series.loc[plane.eq("sagittal") & fluid].copy()


def unique_series_or_fail(
    candidates: pd.DataFrame, study_ids: list[str]
) -> dict[str, str]:
    """Exactly one sagittal fluid-sensitive series per requested study."""
    lookup: dict[str, str] = {}
    grouped = candidates.groupby(candidates[STUDY_ID_COLUMN].astype(str), sort=True)
    n_zero = 0
    n_multi = 0
    for study in study_ids:
        if study not in grouped.groups:
            n_zero += 1
            continue
        rows = grouped.get_group(study)
        if len(rows) != 1:
            n_multi += 1
            continue
        lookup[study] = str(rows.iloc[0][SERIES_ID_COLUMN])
    if n_zero or n_multi or len(lookup) != len(study_ids):
        raise Gold58PreflightError(
            STATUS_SERIES,
            (
                f"{SERIES_RULE} did not yield exactly one series per labelled "
                f"study (n_zero={n_zero}, n_multi={n_multi}, "
                f"n_unique={len(lookup)}, n_required={len(study_ids)})"
            ),
        )
    return lookup


def normalize_series_description(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _series_tier(plane: str, fluid: bool) -> int:
    sag = str(plane or "").strip().lower() == "sagittal"
    if sag and fluid:
        return 1
    if sag:
        return 2
    if fluid:
        return 3
    return 4


def _inspect_series_dir(series_dir: Path) -> dict[str, Any]:
    paths = list_dicom_paths(series_dir)
    n_readable = 0
    descriptions: list[str] = []
    patient_ids: list[str] = []
    n_unreadable = 0
    for path in paths:
        try:
            ds = read_header(path, stop_before_pixels=True)
        except Exception:
            n_unreadable += 1
            continue
        n_readable += 1
        descriptions.append(str(getattr(ds, "SeriesDescription", "") or ""))
        pid = str(getattr(ds, "PatientID", "") or "").strip()
        if pid:
            patient_ids.append(pid)
    desc = descriptions[0] if descriptions else ""
    return {
        "n_files": int(len(paths)),
        "n_readable": int(n_readable),
        "n_unreadable": int(n_unreadable),
        "series_description": desc,
        "patient_ids": patient_ids,
        "missing_dir": not series_dir.is_dir(),
    }


def inspect_series_inventory(
    series: pd.DataFrame, data_root: Path, *, split: str = "train"
) -> list[dict[str, Any]]:
    """Filesystem + metadata inventory. Does not use labels."""
    required = {STUDY_ID_COLUMN, SERIES_ID_COLUMN, "Anatomical_Plane", "Fluid_Sensitive"}
    missing = sorted(required - set(series.columns))
    if missing:
        raise Gold58PreflightError(
            STATUS_SERIES, f"series table missing columns: {missing}"
        )
    inventory: list[dict[str, Any]] = []
    studies = series[STUDY_ID_COLUMN].astype(str).tolist()
    series_uids = series[SERIES_ID_COLUMN].astype(str).tolist()
    planes = series["Anatomical_Plane"].astype(str).tolist()
    fluids = series["Fluid_Sensitive"].tolist()
    for study, series_uid, plane, fluid_raw in zip(studies, series_uids, planes, fluids):
        fluid = _truthy(fluid_raw)
        info = _inspect_series_dir(_series_dir(data_root, study, series_uid, split=split))
        inventory.append(
            {
                "study": study,
                "series_uid": series_uid,
                "plane": plane.strip().lower(),
                "fluid_sensitive": bool(fluid),
                "tier": _series_tier(plane, fluid),
                **info,
            }
        )
    return inventory


def select_deterministic_primary_series_v2(
    inventory: list[dict[str, Any]], study_ids: list[str]
) -> tuple[dict[str, str], dict[str, Any]]:
    """Rank readable series with the frozen v2 hierarchy. Labels are not used."""
    by_study: dict[str, list[dict[str, Any]]] = {}
    n_unreadable = 0
    n_missing_dir = 0
    for item in inventory:
        if item["missing_dir"]:
            n_missing_dir += 1
        n_unreadable += int(item["n_unreadable"])
        if int(item["n_readable"]) < 1:
            continue
        by_study.setdefault(item["study"], []).append(item)

    lookup: dict[str, str] = {}
    n_zero = 0
    tier_counts = Counter()
    tie_counts = Counter()
    dicom_counts: Counter[str] = Counter()
    plane_counts: Counter[str] = Counter()
    fluid_counts = Counter()
    for study in study_ids:
        readable = list(by_study.get(study) or [])
        if not readable:
            n_zero += 1
            continue
        best_tier = min(int(item["tier"]) for item in readable)
        tier_rows = [item for item in readable if int(item["tier"]) == best_tier]
        ranked = sorted(
            tier_rows,
            key=lambda item: (
                -int(item["n_readable"]),
                normalize_series_description(item["series_description"]),
                str(item["series_uid"]),
            ),
        )
        winner = ranked[0]
        lookup[study] = str(winner["series_uid"])
        tier_counts[str(best_tier)] += 1
        plane_counts[str(winner["plane"]) or "<empty>"] += 1
        fluid_counts["true" if winner["fluid_sensitive"] else "false"] += 1
        dicom_counts[str(int(winner["n_readable"]))] += 1
        if len(tier_rows) == 1:
            tie_counts["none"] += 1
        else:
            max_n = max(int(item["n_readable"]) for item in tier_rows)
            at_max = [item for item in tier_rows if int(item["n_readable"]) == max_n]
            if len(at_max) == 1:
                tie_counts["dicom_count"] += 1
            else:
                win_desc = normalize_series_description(winner["series_description"])
                same_desc = [
                    item
                    for item in at_max
                    if normalize_series_description(item["series_description"]) == win_desc
                ]
                if len(same_desc) == 1:
                    tie_counts["series_description"] += 1
                else:
                    tie_counts["series_uid"] += 1

    aggregates = {
        "series_selection_rule": SERIES_RULE_V2,
        "n_required": int(len(study_ids)),
        "n_selected": int(len(lookup)),
        "n_zero_readable": int(n_zero),
        "n_unreadable_instances": int(n_unreadable),
        "n_missing_series_dir": int(n_missing_dir),
        "n_tier_1": int(tier_counts.get("1", 0)),
        "n_tier_2": int(tier_counts.get("2", 0)),
        "n_tier_3": int(tier_counts.get("3", 0)),
        "n_tier_4": int(tier_counts.get("4", 0)),
        "n_tie_break_none": int(tie_counts.get("none", 0)),
        "n_tie_break_dicom_count": int(tie_counts.get("dicom_count", 0)),
        "n_tie_break_series_description": int(tie_counts.get("series_description", 0)),
        "n_tie_break_series_uid": int(tie_counts.get("series_uid", 0)),
        "selected_dicom_count_distribution": dict(sorted(dicom_counts.items(), key=lambda kv: int(kv[0]))),
        "selected_plane_counts": dict(sorted(plane_counts.items())),
        "selected_fluid_sensitive_counts": dict(sorted(fluid_counts.items())),
    }
    if n_zero or len(lookup) != len(study_ids):
        raise Gold58PreflightError(
            STATUS_SERIES,
            (
                f"{SERIES_RULE_V2} did not yield exactly one readable series per "
                f"labelled study (n_zero_readable={n_zero}, n_selected={len(lookup)}, "
                f"n_required={len(study_ids)})"
            ),
        )
    return lookup, aggregates


def collect_patient_tokens_across_readable_series(
    study_ids: list[str], inventory: list[dict[str, Any]]
) -> dict[str, str]:
    """PatientID from every readable series in the study, not only the chosen series."""
    by_study: dict[str, list[str]] = {study: [] for study in study_ids}
    for item in inventory:
        study = str(item["study"])
        if study not in by_study:
            continue
        if int(item["n_readable"]) < 1:
            continue
        by_study[study].extend(str(pid) for pid in item["patient_ids"] if str(pid).strip())

    tokens: dict[str, str] = {}
    n_missing = 0
    n_disagree = 0
    for study in study_ids:
        unique = sorted(set(by_study.get(study) or []))
        if not unique:
            n_missing += 1
            continue
        if len(unique) != 1:
            n_disagree += 1
            continue
        tokens[study] = patient_group_token(unique[0])
    if n_missing or n_disagree or len(tokens) != len(study_ids):
        raise Gold58PreflightError(
            STATUS_GROUPING,
            (
                "PatientID incomplete or inconsistent across readable series "
                f"(n_missing={n_missing}, n_disagree={n_disagree})"
            ),
        )
    return tokens


def assert_train_pos_neg_every_target(
    labels: np.ndarray, fold_ids: np.ndarray, n_folds: int
) -> dict[str, Any]:
    """v2 fold gate: every training partition has a known + and known - per target."""
    per_fold: dict[str, Any] = {}
    failures: list[str] = []
    for fold in range(n_folds):
        train_labels = labels[fold_ids != fold]
        fold_report: dict[str, Any] = {}
        for t, name in enumerate(OFFICIAL_TARGET_COLUMNS):
            col = train_labels[:, t]
            known = np.isfinite(col)
            n_pos = int((col[known] == 1).sum())
            n_neg = int((col[known] == 0).sum())
            fold_report[name] = {"n_positive": n_pos, "n_negative": n_neg}
            if n_pos < 1 or n_neg < 1:
                failures.append(f"fold{fold}:{name}:pos={n_pos}:neg={n_neg}")
        per_fold[str(fold)] = fold_report
    if failures:
        n_fail = len(failures)
        raise Gold58PreflightError(
            STATUS_FOLDS,
            (
                "every training partition must contain at least one known positive "
                f"and one known negative for every target (n_failing_target_folds={n_fail})"
            ),
        )
    return {"n_failing_target_folds": 0, "train_class_support": per_fold}


def patient_group_token(patient_id: str) -> str:
    return hashlib.sha256(patient_id.encode("utf-8")).hexdigest()


def read_patient_id_from_series(series_dir: Path) -> str:
    paths = list_dicom_paths(series_dir)
    if not paths:
        return ""
    ds = read_header(paths[0], stop_before_pixels=True)
    return str(getattr(ds, "PatientID", "") or "").strip()


def collect_patient_tokens(
    study_ids: list[str],
    series_lookup: dict[str, str],
    data_root: Path,
) -> dict[str, str]:
    tokens: dict[str, str] = {}
    missing = 0
    for study in study_ids:
        series_uid = series_lookup[study]
        series_dir = data_root / "train_series" / study / series_uid
        patient_id = read_patient_id_from_series(series_dir)
        if not patient_id:
            missing += 1
            continue
        tokens[study] = patient_group_token(patient_id)
    if missing or len(tokens) != len(study_ids):
        raise Gold58PreflightError(
            STATUS_GROUPING,
            f"PatientID incomplete for labelled studies (n_missing={missing})",
        )
    return tokens


def studies_per_patient_distribution(tokens: dict[str, str]) -> dict[str, int]:
    counts = Counter(Counter(tokens.values()).values())
    return {str(k): int(v) for k, v in sorted(counts.items())}


def locate_imagenet_weights(explicit: str | Path | None = None) -> Path | None:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    env = os.environ.get("GOLD58_IMAGENET_WEIGHTS")
    if env:
        candidates.append(Path(env))
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        for top in sorted(kaggle_input.iterdir()):
            if not top.is_dir() or top.name in SKIP_WEIGHT_WALK:
                continue
            for dirpath, dirnames, filenames in os.walk(top):
                dirnames[:] = [name for name in dirnames if name not in SKIP_WEIGHT_WALK]
                if IMAGENET_FILENAME in filenames:
                    candidates.append(Path(dirpath) / IMAGENET_FILENAME)
    local_default = PROJECT_ROOT / "outputs" / "imagenet_resnet18" / IMAGENET_FILENAME
    candidates.append(local_default)
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.is_file():
            return path
    return None


def verify_imagenet_weights(path: Path, expected_sha256: str = IMAGENET_SHA256) -> str:
    digest = sha256_file(path)
    if digest != expected_sha256:
        raise Gold58PreflightError(
            STATUS_WEIGHTS,
            f"ImageNet weight SHA-256 mismatch: got {digest} expected {expected_sha256}",
        )
    return digest


def build_resnet18_from_verified_file(path: Path) -> BaselineStudyModel:
    from torchvision.models import resnet18
    from torch import nn

    verify_imagenet_weights(path)
    backbone = resnet18(weights=None)
    state = torch.load(path, map_location="cpu")
    backbone.load_state_dict(state)
    backbone.fc = nn.Identity()

    class _VerifiedEncoder(nn.Module):
        def __init__(self, net: nn.Module):
            super().__init__()
            self.backbone = net
            self.out_dim = 512

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.size(1) == 1:
                x = x.repeat(1, 3, 1, 1)
            return self.backbone(x)

    return BaselineStudyModel(_VerifiedEncoder(backbone), num_targets=NUM_TARGETS, aggregation="mean")


def _series_dir(data_root: Path, study: str, series: str, split: str = "train") -> Path:
    if split not in {"train", "test"}:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    return Path(data_root) / f"{split}_series" / study / series


def build_folds(
    study_ids: list[str],
    tokens: dict[str, str],
    labels: np.ndarray,
    *,
    n_folds: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if n_folds != 5:
        raise Gold58PreflightError(STATUS_FOLDS, "n_folds is frozen at 5")
    group_ids = np.array([tokens[s] for s in study_ids], dtype=object)
    n_groups = int(len(set(tokens.values())))
    if n_groups < n_folds:
        raise Gold58PreflightError(
            STATUS_FOLDS,
            f"n_unique_patients={n_groups} is fewer than n_folds={n_folds}",
        )
    try:
        fold_ids, metadata = assign_grouped_folds(
            np.array(study_ids, dtype=object),
            group_ids,
            labels,
            n_folds=n_folds,
            seed=seed,
        )
    except FoldError as exc:
        raise Gold58PreflightError(STATUS_FOLDS, str(exc)) from exc
    table = pd.DataFrame(
        {
            STUDY_ID_COLUMN: study_ids,
            "group_id": group_ids,
            "fold": fold_ids,
        }
    )
    assert_no_group_leakage(table)
    offenders = table.groupby("group_id")["fold"].nunique()
    if int((offenders > 1).sum()) != 0:
        raise Gold58PreflightError(STATUS_GROUPING, "patient crossed folds")
    for fold in range(n_folds):
        n_val = int((fold_ids == fold).sum())
        n_train = int((fold_ids != fold).sum())
        if n_val < 1 or n_train < 1:
            raise Gold58PreflightError(
                STATUS_FOLDS, f"fold {fold} has n_train={n_train} n_val={n_val}"
            )
        train_labels = labels[fold_ids != fold]
        n_targets_both = 0
        for t in range(NUM_TARGETS):
            col = train_labels[:, t]
            known = np.isfinite(col)
            if known.sum() == 0:
                continue
            if col[known].min() != col[known].max():
                n_targets_both += 1
        if n_targets_both == 0:
            raise Gold58PreflightError(
                STATUS_FOLDS,
                f"fold {fold} train set has no two-class target",
            )
    metadata["group_source"] = "PatientID"
    metadata["n_unique_patients"] = n_groups
    metadata["allow_study_id_grouping_fallback"] = False
    return table, metadata


def fold_class_counts(labels: np.ndarray, fold_ids: np.ndarray, n_folds: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for fold in range(n_folds):
        mask = fold_ids == fold
        subset = labels[mask]
        per_target = {}
        for i, name in enumerate(OFFICIAL_TARGET_COLUMNS):
            col = subset[:, i]
            known = np.isfinite(col)
            per_target[name] = {
                "n_positive": int((col[known] == 1).sum()),
                "n_negative": int((col[known] == 0).sum()),
                "n_unknown": int((~known).sum()),
            }
        out[str(fold)] = {
            "n_studies": int(mask.sum()),
            "per_target": per_target,
        }
    return out


def patient_bootstrap_aucs(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    group_ids: np.ndarray,
    *,
    n_resamples: int,
    seed: int,
    ci_level: float,
) -> dict[str, Any]:
    groups = np.asarray(group_ids, dtype=object)
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    macros: list[float] = []
    per_target_draws: list[list[float]] = [[] for _ in range(NUM_TARGETS)]
    index = {g: np.where(groups == g)[0] for g in unique}
    for _ in range(n_resamples):
        drawn = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([index[g] for g in drawn])
        macro, per = macro_roc_auc(y_true[rows], y_prob[rows])
        if macro == macro:
            macros.append(float(macro))
        for t, value in enumerate(per):
            if value == value:
                per_target_draws[t].append(float(value))
    alpha = (1.0 - ci_level) / 2.0

    def _ci(values: list[float]) -> tuple[float | None, float | None, int]:
        if len(values) < 2:
            return None, None, len(values)
        arr = np.asarray(values, dtype=np.float64)
        lo = float(np.quantile(arr, alpha))
        hi = float(np.quantile(arr, 1.0 - alpha))
        if not np.isfinite(lo) or not np.isfinite(hi):
            return None, None, len(values)
        return lo, hi, len(values)

    macro_lo, macro_hi, n_macro = _ci(macros)
    per_target_ci = {}
    for i, name in enumerate(OFFICIAL_TARGET_COLUMNS):
        lo, hi, n_ok = _ci(per_target_draws[i])
        per_target_ci[name] = {
            "ci_low": lo,
            "ci_high": hi,
            "n_finite_resamples": n_ok,
        }
    return {
        "method": "patient_percentile_bootstrap",
        "n_resamples": int(n_resamples),
        "seed": int(seed),
        "ci_level": float(ci_level),
        "macro_roc_auc_ci_low": macro_lo,
        "macro_roc_auc_ci_high": macro_hi,
        "n_finite_macro_resamples": n_macro,
        "per_target": per_target_ci,
    }


def assert_probabilities_ok(probs: np.ndarray) -> None:
    if not np.isfinite(probs).all():
        raise Gold58PreflightError(STATUS_PREFLIGHT, "non-finite probabilities")
    if (probs < 0).any() or (probs > 1).any():
        raise Gold58PreflightError(STATUS_PREFLIGHT, "probabilities outside [0, 1]")


def round_trip_predictions(
    model: torch.nn.Module,
    images: torch.Tensor,
    checkpoint_path: Path,
    device: torch.device,
    atol: float = ROUND_TRIP_ATOL,
) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        before = torch.sigmoid(model(images.to(device))).cpu().numpy()
    save_checkpoint(model, checkpoint_path)
    clone = build_baseline_model(smoke=True, encoder_name="tiny", pretrained=False)
    # Tiny clone is only valid when the live model is tiny. Scientific path reloads same class.
    if isinstance(model.encoder, TinySliceEncoder):
        restored = build_baseline_model(smoke=True, encoder_name="tiny", pretrained=False).to(device)
    else:
        raise Gold58PreflightError(
            STATUS_PREFLIGHT,
            "round-trip helper received a non-tiny encoder without reload factory",
        )
    load_checkpoint(restored, checkpoint_path, device)
    restored.eval()
    with torch.no_grad():
        after = torch.sigmoid(restored(images.to(device))).cpu().numpy()
    max_abs = float(np.max(np.abs(before - after)))
    return {
        "max_abs_diff": max_abs,
        "within_tolerance": bool(max_abs <= atol),
        "atol": float(atol),
        "n_rows": int(before.shape[0]),
    }


def round_trip_state_dict(
    model: torch.nn.Module,
    images: torch.Tensor,
    checkpoint_path: Path,
    device: torch.device,
    reload_fn,
    atol: float = ROUND_TRIP_ATOL,
) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        before = torch.sigmoid(model(images.to(device))).cpu().numpy()
    save_checkpoint(model, checkpoint_path)
    restored = reload_fn().to(device)
    load_checkpoint(restored, checkpoint_path, device)
    restored.eval()
    with torch.no_grad():
        after = torch.sigmoid(restored(images.to(device))).cpu().numpy()
    max_abs = float(np.max(np.abs(before - after)))
    return {
        "max_abs_diff": max_abs,
        "within_tolerance": bool(max_abs <= atol),
        "atol": float(atol),
        "n_rows": int(before.shape[0]),
    }


def submission_contract_from_oof(oof: pd.DataFrame, tmp_path: Path) -> dict[str, Any]:
    """Prove the writer accepts an OOF-shaped table. Not a competition submission."""
    sample = pd.DataFrame({STUDY_ID_COLUMN: oof[STUDY_ID_COLUMN].astype(str)})
    for column in OFFICIAL_TARGET_COLUMNS:
        sample[column] = 0.5
    sample = sample.loc[:, list(OFFICIAL_SUBMISSION_COLUMNS)]
    aligned = align_to_sample(oof, sample)
    validate_submission(aligned, sample)
    dest = tmp_path / "contract_check_not_a_submission.csv"
    write_competition_submission(oof, sample, dest)
    dest.unlink()
    return {
        "ok": True,
        "competition_submission_written": False,
        "columns": list(OFFICIAL_SUBMISSION_COLUMNS),
        "n_rows": int(len(aligned)),
    }


def _train_one_fold(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    use_amp: bool,
    seed: int,
) -> dict[str, Any]:
    set_deterministic_seeds(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    history: list[float] = []
    model.train()
    for _epoch in range(epochs):
        running = 0.0
        n_seen = 0
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(images)
                loss = masked_bce_with_logits(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach().cpu()) * images.size(0)
            n_seen += int(images.size(0))
        scheduler.step()
        history.append(running / max(n_seen, 1))
    return {"train_loss_last": history[-1] if history else None, "n_epochs": epochs}


@torch.no_grad()
def _predict_loader(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[list[str], np.ndarray, np.ndarray]:
    model.eval()
    ids: list[str] = []
    probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for batch in loader:
        images = batch["image"].to(device)
        logits = model(images)
        p = torch.sigmoid(logits).cpu().numpy()
        ids.extend(batch[STUDY_ID_COLUMN])
        probs.append(p)
        labels.append(batch["labels"].cpu().numpy())
    return ids, np.concatenate(probs, axis=0), np.concatenate(labels, axis=0)


def _base_payload(
    status: str,
    reason: str | None,
    extra: dict[str, Any] | None = None,
    *,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = cfg or {}
    payload: dict[str, Any] = {
        "protocol": str(cfg.get("protocol") or PROTOCOL),
        "status": status,
        "ok": status == STATUS_COMPLETE,
        "skip_reason": reason,
        "kernel_slug": str(cfg.get("kernel_slug") or KERNEL_SLUG),
        "HARD_WEAK_LABEL_TRAINING_ALLOWED": HARD_WEAK_LABEL_TRAINING_ALLOWED,
        "SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED": SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED,
        "SECOND_READER_SCORE_BEARING_ALLOWED": SECOND_READER_SCORE_BEARING_ALLOWED,
        "gold_second_reader_rerun_authorized": False,
        "forbidden_outcome_not_used": FORBIDDEN_SEMANTIC_PASS,
        "training_run": False,
        "inference_on_hidden_test": False,
        "kaggle_kernel_launched": False,
        "competition_submission_written": False,
        "notebook_07_executed": False,
        "notebook_09_executed": False,
        "notebook_10_executed": False,
        "n_report_only_studies_processed": 0,
        "use_second_reader": False,
        "use_weak_labels": False,
        "allow_study_id_grouping_fallback": False,
        "consumes_submission_slot": False,
        "utc": _utc(),
        "git_head_sha": git_head_sha(),
    }
    if extra:
        payload.update(extra)
    return jsonable(payload)


def series_description_counts(data_root: Path, series_lookup: dict[str, str]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    n_missing_dir = 0
    for study, series_uid in series_lookup.items():
        series_dir = _series_dir(data_root, study, series_uid)
        paths = list_dicom_paths(series_dir)
        if not paths:
            n_missing_dir += 1
            continue
        ds = read_header(paths[0], stop_before_pixels=True)
        desc = str(getattr(ds, "SeriesDescription", "") or "").strip() or "<empty>"
        counts[desc] += 1
    out = dict(counts.most_common())
    out["_n_missing_series_dir"] = int(n_missing_dir)
    return out


def run_gold58_image_oof(
    *,
    data_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    config_path: str | Path | None = None,
    weights_path: str | Path | None = None,
    execute_training: bool = False,
    allow_tiny_encoder_for_tests: bool = False,
    require_cuda: bool | None = None,
    layout: KaggleLayout | None = None,
    config_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fail-closed labelled-image OOF. Training is opt-in after preflight."""
    assert HARD_WEAK_LABEL_TRAINING_ALLOWED is False
    assert SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED is False
    assert SECOND_READER_SCORE_BEARING_ALLOWED is False

    cfg = load_gold58_config(config_path)
    if config_overrides:
        cfg.update(config_overrides)
    out_dir = Path(output_dir or (PROJECT_ROOT / "outputs" / cfg["output_dirname"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    layout = layout or KaggleLayout(output_dirname=str(cfg["output_dirname"]))
    pre = preflight(layout=layout, data_root=data_root, output_dir=out_dir)
    def _payload(status: str, reason: str | None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        return _base_payload(status, reason, extra, cfg=cfg)

    if not pre.ok or pre.data_root is None:
        payload = _payload(STATUS_PREFLIGHT, pre.message or "DATA_NOT_AVAILABLE")
        _write_public(out_dir, payload)
        return payload

    root = Path(pre.data_root)
    data_source = infer_data_source(root, layout=layout)
    train = pd.read_csv(root / "train.csv")
    series = pd.read_csv(root / "train_series.csv")
    rule = str(cfg.get("series_selection_rule") or SERIES_RULE)
    is_v2 = rule == SERIES_RULE_V2
    # Series rule is applied to metadata/files before joining labels.
    labelled, report_only = select_labelled_studies(train)
    study_ids = labelled[STUDY_ID_COLUMN].astype(str).tolist()
    n_labelled = len(study_ids)
    n_report_only_present = int(len(report_only))
    if data_source == DATA_SOURCE_REAL_KAGGLE and n_labelled != EXPECTED_LABELLED_REAL:
        payload = _payload(
            STATUS_PREFLIGHT,
            f"labelled_count_not_{EXPECTED_LABELLED_REAL}:{n_labelled}",
            extra={"n_labelled_studies": n_labelled, "data_source": data_source},
        )
        _write_public(out_dir, payload)
        return payload
    if n_labelled < 2:
        payload = _payload(STATUS_PREFLIGHT, f"too_few_labelled:{n_labelled}")
        _write_public(out_dir, payload)
        return payload

    series_aggregates: dict[str, Any] | None = None
    try:
        if is_v2:
            inventory = inspect_series_inventory(series, root)
            series_lookup, series_aggregates = select_deterministic_primary_series_v2(
                inventory, study_ids
            )
            tokens = collect_patient_tokens_across_readable_series(study_ids, inventory)
        else:
            candidates = select_sagittal_fluid_sensitive_series(series)
            series_lookup = unique_series_or_fail(candidates, study_ids)
            tokens = collect_patient_tokens(study_ids, series_lookup, root)
        labels, _mask = extract_label_matrix(labelled, require_all=True)
        fold_table, fold_meta = build_folds(
            study_ids,
            tokens,
            labels,
            n_folds=int(cfg["n_folds"]),
            seed=int(cfg["seed"]),
        )
        fold_support = None
        if is_v2 or bool(cfg.get("fold_require_pos_neg_every_target")):
            fold_support = assert_train_pos_neg_every_target(
                labels, fold_table["fold"].to_numpy(), int(cfg["n_folds"])
            )
    except Gold58PreflightError as exc:
        extra_fail: dict[str, Any] = {
            "n_labelled_studies": n_labelled,
            "n_report_only_studies_present": n_report_only_present,
            "n_report_only_studies_processed": 0,
            "data_source": data_source,
            "series_selection_rule": rule,
        }
        if series_aggregates:
            extra_fail["series_selection_aggregates"] = series_aggregates
        payload = _payload(exc.status, exc.reason, extra=extra_fail)
        _write_public(out_dir, payload)
        return payload

    prev = prevalence(labels)
    per_target_counts = {
        name: {
            "n_positive": int(prev[name]["n_positive"]),
            "n_negative": int(prev[name]["n_negative"]),
            "n_unknown": int(prev[name]["n_unknown"]),
        }
        for name in OFFICIAL_TARGET_COLUMNS
    }
    fold_ids = fold_table["fold"].to_numpy()
    n_patients = int(len(set(tokens.values())))
    sanitized_preflight = {
        "n_studies": n_labelled,
        "n_unique_patients": n_patients,
        "studies_per_patient_distribution": studies_per_patient_distribution(tokens),
        "per_target_counts": per_target_counts,
        "per_fold": fold_class_counts(labels, fold_ids, int(cfg["n_folds"])),
        "n_report_only_studies_present": n_report_only_present,
        "n_report_only_studies_processed": 0,
        "series_selection_rule": rule,
        "n_chosen_series": len(series_lookup),
        "series_description_counts": series_description_counts(root, series_lookup),
        "fold_metadata_public": {
            "n_folds": fold_meta["n_folds"],
            "seed": fold_meta["seed"],
            "n_studies": fold_meta["n_studies"],
            "n_groups": fold_meta["n_groups"],
            "impossible_full_coverage_targets": fold_meta["impossible_full_coverage_targets"],
            "group_source": "PatientID",
        },
    }
    if series_aggregates:
        sanitized_preflight["series_selection_aggregates"] = series_aggregates
    if fold_support:
        sanitized_preflight["train_class_support_ok"] = True

    scientific = bool(execute_training) and not allow_tiny_encoder_for_tests
    cuda_needed = cfg.get("require_cuda_for_scientific_run", True) if require_cuda is None else require_cuda
    if scientific and cuda_needed and not torch.cuda.is_available():
        payload = _payload(
            STATUS_PREFLIGHT,
            "cuda_required",
            extra={"preflight_aggregates": sanitized_preflight, "data_source": data_source},
        )
        _write_public(out_dir, payload)
        return payload

    weights_file = locate_imagenet_weights(weights_path)
    weight_sha = None
    if scientific:
        if weights_file is None:
            payload = _payload(
                STATUS_WEIGHTS,
                "imagenet_weights_missing",
                extra={"preflight_aggregates": sanitized_preflight},
            )
            _write_public(out_dir, payload)
            return payload
        try:
            weight_sha = verify_imagenet_weights(weights_file, str(cfg["imagenet_weights_sha256"]))
        except Gold58PreflightError as exc:
            payload = _payload(exc.status, exc.reason, extra={"preflight_aggregates": sanitized_preflight})
            _write_public(out_dir, payload)
            return payload

    if not execute_training:
        payload = _payload(
            "PREFLIGHT_PASS_TRAINING_NOT_EXECUTED",
            "execute_training=false",
            extra={
                "preflight_aggregates": sanitized_preflight,
                "data_source": data_source,
                "imagenet_weights_sha256": weight_sha,
                "config_sha256": cfg["_config_sha256"],
            },
        )
        payload["ok"] = True
        _write_public(out_dir, payload)
        return payload

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg.get("amp") == "cuda_only" and device.type == "cuda")
    set_deterministic_seeds(int(cfg["seed"]))
    private_dir = out_dir / "private"
    private_dir.mkdir(parents=True, exist_ok=True)

    oof_ids: list[str] = []
    oof_prob: list[np.ndarray] = []
    oof_true: list[np.ndarray] = []
    oof_groups: list[str] = []
    checkpoint_hashes: dict[str, str] = {}
    fold_losses: dict[str, Any] = {}
    round_trip = None

    def _reload_scientific():
        assert weights_file is not None
        return build_resnet18_from_verified_file(weights_file)

    for fold in range(int(cfg["n_folds"])):
        train_mask = fold_table["fold"].to_numpy() != fold
        val_mask = ~train_mask
        train_ids = [study_ids[i] for i, keep in enumerate(train_mask) if keep]
        val_ids = [study_ids[i] for i, keep in enumerate(val_mask) if keep]
        train_dirs = [_series_dir(root, s, series_lookup[s]) for s in train_ids]
        val_dirs = [_series_dir(root, s, series_lookup[s]) for s in val_ids]
        train_y = labels[train_mask]
        val_y = labels[val_mask]
        interpolation = str(cfg.get("interpolation") or "nearest")
        imagenet_normalize = bool(cfg.get("imagenet_normalize"))
        imagenet_mean = tuple(cfg.get("imagenet_mean") or IMAGENET_MEAN)
        imagenet_std = tuple(cfg.get("imagenet_std") or IMAGENET_STD)
        ds_kwargs = {
            "num_slices": int(cfg["num_slices"]),
            "image_size": int(cfg["image_size"]),
            "percentiles": tuple(cfg["intensity_percentiles"]),
            "interpolation": interpolation,
            "imagenet_normalize": imagenet_normalize,
            "imagenet_mean": imagenet_mean,
            "imagenet_std": imagenet_std,
        }
        train_ds = StudySliceDataset(
            train_ids,
            train_dirs,
            train_y,
            **ds_kwargs,
        )
        val_ds = StudySliceDataset(
            val_ids,
            val_dirs,
            val_y,
            **ds_kwargs,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=int(cfg["batch_size"]),
            shuffle=True,
            collate_fn=collate_studies,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=int(cfg["batch_size"]),
            shuffle=False,
            collate_fn=collate_studies,
            num_workers=0,
        )
        if allow_tiny_encoder_for_tests:
            in_channels = 3 if bool(cfg.get("imagenet_normalize")) else 1
            model = BaselineStudyModel(
                TinySliceEncoder(in_channels=in_channels),
                num_targets=NUM_TARGETS,
                aggregation=str(cfg.get("aggregation") or "mean"),
            ).to(device)
        else:
            model = _reload_scientific().to(device)
        stats = _train_one_fold(
            model,
            train_loader,
            epochs=int(cfg["epochs"]),
            lr=float(cfg["learning_rate"]),
            weight_decay=float(cfg["weight_decay"]),
            device=device,
            use_amp=use_amp,
            seed=int(cfg["seed"]) + fold,
        )
        fold_losses[str(fold)] = stats
        ckpt = private_dir / f"fold{fold}_last.pt"
        save_checkpoint(model, ckpt)
        checkpoint_hashes[f"fold{fold}"] = sha256_file(ckpt)
        ids, probs, trues = _predict_loader(model, val_loader, device)
        oof_ids.extend(ids)
        oof_prob.append(probs)
        oof_true.append(trues)
        oof_groups.extend([tokens[s] for s in ids])
        if round_trip is None:
            sample_batch = next(iter(val_loader))
            reload_fn = (
                (
                    lambda: BaselineStudyModel(
                        TinySliceEncoder(in_channels=3 if bool(cfg.get("imagenet_normalize")) else 1),
                        num_targets=NUM_TARGETS,
                        aggregation=str(cfg.get("aggregation") or "mean"),
                    )
                )
                if allow_tiny_encoder_for_tests
                else _reload_scientific
            )
            round_trip = round_trip_state_dict(
                model,
                sample_batch["image"],
                private_dir / "round_trip.pt",
                device,
                reload_fn,
                atol=float(cfg["round_trip_atol"]),
            )
            if not round_trip["within_tolerance"]:
                payload = _payload(
                    STATUS_PREFLIGHT,
                    f"round_trip_diff={round_trip['max_abs_diff']}",
                    extra={"round_trip": round_trip, "preflight_aggregates": sanitized_preflight},
                )
                _write_public(out_dir, payload)
                return payload

    y_prob = np.concatenate(oof_prob, axis=0)
    y_true = np.concatenate(oof_true, axis=0)
    assert_probabilities_ok(y_prob)
    if len(oof_ids) != n_labelled or len(set(oof_ids)) != n_labelled:
        payload = _payload(
            STATUS_PREFLIGHT,
            f"oof_count_mismatch:{len(oof_ids)}",
            extra={"preflight_aggregates": sanitized_preflight},
        )
        _write_public(out_dir, payload)
        return payload

    order = {sid: i for i, sid in enumerate(oof_ids)}
    ordered_idx = [order[s] for s in study_ids]
    y_prob = y_prob[ordered_idx]
    y_true = y_true[ordered_idx]
    groups = np.array([oof_groups[order[s]] for s in study_ids], dtype=object)

    macro, per_target = macro_roc_auc(y_true, y_prob)
    bootstrap = patient_bootstrap_aucs(
        y_true,
        y_prob,
        groups,
        n_resamples=int(cfg["bootstrap_n_resamples"]),
        seed=int(cfg["bootstrap_seed"]),
        ci_level=float(cfg["bootstrap_ci_level"]),
    )
    rng = np.random.default_rng(int(cfg["seed"]))
    perm = rng.permutation(len(y_true))
    shuffled_macro, _ = macro_roc_auc(y_true[perm], y_prob)

    oof_frame = pd.DataFrame({STUDY_ID_COLUMN: study_ids})
    for i, column in enumerate(OFFICIAL_TARGET_COLUMNS):
        oof_frame[column] = y_prob[:, i]
    oof_path = private_dir / "oof_predictions.csv"
    oof_frame.to_csv(oof_path, index=False)
    contract = submission_contract_from_oof(oof_frame, private_dir)

    per_target_report = {}
    for i, name in enumerate(OFFICIAL_TARGET_COLUMNS):
        per_target_report[name] = {
            "roc_auc": None if per_target[i] != per_target[i] else float(per_target[i]),
            **per_target_counts[name],
            **bootstrap["per_target"][name],
        }

    payload = _payload(
        STATUS_COMPLETE,
        None,
        extra={
            "data_source": data_source,
            "scientific_evaluation": not allow_tiny_encoder_for_tests,
            "training_run": True,
            "kaggle_runtime": is_kaggle_runtime(layout),
            "preflight_aggregates": sanitized_preflight,
            "config_sha256": cfg["_config_sha256"],
            "imagenet_weights_sha256": weight_sha,
            "imagenet_weights_filename": IMAGENET_FILENAME,
            "primary_metric": "macro_roc_auc_combined_oof",
            "macro_roc_auc": None if macro != macro else float(macro),
            "bootstrap": bootstrap,
            "per_target": per_target_report,
            "shuffled_label_macro_roc_auc_diagnostic_only": (
                None if shuffled_macro != shuffled_macro else float(shuffled_macro)
            ),
            "checkpoint_sha256": checkpoint_hashes,
            "oof_predictions_sha256": sha256_file(oof_path),
            "n_oof_rows": int(len(oof_frame)),
            "round_trip": round_trip,
            "submission_contract": contract,
            "fold_losses": fold_losses,
            "device": str(device),
            "amp": use_amp,
            "n_labelled_studies": n_labelled,
            "interpolation": str(cfg.get("interpolation") or "nearest"),
            "imagenet_normalize": bool(cfg.get("imagenet_normalize")),
        },
    )
    _write_public(out_dir, payload)
    return payload


def _write_public(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    public_dir = output_dir / "sanitized"
    public_dir.mkdir(parents=True, exist_ok=True)
    private_keys = {"oof_frame", "patient_tokens", "study_ids"}
    public = {k: v for k, v in payload.items() if k not in private_keys}
    (public_dir / "gold58_image_oof.json").write_text(
        json.dumps(jsonable(public), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def render_protocol_markdown(cfg: dict[str, Any] | None = None) -> str:
    cfg = cfg or load_gold58_config()
    return "\n".join(
        [
            "# Phase 4D.1 gold58 image-only grouped OOF protocol",
            "",
            "Predeclared before execution. No hyperparameter search.",
            "The second reader is not on this path.",
            "",
            f"- protocol: `{PROTOCOL}`",
            f"- kernel: `{KERNEL_SLUG}`",
            f"- encoder: ImageNet ResNet-18 (`{cfg['imagenet_weights_filename']}`)",
            f"- ImageNet SHA-256: `{cfg['imagenet_weights_sha256']}`",
            f"- slices: {cfg['num_slices']} linspace; resize {cfg['image_size']}; percentiles {cfg['intensity_percentiles']}",
            f"- augmentation: `{cfg['augmentation']}`",
            f"- optimizer: {cfg['optimizer']} lr={cfg['learning_rate']} weight_decay={cfg['weight_decay']}",
            f"- schedule: {cfg['lr_schedule']} T_max={cfg['lr_schedule_t_max_epochs']}",
            f"- epochs: {cfg['epochs']}; checkpoint: `{cfg['checkpoint_rule']}`",
            f"- loss: masked BCE; unknown stays NaN",
            f"- AMP: `{cfg['amp']}`; cudnn_deterministic: `{cfg['cudnn_deterministic']}`",
            f"- folds: {cfg['n_folds']}; seed {cfg['seed']}; PatientID required; no StudyInstanceUID fallback",
            f"- series rule: `{SERIES_RULE}` (exactly one sagittal fluid-sensitive series per study)",
            f"- primary metric: combined 58-row OOF macro ROC AUC (not a mean of fold AUCs)",
            f"- bootstrap: patient-level percentile, n={cfg['bootstrap_n_resamples']}, seed={cfg['bootstrap_seed']}",
            "- no report-only rows, no weak labels, no second reader, no hidden test, no submission",
            "",
        ]
    )


def render_protocol_markdown_v2(cfg: dict[str, Any] | None = None) -> str:
    cfg = cfg or load_gold58_config(DEFAULT_CONFIG_PATH_V2)
    return "\n".join(
        [
            "# Phase 4D.2 gold58 image-only grouped OOF protocol v2",
            "",
            "Predeclared before execution. No hyperparameter search.",
            "Does not edit v1 history. Frozen v1 kernel/config/report stay unchanged.",
            "The second reader is not on this path.",
            "",
            f"- protocol: `{PROTOCOL_V2}`",
            f"- kernel: `{KERNEL_SLUG_V2}`",
            f"- encoder: ImageNet ResNet-18 (`{cfg['imagenet_weights_filename']}`)",
            f"- ImageNet SHA-256: `{cfg['imagenet_weights_sha256']}`",
            f"- slices: {cfg['num_slices']} linspace; resize {cfg['image_size']} bilinear; percentiles {cfg['intensity_percentiles']}",
            f"- ImageNet normalize after RGB repeat: mean {cfg.get('imagenet_mean')} std {cfg.get('imagenet_std')}",
            f"- augmentation: `{cfg['augmentation']}`",
            f"- optimizer: {cfg['optimizer']} lr={cfg['learning_rate']} weight_decay={cfg['weight_decay']}",
            f"- schedule: {cfg['lr_schedule']} T_max={cfg['lr_schedule_t_max_epochs']}",
            f"- epochs: {cfg['epochs']}; checkpoint: `{cfg['checkpoint_rule']}`",
            f"- loss: masked BCE; unknown stays NaN",
            f"- AMP: `{cfg['amp']}`; cudnn_deterministic: `{cfg['cudnn_deterministic']}`",
            f"- folds: {cfg['n_folds']}; seed {cfg['seed']}; PatientID from all readable series; no StudyInstanceUID fallback",
            f"- series rule: `{SERIES_RULE_V2}` (tiered ranking; exactly one readable series per labelled study)",
            f"- fold gate: every training partition has known positive and known negative for all 12 targets",
            f"- primary metric: combined 58-row OOF macro ROC AUC (not a mean of fold AUCs)",
            f"- bootstrap: patient-level percentile, n={cfg['bootstrap_n_resamples']}, seed={cfg['bootstrap_seed']}",
            "- no report-only rows, no weak labels, no second reader, no hidden test, no submission",
            "",
        ]
    )
