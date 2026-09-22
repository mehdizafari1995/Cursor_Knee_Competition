"""Gold-58 three-plane fusion OOF. Frozen config; one authorized scientific kernel."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import yaml

from .baseline import load_checkpoint, save_checkpoint
from .baseline_model import BaselineStudyModel, TinySliceEncoder
from .config import NUM_TARGETS, OFFICIAL_TARGET_COLUMNS, PROJECT_ROOT, STUDY_ID_COLUMN
from .dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    _resize2d,
    apply_imagenet_normalization,
)
from .dicom_io import load_series_volume
from .gold58_oof import (
    EXPECTED_LABELLED_REAL,
    IMAGENET_SHA256,
    SERIES_RULE_V2,
    Gold58PreflightError,
    _series_dir,
    assert_probabilities_ok,
    assert_train_pos_neg_every_target,
    build_folds,
    build_resnet18_from_verified_file,
    collect_patient_tokens_across_readable_series,
    fold_class_counts,
    inspect_series_inventory,
    locate_imagenet_weights,
    normalize_series_description,
    patient_bootstrap_aucs,
    select_labelled_studies,
    set_deterministic_seeds,
    studies_per_patient_distribution,
    submission_contract_from_oof,
    verify_imagenet_weights,
)
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
from .phase4f_ssl import (
    FROZEN_V2_CONFIG_SHA256,
    GOLD58_SSL_OOF_EXECUTION_ALLOWED,
    UNLABELED_IMAGE_SSL_EXECUTION_ALLOWED,
)
from .preprocess import percentile_normalize, sample_slices
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
FROZEN_V2_OOF_SHA256 = "79e37e8ccab00de230cfc49069e3380c68710f616a896b057f7f3f2b20b4d896"
V2_WALL_SECONDS = 8168.106
V2_PEAK_GPU_BYTES = 404743680
GOLD58_3PLANE_OOF_EXECUTION_ALLOWED = True
STATUS_PREPARED = "GOLD58_3PLANE_PROTOCOL_PREPARED_NOT_LAUNCHED"
STATUS_NOT_AUTHORIZED = "GOLD58_3PLANE_OOF_NOT_AUTHORIZED"
STATUS_SERIES = "SERIES_SELECTION_PREFLIGHT_FAIL"
STATUS_COMPLETE = "GOLD58_3PLANE_OOF_COMPLETE"
STATUS_RUNTIME = "GOLD58_3PLANE_RUNTIME_PREFLIGHT_FAIL"
STATUS_TRAINING = "GOLD58_3PLANE_TRAINING_FAIL"
STATUS_VALIDATION = "GOLD58_3PLANE_VALIDATION_FAIL"
STATUS_FOLDS = "FOLD_SUPPORT_PREFLIGHT_FAIL"
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
    plane_counts: Counter[str] = Counter()
    n_planes_hist: Counter[str] = Counter()
    n_tie_break = 0
    for study in study_ids:
        readable = list(by_study.get(str(study)) or [])
        chosen: dict[str, str] = {}
        for plane in PLANES:
            rows = [item for item in readable if str(item.get("plane") or "") == plane]
            if not rows:
                continue
            ranked = sorted(rows, key=_rank_key)
            if len(ranked) > 1 and _rank_key(ranked[0])[:2] == _rank_key(ranked[1])[:2]:
                n_tie_break += 1
            winner = ranked[0]
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
        "n_tie_break_description_or_uid": int(n_tie_break),
        "min_planes_required": int(min_planes_required),
        "planes": list(PLANES),
        "mean_planes_per_study": (
            float(sum(int(k) * int(v) for k, v in n_planes_hist.items()) / max(len(study_ids), 1))
        ),
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


def fold_assignment_hash(table: pd.DataFrame) -> str:
    rows = table[[STUDY_ID_COLUMN, "fold"]].copy()
    rows[STUDY_ID_COLUMN] = rows[STUDY_ID_COLUMN].astype(str)
    rows = rows.sort_values(STUDY_ID_COLUMN, kind="mergesort")
    payload = json.dumps(rows.values.tolist(), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def locate_frozen_v2_oof(explicit: str | Path | None = None) -> Path | None:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))
    env = os.environ.get("GOLD58_V2_OOF_PATH")
    if env:
        candidates.append(Path(env))
    for base in (
        Path("/kaggle/working"),
        Path("/kaggle/input"),
        PROJECT_ROOT / "outputs",
    ):
        candidates.append(base / "rsna_gold58_image_oof_v2" / "private" / "oof_predictions.csv")
    for path in candidates:
        if path.is_file():
            return path
    return None


class ThreePlaneStudyDataset(Dataset):
    """Load one series per available plane. Missing planes are omitted, not zero-filled."""

    def __init__(
        self,
        study_ids: list[str],
        plane_dirs: list[dict[str, Path]],
        labels: np.ndarray | None,
        *,
        num_slices: int,
        image_size: int,
        percentiles: tuple[float, float],
        interpolation: str,
        imagenet_normalize: bool,
        imagenet_mean: tuple[float, float, float],
        imagenet_std: tuple[float, float, float],
    ):
        if len(study_ids) != len(plane_dirs):
            raise ValueError("study_ids and plane_dirs must match")
        self.study_ids = list(study_ids)
        self.plane_dirs = plane_dirs
        self.labels = None if labels is None else np.asarray(labels, dtype=np.float32)
        self.num_slices = int(num_slices)
        self.image_size = int(image_size)
        self.percentiles = percentiles
        self.interpolation = interpolation
        self.imagenet_normalize = bool(imagenet_normalize)
        self.imagenet_mean = tuple(float(x) for x in imagenet_mean)
        self.imagenet_std = tuple(float(x) for x in imagenet_std)

    def __len__(self) -> int:
        return len(self.study_ids)

    def _load_plane(self, series_dir: Path) -> torch.Tensor:
        volume, _headers = load_series_volume(series_dir)
        volume = percentile_normalize(volume, *self.percentiles)
        sampled = sample_slices(volume, self.num_slices)
        resized = np.stack(
            [
                _resize2d(slice_, self.image_size, interpolation=self.interpolation)
                for slice_ in sampled
            ],
            axis=0,
        )
        if self.imagenet_normalize:
            return torch.from_numpy(
                apply_imagenet_normalization(
                    resized, mean=self.imagenet_mean, std=self.imagenet_std
                )
            )
        return torch.from_numpy(resized[:, None, :, :])

    def __getitem__(self, idx: int):
        planes: dict[str, torch.Tensor | None] = {}
        for plane in PLANES:
            d = self.plane_dirs[idx].get(plane)
            planes[plane] = None if d is None else self._load_plane(d)
        if self.labels is None:
            target = torch.full((NUM_TARGETS,), float("nan"))
        else:
            target = torch.from_numpy(self.labels[idx].copy())
        return {
            STUDY_ID_COLUMN: self.study_ids[idx],
            "planes": planes,
            "labels": target,
        }


def collate_three_plane(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        STUDY_ID_COLUMN: [item[STUDY_ID_COLUMN] for item in batch],
        "labels": torch.stack([item["labels"] for item in batch], dim=0),
        "plane_images": {},
        "plane_mask": {},
    }
    for plane in PLANES:
        present = [item["planes"][plane] is not None for item in batch]
        ref = next((item["planes"][plane] for item in batch if item["planes"][plane] is not None), None)
        if ref is None:
            out["plane_images"][plane] = torch.zeros(len(batch), 1, 1, 1, 1)
            out["plane_mask"][plane] = torch.zeros(len(batch), dtype=torch.bool)
            continue
        stacked = []
        for item in batch:
            tensor = item["planes"][plane]
            stacked.append(torch.zeros_like(ref) if tensor is None else tensor)
        out["plane_images"][plane] = torch.stack(stacked, dim=0)
        out["plane_mask"][plane] = torch.tensor(present, dtype=torch.bool)
    return out


class ThreePlaneFusionModel(nn.Module):
    """Shared encoder, sequential per-plane forwards, mean of available embeddings."""

    def __init__(self, backbone: BaselineStudyModel, planes: tuple[str, ...] = PLANES):
        super().__init__()
        self.backbone = backbone
        self.planes = list(planes)

    def encode_volume(self, slices: torch.Tensor) -> torch.Tensor:
        batch, n_slices, channels, height, width = slices.shape
        flat = slices.reshape(batch * n_slices, channels, height, width)
        features = self.backbone.encoder(flat)
        features = features.reshape(batch, n_slices, -1)
        return features.mean(dim=1)

    def fuse_embeddings(
        self,
        plane_images: dict[str, torch.Tensor],
        plane_mask: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        present = [p for p in self.planes if int(plane_mask[p].sum()) > 0]
        if not present:
            raise Gold58PreflightError(STATUS_SERIES, "study_with_zero_planes_in_batch")
        ref = plane_images[present[0]]
        batch = int(ref.shape[0])
        dim = int(self.backbone.encoder.out_dim)
        fused = torch.zeros(batch, dim, device=ref.device, dtype=ref.dtype)
        counts = torch.zeros(batch, 1, device=ref.device, dtype=ref.dtype)
        for plane in self.planes:
            mask = plane_mask[plane].to(device=ref.device).bool()
            if not bool(mask.any()):
                continue
            emb = self.encode_volume(plane_images[plane][mask].to(device=ref.device))
            fused = fused.clone()
            fused[mask] = fused[mask] + emb
            counts[mask] = counts[mask] + 1
        if bool((counts < 1).any()):
            raise Gold58PreflightError(STATUS_SERIES, "study_with_zero_planes_in_batch")
        return fused / counts.clamp(min=1)

    def forward(
        self,
        plane_images: dict[str, torch.Tensor],
        plane_mask: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        fused = self.fuse_embeddings(plane_images, plane_mask)
        return self.backbone.head(fused)


def classifier_has_batchnorm1d(model: ThreePlaneFusionModel) -> bool:
    for module in model.backbone.head.modules():
        if isinstance(module, nn.BatchNorm1d):
            return True
    return False


def assert_classifier_has_no_batchnorm1d(model: ThreePlaneFusionModel) -> None:
    if classifier_has_batchnorm1d(model):
        raise Gold58PreflightError(STATUS_TRAINING, "classifier_path_contains_batchnorm1d")


def train_loader_drop_last(*, n_items: int, batch_size: int, has_bn1d: bool) -> bool:
    """Drop a remainder-1 train batch only when a BatchNorm1d classifier exists."""
    if not has_bn1d:
        return False
    size = max(int(batch_size), 1)
    return int(n_items) % size == 1


def validate_oof_table(
    frame: pd.DataFrame,
    *,
    expected_n: int,
    expected_ids: list[str] | None = None,
) -> dict[str, Any]:
    if STUDY_ID_COLUMN not in frame.columns:
        raise Gold58PreflightError(STATUS_VALIDATION, "oof_missing_study_id_column")
    ids = frame[STUDY_ID_COLUMN].astype(str).tolist()
    if len(ids) != int(expected_n):
        raise Gold58PreflightError(STATUS_VALIDATION, f"oof_count_mismatch:{len(ids)}")
    if len(set(ids)) != int(expected_n):
        raise Gold58PreflightError(STATUS_VALIDATION, "oof_duplicate_study_ids")
    missing = [column for column in OFFICIAL_TARGET_COLUMNS if column not in frame.columns]
    if missing:
        raise Gold58PreflightError(STATUS_VALIDATION, "oof_missing_target_columns")
    if expected_ids is not None and ids != [str(item) for item in expected_ids]:
        raise Gold58PreflightError(STATUS_VALIDATION, "oof_row_order_mismatch")
    probs = frame.loc[:, list(OFFICIAL_TARGET_COLUMNS)].to_numpy(dtype=np.float64)
    assert_probabilities_ok(probs)
    return {
        "ok": True,
        "n_rows": int(len(ids)),
        "n_duplicates": 0,
        "n_targets": int(len(OFFICIAL_TARGET_COLUMNS)),
        "finite": True,
        "unit_interval": True,
        "shape": [int(len(ids)), int(len(OFFICIAL_TARGET_COLUMNS))],
    }


def _build_backbone(
    *,
    tiny: bool,
    weights_path: Path | None,
    imagenet_normalize: bool,
    aggregation: str,
) -> BaselineStudyModel:
    if tiny:
        in_channels = 3 if imagenet_normalize else 1
        return BaselineStudyModel(
            TinySliceEncoder(in_channels=in_channels),
            num_targets=NUM_TARGETS,
            aggregation=aggregation,
        )
    assert weights_path is not None
    return build_resnet18_from_verified_file(weights_path)


def _train_one_fold(
    model: ThreePlaneFusionModel,
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
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if hasattr(torch, "amp") else torch.cuda.amp.GradScaler(enabled=use_amp)
    history: list[float] = []
    model.train()
    for _epoch in range(epochs):
        running = 0.0
        n_seen = 0
        for batch in loader:
            labels = batch["labels"].to(device)
            plane_images = {p: t.to(device) for p, t in batch["plane_images"].items()}
            plane_mask = {p: m.to(device) for p, m in batch["plane_mask"].items()}
            optimizer.zero_grad(set_to_none=True)
            autocast = torch.amp.autocast("cuda", enabled=use_amp) if hasattr(torch, "amp") else torch.cuda.amp.autocast(enabled=use_amp)
            with autocast:
                logits = model(plane_images, plane_mask)
                loss = masked_bce_with_logits(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach().cpu()) * labels.size(0)
            n_seen += int(labels.size(0))
        scheduler.step()
        history.append(running / max(n_seen, 1))
    return {"train_loss_last": history[-1] if history else None, "n_epochs": epochs}


@torch.no_grad()
def _predict_loader(
    model: ThreePlaneFusionModel, loader: DataLoader, device: torch.device
) -> tuple[list[str], np.ndarray, np.ndarray]:
    model.eval()
    ids: list[str] = []
    probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for batch in loader:
        plane_images = {p: t.to(device) for p, t in batch["plane_images"].items()}
        plane_mask = {p: m.to(device) for p, m in batch["plane_mask"].items()}
        logits = model(plane_images, plane_mask)
        p = torch.sigmoid(logits).cpu().numpy()
        ids.extend(batch[STUDY_ID_COLUMN])
        probs.append(p)
        labels.append(batch["labels"].cpu().numpy())
    return ids, np.concatenate(probs, axis=0), np.concatenate(labels, axis=0)


def _dummy_batch(cfg: dict[str, Any], n: int = 2) -> dict[str, Any]:
    image_size = int(cfg.get("image_size") or 224)
    num_slices = int(cfg.get("num_slices") or 8)
    channels = 3 if bool(cfg.get("imagenet_normalize")) else 1
    gen = torch.Generator().manual_seed(int(cfg.get("seed") or 42))
    plane_images = {}
    plane_mask = {}
    for plane in PLANES:
        plane_images[plane] = torch.rand(n, num_slices, channels, image_size, image_size, generator=gen)
        plane_mask[plane] = torch.ones(n, dtype=torch.bool)
    labels = torch.randint(0, 2, (n, NUM_TARGETS)).float()
    labels[0, 0] = float("nan")
    return {"plane_images": plane_images, "plane_mask": plane_mask, "labels": labels}


def run_runtime_preflight(
    model: ThreePlaneFusionModel,
    cfg: dict[str, Any],
    *,
    device: torch.device,
    use_amp: bool,
    series_aggregates: dict[str, Any],
    n_labelled: int,
    n_folds: int,
    skip_wall_gate: bool = False,
    reload_factory=None,
    checkpoint_dir: Path | None = None,
) -> dict[str, Any]:
    assert_classifier_has_no_batchnorm1d(model)
    model.train()
    dummy = _dummy_batch(cfg, n=min(2, int(cfg.get("batch_size") or 2)))
    plane_images = {p: t.to(device) for p, t in dummy["plane_images"].items()}
    plane_mask = {p: m.to(device) for p, m in dummy["plane_mask"].items()}
    labels = dummy["labels"].to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    autocast = torch.amp.autocast("cuda", enabled=use_amp) if hasattr(torch, "amp") else torch.cuda.amp.autocast(enabled=use_amp)
    with autocast:
        logits = model(plane_images, plane_mask)
        loss = masked_bce_with_logits(logits, labels)
    loss.backward()
    train_dt = time.perf_counter() - t0
    model.zero_grad(set_to_none=True)
    model.eval()
    t1 = time.perf_counter()
    with torch.no_grad():
        before_logits = model(plane_images, plane_mask)
    eval_dt = time.perf_counter() - t1
    round_trip_max_abs = 0.0
    if reload_factory is not None and checkpoint_dir is not None:
        ckpt = Path(checkpoint_dir) / "runtime_preflight_round_trip.pt"
        save_checkpoint(model, ckpt)
        restored = reload_factory().to(device)
        load_checkpoint(restored, ckpt, device)
        restored.eval()
        with torch.no_grad():
            after_logits = restored(plane_images, plane_mask)
        round_trip_max_abs = float((before_logits - after_logits).abs().max().item())
        if round_trip_max_abs != 0.0:
            raise Gold58PreflightError(
                STATUS_RUNTIME,
                f"preflight_round_trip_diff={round_trip_max_abs}",
            )
    model.train()
    micro_opt = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg.get("learning_rate") or 1e-4),
        weight_decay=float(cfg.get("weight_decay") or 0.0),
    )
    t_micro = time.perf_counter()
    micro_opt.zero_grad(set_to_none=True)
    with autocast:
        micro_logits = model(plane_images, plane_mask)
        micro_loss = masked_bce_with_logits(micro_logits, labels)
    micro_loss.backward()
    micro_opt.step()
    micro_train_dt = time.perf_counter() - t_micro
    model.eval()
    t_micro_eval = time.perf_counter()
    with torch.no_grad():
        _ = model(plane_images, plane_mask)
    micro_eval_dt = time.perf_counter() - t_micro_eval
    batch_n = int(labels.size(0))
    epochs = int(cfg.get("epochs") or 12)
    n_train = int(math.ceil(n_labelled * (n_folds - 1) / n_folds))
    n_val = int(n_labelled - n_train)
    steps_per_epoch = int(math.ceil(n_train / max(int(cfg.get("batch_size") or 2), 1)))
    mean_planes = float(series_aggregates.get("mean_planes_per_study") or 3.0)
    plane_scale = mean_planes / float(len(PLANES))
    projected = (
        n_folds
        * (
            steps_per_epoch * epochs * train_dt * plane_scale
            + math.ceil(n_val / max(int(cfg.get("batch_size") or 2), 1)) * eval_dt * plane_scale
        )
    )
    safety = float(cfg.get("safety_factor") or 1.25)
    reserve = float(cfg.get("shutdown_reserve_seconds") or 3600)
    limit = float(cfg.get("max_wall_seconds") or 39600)
    guarded = projected * safety
    planned = guarded + reserve
    peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    payload = {
        "measured_train_step_seconds": float(train_dt),
        "measured_eval_step_seconds": float(eval_dt),
        "timed_batch_size": batch_n,
        "mean_planes_per_study": mean_planes,
        "projected_seconds": float(projected),
        "guarded_seconds": float(guarded),
        "planned_seconds_with_reserve": float(planned),
        "max_wall_seconds": limit,
        "fits_private_wall": bool(planned <= limit),
        "peak_gpu_memory_bytes": peak,
        "n_train_per_fold_est": n_train,
        "steps_per_epoch_est": steps_per_epoch,
        "preflight_round_trip_max_abs_diff": float(round_trip_max_abs),
        "micro_train_step_seconds": float(micro_train_dt),
        "micro_eval_step_seconds": float(micro_eval_dt),
        "classifier_has_batchnorm1d": False,
    }
    if not skip_wall_gate and planned > limit:
        raise Gold58PreflightError(
            STATUS_RUNTIME,
            f"projected_wall_{planned:.1f}_exceeds_{limit:.1f}",
        )
    return payload


def paired_macro_delta(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_prob_v2: np.ndarray,
    group_ids: np.ndarray,
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    groups = np.asarray(group_ids, dtype=object)
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    index = {g: np.where(groups == g)[0] for g in unique}
    deltas: list[float] = []
    for _ in range(n_resamples):
        drawn = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([index[g] for g in drawn])
        macro, _ = macro_roc_auc(y_true[rows], y_prob[rows])
        macro_v2, _ = macro_roc_auc(y_true[rows], y_prob_v2[rows])
        if macro == macro and macro_v2 == macro_v2:
            deltas.append(float(macro - macro_v2))
    n_ok = len(deltas)
    frac = float(sum(d > 0 for d in deltas) / n_ok) if n_ok else None
    point, _ = macro_roc_auc(y_true, y_prob)
    point_v2, _ = macro_roc_auc(y_true, y_prob_v2)
    return {
        "n_resamples": int(n_resamples),
        "n_finite": n_ok,
        "seed": int(seed),
        "point_delta": None if point != point or point_v2 != point_v2 else float(point - point_v2),
        "fraction_delta_gt_0": frac,
        "v2_oof_available": True,
        "v2_oof_sha256_verified": True,
    }


def render_3plane_protocol_markdown(cfg: dict[str, Any] | None = None) -> str:
    cfg = cfg or load_3plane_config()
    estimate = estimate_3plane_resources(cfg)
    return "\n".join(
        [
            "# Gold-58 image-only three-plane fusion OOF protocol v1",
            "",
            "Authorized Phase 4G scientific path. Frozen v2 primary-series OOF remains the comparator.",
            "",
            f"- protocol: `{PROTOCOL}`",
            f"- kernel slug: `{KERNEL_SLUG}`",
            "- encoder: ImageNet ResNet-18; shared weights across planes",
            f"- ImageNet SHA-256: `{IMAGENET_SHA256}`",
            f"- planes: {', '.join(PLANES)}",
            f"- series rule: `{SERIES_RULE_3PLANE}` (one readable series per plane;",
            "  fluid-sensitive first, then DICOM count, description, SeriesInstanceUID)",
            "- missing plane: average available planes; min_planes_required=1",
            "- fusion: mean of per-plane encoder embeddings with boolean mask",
            "  and denominator clamp; never fuse a missing-plane zero",
            "- classifier: Linear head; no BatchNorm1d; val/inference batch size 1",
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
            "- GOLD58_3PLANE_OOF_EXECUTION_ALLOWED=true",
            "- UNLABELED_IMAGE_SSL_EXECUTION_ALLOWED=false",
            "- HARD_WEAK_LABEL_TRAINING_ALLOWED=false",
            f"- config SHA-256: `{cfg['_config_sha256']}`",
            f"- forbidden outcome not used: `{FORBIDDEN_SEMANTIC_PASS}`",
            "",
        ]
    )


def _write_public(output_dir: Path, payload: dict[str, Any]) -> None:
    public_dir = Path(output_dir) / "sanitized"
    public_dir.mkdir(parents=True, exist_ok=True)
    public = {
        k: v
        for k, v in payload.items()
        if k not in {"oof_frame", "patient_tokens", "study_ids", "v2_oof_rows"}
    }
    (public_dir / "gold58_3plane_oof.json").write_text(
        json.dumps(jsonable(public), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _base_payload(status: str, reason: str | None, extra: dict[str, Any] | None, cfg: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "protocol": PROTOCOL,
        "status": status,
        "ok": status == STATUS_COMPLETE,
        "skip_reason": reason,
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
        "comparator_macro_roc_auc": FROZEN_V2_MACRO_AUC,
        "frozen_v2_config_sha256": FROZEN_V2_CONFIG_SHA256,
        "config_sha256": cfg.get("_config_sha256"),
        "forbidden_outcome_not_used": FORBIDDEN_SEMANTIC_PASS,
        "git_head_sha": git_head_sha(),
        "utc": _utc(),
        "notebook_07_executed": False,
        "notebook_09_executed": False,
        "notebook_10_executed": False,
        "notebook_11_executed": False,
        "notebook_16_executed": False,
        "notebook_17_executed": False,
    }
    if extra:
        payload.update(extra)
    return jsonable(payload)


def run_gold58_3plane_oof(
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
    v2_oof_path: str | Path | None = None,
) -> dict[str, Any]:
    """Three-plane labelled-image OOF. Tiny encoder is tests-only."""
    assert HARD_WEAK_LABEL_TRAINING_ALLOWED is False
    assert SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED is False
    assert SECOND_READER_SCORE_BEARING_ALLOWED is False
    assert UNLABELED_IMAGE_SSL_EXECUTION_ALLOWED is False
    cfg = load_3plane_config(config_path)
    if config_overrides:
        cfg.update(config_overrides)
    estimate = estimate_3plane_resources(cfg)
    out_dir = Path(output_dir or (PROJECT_ROOT / "outputs" / cfg["output_dirname"]))
    out_dir.mkdir(parents=True, exist_ok=True)

    if not GOLD58_3PLANE_OOF_EXECUTION_ALLOWED:
        payload = _base_payload(STATUS_NOT_AUTHORIZED, "GOLD58_3PLANE_OOF_EXECUTION_ALLOWED=false", {"resource_estimate": estimate}, cfg)
        _write_public(out_dir, payload)
        return payload

    if not execute_training and data_root is None:
        payload = _base_payload(
            STATUS_PREPARED,
            "execute_training=false",
            {"resource_estimate": estimate, "kaggle_kernel_launched": False},
            cfg,
        )
        _write_public(out_dir, payload)
        return payload

    layout = layout or KaggleLayout(output_dirname=str(cfg["output_dirname"]))
    pre = preflight(layout=layout, data_root=data_root, output_dir=out_dir)
    if not pre.ok or pre.data_root is None:
        payload = _base_payload(STATUS_RUNTIME, pre.message or "DATA_NOT_AVAILABLE", {"resource_estimate": estimate}, cfg)
        _write_public(out_dir, payload)
        return payload

    root = Path(pre.data_root)
    data_source = infer_data_source(root, layout=layout)
    train = pd.read_csv(root / "train.csv")
    series = pd.read_csv(root / "train_series.csv")
    labelled, report_only = select_labelled_studies(train)
    study_ids = labelled[STUDY_ID_COLUMN].astype(str).tolist()
    n_labelled = len(study_ids)
    if data_source == DATA_SOURCE_REAL_KAGGLE and n_labelled != EXPECTED_LABELLED_REAL:
        payload = _base_payload(
            STATUS_RUNTIME,
            f"labelled_count_not_{EXPECTED_LABELLED_REAL}:{n_labelled}",
            {"n_labelled_studies": n_labelled, "data_source": data_source},
            cfg,
        )
        _write_public(out_dir, payload)
        return payload

    try:
        inventory = inspect_series_inventory(series, root)
        series_lookup, series_aggregates = select_one_series_per_plane(
            inventory, study_ids, min_planes_required=int(cfg.get("min_planes_required") or 1)
        )
        tokens = collect_patient_tokens_across_readable_series(study_ids, inventory)
        labels, _mask = extract_label_matrix(labelled, require_all=True)
        fold_table, fold_meta = build_folds(
            study_ids,
            tokens,
            labels,
            n_folds=int(cfg["n_folds"]),
            seed=int(cfg["seed"]),
        )
        v2_table, _v2_meta = build_folds(
            study_ids,
            tokens,
            labels,
            n_folds=int(cfg["n_folds"]),
            seed=int(cfg["seed"]),
        )
        hash_3 = fold_assignment_hash(fold_table)
        hash_v2 = fold_assignment_hash(v2_table)
        if hash_3 != hash_v2:
            raise Gold58PreflightError(STATUS_FOLDS, "fold_assignment_hash_mismatch_vs_v2")
        fold_support = None
        if bool(cfg.get("fold_require_pos_neg_every_target")):
            fold_support = assert_train_pos_neg_every_target(
                labels, fold_table["fold"].to_numpy(), int(cfg["n_folds"])
            )
    except Gold58PreflightError as exc:
        status = exc.status if exc.status in {STATUS_SERIES, STATUS_FOLDS, STATUS_RUNTIME} else STATUS_RUNTIME
        payload = _base_payload(status, exc.reason, {"n_labelled_studies": n_labelled, "data_source": data_source}, cfg)
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
    sanitized_preflight = {
        "n_studies": n_labelled,
        "n_unique_patients": int(len(set(tokens.values()))),
        "studies_per_patient_distribution": studies_per_patient_distribution(tokens),
        "per_target_counts": per_target_counts,
        "per_fold": fold_class_counts(labels, fold_ids, int(cfg["n_folds"])),
        "n_report_only_studies_present": int(len(report_only)),
        "n_report_only_studies_processed": 0,
        "series_selection_rule": SERIES_RULE_3PLANE,
        "series_selection_aggregates": series_aggregates,
        "fold_assignment_sha256": hash_3,
        "fold_assignment_matches_v2": True,
        "frozen_v2_config_sha256": FROZEN_V2_CONFIG_SHA256,
        "train_class_support_ok": True if fold_support is not None else None,
    }

    scientific = bool(execute_training) and not allow_tiny_encoder_for_tests
    cuda_needed = cfg.get("require_cuda_for_scientific_run", True) if require_cuda is None else require_cuda
    if scientific and cuda_needed and not torch.cuda.is_available():
        payload = _base_payload(STATUS_RUNTIME, "cuda_required", {"preflight_aggregates": sanitized_preflight}, cfg)
        _write_public(out_dir, payload)
        return payload

    weights_file = locate_imagenet_weights(weights_path)
    weight_sha = None
    if scientific:
        if weights_file is None:
            payload = _base_payload(STATUS_RUNTIME, "imagenet_weights_missing", {"preflight_aggregates": sanitized_preflight}, cfg)
            _write_public(out_dir, payload)
            return payload
        try:
            weight_sha = verify_imagenet_weights(weights_file, str(cfg["imagenet_weights_sha256"]))
        except Gold58PreflightError as exc:
            payload = _base_payload(STATUS_RUNTIME, exc.reason, {"preflight_aggregates": sanitized_preflight}, cfg)
            _write_public(out_dir, payload)
            return payload

    if not execute_training:
        payload = _base_payload(
            "PREFLIGHT_PASS_TRAINING_NOT_EXECUTED",
            "execute_training=false",
            {
                "ok": True,
                "preflight_aggregates": sanitized_preflight,
                "data_source": data_source,
                "imagenet_weights_sha256": weight_sha,
                "n_labelled_studies": n_labelled,
            },
            cfg,
        )
        payload["ok"] = True
        _write_public(out_dir, payload)
        return payload

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if scientific and "Tesla T4" not in (torch.cuda.get_device_name(0) if device.type == "cuda" else ""):
        payload = _base_payload(STATUS_RUNTIME, "expected_tesla_t4", {"preflight_aggregates": sanitized_preflight, "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"}, cfg)
        _write_public(out_dir, payload)
        return payload
    use_amp = bool(cfg.get("amp") == "cuda_only" and device.type == "cuda")
    set_deterministic_seeds(int(cfg["seed"]))
    imagenet_normalize = bool(cfg.get("imagenet_normalize"))
    aggregation = str(cfg.get("aggregation") or "mean")
    private_dir = out_dir / "private"
    private_dir.mkdir(parents=True, exist_ok=True)
    write_probe = private_dir / ".write_ok"
    write_probe.write_text("ok\n", encoding="utf-8")
    write_probe.unlink()

    def _reload() -> ThreePlaneFusionModel:
        return ThreePlaneFusionModel(
            _build_backbone(
                tiny=allow_tiny_encoder_for_tests,
                weights_path=weights_file,
                imagenet_normalize=imagenet_normalize,
                aggregation=aggregation,
            )
        )

    try:
        backbone = _build_backbone(
            tiny=allow_tiny_encoder_for_tests,
            weights_path=weights_file,
            imagenet_normalize=imagenet_normalize,
            aggregation=aggregation,
        )
        model = ThreePlaneFusionModel(backbone).to(device)
        assert_classifier_has_no_batchnorm1d(model)
        runtime = run_runtime_preflight(
            model,
            cfg,
            device=device,
            use_amp=use_amp,
            series_aggregates=series_aggregates,
            n_labelled=n_labelled,
            n_folds=int(cfg["n_folds"]),
            skip_wall_gate=allow_tiny_encoder_for_tests,
            reload_factory=_reload,
            checkpoint_dir=private_dir,
        )
    except Gold58PreflightError as exc:
        payload = _base_payload(exc.status if exc.status == STATUS_RUNTIME else STATUS_RUNTIME, exc.reason, {"preflight_aggregates": sanitized_preflight}, cfg)
        _write_public(out_dir, payload)
        return payload
    except Exception as exc:  # noqa: BLE001
        payload = _base_payload(STATUS_RUNTIME, f"{type(exc).__name__}:{exc}", {"preflight_aggregates": sanitized_preflight}, cfg)
        _write_public(out_dir, payload)
        return payload

    ds_kwargs = {
        "num_slices": int(cfg["num_slices"]),
        "image_size": int(cfg["image_size"]),
        "percentiles": tuple(cfg["intensity_percentiles"]),
        "interpolation": str(cfg.get("interpolation") or "bilinear"),
        "imagenet_normalize": imagenet_normalize,
        "imagenet_mean": tuple(cfg.get("imagenet_mean") or IMAGENET_MEAN),
        "imagenet_std": tuple(cfg.get("imagenet_std") or IMAGENET_STD),
    }

    def _dirs_for(ids: list[str]) -> list[dict[str, Path]]:
        out: list[dict[str, Path]] = []
        for sid in ids:
            mapping: dict[str, Path] = {}
            for plane, series_uid in series_lookup[sid].items():
                mapping[plane] = _series_dir(root, sid, series_uid)
            out.append(mapping)
        return out

    oof_ids: list[str] = []
    oof_prob: list[np.ndarray] = []
    oof_true: list[np.ndarray] = []
    oof_groups: list[str] = []
    checkpoint_hashes: dict[str, str] = {}
    fold_losses: dict[str, Any] = {}
    round_trip = None

    try:
        for fold in range(int(cfg["n_folds"])):
            train_mask = fold_table["fold"].to_numpy() != fold
            val_mask = ~train_mask
            train_ids = [study_ids[i] for i, keep in enumerate(train_mask) if keep]
            val_ids = [study_ids[i] for i, keep in enumerate(val_mask) if keep]
            train_ds = ThreePlaneStudyDataset(train_ids, _dirs_for(train_ids), labels[train_mask], **ds_kwargs)
            val_ds = ThreePlaneStudyDataset(val_ids, _dirs_for(val_ids), labels[val_mask], **ds_kwargs)
            model = _reload().to(device)
            assert_classifier_has_no_batchnorm1d(model)
            drop_last = train_loader_drop_last(
                n_items=len(train_ds),
                batch_size=int(cfg["batch_size"]),
                has_bn1d=classifier_has_batchnorm1d(model),
            )
            loader_gen = torch.Generator()
            loader_gen.manual_seed(int(cfg["seed"]) + fold)
            train_loader = DataLoader(
                train_ds,
                batch_size=int(cfg["batch_size"]),
                shuffle=True,
                drop_last=drop_last,
                generator=loader_gen,
                collate_fn=collate_three_plane,
                num_workers=0,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=1,
                shuffle=False,
                collate_fn=collate_three_plane,
                num_workers=0,
            )
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
                model.eval()
                with torch.no_grad():
                    before = torch.sigmoid(
                        model(
                            {p: t.to(device) for p, t in sample_batch["plane_images"].items()},
                            {p: m.to(device) for p, m in sample_batch["plane_mask"].items()},
                        )
                    ).cpu().numpy()
                save_checkpoint(model, private_dir / "round_trip.pt")
                restored = _reload().to(device)
                load_checkpoint(restored, private_dir / "round_trip.pt", device)
                restored.eval()
                with torch.no_grad():
                    after = torch.sigmoid(
                        restored(
                            {p: t.to(device) for p, t in sample_batch["plane_images"].items()},
                            {p: m.to(device) for p, m in sample_batch["plane_mask"].items()},
                        )
                    ).cpu().numpy()
                max_abs = float(np.max(np.abs(before - after)))
                atol = 0.0 if cfg.get("round_trip_atol") is None else float(cfg.get("round_trip_atol"))
                round_trip = {
                    "max_abs_diff": max_abs,
                    "within_tolerance": bool(max_abs <= atol),
                    "atol": atol,
                    "n_rows": int(before.shape[0]),
                }
                if not round_trip["within_tolerance"]:
                    payload = _base_payload(
                        STATUS_VALIDATION,
                        f"round_trip_diff={max_abs}",
                        {"round_trip": round_trip, "preflight_aggregates": sanitized_preflight},
                        cfg,
                    )
                    _write_public(out_dir, payload)
                    return payload
    except Exception as exc:  # noqa: BLE001
        payload = _base_payload(STATUS_TRAINING, f"{type(exc).__name__}:{exc}", {"preflight_aggregates": sanitized_preflight, "runtime_preflight": runtime}, cfg)
        _write_public(out_dir, payload)
        return payload

    y_prob = np.concatenate(oof_prob, axis=0)
    y_true = np.concatenate(oof_true, axis=0)
    try:
        assert_probabilities_ok(y_prob)
        if len(oof_ids) != n_labelled or len(set(oof_ids)) != n_labelled:
            raise Gold58PreflightError(STATUS_VALIDATION, f"oof_count_mismatch:{len(oof_ids)}")
        order = {sid: i for i, sid in enumerate(oof_ids)}
        ordered_idx = [order[s] for s in study_ids]
        y_prob = y_prob[ordered_idx]
        y_true = y_true[ordered_idx]
        groups = np.array([oof_groups[order[s]] for s in study_ids], dtype=object)
        if not np.array_equal(y_true, labels):
            # labels may differ only by float; check finite equality
            if not np.allclose(np.nan_to_num(y_true, nan=-1), np.nan_to_num(labels, nan=-1)):
                raise Gold58PreflightError(STATUS_VALIDATION, "oof_label_order_mismatch")
        macro, per_target = macro_roc_auc(y_true, y_prob)
        bootstrap = patient_bootstrap_aucs(
            y_true,
            y_prob,
            groups,
            n_resamples=int(cfg.get("bootstrap_n_resamples") or 1000),
            seed=int(cfg.get("bootstrap_seed") or 42),
            ci_level=float(cfg.get("bootstrap_ci_level") or 0.95),
        )
        rng = np.random.default_rng(int(cfg["seed"]))
        perm = rng.permutation(len(y_true))
        shuffled_macro, _ = macro_roc_auc(y_true[perm], y_prob)
    except Gold58PreflightError as exc:
        payload = _base_payload(STATUS_VALIDATION, exc.reason, {"preflight_aggregates": sanitized_preflight}, cfg)
        _write_public(out_dir, payload)
        return payload

    oof_frame = pd.DataFrame({STUDY_ID_COLUMN: study_ids})
    for i, column in enumerate(OFFICIAL_TARGET_COLUMNS):
        oof_frame[column] = y_prob[:, i]
    try:
        oof_table_ok = validate_oof_table(
            oof_frame, expected_n=n_labelled, expected_ids=study_ids
        )
    except Gold58PreflightError as exc:
        payload = _base_payload(STATUS_VALIDATION, exc.reason, {"preflight_aggregates": sanitized_preflight}, cfg)
        _write_public(out_dir, payload)
        return payload
    oof_path = private_dir / "oof_predictions.csv"
    oof_frame.to_csv(oof_path, index=False)
    contract = submission_contract_from_oof(oof_frame, private_dir)

    paired = {"v2_oof_available": False}
    v2_path = locate_frozen_v2_oof(v2_oof_path)
    if v2_path is not None:
        digest = sha256_file(v2_path)
        if digest != FROZEN_V2_OOF_SHA256:
            payload = _base_payload(
                STATUS_VALIDATION,
                f"v2_oof_hash_mismatch:{digest}",
                {"preflight_aggregates": sanitized_preflight},
                cfg,
            )
            _write_public(out_dir, payload)
            return payload
        v2_frame = pd.read_csv(v2_path)
        v2_frame[STUDY_ID_COLUMN] = v2_frame[STUDY_ID_COLUMN].astype(str)
        aligned = v2_frame.set_index(STUDY_ID_COLUMN).loc[study_ids, OFFICIAL_TARGET_COLUMNS]
        y_prob_v2 = aligned.to_numpy(dtype=np.float64)
        paired = paired_macro_delta(
            y_true,
            y_prob,
            y_prob_v2,
            groups,
            n_resamples=int(cfg.get("bootstrap_n_resamples") or 1000),
            seed=int(cfg.get("bootstrap_seed") or 42),
        )

    per_target_report = {}
    for i, name in enumerate(OFFICIAL_TARGET_COLUMNS):
        per_target_report[name] = {
            "roc_auc": None if per_target[i] != per_target[i] else float(per_target[i]),
            **per_target_counts[name],
            **bootstrap["per_target"][name],
        }

    payload = _base_payload(
        STATUS_COMPLETE,
        None,
        extra={
            "data_source": data_source,
            "scientific_evaluation": not allow_tiny_encoder_for_tests,
            "training_run": True,
            "OOF_run": True,
            "kaggle_runtime": is_kaggle_runtime(layout),
            "preflight_aggregates": sanitized_preflight,
            "runtime_preflight": runtime,
            "imagenet_weights_sha256": weight_sha,
            "primary_metric": "macro_roc_auc_combined_oof",
            "macro_roc_auc": None if macro != macro else float(macro),
            "comparator_macro_roc_auc": FROZEN_V2_MACRO_AUC,
            "macro_roc_auc_delta_vs_v2_point": (
                None if macro != macro else float(macro) - FROZEN_V2_MACRO_AUC
            ),
            "paired_bootstrap_vs_v2": paired,
            "bootstrap": bootstrap,
            "per_target": per_target_report,
            "shuffled_label_macro_roc_auc_diagnostic_only": (
                None if shuffled_macro != shuffled_macro else float(shuffled_macro)
            ),
            "checkpoint_sha256": checkpoint_hashes,
            "oof_predictions_sha256": sha256_file(oof_path),
            "n_oof_rows": int(len(oof_frame)),
            "oof_table_validation": oof_table_ok,
            "n_labelled_studies": n_labelled,
            "round_trip": round_trip,
            "submission_contract": contract,
            "fold_losses": fold_losses,
            "device": str(device),
            "amp": use_amp,
            "resource_estimate": estimate,
        },
        cfg=cfg,
    )
    _write_public(out_dir, payload)
    return payload
