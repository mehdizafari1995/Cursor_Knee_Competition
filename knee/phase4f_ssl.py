"""Phase 4F image-only SimSiam SSL design.

Planning and synthetic/tiny-tensor smoke only. Real unlabeled-pixel training is
gated off (``UNLABELED_IMAGE_SSL_EXECUTION_ALLOWED=false``). Never uses reports,
target columns, weak labels, second-reader outputs, gold-58 pixels, or hidden
test data. Reuses frozen ``deterministic-primary-series-v2`` without retuning.
"""

from __future__ import annotations

from datetime import datetime, timezone
import inspect
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
import yaml

from .baseline_model import TinySliceEncoder
from .config import (
    OFFICIAL_TARGET_COLUMNS,
    PROJECT_ROOT,
    REPORT_COLUMN,
    STUDY_ID_COLUMN,
    TARGET_KEYS,
)
from .gold58_oof import (
    EXPECTED_LABELLED_REAL,
    IMAGENET_SHA256,
    SERIES_RULE_V2,
    Gold58PreflightError,
    inspect_series_inventory,
    select_deterministic_primary_series_v2,
    select_labelled_studies,
    set_deterministic_seeds,
)
from .kaggle_audit import jsonable
from .kaggle_bundle import sha256_file
from .kaggle_runtime import DATA_SOURCE_REAL_KAGGLE, infer_data_source
from .phase4d_readiness import SECOND_READER_SCORE_BEARING_ALLOWED
from .report_labels import (
    HARD_WEAK_LABEL_TRAINING_ALLOWED,
    SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED,
)
from .second_reader_phase46 import FORBIDDEN_SEMANTIC_PASS, git_head_sha

PROTOCOL = "unlabeled-image-simsiam-ssl-v1"
KERNEL_SLUG = "mahdimzafari/rsna-knee-unlabeled-image-ssl-v1"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "unlabeled_image_simsiam_ssl_v1.yaml"
FROZEN_V1_RECIPE_SHA256 = "92f9d8b52d9488646a4dd19ca9740b8ae658c5f71bf6812a393b91fa3a07af9c"
FROZEN_V2_CONFIG_SHA256 = "4b65bc0d4d56a3dd36b6898c71d630a113c99f6e4b986e51a15ae1b66e1ea093"
FROZEN_V14 = "992af7a4b4483278445a6f3fb363c818611e79005618dff063460d0aed3bd7f7"
EXPECTED_UNLABELED_REAL = 4349
EXPECTED_GOLD_REAL = EXPECTED_LABELLED_REAL

UNLABELED_IMAGE_SSL_PLANNING_ALLOWED = True
UNLABELED_IMAGE_SSL_EXECUTION_ALLOWED = False
GOLD58_SSL_OOF_EXECUTION_ALLOWED = False

STATUS_SMOKE = "SSL_SMOKE_COMPLETE"
STATUS_PLANNING = "SSL_PLANNING_COMPLETE"
STATUS_EXECUTION = "SSL_EXECUTION_NOT_AUTHORIZED"
STATUS_GOLD = "GOLD58_INTERSECTION_FAIL"
STATUS_LABELS = "LABEL_OR_REPORT_INPUT_FAIL"
STATUS_HIDDEN = "HIDDEN_TEST_PATH_FAIL"
STATUS_SERIES = "SERIES_SELECTION_PREFLIGHT_FAIL"
STATUS_COLLAPSE = "SSL_COLLAPSE_FAIL"
STATUS_FINITE = "SSL_NONFINITE_LOSS_FAIL"
STATUS_RUNTIME = "RUNTIME_GUARD_FAIL"
STATUS_OOF = "GOLD58_SSL_OOF_NOT_AUTHORIZED"
STATUS_INTERNET = "INTERNET_PREFLIGHT_FAIL"

FORBIDDEN_TRAINING_KEYS = frozenset(
    {
        "labels",
        "y_true",
        "gold",
        "report",
        REPORT_COLUMN,
        "weak_labels",
        "second_reader",
        *OFFICIAL_TARGET_COLUMNS,
        *TARGET_KEYS,
    }
)

V2_SUPERVISED_SECONDS_PER_STUDY = 8168.106 / 3480.0
SSL_VIEW_FACTOR = 1.5
KAGGLE_GPU_PEAK_BYTES_V2 = 404743680
COMPETITION_SCORING_LIMIT_SECONDS = 32400.0


class Phase4FError(RuntimeError):
    def __init__(self, status: str, reason: str):
        super().__init__(reason)
        self.status = status
        self.reason = reason


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_ssl_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{cfg_path} must contain a mapping")
    payload["_config_path"] = str(cfg_path)
    payload["_config_sha256"] = sha256_file(cfg_path)
    return payload


def assert_no_forbidden_training_inputs(kwargs: dict[str, Any] | None = None) -> None:
    extra = kwargs or {}
    hits = sorted(k for k in extra if k in FORBIDDEN_TRAINING_KEYS)
    if hits:
        raise Phase4FError(STATUS_LABELS, f"forbidden_training_keys:{len(hits)}")


def assert_run_signature_image_only(fn: Any | None = None) -> None:
    target = fn if fn is not None else run_phase4f_ssl
    names = set(inspect.signature(target).parameters)
    overlap = sorted(names & FORBIDDEN_TRAINING_KEYS)
    if overlap:
        raise Phase4FError(STATUS_LABELS, "run_signature_contains_forbidden_keys")


def drop_non_image_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if STUDY_ID_COLUMN not in frame.columns:
        raise Phase4FError(STATUS_LABELS, "missing_study_id")
    return frame.loc[:, [STUDY_ID_COLUMN]].copy()


def select_unlabeled_study_ids(train: pd.DataFrame) -> tuple[list[str], list[str], dict[str, Any]]:
    """Identify unlabeled studies from label missingness, then drop labels."""
    labelled, report_only = select_labelled_studies(train)
    gold_ids = labelled[STUDY_ID_COLUMN].astype(str).tolist()
    ssl_ids = report_only[STUDY_ID_COLUMN].astype(str).tolist()
    overlap = sorted(set(gold_ids) & set(ssl_ids))
    if overlap:
        raise Phase4FError(STATUS_GOLD, f"n_overlap={len(overlap)}")
    stats = {
        "n_gold_excluded": int(len(gold_ids)),
        "n_unlabeled_selected": int(len(ssl_ids)),
        "n_gold_intersection": 0,
    }
    return ssl_ids, gold_ids, stats


def assert_not_hidden_test_path(data_root: Path | None, *, split: str = "train") -> None:
    if split == "test":
        raise Phase4FError(STATUS_HIDDEN, "split_test_forbidden_for_ssl")
    if data_root is None:
        return
    text = str(Path(data_root)).replace("\\", "/").lower()
    if "test_series" in text.split("/"):
        raise Phase4FError(STATUS_HIDDEN, "test_series_path_forbidden_for_ssl")


def projection_mlp(in_dim: int, hidden: int, out: int, *, tiny: bool) -> nn.Sequential:
    if tiny:
        return nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out),
        )
    return nn.Sequential(
        nn.Linear(in_dim, hidden, bias=False),
        nn.BatchNorm1d(hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, hidden, bias=False),
        nn.BatchNorm1d(hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, out, bias=False),
        nn.BatchNorm1d(out, affine=False),
    )


def prediction_mlp(in_dim: int, hidden: int, *, tiny: bool) -> nn.Sequential:
    if tiny:
        return nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_dim),
        )
    return nn.Sequential(
        nn.Linear(in_dim, hidden, bias=False),
        nn.BatchNorm1d(hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, in_dim),
    )


class SimSiamStudyModel(nn.Module):
    """Encode slices, mean-pool, project, predict. Stop-grad is applied in the loss."""

    def __init__(
        self,
        encoder: nn.Module,
        *,
        projector_hidden: int,
        projector_out: int,
        predictor_hidden: int,
        aggregation: str = "mean",
        tiny: bool = False,
    ):
        super().__init__()
        if aggregation != "mean":
            raise ValueError("Phase 4F aggregation is frozen at mean")
        self.encoder = encoder
        self.aggregation = aggregation
        self.projector = projection_mlp(encoder.out_dim, projector_hidden, projector_out, tiny=tiny)
        self.predictor = prediction_mlp(projector_out, predictor_hidden, tiny=tiny)

    def encode(self, slices: torch.Tensor) -> torch.Tensor:
        if slices.ndim != 5:
            raise ValueError(f"expected (B, S, C, H, W), got {tuple(slices.shape)}")
        batch, n_slices, channels, height, width = slices.shape
        flat = slices.reshape(batch * n_slices, channels, height, width)
        features = self.encoder(flat)
        features = features.reshape(batch, n_slices, -1)
        return features.mean(dim=1)

    def forward_views(
        self, view1: torch.Tensor, view2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z1 = self.projector(self.encode(view1))
        z2 = self.projector(self.encode(view2))
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        return p1, z1, p2, z2


def simsiam_loss(
    p1: torch.Tensor, z2: torch.Tensor, p2: torch.Tensor, z1: torch.Tensor
) -> torch.Tensor:
    """Symmetric negative cosine similarity with stop-gradient on z."""
    left = F.cosine_similarity(p1, z2.detach(), dim=-1).mean()
    right = F.cosine_similarity(p2, z1.detach(), dim=-1).mean()
    return -(left + right) / 2.0


def representation_std(z: torch.Tensor) -> float:
    if z.ndim != 2 or z.shape[0] < 2:
        return float("nan")
    return float(z.detach().std(dim=0).mean().clamp(min=0).cpu())


def detect_collapse(z: torch.Tensor, min_std: float) -> bool:
    std = representation_std(z)
    if not math.isfinite(std):
        return True
    return std < float(min_std)


def augment_study(
    image: torch.Tensor,
    *,
    generator: torch.Generator,
    crop_scale: tuple[float, float] = (0.6, 1.0),
    hflip_p: float = 0.5,
    intensity_scale: tuple[float, float] = (0.9, 1.1),
    intensity_shift: tuple[float, float] = (-0.05, 0.05),
) -> torch.Tensor:
    """Study-consistent spatial/intensity jitter. ``image`` is (S, C, H, W)."""
    if image.ndim != 4:
        raise ValueError(f"expected (S, C, H, W), got {tuple(image.shape)}")
    slices, channels, height, width = image.shape
    scale = float(crop_scale[0]) + float(torch.rand((), generator=generator)) * (
        float(crop_scale[1]) - float(crop_scale[0])
    )
    crop_h = max(1, int(round(height * scale)))
    crop_w = max(1, int(round(width * scale)))
    top = int(torch.randint(0, height - crop_h + 1, (1,), generator=generator).item())
    left = int(torch.randint(0, width - crop_w + 1, (1,), generator=generator).item())
    cropped = image[:, :, top : top + crop_h, left : left + crop_w]
    resized = F.interpolate(
        cropped, size=(height, width), mode="bilinear", align_corners=False
    )
    if float(torch.rand((), generator=generator)) < hflip_p:
        resized = torch.flip(resized, dims=[-1])
    gain = float(intensity_scale[0]) + float(torch.rand((), generator=generator)) * (
        float(intensity_scale[1]) - float(intensity_scale[0])
    )
    shift = float(intensity_shift[0]) + float(torch.rand((), generator=generator)) * (
        float(intensity_shift[1]) - float(intensity_shift[0])
    )
    return resized * gain + shift


class UnlabeledStudySliceDataset:
    """Image-only study dataset. Rejects labels/reports in the constructor."""

    def __init__(
        self,
        study_ids: list[str],
        series_dirs: list[Path],
        *,
        num_slices: int = 8,
        image_size: int = 224,
        percentiles: tuple[float, float] = (1.0, 99.0),
        interpolation: str = "bilinear",
        imagenet_normalize: bool = True,
        **forbidden: Any,
    ):
        if forbidden:
            raise Phase4FError(STATUS_LABELS, f"dataset_forbidden_kwargs:{len(forbidden)}")
        if len(study_ids) != len(series_dirs):
            raise ValueError("study_ids and series_dirs must be the same length")
        self.study_ids = list(study_ids)
        self.series_dirs = [Path(p) for p in series_dirs]
        self.num_slices = num_slices
        self.image_size = image_size
        self.percentiles = percentiles
        self.interpolation = interpolation
        self.imagenet_normalize = bool(imagenet_normalize)
        from .dataset import StudySliceDataset as _SSD

        self._loader = _SSD(
            self.study_ids,
            self.series_dirs,
            None,
            num_slices=num_slices,
            image_size=image_size,
            percentiles=percentiles,
            interpolation=interpolation,
            imagenet_normalize=imagenet_normalize,
        )

    def __len__(self) -> int:
        return len(self.study_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self._loader[idx]
        return {"image": item["image"]}


def two_views(
    images: torch.Tensor,
    *,
    generator: torch.Generator,
    crop_scale: tuple[float, float],
    hflip_p: float,
    intensity_scale: tuple[float, float],
    intensity_shift: tuple[float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    views1 = []
    views2 = []
    for row in images:
        views1.append(
            augment_study(
                row,
                generator=generator,
                crop_scale=crop_scale,
                hflip_p=hflip_p,
                intensity_scale=intensity_scale,
                intensity_shift=intensity_shift,
            )
        )
        views2.append(
            augment_study(
                row,
                generator=generator,
                crop_scale=crop_scale,
                hflip_p=hflip_p,
                intensity_scale=intensity_scale,
                intensity_shift=intensity_shift,
            )
        )
    return torch.stack(views1, dim=0), torch.stack(views2, dim=0)


def save_ssl_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_ssl_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    return torch.load(Path(path), map_location=device, weights_only=False)


def estimate_resources(n_studies: int, cfg: dict[str, Any]) -> dict[str, Any]:
    """Estimates derived from frozen Phase 4D.2 / 4E measurements only."""
    epochs = int(cfg.get("epochs") or 2)
    batch = int(cfg.get("batch_size") or 8)
    seconds_per_study = V2_SUPERVISED_SECONDS_PER_STUDY * SSL_VIEW_FACTOR
    projected_epoch = float(n_studies) * seconds_per_study
    projected_total = projected_epoch * epochs
    limit = float(cfg.get("max_wall_seconds") or 39600)
    safety = float(cfg.get("safety_factor") or 1.2)
    guarded = projected_total * safety
    fold_bytes_approx = 207_868_771 / 5.0
    projector_extra = 20 * 1024 * 1024
    return {
        "evidence_v2_wall_seconds": 8168.106,
        "evidence_v2_peak_gpu_memory_bytes": KAGGLE_GPU_PEAK_BYTES_V2,
        "evidence_v2_n_study_presentations_approx": 3480,
        "seconds_per_study_unlabeled_estimate": seconds_per_study,
        "n_studies": int(n_studies),
        "epochs": epochs,
        "batch_size": batch,
        "projected_seconds_per_epoch": projected_epoch,
        "projected_seconds_total": projected_total,
        "projected_seconds_with_safety": guarded,
        "max_wall_seconds": limit,
        "fits_private_max_wall": bool(guarded <= limit),
        "fits_competition_scoring_9h": bool(guarded <= COMPETITION_SCORING_LIMIT_SECONDS),
        "note": (
            "Throughput is scaled from gold58 v2 wall 8168.106s over ~3480 "
            "study presentations, times 1.5 for two SSL views. DICOM I/O for "
            "4349 studies is not separately measured. Pixel cache is not used: "
            "float32 grayscale 8x224 cache would be about 1.75 GiB and is not "
            "predeclared. Checkpoint size is encoder (~41.6 MiB from 4D.2 fold "
            "mean) plus projector/predictor (~20 MiB)."
        ),
        "checkpoint_bytes_estimate": int(fold_bytes_approx + projector_extra),
        "working_disk_pixel_cache_planned": False,
    }


def project_runtime(n_studies: int, cfg: dict[str, Any]) -> dict[str, Any]:
    info = estimate_resources(n_studies, cfg)
    if n_studies > 0 and not info["fits_private_max_wall"]:
        raise Phase4FError(
            STATUS_RUNTIME,
            (
                "projected_ssl_wall_exceeds_limit:"
                f"{info['projected_seconds_with_safety']:.1f}"
            ),
        )
    return info


def _tiny_batch(
    *,
    batch_size: int = 4,
    num_slices: int = 2,
    image_size: int = 16,
    seed: int = 0,
) -> torch.Tensor:
    rng = torch.Generator().manual_seed(seed)
    return torch.rand(batch_size, num_slices, 3, image_size, image_size, generator=rng)


def _build_tiny_model(cfg: dict[str, Any]) -> SimSiamStudyModel:
    encoder = TinySliceEncoder(in_channels=3, width=8)
    return SimSiamStudyModel(
        encoder,
        projector_hidden=int(cfg.get("tiny_projector_hidden") or 16),
        projector_out=int(cfg.get("tiny_projector_out") or 16),
        predictor_hidden=int(cfg.get("tiny_predictor_hidden") or 8),
        aggregation="mean",
        tiny=True,
    )


def run_synthetic_smoke(
    cfg: dict[str, Any],
    *,
    output_dir: Path,
    steps: int = 3,
    collapse_fixture: bool = False,
) -> dict[str, Any]:
    set_deterministic_seeds(int(cfg.get("seed") or 42))
    device = torch.device("cpu")
    model = _build_tiny_model(cfg).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(cfg.get("learning_rate") or 0.03),
        momentum=float(cfg.get("momentum") or 0.9),
        weight_decay=float(cfg.get("weight_decay") or 1e-4),
    )
    crop = tuple(cfg.get("crop_scale") or (0.6, 1.0))
    intensity_scale = tuple(cfg.get("intensity_scale") or (0.9, 1.1))
    intensity_shift = tuple(cfg.get("intensity_shift") or (-0.05, 0.05))
    min_std = float(cfg.get("collapse_std_min") or 0.001)
    generator = torch.Generator().manual_seed(int(cfg.get("seed") or 42))
    losses: list[float] = []
    stds: list[float] = []
    ckpt_path = Path(output_dir) / "private" / "ssl_tiny_last.pt"
    mid_path = Path(output_dir) / "private" / "ssl_tiny_step1.pt"

    def _step(step_i: int) -> tuple[float, float]:
        images = _tiny_batch(seed=int(cfg.get("seed") or 42) + step_i)
        if collapse_fixture:
            images = torch.zeros_like(images)
        v1, v2 = two_views(
            images,
            generator=generator,
            crop_scale=(float(crop[0]), float(crop[1])),
            hflip_p=float(cfg.get("hflip_p") or 0.5),
            intensity_scale=(float(intensity_scale[0]), float(intensity_scale[1])),
            intensity_shift=(float(intensity_shift[0]), float(intensity_shift[1])),
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        p1, z1, p2, z2 = model.forward_views(v1, v2)
        if collapse_fixture:
            z1 = torch.zeros_like(z1)
            z2 = torch.zeros_like(z2)
            p1 = torch.zeros_like(p1)
            p2 = torch.zeros_like(p2)
        if detect_collapse(torch.cat([z1, z2], dim=0), min_std):
            raise Phase4FError(STATUS_COLLAPSE, "representation_std_below_min")
        loss = simsiam_loss(p1, z2, p2, z1)
        if not torch.isfinite(loss):
            raise Phase4FError(STATUS_FINITE, "nonfinite_ssl_loss")
        loss.backward()
        optimizer.step()
        return float(loss.detach().cpu()), representation_std(torch.cat([z1, z2], dim=0))

    loss1, std1 = _step(0)
    losses.append(loss1)
    stds.append(std1)
    save_ssl_checkpoint(
        mid_path,
        {
            "encoder": model.encoder.state_dict(),
            "projector": model.projector.state_dict(),
            "predictor": model.predictor.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": 1,
            "epoch": 0,
            "seed": int(cfg.get("seed") or 42),
            "protocol": PROTOCOL,
        },
    )
    for step_i in range(1, steps):
        loss_i, std_i = _step(step_i)
        losses.append(loss_i)
        stds.append(std_i)

    save_ssl_checkpoint(
        ckpt_path,
        {
            "encoder": model.encoder.state_dict(),
            "projector": model.projector.state_dict(),
            "predictor": model.predictor.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": steps,
            "epoch": 0,
            "seed": int(cfg.get("seed") or 42),
            "protocol": PROTOCOL,
        },
    )

    resumed = _build_tiny_model(cfg).to(device)
    opt2 = torch.optim.SGD(
        resumed.parameters(),
        lr=float(cfg.get("learning_rate") or 0.03),
        momentum=float(cfg.get("momentum") or 0.9),
        weight_decay=float(cfg.get("weight_decay") or 1e-4),
    )
    loaded = load_ssl_checkpoint(mid_path, device)
    resumed.encoder.load_state_dict(loaded["encoder"])
    resumed.projector.load_state_dict(loaded["projector"])
    resumed.predictor.load_state_dict(loaded["predictor"])
    opt2.load_state_dict(loaded["optimizer"])
    last = load_ssl_checkpoint(ckpt_path, device)
    encoder_round_trip = max(
        float((a - b).abs().max())
        for a, b in zip(model.encoder.state_dict().values(), last["encoder"].values())
    )
    projector_round_trip = max(
        float((a - b).abs().max())
        for a, b in zip(model.projector.state_dict().values(), last["projector"].values())
    )
    resume_encoder_match = max(
        float((a - b).abs().max())
        for a, b in zip(
            resumed.encoder.state_dict().values(),
            load_ssl_checkpoint(mid_path, device)["encoder"].values(),
        )
    )
    return {
        "losses": losses,
        "representation_stds": stds,
        "n_steps": steps,
        "checkpoint_path_basename": ckpt_path.name,
        "resume_checkpoint_basename": mid_path.name,
        "encoder_round_trip_max_abs": encoder_round_trip,
        "projector_round_trip_max_abs": projector_round_trip,
        "resume_encoder_match_max_abs": resume_encoder_match,
        "backprop_ran": True,
        "collapse_detected": False,
    }


def plan_unlabeled_ssl_cohort(
    train: pd.DataFrame,
    series: pd.DataFrame,
    data_root: Path,
) -> dict[str, Any]:
    """Metadata + header-only series ranking. Does not pass labels to a dataset."""
    assert_not_hidden_test_path(data_root, split="train")
    ssl_ids, _gold_ids, stats = select_unlabeled_study_ids(train)
    image_frame = drop_non_image_columns(train)
    if any(col in image_frame.columns for col in FORBIDDEN_TRAINING_KEYS):
        raise Phase4FError(STATUS_LABELS, "image_frame_still_has_forbidden_columns")
    if REPORT_COLUMN in series.columns:
        raise Phase4FError(STATUS_LABELS, "series_table_has_report")
    inventory = inspect_series_inventory(series, data_root, split="train")
    try:
        lookup, aggregates = select_deterministic_primary_series_v2(inventory, ssl_ids)
    except Gold58PreflightError as exc:
        raise Phase4FError(STATUS_SERIES, str(exc)) from exc
    return {
        **stats,
        "n_selected": int(len(lookup)),
        "series_selection_aggregates": aggregates,
        "n_image_frame_columns": int(image_frame.shape[1]),
    }


def _payload(status: str, reason: str | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {
        "status": status,
        "ok": status in {STATUS_SMOKE, STATUS_PLANNING},
        "skip_reason": reason,
        "protocol": PROTOCOL,
        "kernel_slug": KERNEL_SLUG,
        "ssl_objective": "simsiam",
        "series_selection_rule": SERIES_RULE_V2,
        "UNLABELED_IMAGE_SSL_PLANNING_ALLOWED": UNLABELED_IMAGE_SSL_PLANNING_ALLOWED,
        "UNLABELED_IMAGE_SSL_EXECUTION_ALLOWED": UNLABELED_IMAGE_SSL_EXECUTION_ALLOWED,
        "GOLD58_SSL_OOF_EXECUTION_ALLOWED": GOLD58_SSL_OOF_EXECUTION_ALLOWED,
        "HARD_WEAK_LABEL_TRAINING_ALLOWED": HARD_WEAK_LABEL_TRAINING_ALLOWED,
        "SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED": SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED,
        "SECOND_READER_SCORE_BEARING_ALLOWED": SECOND_READER_SCORE_BEARING_ALLOWED,
        "competition_submission_allowed": False,
        "n_real_unlabelled_studies_processed": 0,
        "training_run": False,
        "OOF_run": False,
        "inference_run": False,
        "competition_submission_written": False,
        "consumes_submission_slot": False,
        "open_unlabeled_pixels": False,
        "forbidden_outcome_not_used": FORBIDDEN_SEMANTIC_PASS,
        "git_head_sha": git_head_sha(),
        "utc": _utc(),
    }
    if extra:
        body.update(extra)
    return body


def run_phase4f_ssl(
    *,
    data_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    config_path: str | Path | None = None,
    execute_training: bool = False,
    allow_tiny_encoder_for_tests: bool = False,
    internet_enabled: bool = False,
    collapse_fixture: bool = False,
    run_oof: bool = False,
    config_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Planning or tiny-tensor smoke. Real unlabeled SSL is not authorized."""
    assert_run_signature_image_only()
    assert HARD_WEAK_LABEL_TRAINING_ALLOWED is False
    assert SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED is False
    assert SECOND_READER_SCORE_BEARING_ALLOWED is False
    assert UNLABELED_IMAGE_SSL_PLANNING_ALLOWED is True
    assert UNLABELED_IMAGE_SSL_EXECUTION_ALLOWED is False
    assert GOLD58_SSL_OOF_EXECUTION_ALLOWED is False
    if internet_enabled:
        return _payload(STATUS_INTERNET, "internet_must_be_off")
    if run_oof:
        return _payload(STATUS_OOF, "gold58_ssl_oof_not_authorized")
    cfg = load_ssl_config(config_path)
    if config_overrides:
        assert_no_forbidden_training_inputs(config_overrides)
        cfg.update(config_overrides)
    out = Path(output_dir) if output_dir else PROJECT_ROOT / "outputs" / cfg["output_dirname"]
    out.mkdir(parents=True, exist_ok=True)
    extra = {
        "config_sha256": cfg["_config_sha256"],
        "imagenet_weights_sha256": IMAGENET_SHA256,
        "frozen_v1_recipe_sha256": FROZEN_V1_RECIPE_SHA256,
        "frozen_v2_recipe_sha256": FROZEN_V2_CONFIG_SHA256,
        "resource_estimate_n4349": estimate_resources(EXPECTED_UNLABELED_REAL, cfg),
        "notebook_07_executed": False,
        "notebook_09_executed": False,
        "notebook_10_executed": False,
        "notebook_11_executed": False,
        "notebook_12_executed": False,
        "kaggle_kernel_launched": False,
    }
    root = Path(data_root) if data_root is not None else None
    if root is not None:
        try:
            assert_not_hidden_test_path(root, split="train")
        except Phase4FError as exc:
            payload = _payload(exc.status, exc.reason, extra=extra)
            _write_public(out, payload)
            return payload
        source = infer_data_source(root)
        extra["data_source"] = source
        if execute_training and source == DATA_SOURCE_REAL_KAGGLE:
            payload = _payload(
                STATUS_EXECUTION,
                "UNLABELED_IMAGE_SSL_EXECUTION_ALLOWED=false",
                extra=extra,
            )
            _write_public(out, payload)
            return payload

    if execute_training and not allow_tiny_encoder_for_tests:
        payload = _payload(
            STATUS_EXECUTION,
            "UNLABELED_IMAGE_SSL_EXECUTION_ALLOWED=false",
            extra=extra,
        )
        _write_public(out, payload)
        return payload

    if allow_tiny_encoder_for_tests and execute_training:
        try:
            smoke = run_synthetic_smoke(
                cfg, output_dir=out, collapse_fixture=collapse_fixture
            )
        except Phase4FError as exc:
            payload = _payload(exc.status, exc.reason, extra=extra)
            _write_public(out, payload)
            return payload
        extra.update(
            {
                "smoke": smoke,
                "scientific_evaluation": False,
                "allow_tiny_encoder_for_tests": True,
            }
        )
        payload = _payload(STATUS_SMOKE, None, extra=extra)
        _write_public(out, payload)
        return payload

    extra["scientific_evaluation"] = False
    payload = _payload(STATUS_PLANNING, None, extra=extra)
    _write_public(out, payload)
    return payload


def _write_public(output_dir: Path, payload: dict[str, Any]) -> None:
    public_dir = Path(output_dir) / "sanitized"
    public_dir.mkdir(parents=True, exist_ok=True)
    path = public_dir / "phase4f_ssl.json"
    path.write_text(json_dumps(payload), encoding="utf-8")


def json_dumps(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n"


def render_protocol_markdown(cfg: dict[str, Any] | None = None) -> str:
    cfg = cfg or load_ssl_config()
    estimate = estimate_resources(EXPECTED_UNLABELED_REAL, cfg)
    return "\n".join(
        [
            "# Phase 4F image-only SimSiam SSL protocol",
            "",
            "Prepared, not executed on real unlabeled pixels.",
            "",
            f"- protocol: `{PROTOCOL}`",
            f"- kernel slug (prepared, not launched): `{KERNEL_SLUG}`",
            "- SSL objective: SimSiam (Chen & He, CVPR 2021)",
            "- rationale: Tesla T4 16 GiB; gold58 v2 ResNet-18 training peaked at",
            f"  {KAGGLE_GPU_PEAK_BYTES_V2} bytes with batch 2. SimSiam uses two views,",
            "  a projector, and a predictor with stop-gradient. It needs no negative",
            "  queue (SimCLR) and no momentum encoder (MoCo/BYOL), so peak memory",
            "  stays compatible with T4. Frozen batch_size=8 (two views) is expected",
            "  well under 16 GiB. SimCLR is rejected for large-batch negatives.",
            f"- encoder init: ImageNet ResNet-18 SHA-256 `{IMAGENET_SHA256}`",
            f"- series rule: `{SERIES_RULE_V2}` (frozen, not retuned)",
            "- cohort: 4349 non-gold released training studies; gold 58 excluded",
            "- never load Report, target columns, weak labels, second-reader files,",
            "  or hidden-test paths into the dataset or trainer",
            "- projector: 3-layer MLP 512→2048→2048→2048 with BN (tiny smoke: 8→16)",
            "- predictor: 2-layer MLP 2048→512→2048 (tiny smoke: 16→8→16)",
            "- augmentations: study-consistent random resized crop scale [0.6, 1.0],",
            "  hflip p=0.5, intensity scale [0.9, 1.1], shift [-0.05, 0.05]",
            "- crop size / slices: 224, 8 linspace slices, bilinear, ImageNet norm",
            "- optimizer: SGD lr=0.03 momentum=0.9 weight_decay=1e-4 cosine, epochs=2",
            "- AMP: cuda_only; seed=42; checkpoint every 200 steps plus last; resume true",
            "- collapse: mean per-dim std of z < 0.001 fails closed",
            f"- config SHA-256: `{cfg['_config_sha256']}`",
            "",
            "## Resource estimate (from frozen v2/v4E evidence, not a new measurement)",
            "",
            f"- seconds/study unlabeled estimate: {estimate['seconds_per_study_unlabeled_estimate']:.4f}",
            f"- projected seconds / epoch for n=4349: {estimate['projected_seconds_per_epoch']:.1f}",
            f"- projected total (2 epochs): {estimate['projected_seconds_total']:.1f}",
            f"- with safety 1.2: {estimate['projected_seconds_with_safety']:.1f}",
            f"- fits max_wall 39600s: {estimate['fits_private_max_wall']}",
            f"- fits competition 9h scoring: {estimate['fits_competition_scoring_9h']}",
            "  (SSL is not a scoring notebook)",
            f"- checkpoint bytes estimate: {estimate['checkpoint_bytes_estimate']}",
            f"- {estimate['note']}",
            "",
            "## Later OOF (not authorized, not run)",
            "",
            "- identical gold58 v2 folds, bilinear 224, ImageNet preprocess, 8 slices,",
            "  mean aggregation, Adam 1e-4 cosine 12 epochs, official combined OOF",
            "  macro ROC AUC",
            "- compare SSL-v1 OOF to frozen v2 with paired patient bootstrap",
            "- no target-specific weights, sign flips, or OOF-derived calibration",
            "- `GOLD58_SSL_OOF_EXECUTION_ALLOWED=false`",
            "",
            "## Gates",
            "",
            f"- UNLABELED_IMAGE_SSL_PLANNING_ALLOWED={str(UNLABELED_IMAGE_SSL_PLANNING_ALLOWED).lower()}",
            f"- UNLABELED_IMAGE_SSL_EXECUTION_ALLOWED={str(UNLABELED_IMAGE_SSL_EXECUTION_ALLOWED).lower()}",
            f"- GOLD58_SSL_OOF_EXECUTION_ALLOWED={str(GOLD58_SSL_OOF_EXECUTION_ALLOWED).lower()}",
            "- HARD_WEAK_LABEL_TRAINING_ALLOWED=false",
            "- no `/kaggle/working/submission.csv`",
            f"- forbidden outcome not used: `{FORBIDDEN_SEMANTIC_PASS}`",
            "",
        ]
    )


def render_readiness_markdown(result: dict[str, Any], cfg: dict[str, Any] | None = None) -> str:
    cfg = cfg or load_ssl_config()
    smoke = result.get("smoke") or {}
    estimate = result.get("resource_estimate_n4349") or estimate_resources(
        EXPECTED_UNLABELED_REAL, cfg
    )
    return "\n".join(
        [
            "# Phase 4F SSL readiness (planning + synthetic smoke)",
            "",
            "No Kaggle kernel. No real unlabeled pixels. No OOF. No test inference.",
            "No competition submission.",
            "",
            f"- status: `{result.get('status')}`",
            f"- protocol: `{PROTOCOL}`",
            f"- kernel prepared, not launched: `{KERNEL_SLUG}`",
            f"- config SHA-256: `{cfg.get('_config_sha256')}`",
            f"- v1 recipe SHA-256 unchanged: `{FROZEN_V1_RECIPE_SHA256}`",
            f"- v2 recipe SHA-256 unchanged: `{FROZEN_V2_CONFIG_SHA256}`",
            f"- v1.4 config SHA-256 unchanged: `{FROZEN_V14}`",
            f"- ImageNet SHA-256 pin: `{IMAGENET_SHA256}`",
            f"- n_real_unlabelled_studies_processed: {result.get('n_real_unlabelled_studies_processed')}",
            f"- training_run: {str(result.get('training_run')).lower()}",
            f"- OOF_run: {str(result.get('OOF_run')).lower()}",
            f"- inference_run: {str(result.get('inference_run')).lower()}",
            f"- competition_submission_written: {str(result.get('competition_submission_written')).lower()}",
            f"- kaggle_kernel_launched: {str(result.get('kaggle_kernel_launched')).lower()}",
            f"- synthetic smoke steps: {smoke.get('n_steps')}",
            f"- smoke backprop_ran: {smoke.get('backprop_ran')}",
            f"- smoke losses: {smoke.get('losses')}",
            f"- smoke representation_stds: {smoke.get('representation_stds')}",
            f"- encoder_round_trip_max_abs: {smoke.get('encoder_round_trip_max_abs')}",
            f"- projected_seconds_with_safety n=4349: {estimate.get('projected_seconds_with_safety')}",
            "- notebooks 07, 09, 10, 11, 12 were not executed",
            f"- forbidden outcome not used: `{FORBIDDEN_SEMANTIC_PASS}`",
            "",
        ]
    )
