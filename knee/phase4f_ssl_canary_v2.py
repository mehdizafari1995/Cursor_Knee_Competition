"""Phase 4F.2 AMP-safe bounded image-only SimSiam SSL canary.

Versioned separately from frozen Phase 4F v1 and Phase 4F.1. Correct preprocess order is
percentile [0, 1] → spatial/intensity augs on unnormalized tensors → clamp →
ImageNet last. Authorized real-data scope is exactly 64 non-gold studies and
40 optimizer steps with a forced resume at step 20. The full 4,349-study SSL
job remains unauthorized.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
import yaml

from .config import PROJECT_ROOT, REPORT_COLUMN, STUDY_ID_COLUMN
from .dataset import IMAGENET_MEAN, IMAGENET_STD
from .gold58_oof import (
    EXPECTED_LABELLED_REAL,
    IMAGENET_SHA256,
    SERIES_RULE_V2,
    Gold58PreflightError,
    _series_dir,
    build_resnet18_from_verified_file,
    inspect_series_inventory,
    locate_imagenet_weights,
    select_deterministic_primary_series_v2,
    set_deterministic_seeds,
    verify_imagenet_weights,
)
from .kaggle_audit import jsonable
from .kaggle_bundle import sha256_file
from .kaggle_runtime import DATA_SOURCE_REAL_KAGGLE, infer_data_source
from .phase4d_readiness import SECOND_READER_SCORE_BEARING_ALLOWED
from .phase4f_ssl import (
    EXPECTED_UNLABELED_REAL,
    FORBIDDEN_TRAINING_KEYS,
    GOLD58_SSL_OOF_EXECUTION_ALLOWED,
    UNLABELED_IMAGE_SSL_EXECUTION_ALLOWED,
    UNLABELED_IMAGE_SSL_PLANNING_ALLOWED,
    Phase4FError,
    UnlabeledStudySliceDataset,
    SimSiamStudyModel,
    assert_no_forbidden_training_inputs,
    assert_not_hidden_test_path,
    augment_study,
    drop_non_image_columns,
    load_ssl_checkpoint,
    representation_std,
    save_ssl_checkpoint,
    select_unlabeled_study_ids,
    simsiam_loss,
)
from .report_labels import (
    HARD_WEAK_LABEL_TRAINING_ALLOWED,
    SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED,
)
from .second_reader_phase46 import FORBIDDEN_SEMANTIC_PASS, git_head_sha

PROTOCOL = "unlabeled-image-simsiam-ssl-canary-v2"
KERNEL_SLUG = "mahdimzafari/rsna-knee-unlabeled-image-ssl-canary-v2"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "unlabeled_image_simsiam_ssl_canary_v2.yaml"
FROZEN_V1_RECIPE_SHA256 = "92f9d8b52d9488646a4dd19ca9740b8ae658c5f71bf6812a393b91fa3a07af9c"
FROZEN_V2_CONFIG_SHA256 = "4b65bc0d4d56a3dd36b6898c71d630a113c99f6e4b986e51a15ae1b66e1ea093"
FROZEN_V14 = "992af7a4b4483278445a6f3fb363c818611e79005618dff063460d0aed3bd7f7"
FROZEN_SSL_V1_CONFIG_SHA256 = "d0c714f2ec1dc818f140abee21464cc4cf8a25f9a2a882195ae6860fda48ce43"
FROZEN_CANARY_V1_CONFIG_SHA256 = "3c8ad505be9e2c71c1ad3c15f894d7188f0ca1f8cf1752ad4df7d84d7ce25c4c"
CANARY_SOURCE_PATH = PROJECT_ROOT / "knee" / "phase4f_ssl_canary_v2.py"

UNLABELED_IMAGE_SSL_CANARY_V2_EXECUTION_ALLOWED = True
UNLABELED_IMAGE_SSL_CANARY_EXECUTION_ALLOWED = True  # 4F.1 consumed; v2 uses V2 flag

STATUS_PASS = "SSL_CANARY_PASS_FULL_RUN_NOT_AUTHORIZED"
STATUS_FAIL = "SSL_CANARY_FAIL"
STATUS_MID = "SSL_CANARY_MID_CHECKPOINT"
STATUS_GOLD = "GOLD58_INTERSECTION_FAIL"
STATUS_LABELS = "LABEL_OR_REPORT_INPUT_FAIL"
STATUS_HIDDEN = "HIDDEN_TEST_PATH_FAIL"
STATUS_SERIES = "SERIES_SELECTION_PREFLIGHT_FAIL"
STATUS_COLLAPSE = "SSL_COLLAPSE_FAIL"
STATUS_FINITE = "SSL_NONFINITE_LOSS_FAIL"
STATUS_PREPROCESS = "SSL_PREPROCESS_ORDER_FAIL"
STATUS_WEIGHTS = "IMAGENET_WEIGHTS_PREFLIGHT_FAIL"
STATUS_RESUME = "SSL_RESUME_MISMATCH_FAIL"
STATUS_EMPTY = "SSL_EMPTY_BATCH_FAIL"
STATUS_INTERNET = "INTERNET_PREFLIGHT_FAIL"
STATUS_SCOPE = "SSL_CANARY_SCOPE_FAIL"
STATUS_OOF = "GOLD58_SSL_OOF_NOT_AUTHORIZED"
STATUS_FULL = "SSL_EXECUTION_NOT_AUTHORIZED"
STATUS_GPU_PREFLIGHT = "SSL_CANARY_GPU_PREFLIGHT_FAIL"
STATUS_GPU_PREFLIGHT_PASS = "SSL_CANARY_GPU_PREFLIGHT_PASS"

AUTHORIZED_N_STUDIES = 64
AUTHORIZED_STEPS = 40
AUTHORIZED_CHECKPOINT_STEP = 20
AUTHORIZED_BATCH = 8
PROBE_EIGVALSH_MAX_DIM = 16
LAST_EIGVALSH_META: dict[str, Any] = {}

REQUIRED_CHECKPOINT_KEYS = (
    "encoder",
    "projector",
    "predictor",
    "optimizer",
    "scheduler",
    "amp_scaler",
    "epoch",
    "step",
    "sampler_order",
    "sampler_position",
    "python_rng",
    "numpy_rng",
    "torch_rng",
    "torch_cuda_rng",
    "aug_generator_rng",
    "recipe_sha256",
    "source_sha256",
    "config_sha256",
    "protocol",
    "losses",
    "grad_norms",
    "timings",
)


class Phase4FCanaryError(RuntimeError):
    def __init__(self, status: str, reason: str):
        super().__init__(reason)
        self.status = status
        self.reason = reason


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n"


def load_canary_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{cfg_path} must contain a mapping")
    payload["_config_path"] = str(cfg_path)
    payload["_config_sha256"] = sha256_file(cfg_path)
    return payload


def canary_source_sha256() -> str:
    return sha256_file(CANARY_SOURCE_PATH)


def assert_run_signature_image_only(fn: Any | None = None) -> None:
    target = fn if fn is not None else run_phase4f_ssl_canary_v2
    names = set(inspect.signature(target).parameters)
    overlap = sorted(names & FORBIDDEN_TRAINING_KEYS)
    if overlap:
        raise Phase4FCanaryError(STATUS_LABELS, "run_signature_contains_forbidden_keys")


def select_canary_study_ids(
    unlabeled_ids: list[str],
    *,
    n: int,
    seed: int,
) -> list[str]:
    """Deterministic SHA-256 order. Never logs identifiers."""
    if int(n) < 1:
        raise Phase4FCanaryError(STATUS_SCOPE, "n_canary_studies_invalid")
    unique: list[str] = []
    seen: set[str] = set()
    for uid in unlabeled_ids:
        text = str(uid)
        if text in seen:
            continue
        seen.add(text)
        unique.append(text)
    if len(unique) < int(n):
        raise Phase4FCanaryError(
            STATUS_SCOPE, f"unlabeled_pool_smaller_than_canary:{len(unique)}<{int(n)}"
        )
    ranked = sorted(
        unique,
        key=lambda uid: hashlib.sha256(f"{int(seed)}|{uid}".encode("utf-8")).hexdigest(),
    )
    return ranked[: int(n)]


def assert_zero_gold_overlap(canary_ids: list[str], gold_ids: list[str]) -> int:
    overlap = set(map(str, canary_ids)) & set(map(str, gold_ids))
    n_overlap = int(len(overlap))
    if n_overlap:
        raise Phase4FCanaryError(STATUS_GOLD, f"n_overlap={n_overlap}")
    return 0


def assert_unit_interval_for_intensity_jitter(
    image: torch.Tensor, *, eps: float = 1e-5
) -> None:
    """Fail if intensity jitter would run on ImageNet-normalized tensors."""
    if image.numel() == 0:
        raise Phase4FCanaryError(STATUS_EMPTY, "empty_tensor_before_intensity_jitter")
    if image.ndim != 4:
        raise Phase4FCanaryError(STATUS_PREPROCESS, "expected_schw_before_intensity_jitter")
    if int(image.size(1)) != 1:
        raise Phase4FCanaryError(
            STATUS_PREPROCESS,
            f"intensity_jitter_requires_grayscale_got_{int(image.size(1))}",
        )
    lo = float(image.min())
    hi = float(image.max())
    if (not math.isfinite(lo)) or (not math.isfinite(hi)):
        raise Phase4FCanaryError(STATUS_PREPROCESS, "nonfinite_before_intensity_jitter")
    if lo < -float(eps) or hi > 1.0 + float(eps):
        raise Phase4FCanaryError(
            STATUS_PREPROCESS,
            f"intensity_jitter_not_on_unit_interval:min={lo:.6f}:max={hi:.6f}",
        )


def apply_imagenet_last(
    image: torch.Tensor,
    *,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> torch.Tensor:
    """Repeat grayscale to RGB if needed, then ImageNet mean/std. Last step."""
    if image.ndim != 4:
        raise ValueError(f"expected (S, C, H, W), got {tuple(image.shape)}")
    if image.size(1) == 1:
        image = image.repeat(1, 3, 1, 1)
    elif image.size(1) != 3:
        raise ValueError(f"expected 1 or 3 channels, got {int(image.size(1))}")
    mean_t = image.new_tensor(mean).view(1, 3, 1, 1)
    std_t = image.new_tensor(std).view(1, 3, 1, 1)
    if bool((std_t == 0).any()):
        raise Phase4FCanaryError(STATUS_PREPROCESS, "imagenet_std_zero")
    return (image - mean_t) / std_t


def augment_canary_study(
    image: torch.Tensor,
    *,
    generator: torch.Generator,
    crop_scale: tuple[float, float],
    hflip_p: float,
    intensity_scale: tuple[float, float],
    intensity_shift: tuple[float, float],
    clamp_min: float,
    clamp_max: float,
    unit_interval_eps: float,
) -> torch.Tensor:
    assert_unit_interval_for_intensity_jitter(image, eps=unit_interval_eps)
    jittered = augment_study(
        image,
        generator=generator,
        crop_scale=crop_scale,
        hflip_p=hflip_p,
        intensity_scale=intensity_scale,
        intensity_shift=intensity_shift,
    )
    return torch.clamp(jittered, min=float(clamp_min), max=float(clamp_max))


def two_views_canary(
    images: torch.Tensor,
    *,
    generator: torch.Generator,
    cfg: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    crop = tuple(cfg.get("crop_scale") or (0.6, 1.0))
    intensity_scale = tuple(cfg.get("intensity_scale") or (0.9, 1.1))
    intensity_shift = tuple(cfg.get("intensity_shift") or (-0.05, 0.05))
    views1 = []
    views2 = []
    for row in images:
        left = augment_canary_study(
            row,
            generator=generator,
            crop_scale=(float(crop[0]), float(crop[1])),
            hflip_p=float(cfg.get("hflip_p") or 0.5),
            intensity_scale=(float(intensity_scale[0]), float(intensity_scale[1])),
            intensity_shift=(float(intensity_shift[0]), float(intensity_shift[1])),
            clamp_min=float(cfg.get("clamp_min") or 0.0),
            clamp_max=float(cfg.get("clamp_max") or 1.0),
            unit_interval_eps=float(cfg.get("unit_interval_eps") or 1e-5),
        )
        right = augment_canary_study(
            row,
            generator=generator,
            crop_scale=(float(crop[0]), float(crop[1])),
            hflip_p=float(cfg.get("hflip_p") or 0.5),
            intensity_scale=(float(intensity_scale[0]), float(intensity_scale[1])),
            intensity_shift=(float(intensity_shift[0]), float(intensity_shift[1])),
            clamp_min=float(cfg.get("clamp_min") or 0.0),
            clamp_max=float(cfg.get("clamp_max") or 1.0),
            unit_interval_eps=float(cfg.get("unit_interval_eps") or 1e-5),
        )
        views1.append(apply_imagenet_last(left))
        views2.append(apply_imagenet_last(right))
    return torch.stack(views1, dim=0), torch.stack(views2, dim=0)


class TinyCanarySliceEncoder(nn.Module):
    """CPU test encoder. Adaptive 2x2 pool keeps enough spatial diversity."""

    def __init__(self, in_channels: int = 3, width: int = 16):
        super().__init__()
        self.out_dim = width * 4
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(2),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CanaryUnlabeledStudySliceDataset:
    """[0, 1] grayscale only. Rejects ImageNet-in-dataset and label kwargs."""

    def __init__(
        self,
        study_ids: list[str],
        series_dirs: list[Path],
        *,
        num_slices: int = 8,
        image_size: int = 224,
        percentiles: tuple[float, float] = (1.0, 99.0),
        interpolation: str = "bilinear",
        imagenet_normalize: bool = False,
        **forbidden: Any,
    ):
        if forbidden:
            raise Phase4FCanaryError(STATUS_LABELS, f"dataset_forbidden_kwargs:{len(forbidden)}")
        if imagenet_normalize:
            raise Phase4FCanaryError(
                STATUS_PREPROCESS, "canary_dataset_must_not_imagenet_normalize"
            )
        self._inner = UnlabeledStudySliceDataset(
            study_ids,
            series_dirs,
            num_slices=num_slices,
            image_size=image_size,
            percentiles=percentiles,
            interpolation=interpolation,
            imagenet_normalize=False,
        )

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self._inner[idx]
        image = item["image"]
        assert_unit_interval_for_intensity_jitter(image)
        if image.size(1) != 1:
            raise Phase4FCanaryError(
                STATUS_PREPROCESS, f"expected_grayscale_channel_got_{int(image.size(1))}"
            )
        return {"image": image}


def mean_abs_pairwise_cosine(z: torch.Tensor) -> float:
    if z.ndim != 2 or z.shape[0] < 2:
        return float("nan")
    normed = F.normalize(z.detach(), dim=-1, eps=1e-8)
    sim = normed @ normed.T
    n = int(sim.size(0))
    mask = ~torch.eye(n, dtype=torch.bool, device=sim.device)
    values = sim[mask]
    if values.numel() < 1:
        return float("nan")
    return float(values.abs().mean().cpu())


def _eigvalsh_float32_gram(gram: torch.Tensor) -> torch.Tensor:
    """Record dtype/shape then eigendecompose a float32 n×n Gram matrix."""
    if gram.dtype != torch.float32:
        gram = gram.float()
    LAST_EIGVALSH_META.clear()
    LAST_EIGVALSH_META.update(
        {
            "dtype": str(gram.dtype),
            "is_float32": bool(gram.dtype == torch.float32),
            "shape": [int(dim) for dim in gram.shape],
            "device_type": gram.device.type,
            "max_dim": int(max(gram.shape)) if gram.ndim else 0,
        }
    )
    return torch.linalg.eigvalsh(gram)


def effective_rank(z: torch.Tensor) -> float:
    """Effective rank from the centered sample Gram ``X @ X.T / (n-1)`` in float32.

    The nonzero spectrum matches feature covariance. The canary probe has 16 rows,
    so this is a 16×16 solve instead of a 2048×2048 feature covariance.
    """
    z = z.detach().float()
    if z.ndim != 2 or z.shape[0] < 2 or z.shape[1] < 1:
        return float("nan")
    n = int(z.shape[0])
    centered = z - z.mean(dim=0, keepdim=True)
    denom = max(n - 1, 1)
    gram = (centered @ centered.T) / float(denom)
    eigs = _eigvalsh_float32_gram(gram).clamp(min=0)
    total = float(eigs.sum().cpu())
    if not math.isfinite(total) or total <= 0.0:
        return 0.0
    probs = (eigs / eigs.sum()).clamp(min=0)
    nonzero = probs[probs > 0]
    if nonzero.numel() < 1:
        return 0.0
    entropy = float((-(nonzero * nonzero.log())).sum().cpu())
    return float(math.exp(entropy))


def collapse_thresholds(cfg: dict[str, Any]) -> dict[str, float]:
    return {
        "encoder_std_min": float(cfg.get("collapse_encoder_std_min") or 0.001),
        "projector_std_min": float(cfg.get("collapse_projector_std_min") or 0.001),
        "predictor_std_min": float(cfg.get("collapse_predictor_std_min") or 0.001),
        "mean_abs_cosine_max": float(cfg.get("collapse_mean_abs_cosine_max") or 0.99),
        "effective_rank_min": float(cfg.get("collapse_effective_rank_min") or 1.5),
    }


def compute_collapse_diagnostics(
    encoder_z: torch.Tensor,
    projector_z: torch.Tensor,
    predictor_p: torch.Tensor,
    *,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    encoder_z = encoder_z.detach().float()
    projector_z = projector_z.detach().float()
    predictor_p = predictor_p.detach().float()
    thresholds = collapse_thresholds(cfg)
    encoder_std = representation_std(encoder_z)
    projector_std = representation_std(projector_z)
    predictor_std = representation_std(predictor_p)
    cosine = mean_abs_pairwise_cosine(projector_z)
    rank = effective_rank(projector_z)
    reasons: list[str] = []
    if (not math.isfinite(encoder_std)) or encoder_std < thresholds["encoder_std_min"]:
        reasons.append("encoder_std")
    if (not math.isfinite(projector_std)) or projector_std < thresholds["projector_std_min"]:
        reasons.append("projector_std")
    if (not math.isfinite(predictor_std)) or predictor_std < thresholds["predictor_std_min"]:
        reasons.append("predictor_std")
    if (not math.isfinite(cosine)) or cosine > thresholds["mean_abs_cosine_max"]:
        reasons.append("mean_abs_cosine")
    if (not math.isfinite(rank)) or rank < thresholds["effective_rank_min"]:
        reasons.append("effective_rank")
    return {
        "encoder_std": encoder_std,
        "projector_std": projector_std,
        "predictor_std": predictor_std,
        "mean_abs_pairwise_cosine": cosine,
        "effective_rank": rank,
        "thresholds": thresholds,
        "collapsed": bool(reasons),
        "reasons": reasons,
    }


def assert_collapse_ok(diagnostics: dict[str, Any]) -> None:
    if diagnostics.get("collapsed"):
        reasons = ",".join(diagnostics.get("reasons") or [])
        raise Phase4FCanaryError(STATUS_COLLAPSE, f"collapse_gate:{reasons}")


def std_only_diagnostics(
    encoder_z: torch.Tensor,
    projector_z: torch.Tensor,
    predictor_p: torch.Tensor,
    *,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Per-step std gates only.

    Must not call pairwise cosine, covariance, SVD, eigendecomposition, or the
    full probe function. Cosine and effective rank run once on the frozen probe
    set after the training segment.
    """
    encoder_z = encoder_z.detach().float()
    projector_z = projector_z.detach().float()
    predictor_p = predictor_p.detach().float()
    thresholds = collapse_thresholds(cfg)
    encoder_std = representation_std(encoder_z)
    projector_std = representation_std(projector_z)
    predictor_std = representation_std(predictor_p)
    reasons: list[str] = []
    if (not math.isfinite(encoder_std)) or encoder_std < thresholds["encoder_std_min"]:
        reasons.append("encoder_std")
    if (not math.isfinite(projector_std)) or projector_std < thresholds["projector_std_min"]:
        reasons.append("projector_std")
    if (not math.isfinite(predictor_std)) or predictor_std < thresholds["predictor_std_min"]:
        reasons.append("predictor_std")
    return {
        "encoder_std": encoder_std,
        "projector_std": projector_std,
        "predictor_std": predictor_std,
        "thresholds": {
            "encoder_std_min": thresholds["encoder_std_min"],
            "projector_std_min": thresholds["projector_std_min"],
            "predictor_std_min": thresholds["predictor_std_min"],
        },
        "collapsed": bool(reasons),
        "reasons": reasons,
        "std_only": True,
    }


def simsiam_loss_amp_safe(
    p1: torch.Tensor,
    z2: torch.Tensor,
    p2: torch.Tensor,
    z1: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Symmetric negative cosine similarity in float32. Stop-grad on z.

    Does not call the frozen Phase 4F v1 ``simsiam_loss``. Gradients flow back
    to the autocast predictor/projector outputs through the float32 casts.
    """
    # Keep the frozen v1 symbol imported so tests can prove this path never
    # calls it. Do not invoke ``simsiam_loss`` here.
    assert simsiam_loss is not None
    p1_f = p1.float()
    p2_f = p2.float()
    z1_f = z1.detach().float()
    z2_f = z2.detach().float()
    left = F.cosine_similarity(p1_f, z2_f, dim=-1, eps=float(eps)).mean()
    right = F.cosine_similarity(p2_f, z1_f, dim=-1, eps=float(eps)).mean()
    return (-(left + right) / 2.0).float()


def _grads_all_finite(model: nn.Module) -> bool:
    for param in model.parameters():
        if param.grad is None:
            continue
        if not torch.isfinite(param.grad).all():
            return False
    return True


def validate_collapse_fixtures(cfg: dict[str, Any]) -> dict[str, Any]:
    """Frozen thresholds vs non-collapsed and collapsed synthetic tensors only."""
    rng = torch.Generator().manual_seed(0)
    diverse = torch.randn(16, 32, generator=rng)
    collapsed = torch.ones(16, 32)
    ok = compute_collapse_diagnostics(diverse, diverse, diverse, cfg=cfg)
    bad = compute_collapse_diagnostics(collapsed, collapsed, collapsed, cfg=cfg)
    if ok["collapsed"]:
        raise Phase4FCanaryError(STATUS_COLLAPSE, "collapse_fixture_false_positive")
    if not bad["collapsed"]:
        raise Phase4FCanaryError(STATUS_COLLAPSE, "collapse_fixture_false_negative")
    return {
        "non_collapsed_detected": False,
        "collapsed_detected": True,
        "non_collapsed": {k: ok[k] for k in ("encoder_std", "projector_std", "predictor_std", "mean_abs_pairwise_cosine", "effective_rank")},
        "collapsed": {k: bad[k] for k in ("encoder_std", "projector_std", "predictor_std", "mean_abs_pairwise_cosine", "effective_rank")},
        "thresholds": collapse_thresholds(cfg),
    }


def assert_not_empty_batch(images: torch.Tensor) -> None:
    if images.ndim < 1 or int(images.size(0)) < 1:
        raise Phase4FCanaryError(STATUS_EMPTY, "empty_batch")


def finite_grad_norm(model: nn.Module) -> float:
    total = 0.0
    for param in model.parameters():
        if param.grad is None:
            continue
        if not torch.isfinite(param.grad).all():
            raise Phase4FCanaryError(STATUS_FINITE, "nonfinite_gradient")
        total += float(param.grad.detach().float().norm(2).cpu() ** 2)
    norm = math.sqrt(total)
    if not math.isfinite(norm):
        raise Phase4FCanaryError(STATUS_FINITE, "nonfinite_grad_norm")
    return norm


def capture_rng_states(generator: torch.Generator) -> dict[str, Any]:
    cuda_state: list[Any] = []
    if torch.cuda.is_available():
        cuda_state = [state.detach().cpu().clone() for state in torch.cuda.get_rng_state_all()]
    return {
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state().detach().cpu().clone(),
        "torch_cuda_rng": cuda_state,
        "aug_generator_rng": generator.get_state().detach().cpu().clone(),
    }


def restore_rng_states(payload: dict[str, Any], generator: torch.Generator) -> None:
    python_rng = payload.get("python_rng")
    numpy_rng = payload.get("numpy_rng")
    torch_rng = payload.get("torch_rng")
    cuda_rng = payload.get("torch_cuda_rng")
    aug_rng = payload.get("aug_generator_rng")
    if python_rng is None or numpy_rng is None or torch_rng is None or aug_rng is None:
        raise Phase4FCanaryError(STATUS_RESUME, "missing_rng_state")
    random.setstate(python_rng)
    np.random.set_state(numpy_rng)
    torch.set_rng_state(torch_rng)
    if torch.cuda.is_available() and cuda_rng:
        torch.cuda.set_rng_state_all(cuda_rng)
    generator.set_state(aug_rng)


def build_checkpoint_payload(
    *,
    model: SimSiamStudyModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    step: int,
    sampler_order: list[int],
    sampler_position: int,
    generator: torch.Generator,
    cfg: dict[str, Any],
    losses: list[float],
    grad_norms: list[float],
    timings: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "encoder": {k: v.detach().cpu().clone() for k, v in model.encoder.state_dict().items()},
        "projector": {k: v.detach().cpu().clone() for k, v in model.projector.state_dict().items()},
        "predictor": {k: v.detach().cpu().clone() for k, v in model.predictor.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "amp_scaler": scaler.state_dict(),
        "epoch": int(epoch),
        "step": int(step),
        "sampler_order": list(sampler_order),
        "sampler_position": int(sampler_position),
        "recipe_sha256": str(cfg["_config_sha256"]),
        "source_sha256": canary_source_sha256(),
        "config_sha256": str(cfg["_config_sha256"]),
        "protocol": PROTOCOL,
        "losses": list(losses),
        "grad_norms": list(grad_norms),
        "timings": dict(timings),
        "imagenet_weights_sha256": str(cfg.get("imagenet_weights_sha256") or IMAGENET_SHA256),
    }
    payload.update(capture_rng_states(generator))
    return payload


def assert_checkpoint_complete(payload: dict[str, Any]) -> None:
    missing = [key for key in REQUIRED_CHECKPOINT_KEYS if key not in payload]
    if missing:
        raise Phase4FCanaryError(STATUS_RESUME, f"missing_checkpoint_keys:{len(missing)}")


def assert_resume_hashes(payload: dict[str, Any], cfg: dict[str, Any]) -> None:
    if str(payload.get("protocol")) != PROTOCOL:
        raise Phase4FCanaryError(STATUS_RESUME, "checkpoint_protocol_mismatch")
    if str(payload.get("recipe_sha256")) != str(cfg["_config_sha256"]):
        raise Phase4FCanaryError(STATUS_RESUME, "checkpoint_recipe_hash_mismatch")
    if str(payload.get("config_sha256")) != str(cfg["_config_sha256"]):
        raise Phase4FCanaryError(STATUS_RESUME, "checkpoint_config_hash_mismatch")
    if str(payload.get("source_sha256")) != canary_source_sha256():
        raise Phase4FCanaryError(STATUS_RESUME, "checkpoint_source_hash_mismatch")


def epoch_order(n_studies: int, *, seed: int, epoch: int) -> list[int]:
    generator = torch.Generator().manual_seed(int(seed) + 10007 * int(epoch))
    return torch.randperm(int(n_studies), generator=generator).tolist()


def next_batch_indices(
    *,
    n_studies: int,
    batch_size: int,
    seed: int,
    epoch: int,
    position: int,
    order: list[int],
) -> tuple[list[int], int, int, list[int]]:
    if n_studies < 1:
        raise Phase4FCanaryError(STATUS_EMPTY, "no_studies_in_sampler")
    cur_epoch = int(epoch)
    cur_pos = int(position)
    cur_order = list(order) if order else epoch_order(n_studies, seed=seed, epoch=cur_epoch)
    indices: list[int] = []
    while len(indices) < int(batch_size):
        if cur_pos >= len(cur_order):
            cur_epoch += 1
            cur_order = epoch_order(n_studies, seed=seed, epoch=cur_epoch)
            cur_pos = 0
        indices.append(int(cur_order[cur_pos]))
        cur_pos += 1
    return indices, cur_epoch, cur_pos, cur_order


def project_full_run_from_canary(
    timings: dict[str, Any], cfg: dict[str, Any]
) -> dict[str, Any]:
    n_full = int(cfg.get("expected_n_unlabeled_real") or EXPECTED_UNLABELED_REAL)
    epochs = int(cfg.get("full_run_epochs") or 2)
    batch = int(cfg.get("batch_size") or AUTHORIZED_BATCH)
    steps_per_epoch = int(math.ceil(n_full / max(batch, 1)))
    n_steps = steps_per_epoch * epochs
    ckpt_every = int(cfg.get("full_checkpoint_every_steps") or 200)
    n_ckpts = (n_steps // max(ckpt_every, 1)) + 1
    setup = float(timings.get("dataset_series_setup_seconds") or 0.0)
    dicom_per_study = float(timings.get("dicom_seconds_per_study") or 0.0)
    step_s = float(timings.get("seconds_per_optimizer_step") or 0.0)
    retry_s = float(timings.get("amp_retry_seconds_per_applied_step") or 0.0)
    save_s = float(timings.get("checkpoint_save_seconds") or 0.0)
    load_s = float(timings.get("checkpoint_load_seconds") or 0.0)
    projected = (
        setup
        + dicom_per_study * n_full * epochs
        + (step_s + retry_s) * n_steps
        + save_s * n_ckpts
        + load_s
    )
    safety = float(cfg.get("safety_factor") or 1.25)
    guarded = projected * safety
    limit = float(cfg.get("max_wall_seconds") or 39600)
    reserve_need = float(cfg.get("shutdown_reserve_seconds") or 3600)
    reserve = limit - guarded
    return {
        "n_full_studies": n_full,
        "full_run_epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "n_optimizer_steps_full": n_steps,
        "n_checkpoints_full": n_ckpts,
        "projected_seconds": projected,
        "safety_factor": safety,
        "projected_seconds_with_safety": guarded,
        "max_wall_seconds": limit,
        "shutdown_reserve_seconds_required": reserve_need,
        "shutdown_reserve_seconds_remaining": reserve,
        "fits_guarded_wall": bool(guarded <= limit),
        "fits_shutdown_reserve": bool(reserve >= reserve_need),
        "full_run_fits_private_wall": bool(guarded <= limit and reserve >= reserve_need),
        "amp_retry_seconds_per_applied_step": retry_s,
        "formula": (
            "setup + dicom_seconds_per_study*4349*2 + "
            "(seconds_per_applied_step + amp_retry_seconds_per_applied_step)*n_steps "
            "+ checkpoint_save*n_ckpts + checkpoint_load, then *1.25; "
            "require remaining reserve >= 3600 within 39600"
        ),
    }


def _aug_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    crop = tuple(cfg.get("crop_scale") or (0.6, 1.0))
    intensity_scale = tuple(cfg.get("intensity_scale") or (0.9, 1.1))
    intensity_shift = tuple(cfg.get("intensity_shift") or (-0.05, 0.05))
    return {
        "crop_scale": (float(crop[0]), float(crop[1])),
        "hflip_p": float(cfg.get("hflip_p") or 0.5),
        "intensity_scale": (float(intensity_scale[0]), float(intensity_scale[1])),
        "intensity_shift": (float(intensity_shift[0]), float(intensity_shift[1])),
    }


def build_canary_model(
    cfg: dict[str, Any],
    *,
    tiny: bool,
    weights_path: Path | None,
) -> SimSiamStudyModel:
    if tiny:
        encoder: nn.Module = TinyCanarySliceEncoder(in_channels=3, width=16)
        return SimSiamStudyModel(
            encoder,
            projector_hidden=int(cfg.get("tiny_projector_hidden") or 32),
            projector_out=int(cfg.get("tiny_projector_out") or 32),
            predictor_hidden=int(cfg.get("tiny_predictor_hidden") or 16),
            aggregation="mean",
            tiny=True,
        )
    if weights_path is None:
        raise Phase4FCanaryError(STATUS_WEIGHTS, "imagenet_weights_missing")
    supervised = build_resnet18_from_verified_file(weights_path)
    return SimSiamStudyModel(
        supervised.encoder,
        projector_hidden=int(cfg.get("projector_hidden_dim") or 2048),
        projector_out=int(cfg.get("projector_out_dim") or 2048),
        predictor_hidden=int(cfg.get("predictor_hidden_dim") or 512),
        aggregation="mean",
        tiny=False,
    )


def _make_optimizer_bundle(
    model: SimSiamStudyModel, cfg: dict[str, Any], *, device: torch.device
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.CosineAnnealingLR, torch.cuda.amp.GradScaler, bool]:
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(cfg.get("learning_rate") or 0.03),
        momentum=float(cfg.get("momentum") or 0.9),
        weight_decay=float(cfg.get("weight_decay") or 1e-4),
    )
    t_max = max(int(cfg.get("max_optimizer_steps") or AUTHORIZED_STEPS), 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
    use_amp = bool(cfg.get("amp") == "cuda_only" and device.type == "cuda")
    init_scale = float(cfg.get("amp_scaler_init_scale") or 1024.0)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
        init_scale=init_scale,
        growth_factor=float(cfg.get("amp_scaler_growth_factor") or 2.0),
        backoff_factor=float(cfg.get("amp_scaler_backoff_factor") or 0.5),
        growth_interval=int(cfg.get("amp_scaler_growth_interval") or 2000),
    )
    return optimizer, scheduler, scaler, use_amp


def restore_training_state(
    *,
    model: SimSiamStudyModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    generator: torch.Generator,
    payload: dict[str, Any],
    cfg: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    assert_checkpoint_complete(payload)
    assert_resume_hashes(payload, cfg)
    model.encoder.load_state_dict(payload["encoder"])
    model.projector.load_state_dict(payload["projector"])
    model.predictor.load_state_dict(payload["predictor"])
    model.to(device)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    scaler.load_state_dict(payload["amp_scaler"])
    restore_rng_states(payload, generator)
    return payload


def _one_optimizer_attempt(
    *,
    model: SimSiamStudyModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    batch: torch.Tensor,
    generator: torch.Generator,
    cfg: dict[str, Any],
    device: torch.device,
    use_amp: bool,
    collapse_fixture: bool,
    force_overflow: bool,
    fail_status: str,
) -> dict[str, Any]:
    """One AMP attempt. Caller restores the aug generator before retries.

    Does not call frozen ``simsiam_loss``. An applied step is one that actually
    updates parameters. Scheduler steps only after an applied optimizer step.
    """
    assert_not_empty_batch(batch)
    batch = batch.to(device)
    scale_before = float(scaler.get_scale())
    v1, v2 = two_views_canary(batch, generator=generator, cfg=cfg)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", enabled=use_amp):
        p1, z1, p2, z2 = model.forward_views(v1, v2)
        encoder_z = torch.cat([model.encode(v1), model.encode(v2)], dim=0)
        projector_z = torch.cat([z1, z2], dim=0)
        predictor_p = torch.cat([p1, p2], dim=0)
        if collapse_fixture:
            encoder_z = torch.zeros_like(encoder_z)
            projector_z = torch.zeros_like(projector_z)
            predictor_p = torch.zeros_like(predictor_p)
            z1 = torch.zeros_like(z1)
            z2 = torch.zeros_like(z2)
            p1 = torch.zeros_like(p1)
            p2 = torch.zeros_like(p2)
    diagnostics = std_only_diagnostics(encoder_z, projector_z, predictor_p, cfg=cfg)
    assert_collapse_ok(diagnostics)
    loss = simsiam_loss_amp_safe(
        p1, z2, p2, z1, eps=float(cfg.get("amp_loss_eps") or 1e-6)
    )
    loss_dtype = str(loss.dtype)
    if (not torch.isfinite(loss)) or force_overflow:
        if torch.isfinite(loss):
            scaler.scale(loss).backward()
        injected = False
        for param in model.parameters():
            if param.grad is None:
                param.grad = torch.full_like(param, float("inf"))
            else:
                param.grad.fill_(float("inf"))
            injected = True
            break
        if injected:
            scaler.unscale_(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        return {
            "applied": False,
            "skipped": True,
            "reason": "amp_overflow_or_nonfinite_loss",
            "scale_before": scale_before,
            "scale_after": float(scaler.get_scale()),
            "loss": float("nan") if not torch.isfinite(loss) else float(loss.detach().cpu()),
            "loss_dtype": loss_dtype,
            "grad_norm": float("nan"),
            "diagnostics": diagnostics,
        }
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    if not _grads_all_finite(model):
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        return {
            "applied": False,
            "skipped": True,
            "reason": "amp_overflow_nonfinite_gradient",
            "scale_before": scale_before,
            "scale_after": float(scaler.get_scale()),
            "loss": float(loss.detach().cpu()),
            "loss_dtype": loss_dtype,
            "grad_norm": float("nan"),
            "diagnostics": diagnostics,
        }
    grad_norm = finite_grad_norm(model)
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    return {
        "applied": True,
        "skipped": False,
        "reason": None,
        "scale_before": scale_before,
        "scale_after": float(scaler.get_scale()),
        "loss": float(loss.detach().cpu()),
        "loss_dtype": loss_dtype,
        "grad_norm": grad_norm,
        "diagnostics": diagnostics,
    }


def _one_logical_optimizer_step(
    *,
    model: SimSiamStudyModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    batch: torch.Tensor,
    generator: torch.Generator,
    cfg: dict[str, Any],
    device: torch.device,
    use_amp: bool,
    collapse_fixture: bool,
    force_overflow_attempts: int = 0,
    fail_status: str = STATUS_FAIL,
) -> dict[str, Any]:
    """Retry the same logical step after AMP overflow. Sampler is not advanced here."""
    max_backoffs = int(cfg.get("max_consecutive_amp_backoffs") or 3)
    gen_state = generator.get_state().detach().cpu().clone()
    skips: list[dict[str, Any]] = []
    last: dict[str, Any] = {}
    retry_seconds = 0.0
    applied_seconds = 0.0
    for attempt in range(max_backoffs + 1):
        generator.set_state(gen_state)
        snapshot = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
        t0 = time.perf_counter()
        last = _one_optimizer_attempt(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            batch=batch,
            generator=generator,
            cfg=cfg,
            device=device,
            use_amp=use_amp,
            collapse_fixture=collapse_fixture,
            force_overflow=bool(attempt < int(force_overflow_attempts)),
            fail_status=fail_status,
        )
        elapsed = time.perf_counter() - t0
        if last.get("applied"):
            applied_seconds = elapsed
            last["n_attempts"] = attempt + 1
            last["n_backoffs"] = attempt
            last["skips"] = skips
            last["applied_seconds"] = applied_seconds
            last["retry_seconds"] = retry_seconds
            return last
        model.load_state_dict(snapshot)
        model.to(device)
        retry_seconds += elapsed
        skips.append(
            {
                "attempt": attempt + 1,
                "reason": last.get("reason"),
                "scale_before": last.get("scale_before"),
                "scale_after": last.get("scale_after"),
                "seconds": elapsed,
            }
        )
    raise Phase4FCanaryError(
        fail_status,
        f"amp_backoff_exhausted:{len(skips)}",
    )


def _one_optimizer_step(
    *,
    model: SimSiamStudyModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    batch: torch.Tensor,
    generator: torch.Generator,
    cfg: dict[str, Any],
    device: torch.device,
    use_amp: bool,
    collapse_fixture: bool,
    force_overflow_attempts: int = 0,
    fail_status: str = STATUS_FAIL,
) -> tuple[float, float, dict[str, Any]]:
    """Compatibility wrapper: returns loss, grad_norm, diagnostics after an applied step."""
    result = _one_logical_optimizer_step(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        batch=batch,
        generator=generator,
        cfg=cfg,
        device=device,
        use_amp=use_amp,
        collapse_fixture=collapse_fixture,
        force_overflow_attempts=force_overflow_attempts,
        fail_status=fail_status,
    )
    diagnostics = dict(result["diagnostics"])
    diagnostics["amp"] = {
        "applied": True,
        "n_attempts": result["n_attempts"],
        "n_backoffs": result["n_backoffs"],
        "scale_before": result["scale_before"],
        "scale_after": result["scale_after"],
        "loss_dtype": result["loss_dtype"],
        "skips": result["skips"],
        "applied_seconds": result.get("applied_seconds"),
        "retry_seconds": result.get("retry_seconds"),
    }
    return float(result["loss"]), float(result["grad_norm"]), diagnostics


def _sanitize_gpu_preflight(result: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "status",
        "ok",
        "skip_reason",
        "gpu_name",
        "use_amp",
        "device_type",
        "eigvalsh",
        "eigvalsh_calls_during_optimizer_step",
        "imagenet_weights_sha256_verified",
        "n_real_unlabelled_studies_pixels_opened",
        "n_report_text_rows_processed",
        "n_second_reader_rows_processed",
        "loss_finite",
        "loss_dtype",
        "grad_norm_finite",
        "applied_optimizer_steps",
        "skipped_attempts",
        "n_backoffs",
        "amp_scaler_init_scale",
        "scaler_initial_scale",
        "scaler_final_scale",
        "peak_gpu_memory_bytes",
        "preflight_batch_shape",
        "std_only",
        "probe_collapsed",
        "probe_effective_rank",
        "probe_mean_abs_pairwise_cosine",
    )
    return {key: result.get(key) for key in keys}


def run_canary_amp_software_preflight(
    *,
    cfg: dict[str, Any],
    model: SimSiamStudyModel,
    device: torch.device,
    use_amp: bool,
    scientific_shape: bool = True,
) -> dict[str, Any]:
    """Zero-pixel AMP software check. Must not retune thresholds or the recipe."""
    model = model.to(device)
    set_deterministic_seeds(int(cfg.get("seed") or 42))
    optimizer, scheduler, scaler, _computed_amp = _make_optimizer_bundle(
        model, cfg, device=device
    )
    if device.type != "cuda":
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=False,
            init_scale=float(cfg.get("amp_scaler_init_scale") or 1024.0),
            growth_factor=float(cfg.get("amp_scaler_growth_factor") or 2.0),
            backoff_factor=float(cfg.get("amp_scaler_backoff_factor") or 0.5),
            growth_interval=int(cfg.get("amp_scaler_growth_interval") or 2000),
        )
        use_amp = False
    declared_init = float(cfg.get("amp_scaler_init_scale") or 1024.0)
    initial_scale = float(scaler.get_scale()) if scaler.is_enabled() else declared_init
    generator = torch.Generator(device="cpu").manual_seed(int(cfg.get("seed") or 42))
    if scientific_shape:
        n_b = int(cfg.get("gpu_preflight_batch_size") or AUTHORIZED_BATCH)
        n_s = int(cfg.get("gpu_preflight_num_slices") or int(cfg.get("num_slices") or 8))
        n_h = int(cfg.get("gpu_preflight_image_size") or int(cfg.get("image_size") or 224))
    else:
        n_b, n_s, n_h = 2, 2, 16
    shape_gen = torch.Generator().manual_seed(int(cfg.get("seed") or 42))
    batch = torch.rand(n_b, n_s, 1, n_h, n_h, generator=shape_gen)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    eig_calls = {"n": 0}
    orig_eig = torch.linalg.eigvalsh

    def _count_eig(*args: Any, **kwargs: Any) -> torch.Tensor:
        eig_calls["n"] += 1
        return orig_eig(*args, **kwargs)

    torch.linalg.eigvalsh = _count_eig  # type: ignore[method-assign]
    try:
        step = _one_logical_optimizer_step(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            batch=batch,
            generator=generator,
            cfg=cfg,
            device=device,
            use_amp=use_amp,
            collapse_fixture=False,
            force_overflow_attempts=0,
            fail_status=STATUS_GPU_PREFLIGHT,
        )
    finally:
        torch.linalg.eigvalsh = orig_eig  # type: ignore[method-assign]
    if eig_calls["n"] != 0:
        raise Phase4FCanaryError(
            STATUS_GPU_PREFLIGHT, "eigvalsh_called_during_std_only_step"
        )
    if not step.get("applied"):
        raise Phase4FCanaryError(STATUS_GPU_PREFLIGHT, "preflight_step_not_applied")
    if str(step.get("loss_dtype") or "") not in {"torch.float32", "float32"}:
        raise Phase4FCanaryError(STATUS_GPU_PREFLIGHT, f"loss_dtype_{step.get('loss_dtype')}")
    if not math.isfinite(float(step["loss"])) or not math.isfinite(float(step["grad_norm"])):
        raise Phase4FCanaryError(STATUS_GPU_PREFLIGHT, "nonfinite_preflight_loss_or_grad")
    max_backoffs = int(cfg.get("max_consecutive_amp_backoffs") or 3)
    if int(step.get("n_backoffs") or 0) > max_backoffs:
        raise Phase4FCanaryError(STATUS_GPU_PREFLIGHT, "amp_backoff_limit")

    std_diag = step["diagnostics"]
    probe_views = two_views_canary(batch[:1].to(device), generator=generator, cfg=cfg)
    with torch.no_grad():
        z_probe = model.projector(model.encode(probe_views[0]))
    pout = int(z_probe.shape[1])
    half_z = torch.randn(PROBE_EIGVALSH_MAX_DIM, pout, dtype=torch.float16, device=device)
    LAST_EIGVALSH_META.clear()
    probe_diag = compute_collapse_diagnostics(half_z, half_z, half_z, cfg=cfg)
    eig_meta = dict(LAST_EIGVALSH_META)
    shape = eig_meta.get("shape") or []
    if not eig_meta.get("is_float32"):
        raise Phase4FCanaryError(STATUS_GPU_PREFLIGHT, "eigvalsh_not_float32")
    if not shape or int(max(int(dim) for dim in shape)) > PROBE_EIGVALSH_MAX_DIM:
        raise Phase4FCanaryError(STATUS_GPU_PREFLIGHT, f"eigvalsh_dim_{shape}")
    peak_gpu = 0
    if device.type == "cuda":
        peak_gpu = int(torch.cuda.max_memory_allocated(device))
    return {
        "status": STATUS_GPU_PREFLIGHT_PASS,
        "ok": True,
        "skip_reason": None,
        "use_amp": bool(use_amp),
        "device_type": device.type,
        "std_only": {
            "encoder_std": std_diag["encoder_std"],
            "projector_std": std_diag["projector_std"],
            "predictor_std": std_diag["predictor_std"],
            "collapsed": std_diag["collapsed"],
            "reasons": std_diag["reasons"],
        },
        "probe_collapsed": bool(probe_diag.get("collapsed")),
        "probe_effective_rank": probe_diag.get("effective_rank"),
        "probe_mean_abs_pairwise_cosine": probe_diag.get("mean_abs_pairwise_cosine"),
        "eigvalsh": eig_meta,
        "eigvalsh_calls_during_optimizer_step": 0,
        "loss_finite": True,
        "loss_dtype": step.get("loss_dtype"),
        "grad_norm_finite": True,
        "applied_optimizer_steps": 1,
        "skipped_attempts": int(step.get("n_backoffs") or 0),
        "n_backoffs": int(step.get("n_backoffs") or 0),
        "amp_scaler_init_scale": declared_init,
        "scaler_initial_scale": initial_scale,
        "scaler_final_scale": float(scaler.get_scale()) if scaler.is_enabled() else declared_init,
        "peak_gpu_memory_bytes": peak_gpu,
        "preflight_batch_shape": [n_b, n_s, 1, n_h, n_h],
        "n_real_unlabelled_studies_pixels_opened": 0,
        "n_report_text_rows_processed": 0,
        "n_second_reader_rows_processed": 0,
    }


def run_canary_gpu_preflight(
    *,
    cfg: dict[str, Any],
    require_tesla_t4: bool = True,
    require_real_resnet18: bool = True,
    weights_path: Path | None = None,
) -> dict[str, Any]:
    """Mandatory CUDA AMP preflight. Fail closed with zero real pixels opened."""
    extra: dict[str, Any] = {
        "n_real_unlabelled_studies_pixels_opened": 0,
        "n_report_text_rows_processed": 0,
        "n_second_reader_rows_processed": 0,
        "report_text_never_read": True,
    }
    try:
        if not torch.cuda.is_available():
            raise Phase4FCanaryError(STATUS_GPU_PREFLIGHT, "cuda_required")
        gpu_name = torch.cuda.get_device_name(0)
        extra["gpu_name"] = gpu_name
        if require_tesla_t4 and "Tesla T4" not in gpu_name:
            raise Phase4FCanaryError(STATUS_GPU_PREFLIGHT, "expected_tesla_t4")
        if require_real_resnet18:
            path = Path(weights_path) if weights_path is not None else locate_imagenet_weights()
            if path is None:
                raise Phase4FCanaryError(STATUS_GPU_PREFLIGHT, "imagenet_weights_missing")
            try:
                extra["imagenet_weights_sha256_verified"] = verify_imagenet_weights(
                    path, str(cfg.get("imagenet_weights_sha256") or IMAGENET_SHA256)
                )
            except Gold58PreflightError as exc:
                raise Phase4FCanaryError(STATUS_GPU_PREFLIGHT, str(exc.reason)) from exc
            model = build_canary_model(cfg, tiny=False, weights_path=path)
        else:
            model = build_canary_model(cfg, tiny=True, weights_path=None)
        device = torch.device("cuda")
        use_amp = bool(cfg.get("amp") == "cuda_only")
        software = run_canary_amp_software_preflight(
            cfg=cfg, model=model, device=device, use_amp=use_amp
        )
        extra.update(software)
        extra["ok"] = True
        extra["status"] = STATUS_GPU_PREFLIGHT_PASS
        return extra
    except Exception as exc:  # noqa: BLE001 - fail closed before pixels
        status = getattr(exc, "status", STATUS_GPU_PREFLIGHT) or STATUS_GPU_PREFLIGHT
        if status != STATUS_GPU_PREFLIGHT:
            status = STATUS_GPU_PREFLIGHT
        reason = getattr(exc, "reason", None) or f"uncaught_{type(exc).__name__}"
        extra.update(
            {
                "status": STATUS_GPU_PREFLIGHT,
                "ok": False,
                "skip_reason": str(reason)[:200],
            }
        )
        return extra


def _write_public(output_dir: Path, payload: dict[str, Any]) -> None:
    public_dir = Path(output_dir) / "sanitized"
    public_dir.mkdir(parents=True, exist_ok=True)
    (public_dir / "phase4f_ssl_canary_v2.json").write_text(
        json_dumps(payload), encoding="utf-8"
    )


def _payload(
    status: str,
    reason: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "status": status,
        "ok": status == STATUS_PASS,
        "skip_reason": reason,
        "protocol": PROTOCOL,
        "kernel_slug": KERNEL_SLUG,
        "ssl_objective": "simsiam",
        "series_selection_rule": SERIES_RULE_V2,
        "UNLABELED_IMAGE_SSL_PLANNING_ALLOWED": UNLABELED_IMAGE_SSL_PLANNING_ALLOWED,
        "UNLABELED_IMAGE_SSL_EXECUTION_ALLOWED": UNLABELED_IMAGE_SSL_EXECUTION_ALLOWED,
        "UNLABELED_IMAGE_SSL_CANARY_EXECUTION_ALLOWED": UNLABELED_IMAGE_SSL_CANARY_V2_EXECUTION_ALLOWED,
        "UNLABELED_IMAGE_SSL_CANARY_V2_EXECUTION_ALLOWED": UNLABELED_IMAGE_SSL_CANARY_V2_EXECUTION_ALLOWED,
        "GOLD58_SSL_OOF_EXECUTION_ALLOWED": GOLD58_SSL_OOF_EXECUTION_ALLOWED,
        "HARD_WEAK_LABEL_TRAINING_ALLOWED": HARD_WEAK_LABEL_TRAINING_ALLOWED,
        "SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED": SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED,
        "SECOND_READER_SCORE_BEARING_ALLOWED": SECOND_READER_SCORE_BEARING_ALLOWED,
        "competition_submission_allowed": False,
        "n_real_unlabelled_studies_pixels_opened": 0,
        "n_report_only_studies_processed": 0,
        "n_report_text_rows_processed": 0,
        "n_second_reader_rows_processed": 0,
        "report_text_never_read": True,
        "training_run": False,
        "OOF_run": False,
        "inference_run": False,
        "competition_submission_written": False,
        "consumes_submission_slot": False,
        "kaggle_kernel_launched": False,
        "full_ssl_run_authorized": False,
        "forbidden_outcome_not_used": FORBIDDEN_SEMANTIC_PASS,
        "git_head_sha": git_head_sha(),
        "utc": _utc(),
        "notebook_07_executed": False,
        "notebook_09_executed": False,
        "notebook_10_executed": False,
        "notebook_11_executed": False,
        "notebook_12_executed": False,
        "notebook_13_executed": False,
        "notebook_14_executed": False,
        "notebook_15_executed": False,
        "frozen_canary_v1_config_sha256": FROZEN_CANARY_V1_CONFIG_SHA256,
        "frozen_ssl_v1_config_sha256": FROZEN_SSL_V1_CONFIG_SHA256,
    }
    if extra:
        body.update(extra)
    return body


def _assert_scientific_scope(cfg: dict[str, Any], *, tiny: bool) -> None:
    if tiny:
        return
    if int(cfg.get("n_canary_studies") or 0) != AUTHORIZED_N_STUDIES:
        raise Phase4FCanaryError(STATUS_SCOPE, "scientific_canary_must_be_64_studies")
    if int(cfg.get("max_optimizer_steps") or 0) != AUTHORIZED_STEPS:
        raise Phase4FCanaryError(STATUS_SCOPE, "scientific_canary_must_be_40_steps")
    if int(cfg.get("checkpoint_at_step") or 0) != AUTHORIZED_CHECKPOINT_STEP:
        raise Phase4FCanaryError(STATUS_SCOPE, "scientific_canary_must_checkpoint_at_20")
    if int(cfg.get("batch_size") or 0) != AUTHORIZED_BATCH:
        raise Phase4FCanaryError(STATUS_SCOPE, "scientific_canary_batch_size_must_be_8")
    if int(cfg.get("n_canary_studies") or 0) >= EXPECTED_UNLABELED_REAL:
        raise Phase4FCanaryError(STATUS_FULL, "full_unlabeled_ssl_not_authorized")


def _load_cached_images(
    dataset: CanaryUnlabeledStudySliceDataset,
) -> tuple[list[torch.Tensor], float]:
    started = time.perf_counter()
    cache: list[torch.Tensor] = []
    for idx in range(len(dataset)):
        try:
            item = dataset[idx]
        except Exception as exc:  # noqa: BLE001 - fail closed on unreadable pixels
            raise Phase4FCanaryError(STATUS_FAIL, "unreadable_selected_series") from exc
        cache.append(item["image"].contiguous())
    elapsed = time.perf_counter() - started
    if len(cache) != len(dataset) or any(row.numel() < 1 for row in cache):
        raise Phase4FCanaryError(STATUS_FAIL, "unreadable_selected_series")
    return cache, elapsed


def _batch_from_cache(cache: list[torch.Tensor], indices: list[int]) -> torch.Tensor:
    if not indices:
        raise Phase4FCanaryError(STATUS_EMPTY, "empty_batch")
    return torch.stack([cache[int(i)] for i in indices], dim=0)


def _synthetic_unit_cache(
    *, n: int, num_slices: int, image_size: int, seed: int
) -> list[torch.Tensor]:
    rng = torch.Generator().manual_seed(int(seed))
    return [
        torch.rand(num_slices, 1, image_size, image_size, generator=rng)
        for _ in range(int(n))
    ]


def plan_canary_cohort(
    train: pd.DataFrame,
    series: pd.DataFrame,
    data_root: Path,
    *,
    n_canary: int,
    seed: int,
    require_real_counts: bool,
) -> dict[str, Any]:
    assert_not_hidden_test_path(data_root, split="train")
    ssl_ids, gold_ids, stats = select_unlabeled_study_ids(train)
    image_frame = drop_non_image_columns(train)
    if any(col in image_frame.columns for col in FORBIDDEN_TRAINING_KEYS):
        raise Phase4FCanaryError(STATUS_LABELS, "image_frame_still_has_forbidden_columns")
    if REPORT_COLUMN in series.columns:
        raise Phase4FCanaryError(STATUS_LABELS, "series_table_has_report")
    if require_real_counts:
        if int(stats["n_gold_excluded"]) != EXPECTED_LABELLED_REAL:
            raise Phase4FCanaryError(
                STATUS_SCOPE, f"expected_gold_{EXPECTED_LABELLED_REAL}_got_{stats['n_gold_excluded']}"
            )
        if int(stats["n_unlabeled_selected"]) != EXPECTED_UNLABELED_REAL:
            raise Phase4FCanaryError(
                STATUS_SCOPE,
                f"expected_unlabeled_{EXPECTED_UNLABELED_REAL}_got_{stats['n_unlabeled_selected']}",
            )
    canary_ids = select_canary_study_ids(ssl_ids, n=n_canary, seed=seed)
    n_overlap = assert_zero_gold_overlap(canary_ids, gold_ids)
    canary_set = set(canary_ids)
    series_canary = series.loc[series[STUDY_ID_COLUMN].astype(str).isin(canary_set)].copy()
    inventory = inspect_series_inventory(series_canary, data_root, split="train")
    try:
        lookup, aggregates = select_deterministic_primary_series_v2(inventory, canary_ids)
    except Gold58PreflightError as exc:
        raise Phase4FCanaryError(STATUS_SERIES, str(exc)) from exc
    if int(aggregates.get("n_zero_readable") or 0) != 0:
        raise Phase4FCanaryError(
            STATUS_FAIL, f"n_zero_readable={int(aggregates['n_zero_readable'])}"
        )
    if int(aggregates.get("n_selected") or 0) != int(n_canary):
        raise Phase4FCanaryError(
            STATUS_FAIL, f"n_selected={aggregates.get('n_selected')}!=n_canary"
        )
    series_dirs = [_series_dir(data_root, sid, lookup[sid], split="train") for sid in canary_ids]
    return {
        "canary_ids": canary_ids,
        "gold_ids": gold_ids,
        "series_dirs": series_dirs,
        "n_gold_excluded": int(stats["n_gold_excluded"]),
        "n_unlabeled_pool": int(stats["n_unlabeled_selected"]),
        "n_selected": int(len(canary_ids)),
        "n_gold_intersection": int(n_overlap),
        "n_zero_readable": int(aggregates.get("n_zero_readable") or 0),
        "series_selection_aggregates": aggregates,
    }


def _round_trip_max_abs(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> float:
    return max(float((a - b).abs().max()) for a, b in zip(left.values(), right.values()))


def run_canary_training_segment(
    *,
    cache: list[torch.Tensor],
    cfg: dict[str, Any],
    output_dir: Path,
    device: torch.device,
    tiny: bool,
    weights_path: Path | None,
    segment: str,
    collapse_fixture: bool,
    timings: dict[str, Any],
) -> dict[str, Any]:
    n_studies = len(cache)
    batch_size = int(cfg.get("batch_size") or AUTHORIZED_BATCH)
    max_steps = int(cfg.get("max_optimizer_steps") or AUTHORIZED_STEPS)
    ckpt_step = int(cfg.get("checkpoint_at_step") or AUTHORIZED_CHECKPOINT_STEP)
    seed = int(cfg.get("seed") or 42)
    private_dir = Path(output_dir) / "private"
    private_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = private_dir / "ssl_canary_step20.pt"
    end_path = private_dir / "ssl_canary_step40.pt"

    set_deterministic_seeds(seed)
    model = build_canary_model(cfg, tiny=tiny, weights_path=weights_path).to(device)
    optimizer, scheduler, scaler, use_amp = _make_optimizer_bundle(model, cfg, device=device)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    epoch = 0
    position = 0
    order = epoch_order(n_studies, seed=seed, epoch=0)
    start_step = 0
    losses: list[float] = []
    grad_norms: list[float] = []
    last_diag: dict[str, Any] = {}
    resumed_step = None
    checkpoint_round_trip = None
    scaler_declared_init = float(cfg.get("amp_scaler_init_scale") or 1024.0)
    scaler_initial_scale = (
        float(scaler.get_scale()) if scaler.is_enabled() else scaler_declared_init
    )
    skipped_attempts = 0
    applied_seconds: list[float] = []
    retry_seconds: list[float] = []

    if segment in {"resume_to_end", "full_with_fresh_restore"} and ckpt_path.is_file():
        load_t0 = time.perf_counter()
        loaded = load_ssl_checkpoint(ckpt_path, torch.device("cpu"))
        timings["checkpoint_load_seconds"] = time.perf_counter() - load_t0
        restore_training_state(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            generator=generator,
            payload=loaded,
            cfg=cfg,
            device=device,
        )
        start_step = int(loaded["step"])
        resumed_step = start_step
        epoch = int(loaded["epoch"])
        position = int(loaded["sampler_position"])
        order = list(loaded["sampler_order"])
        losses = list(loaded.get("losses") or [])
        grad_norms = list(loaded.get("grad_norms") or [])
        if start_step != ckpt_step:
            raise Phase4FCanaryError(
                STATUS_RESUME, f"resumed_step_{start_step}_expected_{ckpt_step}"
            )
        encoder_rt = _round_trip_max_abs(model.encoder.state_dict(), loaded["encoder"])
        projector_rt = _round_trip_max_abs(model.projector.state_dict(), loaded["projector"])
        predictor_rt = _round_trip_max_abs(model.predictor.state_dict(), loaded["predictor"])
        checkpoint_round_trip = {
            "encoder_max_abs": encoder_rt,
            "projector_max_abs": projector_rt,
            "predictor_max_abs": predictor_rt,
            "resumed_step": resumed_step,
        }
        atol = float(cfg.get("round_trip_atol") or 1e-5)
        if encoder_rt > atol or projector_rt > atol or predictor_rt > atol:
            raise Phase4FCanaryError(STATUS_RESUME, "checkpoint_round_trip_exceeds_atol")

    stop_at = ckpt_step if segment == "train_to_mid" else max_steps
    if start_step >= stop_at and segment != "train_to_mid":
        raise Phase4FCanaryError(STATUS_SCOPE, "canary_already_complete_refusing_full_run")

    step_times: list[float] = []
    peak_gpu = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for step in range(start_step, stop_at):
        indices, epoch, position, order = next_batch_indices(
            n_studies=n_studies,
            batch_size=batch_size,
            seed=seed,
            epoch=epoch,
            position=position,
            order=order,
        )
        batch = _batch_from_cache(cache, indices)
        t0 = time.perf_counter()
        loss, grad_norm, last_diag = _one_optimizer_step(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            batch=batch,
            generator=generator,
            cfg=cfg,
            device=device,
            use_amp=use_amp,
            collapse_fixture=collapse_fixture,
            fail_status=STATUS_FAIL,
        )
        wall = time.perf_counter() - t0
        step_times.append(wall)
        amp_meta = (last_diag or {}).get("amp") or {}
        skipped_attempts += int(amp_meta.get("n_backoffs") or 0)
        applied_seconds.append(float(amp_meta.get("applied_seconds") or wall))
        retry_seconds.append(float(amp_meta.get("retry_seconds") or 0.0))
        losses.append(loss)
        grad_norms.append(grad_norm)
        completed = step + 1
        if completed == ckpt_step and segment in {"train_to_mid", "full_with_fresh_restore"}:
            payload = build_checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                step=completed,
                sampler_order=order,
                sampler_position=position,
                generator=generator,
                cfg=cfg,
                losses=losses,
                grad_norms=grad_norms,
                timings=timings,
            )
            save_t0 = time.perf_counter()
            save_ssl_checkpoint(ckpt_path, payload)
            timings["checkpoint_save_seconds"] = time.perf_counter() - save_t0
            if segment == "train_to_mid":
                break
            # Fresh objects in this runtime section, then continue.
            del model, optimizer, scheduler, scaler
            model = build_canary_model(cfg, tiny=tiny, weights_path=weights_path).to(device)
            optimizer, scheduler, scaler, use_amp = _make_optimizer_bundle(
                model, cfg, device=device
            )
            generator = torch.Generator(device="cpu").manual_seed(seed)
            load_t0 = time.perf_counter()
            loaded = load_ssl_checkpoint(ckpt_path, torch.device("cpu"))
            timings["checkpoint_load_seconds"] = time.perf_counter() - load_t0
            restore_training_state(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                generator=generator,
                payload=loaded,
                cfg=cfg,
                device=device,
            )
            resumed_step = int(loaded["step"])
            epoch = int(loaded["epoch"])
            position = int(loaded["sampler_position"])
            order = list(loaded["sampler_order"])
            encoder_rt = _round_trip_max_abs(model.encoder.state_dict(), loaded["encoder"])
            projector_rt = _round_trip_max_abs(model.projector.state_dict(), loaded["projector"])
            predictor_rt = _round_trip_max_abs(model.predictor.state_dict(), loaded["predictor"])
            checkpoint_round_trip = {
                "encoder_max_abs": encoder_rt,
                "projector_max_abs": projector_rt,
                "predictor_max_abs": predictor_rt,
                "resumed_step": resumed_step,
            }
            atol = float(cfg.get("round_trip_atol") or 1e-5)
            if encoder_rt > atol or projector_rt > atol or predictor_rt > atol:
                raise Phase4FCanaryError(STATUS_RESUME, "checkpoint_round_trip_exceeds_atol")

    if device.type == "cuda":
        peak_gpu = int(torch.cuda.max_memory_allocated(device))

    if step_times:
        timings["optimizer_steps_seconds"] = float(sum(step_times))
        timings["seconds_per_optimizer_step"] = float(
            sum(applied_seconds) / max(len(applied_seconds), 1)
        )
        timings["amp_retry_seconds_total"] = float(sum(retry_seconds))
        timings["amp_retry_seconds_per_applied_step"] = float(
            sum(retry_seconds) / max(len(applied_seconds), 1)
        )
        timings["n_optimizer_steps_this_segment"] = int(len(step_times))
        timings["n_applied_optimizer_steps"] = int(len(applied_seconds))
        timings["n_skipped_attempts"] = int(skipped_attempts)
        timings["scaler_initial_scale"] = scaler_initial_scale
        timings["scaler_final_scale"] = (
            float(scaler.get_scale()) if scaler.is_enabled() else scaler_declared_init
        )
        timings["amp_scaler_init_scale"] = scaler_declared_init

    completed_step = start_step + len(step_times)
    if segment != "train_to_mid" and completed_step == max_steps:
        end_payload = build_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            step=completed_step,
            sampler_order=order,
            sampler_position=position,
            generator=generator,
            cfg=cfg,
            losses=losses,
            grad_norms=grad_norms,
            timings=timings,
        )
        save_ssl_checkpoint(end_path, end_payload)

    probe_n = min(int(cfg.get("probe_batch_size") or batch_size), n_studies)
    probe = _batch_from_cache(cache, list(range(probe_n)))
    model.eval()
    with torch.no_grad():
        v1, v2 = two_views_canary(probe.to(device), generator=generator, cfg=cfg)
        p1, z1, p2, z2 = model.forward_views(v1, v2)
        probe_diag = compute_collapse_diagnostics(
            torch.cat([model.encode(v1), model.encode(v2)], dim=0),
            torch.cat([z1, z2], dim=0),
            torch.cat([p1, p2], dim=0),
            cfg=cfg,
        )
    if not collapse_fixture:
        assert_collapse_ok(probe_diag)

    return {
        "completed_step": int(completed_step),
        "resumed_step": resumed_step,
        "losses": losses,
        "grad_norms": grad_norms,
        "last_step_diagnostics": last_diag,
        "probe_diagnostics": probe_diag,
        "checkpoint_round_trip": checkpoint_round_trip,
        "peak_gpu_memory_bytes": peak_gpu,
        "checkpoint_path_basename": ckpt_path.name if ckpt_path.exists() else None,
        "end_checkpoint_basename": end_path.name if end_path.exists() else None,
        "n_losses": int(len(losses)),
        "all_losses_finite": bool(all(math.isfinite(v) for v in losses)),
        "all_grad_norms_finite": bool(all(math.isfinite(v) for v in grad_norms)),
        "applied_optimizer_steps": int(len(applied_seconds)),
        "skipped_attempts": int(skipped_attempts),
        "scaler_initial_scale": scaler_initial_scale,
        "scaler_final_scale": (
            float(scaler.get_scale()) if scaler.is_enabled() else scaler_declared_init
        ),
        "amp_scaler_init_scale": scaler_declared_init,
    }


def run_uninterrupted_synthetic(
    cfg: dict[str, Any],
    *,
    output_dir: Path,
    steps: int,
    n_studies: int,
    batch_size: int,
    image_size: int = 16,
    num_slices: int = 2,
) -> dict[str, Any]:
    """Same-process 0→steps baseline used to compare against interrupted resume."""
    local = dict(cfg)
    local["max_optimizer_steps"] = int(steps)
    local["checkpoint_at_step"] = int(steps) + 1
    local["batch_size"] = int(batch_size)
    local["n_canary_studies"] = int(n_studies)
    cache = _synthetic_unit_cache(
        n=n_studies, num_slices=num_slices, image_size=image_size, seed=int(local.get("seed") or 42)
    )
    device = torch.device("cpu")
    return run_canary_training_segment(
        cache=cache,
        cfg=local,
        output_dir=output_dir,
        device=device,
        tiny=True,
        weights_path=None,
        segment="resume_to_end",
        collapse_fixture=False,
        timings={},
    )


def run_interrupted_synthetic(
    cfg: dict[str, Any],
    *,
    output_dir: Path,
    steps: int,
    checkpoint_at: int,
    n_studies: int,
    batch_size: int,
    image_size: int = 16,
    num_slices: int = 2,
) -> dict[str, Any]:
    local = dict(cfg)
    local["max_optimizer_steps"] = int(steps)
    local["checkpoint_at_step"] = int(checkpoint_at)
    local["batch_size"] = int(batch_size)
    local["n_canary_studies"] = int(n_studies)
    cache = _synthetic_unit_cache(
        n=n_studies, num_slices=num_slices, image_size=image_size, seed=int(local.get("seed") or 42)
    )
    device = torch.device("cpu")
    mid = run_canary_training_segment(
        cache=cache,
        cfg=local,
        output_dir=output_dir,
        device=device,
        tiny=True,
        weights_path=None,
        segment="train_to_mid",
        collapse_fixture=False,
        timings={},
    )
    end = run_canary_training_segment(
        cache=cache,
        cfg=local,
        output_dir=output_dir,
        device=device,
        tiny=True,
        weights_path=None,
        segment="resume_to_end",
        collapse_fixture=False,
        timings={},
    )
    return {"mid": mid, "end": end}


def run_phase4f_ssl_canary_v2(
    *,
    data_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    config_path: str | Path | None = None,
    execute_training: bool = True,
    allow_tiny_encoder_for_tests: bool = False,
    internet_enabled: bool = False,
    collapse_fixture: bool = False,
    run_oof: bool = False,
    run_full_unlabeled: bool = False,
    segment: str = "full_with_fresh_restore",
    config_overrides: dict[str, Any] | None = None,
    synthetic_tensors: bool = False,
) -> dict[str, Any]:
    """Authorized 64-study / 40-step canary, or tiny synthetic tests."""
    assert_run_signature_image_only()
    assert HARD_WEAK_LABEL_TRAINING_ALLOWED is False
    assert SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED is False
    assert SECOND_READER_SCORE_BEARING_ALLOWED is False
    assert UNLABELED_IMAGE_SSL_EXECUTION_ALLOWED is False
    assert GOLD58_SSL_OOF_EXECUTION_ALLOWED is False
    assert UNLABELED_IMAGE_SSL_CANARY_V2_EXECUTION_ALLOWED is True
    if internet_enabled:
        return _payload(STATUS_INTERNET, "internet_must_be_off")
    if run_oof:
        return _payload(STATUS_OOF, "gold58_ssl_oof_not_authorized")
    if run_full_unlabeled:
        return _payload(STATUS_FULL, "UNLABELED_IMAGE_SSL_EXECUTION_ALLOWED=false")

    cfg = load_canary_config(config_path)
    if config_overrides:
        assert_no_forbidden_training_inputs(config_overrides)
        cfg.update(config_overrides)
    out = Path(output_dir) if output_dir else PROJECT_ROOT / "outputs" / cfg["output_dirname"]
    out.mkdir(parents=True, exist_ok=True)
    extra: dict[str, Any] = {
        "config_sha256": cfg["_config_sha256"],
        "source_sha256": canary_source_sha256(),
        "imagenet_weights_sha256": IMAGENET_SHA256,
        "frozen_ssl_v1_config_sha256": FROZEN_SSL_V1_CONFIG_SHA256,
        "frozen_canary_v1_config_sha256": FROZEN_CANARY_V1_CONFIG_SHA256,
        "frozen_v1_recipe_sha256": FROZEN_V1_RECIPE_SHA256,
        "frozen_v2_recipe_sha256": FROZEN_V2_CONFIG_SHA256,
        "amp_scaler_init_scale": float(cfg.get("amp_scaler_init_scale") or 1024.0),
        "max_consecutive_amp_backoffs": int(cfg.get("max_consecutive_amp_backoffs") or 3),
        "amp_loss_eps": float(cfg.get("amp_loss_eps") or 1e-6),
        "segment": segment,
    }
    try:
        fixture_check = validate_collapse_fixtures(cfg)
        extra["collapse_fixture_validation"] = fixture_check
        _assert_scientific_scope(cfg, tiny=allow_tiny_encoder_for_tests)
    except Phase4FCanaryError as exc:
        payload = _payload(exc.status if exc.status != STATUS_COLLAPSE else STATUS_FAIL, exc.reason, extra=extra)
        if exc.status == STATUS_COLLAPSE:
            payload["status"] = STATUS_COLLAPSE
        _write_public(out, payload)
        return payload

    root = Path(data_root) if data_root is not None else None
    data_source = infer_data_source(root) if root is not None else "software_fixture"
    extra["data_source"] = data_source
    scientific = (
        (not allow_tiny_encoder_for_tests)
        and (not synthetic_tensors)
        and data_source == DATA_SOURCE_REAL_KAGGLE
    )
    extra["scientific_evaluation"] = bool(scientific)
    extra["n_report_text_rows_processed"] = 0
    extra["n_second_reader_rows_processed"] = 0
    extra["report_text_never_read"] = True

    if not execute_training:
        payload = _payload(STATUS_FAIL, "execute_training=false_for_authorized_canary", extra=extra)
        _write_public(out, payload)
        return payload

    timings: dict[str, Any] = {}
    cache: list[torch.Tensor]
    n_pixels_opened = 0
    cohort_public: dict[str, Any] = {}
    weights_file: Path | None = None
    weight_sha = None

    if scientific:
        gpu_pre = run_canary_gpu_preflight(
            cfg=cfg,
            require_tesla_t4=True,
            require_real_resnet18=True,
        )
        extra["gpu_preflight"] = _sanitize_gpu_preflight(gpu_pre)
        extra["gpu_preflight_ok"] = bool(gpu_pre.get("ok"))
        if not gpu_pre.get("ok"):
            payload = _payload(
                STATUS_GPU_PREFLIGHT,
                str(gpu_pre.get("skip_reason") or "gpu_preflight_failed"),
                extra=extra,
            )
            payload["n_real_unlabelled_studies_pixels_opened"] = 0
            payload["n_report_text_rows_processed"] = 0
            payload["n_second_reader_rows_processed"] = 0
            _write_public(out, payload)
            return payload

    try:
        if scientific or (root is not None and not synthetic_tensors):
            assert_not_hidden_test_path(root, split="train")
            assert root is not None
            setup_t0 = time.perf_counter()
            train = pd.read_csv(root / "train.csv")
            series = pd.read_csv(root / "train_series.csv")
            planned = plan_canary_cohort(
                train,
                series,
                root,
                n_canary=int(cfg["n_canary_studies"]),
                seed=int(cfg.get("cohort_hash_seed") or cfg.get("seed") or 42),
                require_real_counts=scientific,
            )
            timings["dataset_series_setup_seconds"] = time.perf_counter() - setup_t0
            dataset = CanaryUnlabeledStudySliceDataset(
                planned["canary_ids"],
                planned["series_dirs"],
                num_slices=int(cfg.get("num_slices") or 8),
                image_size=int(cfg.get("image_size") or 224),
                percentiles=tuple(cfg.get("intensity_percentiles") or (1.0, 99.0)),
                interpolation=str(cfg.get("interpolation") or "bilinear"),
                imagenet_normalize=False,
            )
            cache, dicom_s = _load_cached_images(dataset)
            n_pixels_opened = int(len(cache))
            timings["dicom_loading_seconds"] = dicom_s
            timings["dicom_seconds_per_study"] = dicom_s / max(n_pixels_opened, 1)
            timings["seconds_per_study"] = timings["dicom_seconds_per_study"]
            cohort_public = {
                "n_selected": planned["n_selected"],
                "n_zero_readable": planned["n_zero_readable"],
                "n_gold_intersection": planned["n_gold_intersection"],
                "n_gold_excluded": planned["n_gold_excluded"],
                "n_unlabeled_pool": planned["n_unlabeled_pool"],
                "series_selection_aggregates": planned["series_selection_aggregates"],
            }
            if scientific and n_pixels_opened != AUTHORIZED_N_STUDIES:
                raise Phase4FCanaryError(
                    STATUS_FAIL, f"pixels_opened_{n_pixels_opened}_expected_64"
                )
        else:
            n_syn = int(cfg.get("n_canary_studies") or 8)
            cache = _synthetic_unit_cache(
                n=n_syn,
                num_slices=int(cfg.get("tiny_num_slices") or 2),
                image_size=int(cfg.get("tiny_image_size") or 16),
                seed=int(cfg.get("seed") or 42),
            )
            timings["dataset_series_setup_seconds"] = 0.0
            timings["dicom_loading_seconds"] = 0.0
            timings["dicom_seconds_per_study"] = 0.0
            timings["seconds_per_study"] = 0.0
            n_pixels_opened = 0
            cohort_public = {
                "n_selected": n_syn,
                "n_zero_readable": 0,
                "n_gold_intersection": 0,
                "synthetic_tensors": True,
            }

        if not allow_tiny_encoder_for_tests:
            weights_file = locate_imagenet_weights()
            if weights_file is None:
                raise Phase4FCanaryError(STATUS_WEIGHTS, "imagenet_weights_missing")
            weight_sha = verify_imagenet_weights(
                weights_file, str(cfg.get("imagenet_weights_sha256") or IMAGENET_SHA256)
            )
            extra["imagenet_weights_sha256_verified"] = weight_sha

        device = torch.device("cuda" if (scientific and torch.cuda.is_available()) else "cpu")
        if scientific:
            device = torch.device("cuda")
        train_out = run_canary_training_segment(
            cache=cache,
            cfg=cfg,
            output_dir=out,
            device=device,
            tiny=allow_tiny_encoder_for_tests,
            weights_path=weights_file,
            segment=segment,
            collapse_fixture=collapse_fixture,
            timings=timings,
        )
    except Exception as exc:  # noqa: BLE001 - canary must fail closed, not crash the kernel
        status = getattr(exc, "status", STATUS_FAIL)
        reason = getattr(exc, "reason", None) or f"uncaught_{type(exc).__name__}"
        extra.update(
            {
                "timings": timings,
                "cohort": cohort_public,
                "n_real_unlabelled_studies_pixels_opened": n_pixels_opened if scientific else 0,
                "n_report_text_rows_processed": 0,
                "n_second_reader_rows_processed": 0,
                "report_text_never_read": True,
            }
        )
        payload = _payload(status if status else STATUS_FAIL, str(reason)[:200], extra=extra)
        _write_public(out, payload)
        return payload

    extra.update(
        {
            "timings": timings,
            "cohort": cohort_public,
            "training": {
                "completed_step": train_out["completed_step"],
                "resumed_step": train_out["resumed_step"],
                "losses": train_out["losses"],
                "grad_norms": train_out["grad_norms"],
                "probe_diagnostics": train_out["probe_diagnostics"],
                "last_step_diagnostics": train_out["last_step_diagnostics"],
                "checkpoint_round_trip": train_out["checkpoint_round_trip"],
                "peak_gpu_memory_bytes": train_out["peak_gpu_memory_bytes"],
                "checkpoint_path_basename": train_out["checkpoint_path_basename"],
                "all_losses_finite": train_out["all_losses_finite"],
                "all_grad_norms_finite": train_out["all_grad_norms_finite"],
                "applied_optimizer_steps": train_out.get("applied_optimizer_steps"),
                "skipped_attempts": train_out.get("skipped_attempts"),
                "scaler_initial_scale": train_out.get("scaler_initial_scale"),
                "scaler_final_scale": train_out.get("scaler_final_scale"),
            },
            "imagenet_weights_sha256_verified": weight_sha,
        }
    )
    extra["n_real_unlabelled_studies_pixels_opened"] = n_pixels_opened if scientific else 0
    extra["n_fixture_studies_pixels_opened"] = 0 if scientific else n_pixels_opened
    extra["n_report_text_rows_processed"] = 0
    extra["n_second_reader_rows_processed"] = 0
    extra["report_text_never_read"] = True
    extra["unlabeled_image_rows_opened_without_reading_report_text"] = (
        extra["n_real_unlabelled_studies_pixels_opened"]
    )

    if segment == "train_to_mid":
        extra["training_run"] = True
        payload = _payload(STATUS_MID, None, extra=extra)
        payload["ok"] = True
        payload["training_run"] = True
        payload["n_real_unlabelled_studies_pixels_opened"] = extra[
            "n_real_unlabelled_studies_pixels_opened"
        ]
        _write_public(out, payload)
        return payload

    projection = project_full_run_from_canary(timings, cfg)
    extra["full_run_projection"] = projection
    fail_reasons: list[str] = []
    if not train_out["all_losses_finite"] or not train_out["all_grad_norms_finite"]:
        fail_reasons.append("nonfinite_loss_or_grad")
    if int(train_out["completed_step"]) != int(cfg["max_optimizer_steps"]):
        fail_reasons.append("did_not_reach_max_steps")
    if int(cfg["max_optimizer_steps"]) >= AUTHORIZED_CHECKPOINT_STEP and train_out["resumed_step"] != int(
        cfg["checkpoint_at_step"]
    ):
        fail_reasons.append("resume_step_mismatch")
    if scientific and int(cohort_public.get("n_selected") or 0) != AUTHORIZED_N_STUDIES:
        fail_reasons.append("selected_count_not_64")
    if int(cohort_public.get("n_gold_intersection") or 0) != 0:
        fail_reasons.append("gold_overlap")
    if int(cohort_public.get("n_zero_readable") or 0) != 0:
        fail_reasons.append("zero_readable")
    if not projection["full_run_fits_private_wall"]:
        fail_reasons.append("full_run_projection_guard_fail")
    if fail_reasons:
        payload = _payload(STATUS_FAIL, ",".join(fail_reasons), extra=extra)
        payload["training_run"] = True
        payload["n_real_unlabelled_studies_pixels_opened"] = extra[
            "n_real_unlabelled_studies_pixels_opened"
        ]
        _write_public(out, payload)
        return payload

    payload = _payload(STATUS_PASS, None, extra=extra)
    payload["training_run"] = True
    payload["n_real_unlabelled_studies_pixels_opened"] = extra[
        "n_real_unlabelled_studies_pixels_opened"
    ]
    _write_public(out, payload)
    return payload


def render_canary_protocol_markdown(cfg: dict[str, Any] | None = None) -> str:
    cfg = cfg or load_canary_config()
    return "\n".join(
        [
            "# Phase 4F.2 AMP-safe unlabeled-image SimSiam SSL canary protocol",
            "",
            "Bounded real-data canary only. The full 4,349-study SSL job is not authorized.",
            "Phase 4F v1 config, report, notebook 13, and synthetic results stay frozen.",
            "",
            f"- protocol: `{PROTOCOL}`",
            f"- kernel slug: `{KERNEL_SLUG}`",
            "- SSL objective: SimSiam (Chen & He, CVPR 2021)",
            "- preprocess: DICOM → percentile/window to [0, 1] → spatial+intensity",
            "  augs on unnormalized [0, 1] → clamp [0, 1] → ImageNet last",
            "- intensity jitter is never applied to ImageNet-normalized tensors",
            f"- encoder init: ImageNet ResNet-18 SHA-256 `{IMAGENET_SHA256}`",
            f"- series rule: `{SERIES_RULE_V2}` (frozen, not retuned)",
            "- cohort: 64 non-gold studies, SHA-256 order, seed 42; do not log IDs",
            "- gold-58 intersection must be 0 before pixels open",
            "- train exactly 40 optimizer steps at frozen batch_size=8",
            "- force checkpoint at step 20; restore in a fresh runtime section; continue to 40",
            "- never continue into a full SSL run",
            f"- collapse thresholds frozen: encoder/projector/predictor std min "
            f"{cfg.get('collapse_encoder_std_min')}/"
            f"{cfg.get('collapse_projector_std_min')}/"
            f"{cfg.get('collapse_predictor_std_min')}; "
            f"mean_abs_cosine_max={cfg.get('collapse_mean_abs_cosine_max')}; "
            f"effective_rank_min={cfg.get('collapse_effective_rank_min')}",
            "- per-step diagnostics: encoder/projector/predictor std only; no cosine,",
            "  covariance, SVD, or eigendecomposition during optimizer steps",
            "- probe cosine and effective rank run once after the training segment",
            "- effective rank uses the float32 n×n centered Gram `X @ X.T / (n-1)`",
            "  (probe n=16), not 2048×2048 feature covariance",
            "- GPU AMP preflight before train.csv / series / DICOM: Tesla T4, ImageNet",
            "  hash, synthetic ResNet-18 AMP step on scientific shape 8x8x1x224x224,",
            "  AMP-safe float32 SimSiam loss, GradScaler init_scale=1024, std-only on",
            "  AMP outputs, probe eigvalsh float32 dim<=16; fail",
            "  `SSL_CANARY_GPU_PREFLIGHT_FAIL` with zero real pixels if any check fails",
            "- AMP-safe loss: predictor and stop-grad projector in float32; cosine/mean",
            "  in float32 with eps 1e-6; frozen Phase 4F v1 `simsiam_loss` is not called",
            "- an optimizer step counts only when parameters were actually updated;",
            "  scheduler steps only after an applied optimizer step",
            "- AMP overflow: restore aug-generator, skip sampler/optimizer/scheduler,",
            "  at most 3 consecutive backoffs, then fail closed",
            "- pass: `SSL_CANARY_PASS_FULL_RUN_NOT_AUTHORIZED`",
            "- fail: `SSL_CANARY_FAIL`",
            "- GPU preflight fail: `SSL_CANARY_GPU_PREFLIGHT_FAIL`",
            f"- config SHA-256: `{cfg['_config_sha256']}`",
            f"- frozen Phase 4F v1 config SHA-256: `{FROZEN_SSL_V1_CONFIG_SHA256}`",
            f"- frozen Phase 4F.1 canary config SHA-256: `{FROZEN_CANARY_V1_CONFIG_SHA256}`",
            "- UNLABELED_IMAGE_SSL_EXECUTION_ALLOWED=false",
            "- GOLD58_SSL_OOF_EXECUTION_ALLOWED=false",
            "- HARD_WEAK_LABEL_TRAINING_ALLOWED=false",
            "- no `/kaggle/working/submission.csv`",
            f"- forbidden outcome not used: `{FORBIDDEN_SEMANTIC_PASS}`",
            "",
        ]
    )
