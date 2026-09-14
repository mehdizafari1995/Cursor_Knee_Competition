"""Baseline training / inference helpers.

Full training on real studies is opt-in (``allow_training=True``). The default
path is a CPU smoke test on generated tensors, which is software verification
and must not be reported as a scientific result.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam

from .baseline_config import BaselineConfig
from .baseline_model import build_baseline_model, smoke_batch
from .loss import masked_bce_with_logits
from .metrics import macro_roc_auc


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict_logits(model: torch.nn.Module, images: torch.Tensor, device: torch.device) -> np.ndarray:
    model.eval()
    logits = model(images.to(device))
    return torch.sigmoid(logits).cpu().numpy()


def run_smoke(cfg: BaselineConfig) -> dict:
    """One optimiser step on generated tensors. Not a scientific evaluation."""
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    model = build_baseline_model(
        smoke=True,
        encoder_name="tiny",
        pretrained=False,
        aggregation=cfg.aggregation,
        num_targets=cfg.num_targets,
    ).to(device)
    images, labels = smoke_batch(
        batch_size=max(2, cfg.batch_size),
        num_slices=max(2, min(cfg.num_slices, 4)),
        image_size=min(cfg.image_size, 16),
        num_targets=cfg.num_targets,
        seed=cfg.seed,
    )
    images, labels = images.to(device), labels.to(device)
    optimizer = Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    model.train()
    optimizer.zero_grad()
    logits = model(images)
    loss = masked_bce_with_logits(logits, labels)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        probs = torch.sigmoid(model(images)).cpu().numpy()
    y_true = labels.cpu().numpy()
    macro, per_target = macro_roc_auc(y_true, probs)
    return {
        "mode": "software_smoke",
        "scientific_evaluation": False,
        "note": (
            "CPU smoke test used tiny generated tensors only. "
            "The resulting AUC is not a competition or scientific result."
        ),
        "train_loss": float(loss.detach().cpu()),
        "macro_roc_auc": None if macro != macro else float(macro),
        "per_target_roc_auc": [
            None if v != v else float(v) for v in per_target
        ],
        "batch_shape": list(images.shape),
        "num_targets": cfg.num_targets,
    }


def save_checkpoint(model: torch.nn.Module, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model: torch.nn.Module, path: Path, device: torch.device) -> torch.nn.Module:
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    return model
