"""Training and evaluation loops for the multi-label knee MRI model.

The competition metric is the macro-averaged ROC AUC across the twelve targets,
so training minimizes binary cross-entropy per target and evaluation reports
per-target and macro ROC AUC.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from .config import Config
from .model import KneeMRINet


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def macro_roc_auc(y_true: np.ndarray, y_prob: np.ndarray):
    """Macro-averaged ROC AUC across targets (the official metric).

    Targets that are single-class in the evaluation split are skipped (AUC is
    undefined) and reported as NaN, matching how such degenerate columns are
    handled in practice.
    """
    per_target = []
    valid = []
    for j in range(y_true.shape[1]):
        col = y_true[:, j]
        if col.min() == col.max():
            per_target.append(float("nan"))
            continue
        auc = float(roc_auc_score(col, y_prob[:, j]))
        per_target.append(auc)
        valid.append(auc)
    macro = float(np.mean(valid)) if valid else float("nan")
    return macro, per_target


@torch.no_grad()
def predict_proba(model: nn.Module, loader: DataLoader, device: torch.device):
    """Image-only inference: return (probabilities, y_true) as arrays."""
    model.eval()
    probs, trues = [], []
    for images, labels in loader:
        images = images.to(device)
        p = torch.sigmoid(model(images)).cpu().numpy()
        probs.append(p)
        trues.append(labels.numpy())
    return np.concatenate(probs), np.concatenate(trues)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    """Return (macro_auc, per_target_auc, y_true, y_prob)."""
    y_prob, y_true = predict_proba(model, loader, device)
    macro, per_target = macro_roc_auc(y_true, y_prob)
    return macro, per_target, y_true, y_prob


def train_model(
    cfg: Config,
    train_images: np.ndarray,
    train_labels: np.ndarray,
    val_loader: DataLoader,
    device: torch.device,
) -> KneeMRINet:
    """Train the multi-label CNN.

    `train_labels` may be weak labels derived from multilingual reports rather
    than expert annotations; the model itself only ever sees images.
    """
    from .data import KneeMRIDataset
    from torch.utils.data import DataLoader as _DataLoader

    set_seed(cfg.seed)
    train_loader = _DataLoader(
        KneeMRIDataset(train_images, train_labels),
        batch_size=cfg.batch_size,
        shuffle=True,
    )

    model = KneeMRINet(cfg.num_targets, cfg.image_size).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running += loss.item() * images.size(0)

        train_loss = running / len(train_loader.dataset)
        val_macro, _, _, _ = evaluate(model, val_loader, device)
        print(
            f"epoch {epoch:2d}/{cfg.epochs} | train_loss {train_loss:.4f} | "
            f"val_macro_auc {val_macro:.3f}",
            flush=True,
        )

    return model
