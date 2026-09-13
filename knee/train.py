"""Training and evaluation loops for the knee CNN."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import Config
from .model import KneeCNN


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    """Return (accuracy, y_true, y_pred) over a dataloader."""
    model.eval()
    y_true, y_pred = [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1).cpu().numpy()
        y_pred.extend(preds.tolist())
        y_true.extend(labels.numpy().tolist())
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    acc = float((y_true_arr == y_pred_arr).mean()) if len(y_true_arr) else 0.0
    return acc, y_true_arr, y_pred_arr


def train_model(cfg: Config, loaders: dict[str, DataLoader], device: torch.device) -> KneeCNN:
    set_seed(cfg.seed)
    model = KneeCNN(cfg.num_classes, cfg.image_size).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        for images, labels in loaders["train"]:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running += loss.item() * images.size(0)

        train_loss = running / len(loaders["train"].dataset)
        val_acc, _, _ = evaluate(model, loaders["val"], device)
        print(
            f"epoch {epoch:2d}/{cfg.epochs} | train_loss {train_loss:.4f} | val_acc {val_acc:.3f}",
            flush=True,
        )

    return model
