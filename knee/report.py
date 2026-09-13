"""Artifact generation: sample montage, confusion matrix, and submission file."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend for CI / cloud agents
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from .config import CLASS_NAMES, Config


def save_sample_montage(images: np.ndarray, labels: np.ndarray, path: Path) -> None:
    """Save one representative synthetic radiograph per KL grade."""
    n = len(CLASS_NAMES)
    fig, axes = plt.subplots(1, n, figsize=(2.2 * n, 2.6))
    for grade, ax in enumerate(axes):
        idx = int(np.where(labels == grade)[0][0])
        ax.imshow(images[idx, 0], cmap="gray", vmin=0, vmax=1)
        ax.set_title(CLASS_NAMES[grade], fontsize=9)
        ax.axis("off")
    fig.suptitle("Synthetic knee radiographs by KL grade", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(CLASS_NAMES, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Test-set confusion matrix")
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, int(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black", fontsize=9,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def write_submission(y_pred: np.ndarray, cfg: Config, path: Path) -> pd.DataFrame:
    """Write a Kaggle-style submission.csv (image_id, kl_grade)."""
    df = pd.DataFrame(
        {"image_id": [f"test_{i:04d}" for i in range(len(y_pred))], "kl_grade": y_pred}
    )
    df.to_csv(path, index=False)
    return df


def text_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    return classification_report(
        y_true, y_pred, target_names=list(CLASS_NAMES), zero_division=0
    )
