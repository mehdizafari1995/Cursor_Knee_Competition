"""Artifact generation: MRI sample montage, per-target AUC chart, submission."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend for CI / cloud agents
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import Config, TARGET_KEYS, TARGET_NAMES


def save_sample_montage(images: np.ndarray, labels: np.ndarray, path: Path, n: int = 6) -> None:
    """Save a few synthetic MRI studies annotated with their positive findings."""
    n = min(n, len(images))
    fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 3.0))
    if n == 1:
        axes = [axes]
    for k, ax in enumerate(axes):
        ax.imshow(images[k, 0], cmap="gray", vmin=0, vmax=1)
        pos = [TARGET_NAMES[i] for i in range(len(TARGET_NAMES)) if labels[k, i] > 0.5]
        title = "\n".join(pos) if pos else "No findings"
        ax.set_title(title, fontsize=7)
        ax.axis("off")
    fig.suptitle("Synthetic knee MRI studies (multi-label findings)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_auc_barchart(per_target_auc: list[float], macro_auc: float, path: Path) -> None:
    """Horizontal bar chart of per-target ROC AUC with the macro average line."""
    fig, ax = plt.subplots(figsize=(7, 6))
    y = np.arange(len(TARGET_NAMES))
    values = [0.0 if (v != v) else v for v in per_target_auc]  # NaN -> 0 for display
    ax.barh(y, values, color="#4C78A8")
    ax.set_yticks(y)
    ax.set_yticklabels(TARGET_NAMES, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("ROC AUC")
    ax.axvline(macro_auc, color="#E45756", linestyle="--", label=f"macro AUC = {macro_auc:.3f}")
    ax.set_title("Per-target ROC AUC (test set)")
    ax.legend(loc="lower right")
    for yi, v in zip(y, values):
        ax.text(min(v + 0.01, 0.98), yi, f"{v:.2f}", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def write_submission(y_prob: np.ndarray, cfg: Config, path: Path) -> pd.DataFrame:
    """Write an RSNA-style submission: one row per study, 12 probability columns."""
    df = pd.DataFrame(y_prob, columns=list(TARGET_KEYS))
    df.insert(0, "study_id", [f"study_{i:05d}" for i in range(len(y_prob))])
    df.to_csv(path, index=False)
    return df


def text_report(macro_auc: float, per_target_auc: list[float]) -> str:
    lines = ["Per-target ROC AUC:"]
    for name, auc in zip(TARGET_NAMES, per_target_auc):
        shown = "  n/a" if auc != auc else f"{auc:.3f}"
        lines.append(f"  {name:<36} {shown}")
    lines.append(f"\nMacro-averaged ROC AUC: {macro_auc:.4f}")
    return "\n".join(lines)
