"""Synthetic knee MRI dataset generation for RSNA Knee Abnormality Detection.

The real competition provides knee MRI studies, each labelled with twelve
independent binary findings. We synthesize single grayscale MRI-like slices in
which each finding is encoded as a bright lesion at a fixed, finding-specific
anatomical location, so a small CNN has a genuinely learnable multi-label
signal — without shipping any real patient data.

Every training study also carries a multilingual radiology report (for weak
supervision). Reports are intentionally *not* returned from the inference path:
test-time prediction uses images only.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .config import Config, NUM_TARGETS
from .reports import generate_report

# Per-finding lesion centers laid out on a 4x3 grid over the normalized image.
# Each of the twelve findings lights up a distinct region when present.
_GRID = [(r, c) for r in (0.2, 0.4, 0.6, 0.8) for c in (0.25, 0.5, 0.75)]

# Per-finding base prevalence (fraction of studies that are positive).
_PREVALENCE = np.array(
    [0.30, 0.25, 0.35, 0.30, 0.30, 0.28, 0.32, 0.40, 0.25, 0.22, 0.28, 0.15],
    dtype=np.float32,
)


def _add_lesion(img: np.ndarray, center_rc: tuple[float, float], rng: np.random.Generator) -> None:
    """Add a soft bright Gaussian lesion at a normalized (row, col) center."""
    size = img.shape[0]
    cy, cx = int(center_rc[0] * size), int(center_rc[1] * size)
    sigma = size * 0.06
    yy, xx = np.mgrid[0:size, 0:size]
    blob = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2)))
    intensity = 0.6 + 0.2 * rng.random()
    img += intensity * blob


def generate_study(size: int, rng: np.random.Generator):
    """Create one synthetic MRI study: (image, label_vector, report).

    Returns:
      image:  float32 array shape (1, size, size), values in [0, 1]
      labels: float32 array shape (12,), each in {0, 1}
      report: multilingual free-text string (weak-supervision source)
    """
    labels = (rng.random(NUM_TARGETS) < _PREVALENCE).astype(np.float32)

    img = rng.normal(0.25, 0.06, size=(size, size))  # soft-tissue MRI background
    for i in range(NUM_TARGETS):
        if labels[i] > 0.5:
            _add_lesion(img, _GRID[i], rng)
    img += rng.normal(0.0, 0.02, size=(size, size))  # acquisition noise
    img = np.clip(img, 0.0, 1.0).astype(np.float32)

    report = generate_report(labels, rng)
    return img[None, :, :], labels, report


def make_split(n: int, size: int, rng: np.random.Generator):
    """Generate a split of studies.

    Returns (images NCHW, labels N x 12, reports list[str])."""
    images, labels, reports = [], [], []
    for _ in range(n):
        img, lab, rep = generate_study(size, rng)
        images.append(img)
        labels.append(lab)
        reports.append(rep)
    return (
        np.stack(images).astype(np.float32),
        np.stack(labels).astype(np.float32),
        reports,
    )


class KneeMRIDataset(Dataset):
    """Image-only dataset used for both training and (image-only) inference."""

    def __init__(self, images: np.ndarray, labels: np.ndarray):
        self.images = torch.from_numpy(images)
        self.labels = torch.from_numpy(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.images[idx], self.labels[idx]


def build_dataloaders(cfg: Config):
    """Create reproducible train/val/test dataloaders and keep raw arrays.

    The raw dict also exposes the multilingual training reports so the pipeline
    can derive weak labels; val/test reports are never used for inference.
    """
    rng = np.random.default_rng(cfg.seed)
    train = make_split(cfg.n_train, cfg.image_size, rng)
    val = make_split(cfg.n_val, cfg.image_size, rng)
    test = make_split(cfg.n_test, cfg.image_size, rng)

    loaders = {
        "train": DataLoader(KneeMRIDataset(train[0], train[1]), batch_size=cfg.batch_size, shuffle=True),
        "val": DataLoader(KneeMRIDataset(val[0], val[1]), batch_size=cfg.batch_size, shuffle=False),
        "test": DataLoader(KneeMRIDataset(test[0], test[1]), batch_size=cfg.batch_size, shuffle=False),
    }
    raw = {"train": train, "val": val, "test": test}
    return loaders, raw
