"""Synthetic knee X-ray dataset generation.

Real knee OA competitions grade radiographs on the Kellgren-Lawrence (KL)
scale, where higher grades show a *narrower joint space* and *more osteophytes*
(bony spurs). We synthesize grayscale images that encode exactly those signals
so a small CNN has something learnable, without shipping any real medical data.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .config import Config


def _draw_bone(canvas: np.ndarray, top: int, bottom: int, rng: np.random.Generator) -> None:
    """Fill a horizontal bone region with bright, slightly textured intensity."""
    h = canvas.shape[0]
    top = max(0, top)
    bottom = min(h, bottom)
    if bottom <= top:
        return
    texture = rng.normal(0.75, 0.05, size=(bottom - top, canvas.shape[1]))
    canvas[top:bottom, :] = np.clip(texture, 0.0, 1.0)


def generate_image(grade: int, size: int, rng: np.random.Generator) -> np.ndarray:
    """Create one synthetic knee radiograph for a given KL grade.

    Signal encoded:
      * joint-space gap width shrinks as grade increases (KL0 wide, KL4 tight)
      * number/intensity of osteophyte spots grows with grade
    """
    img = rng.normal(0.15, 0.05, size=(size, size))  # soft-tissue background

    center = size // 2
    # Joint space: ~40% of height at KL0 down to ~6% at KL4.
    gap_frac = 0.40 - 0.085 * grade
    gap = max(1, int(round(gap_frac * size)))
    half_gap = gap // 2

    # Femur (top bone) and tibia (bottom bone) flanking the joint space.
    _draw_bone(img, 0, center - half_gap, rng)
    _draw_bone(img, center + half_gap, size, rng)

    # Osteophytes: bright spurs near the joint margins, count grows with grade.
    n_spurs = int(rng.integers(0, grade + 1) + grade)
    for _ in range(n_spurs):
        edge = center - half_gap if rng.random() < 0.5 else center + half_gap
        y = int(np.clip(edge + rng.integers(-2, 3), 0, size - 1))
        x = int(rng.integers(0, size))
        r = int(rng.integers(1, 3))
        y0, y1 = max(0, y - r), min(size, y + r + 1)
        x0, x1 = max(0, x - r), min(size, x + r + 1)
        img[y0:y1, x0:x1] = np.clip(img[y0:y1, x0:x1] + 0.6, 0.0, 1.0)

    img += rng.normal(0.0, 0.03, size=(size, size))  # sensor noise
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def make_split(n: int, size: int, rng: np.random.Generator):
    """Generate a balanced split of images and integer KL-grade labels."""
    labels = rng.integers(0, len(_grades()), size=n)
    images = np.stack([generate_image(int(g), size, rng) for g in labels])
    return images[:, None, :, :], labels.astype(np.int64)  # NCHW, single channel


def _grades():
    from .config import KL_GRADES

    return KL_GRADES


class KneeDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray):
        self.images = torch.from_numpy(images)
        self.labels = torch.from_numpy(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.images[idx], self.labels[idx]


def build_dataloaders(cfg: Config):
    """Create reproducible train/val/test dataloaders from synthetic data."""
    rng = np.random.default_rng(cfg.seed)
    train = make_split(cfg.n_train, cfg.image_size, rng)
    val = make_split(cfg.n_val, cfg.image_size, rng)
    test = make_split(cfg.n_test, cfg.image_size, rng)

    loaders = {
        "train": DataLoader(KneeDataset(*train), batch_size=cfg.batch_size, shuffle=True),
        "val": DataLoader(KneeDataset(*val), batch_size=cfg.batch_size, shuffle=False),
        "test": DataLoader(KneeDataset(*test), batch_size=cfg.batch_size, shuffle=False),
    }
    return loaders, {"train": train, "val": val, "test": test}
