"""Fast smoke tests for the knee OA pipeline (kept tiny for CI speed)."""

from __future__ import annotations

import numpy as np
import torch

from knee.config import Config, KL_GRADES
from knee.data import build_dataloaders, generate_image
from knee.model import KneeCNN
from knee.pipeline import run


def _tiny_cfg(tmp_path) -> Config:
    cfg = Config()
    cfg.n_train, cfg.n_val, cfg.n_test = 40, 20, 20
    cfg.epochs = 1
    cfg.output_dir = tmp_path
    return cfg


def test_generate_image_shape_and_range():
    rng = np.random.default_rng(0)
    img = generate_image(3, size=64, rng=rng)
    assert img.shape == (64, 64)
    assert img.dtype == np.float32
    assert 0.0 <= img.min() and img.max() <= 1.0


def test_joint_space_narrows_with_grade():
    # Higher KL grade should, on average, brighten the joint-space center row
    # (bones move closer together / touch), a core learnable signal.
    rng = np.random.default_rng(1)
    size = 64
    center = size // 2
    low = np.mean([generate_image(0, size, rng)[center].mean() for _ in range(30)])
    high = np.mean([generate_image(4, size, rng)[center].mean() for _ in range(30)])
    assert high > low


def test_model_forward_shape():
    model = KneeCNN(num_classes=len(KL_GRADES), image_size=64)
    out = model(torch.zeros(4, 1, 64, 64))
    assert out.shape == (4, len(KL_GRADES))


def test_dataloaders_batches():
    cfg = Config()
    cfg.n_train, cfg.n_val, cfg.n_test = 32, 16, 16
    loaders, _ = build_dataloaders(cfg)
    images, labels = next(iter(loaders["train"]))
    assert images.shape[1:] == (1, cfg.image_size, cfg.image_size)
    assert labels.max() < cfg.num_classes


def test_end_to_end_pipeline(tmp_path):
    metrics = run(_tiny_cfg(tmp_path))
    assert (tmp_path / "submission.csv").exists()
    assert (tmp_path / "confusion_matrix.png").exists()
    assert (tmp_path / "metrics.json").exists()
    assert 0.0 <= metrics["test_accuracy"] <= 1.0
