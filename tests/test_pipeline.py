"""Fast smoke tests for the RSNA Knee Abnormality Detection pipeline.

Kept tiny for CI speed. These exercise data generation, multilingual weak-label
extraction, the multi-label model, the macro ROC AUC metric, and a full
end-to-end run on synthetic data (no real data download, no real training).
"""

from __future__ import annotations

import numpy as np
import torch

from knee.config import Config, NUM_TARGETS, TARGET_KEYS
from knee.data import build_dataloaders, generate_study
from knee.model import KneeMRINet
from knee.pipeline import run
from knee.reports import extract_weak_labels, generate_report
from knee.train import macro_roc_auc


def _tiny_cfg(tmp_path) -> Config:
    cfg = Config()
    cfg.n_train, cfg.n_val, cfg.n_test = 120, 60, 60
    cfg.epochs = 1
    cfg.output_dir = tmp_path
    return cfg


def test_twelve_targets():
    assert NUM_TARGETS == 12
    assert len(TARGET_KEYS) == 12


def test_generate_study_shapes_and_range():
    rng = np.random.default_rng(0)
    img, labels, report = generate_study(size=64, rng=rng)
    assert img.shape == (1, 64, 64)
    assert img.dtype == np.float32
    assert 0.0 <= img.min() and img.max() <= 1.0
    assert labels.shape == (12,)
    assert set(np.unique(labels)).issubset({0.0, 1.0})
    assert isinstance(report, str) and len(report) > 0


def test_finding_creates_image_signal():
    # More positive findings mean more lesions and thus a brighter image, so
    # finding count and mean intensity should be positively correlated. This
    # confirms the multi-label signal is genuinely learnable.
    rng = np.random.default_rng(1)
    counts, brightness = [], []
    for _ in range(120):
        img, labels, _ = generate_study(64, rng)
        counts.append(float(labels.sum()))
        brightness.append(float(img.mean()))
    corr = np.corrcoef(counts, brightness)[0, 1]
    assert corr > 0.5


def test_weak_label_roundtrip_multilingual():
    # A report synthesized from a known label vector should round-trip back to
    # the same 12-dim vector via multilingual keyword extraction.
    rng = np.random.default_rng(7)
    labels = np.array([1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0], dtype=np.float32)
    report = generate_report(labels, rng)
    recovered = extract_weak_labels(report, NUM_TARGETS)
    assert np.array_equal(recovered, labels)


def test_model_forward_shape():
    model = KneeMRINet(num_targets=NUM_TARGETS, image_size=64)
    out = model(torch.zeros(4, 1, 64, 64))
    assert out.shape == (4, NUM_TARGETS)


def test_macro_roc_auc_perfect():
    y_true = np.array([[0, 1], [1, 0], [1, 1], [0, 0]], dtype=np.float32)
    y_prob = y_true.copy()  # perfect predictions
    macro, per_target = macro_roc_auc(y_true, y_prob)
    assert macro == 1.0
    assert len(per_target) == 2


def test_dataloaders_batches():
    cfg = Config()
    cfg.n_train, cfg.n_val, cfg.n_test = 32, 16, 16
    loaders, raw = build_dataloaders(cfg)
    images, labels = next(iter(loaders["train"]))
    assert images.shape[1:] == (1, cfg.image_size, cfg.image_size)
    assert labels.shape[1] == cfg.num_targets
    # Raw split exposes multilingual reports for weak supervision.
    assert len(raw["train"][2]) == cfg.n_train


def test_end_to_end_pipeline(tmp_path):
    metrics = run(_tiny_cfg(tmp_path))
    assert (tmp_path / "submission.csv").exists()
    assert (tmp_path / "per_target_auc.png").exists()
    assert (tmp_path / "metrics.json").exists()
    assert metrics["num_targets"] == 12
    assert 0.0 <= metrics["macro_roc_auc"] <= 1.0

    import pandas as pd

    sub = pd.read_csv(tmp_path / "submission.csv")
    assert list(sub.columns) == ["study_id", *TARGET_KEYS]
    assert len(sub) == 60
