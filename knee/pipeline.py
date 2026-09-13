"""End-to-end RSNA Knee Abnormality Detection pipeline.

Run with:  python -m knee.pipeline

Stages: generate synthetic MRI studies -> derive weak labels from multilingual
reports (train only) -> train a multi-label CNN on images -> run image-only
inference on the held-out test studies -> report macro-averaged ROC AUC and
write submission.csv plus diagnostic plots into ./outputs.

Note: this pipeline validates the environment on synthetic data only. It does
not download the competition dataset or train on real studies.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from .config import Config, TARGET_KEYS
from .data import build_dataloaders
from .report import (
    save_auc_barchart,
    save_sample_montage,
    text_report,
    write_submission,
)
from .reports import extract_weak_label_matrix
from .train import evaluate, train_model


def run(cfg: Config) -> dict:
    device = torch.device("cpu")
    print(f"Device: {device} | torch {torch.__version__}", flush=True)

    t0 = time.time()
    loaders, raw = build_dataloaders(cfg)
    train_images, train_true, train_reports = raw["train"]
    print(
        f"Data: {len(train_true)} train / {cfg.n_val} val / {cfg.n_test} test "
        f"studies ({cfg.image_size}x{cfg.image_size} MRI, {cfg.num_targets} targets)",
        flush=True,
    )

    # Weak supervision: derive training labels from multilingual reports.
    if cfg.use_weak_supervision:
        train_labels = extract_weak_label_matrix(train_reports, cfg.num_targets)
        agreement = float((train_labels == train_true).mean())
        print(
            f"Weak supervision: training labels extracted from multilingual "
            f"reports (agreement with latent labels {agreement:.3f})",
            flush=True,
        )
    else:
        train_labels = train_true

    save_sample_montage(train_images, train_true, cfg.output_dir / "samples.png")

    model = train_model(cfg, train_images, train_labels, loaders["val"], device)

    # Test-time inference is image-only (no report text is consumed).
    test_macro, per_target, y_true, y_prob = evaluate(model, loaders["test"], device)
    print(f"\nTest macro-averaged ROC AUC: {test_macro:.4f}\n", flush=True)
    print(text_report(test_macro, per_target), flush=True)

    save_auc_barchart(per_target, test_macro, cfg.output_dir / "per_target_auc.png")
    submission = write_submission(y_prob, cfg, cfg.output_dir / "submission.csv")
    torch.save(model.state_dict(), cfg.output_dir / "model.pt")

    elapsed = time.time() - t0
    metrics = {
        "macro_roc_auc": round(test_macro, 4),
        "per_target_roc_auc": {
            key: (None if v != v else round(v, 4))
            for key, v in zip(TARGET_KEYS, per_target)
        },
        "num_targets": cfg.num_targets,
        "epochs": cfg.epochs,
        "weak_supervision": cfg.use_weak_supervision,
        "elapsed_seconds": round(elapsed, 2),
        "submission_rows": int(len(submission)),
    }
    (cfg.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Artifacts written to {cfg.output_dir}/ in {elapsed:.1f}s", flush=True)
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="RSNA Knee Abnormality Detection pipeline")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--no-weak-supervision",
        action="store_true",
        help="Train on latent labels instead of report-derived weak labels.",
    )
    args = parser.parse_args()

    cfg = Config()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.seed is not None:
        cfg.seed = args.seed
    if args.no_weak_supervision:
        cfg.use_weak_supervision = False
    run(cfg)


if __name__ == "__main__":
    main()
