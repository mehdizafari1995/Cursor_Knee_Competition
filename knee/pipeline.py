"""End-to-end knee OA grading pipeline.

Run with:  python -m knee.pipeline

Stages: generate synthetic data -> train CNN -> evaluate on held-out test set
-> write submission.csv and diagnostic plots into ./outputs.
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from .config import Config
from .data import build_dataloaders
from .report import (
    save_confusion_matrix,
    save_sample_montage,
    text_report,
    write_submission,
)
from .train import evaluate, train_model


def run(cfg: Config) -> dict:
    device = torch.device("cpu")
    print(f"Device: {device} | torch {torch.__version__}", flush=True)

    t0 = time.time()
    loaders, raw = build_dataloaders(cfg)
    train_images, train_labels = raw["train"]
    print(
        f"Data: {len(train_labels)} train / {cfg.n_val} val / {cfg.n_test} test "
        f"({cfg.image_size}x{cfg.image_size} grayscale)",
        flush=True,
    )

    save_sample_montage(train_images, train_labels, cfg.output_dir / "samples.png")

    model = train_model(cfg, loaders, device)

    test_acc, y_true, y_pred = evaluate(model, loaders["test"], device)
    print(f"\nTest accuracy: {test_acc:.3f}\n", flush=True)
    print(text_report(y_true, y_pred), flush=True)

    save_confusion_matrix(y_true, y_pred, cfg.output_dir / "confusion_matrix.png")
    submission = write_submission(y_pred, cfg, cfg.output_dir / "submission.csv")
    torch.save(model.state_dict(), cfg.output_dir / "model.pt")

    elapsed = time.time() - t0
    metrics = {
        "test_accuracy": round(test_acc, 4),
        "num_classes": cfg.num_classes,
        "epochs": cfg.epochs,
        "elapsed_seconds": round(elapsed, 2),
        "submission_rows": int(len(submission)),
    }
    (cfg.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Artifacts written to {cfg.output_dir}/ in {elapsed:.1f}s", flush=True)
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Knee OA KL-grade pipeline")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.seed is not None:
        cfg.seed = args.seed
    run(cfg)


if __name__ == "__main__":
    main()
