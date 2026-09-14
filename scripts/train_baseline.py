#!/usr/bin/env python3
"""Train the study-level baseline — smoke by default, real training opt-in.

Without --smoke this script requires a competition data root. It still will not
run a full training loop unless --allow-training is passed. CI and this phase
use --smoke with tiny generated tensors only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from knee.baseline import run_smoke  # noqa: E402
from knee.baseline_config import load_baseline_config  # noqa: E402
from knee.paths import DataNotAvailable, get_data_root  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "baseline.yaml"))
    parser.add_argument("--data-root", default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Software verification on tiny generated tensors. Not a scientific run.",
    )
    parser.add_argument(
        "--allow-training",
        action="store_true",
        help="Opt in to optimizer steps on real studies. Off by default.",
    )
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg = load_baseline_config(
        args.config,
        smoke=True if args.smoke else None,
        allow_training=True if args.allow_training else None,
        output_dir=args.output_dir,
    )
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke or cfg.smoke:
        metrics = run_smoke(cfg)
        (output_dir / "smoke_metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n"
        )
        print(json.dumps(metrics, indent=2))
        print(
            "SMOKE MODE: generated tensors only. Do not treat these numbers as "
            "competition performance."
        )
        return 0

    try:
        root = get_data_root(args.data_root)
    except DataNotAvailable as extra:
        print(extra)
        return 2

    if not cfg.allow_training:
        print(
            f"Data root resolved to {root}. Full training is disabled "
            "(allow_training=false). Pass --allow-training to opt in. "
            "This phase does not begin full training."
        )
        # Still construct the real encoder only if the operator asked for it;
        # default is to refuse loading ImageNet weights in this phase.
        print("No optimizer steps were taken.")
        (output_dir / "train_skipped.json").write_text(
            json.dumps(
                {
                    "status": "TRAINING_SKIPPED",
                    "data_root": str(root),
                    "allow_training": False,
                    "scientific_evaluation": False,
                },
                indent=2,
            )
            + "\n"
        )
        return 0

    # Opt-in path is implemented so the script exists, but this phase must not
    # start a full training run. Keep a hard guard.
    print(
        "Refusing to start full training in this phase even with "
        "--allow-training. Use a later training phase."
    )
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
