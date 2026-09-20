#!/usr/bin/env python3
"""Phase 4D.2 gold-58 image-only grouped OOF trainer (series-policy v2).

Does not overwrite v1. Does not weaken scripts/train_baseline.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from knee.gold58_oof import (  # noqa: E402
    DEFAULT_CONFIG_PATH_V2,
    run_gold58_image_oof,
)
from knee.kaggle_audit import jsonable  # noqa: E402
from knee.phase4d_readiness import SECOND_READER_SCORE_BEARING_ALLOWED  # noqa: E402
from knee.report_labels import (  # noqa: E402
    HARD_WEAK_LABEL_TRAINING_ALLOWED,
    SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH_V2))
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--execute-training", action="store_true")
    parser.add_argument("--allow-tiny-encoder", action="store_true")
    parser.add_argument("--no-require-cuda", action="store_true")
    args = parser.parse_args()
    assert HARD_WEAK_LABEL_TRAINING_ALLOWED is False
    assert SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED is False
    assert SECOND_READER_SCORE_BEARING_ALLOWED is False
    result = run_gold58_image_oof(
        data_root=args.data_root,
        output_dir=args.output_dir,
        config_path=args.config,
        weights_path=args.weights,
        execute_training=args.execute_training,
        allow_tiny_encoder_for_tests=args.allow_tiny_encoder,
        require_cuda=False if args.no_require_cuda else None,
    )
    print(
        json.dumps(
            jsonable(
                {
                    "status": result.get("status"),
                    "ok": result.get("ok"),
                    "skip_reason": result.get("skip_reason"),
                    "training_run": result.get("training_run"),
                    "n_labelled_studies": result.get("n_labelled_studies"),
                    "n_report_only_studies_processed": result.get(
                        "n_report_only_studies_processed"
                    ),
                    "macro_roc_auc": result.get("macro_roc_auc"),
                    "competition_submission_written": result.get(
                        "competition_submission_written"
                    ),
                    "SECOND_READER_SCORE_BEARING_ALLOWED": result.get(
                        "SECOND_READER_SCORE_BEARING_ALLOWED"
                    ),
                }
            ),
            sort_keys=True,
        )
    )
    status = str(result.get("status") or "")
    if status.endswith("PREFLIGHT_FAIL") or status == "PREFLIGHT_FAIL":
        return 2
    if result.get("ok"):
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
