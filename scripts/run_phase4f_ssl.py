#!/usr/bin/env python3
"""Phase 4F image-only SSL planning / synthetic smoke. Does not train on real pixels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from knee.kaggle_audit import jsonable  # noqa: E402
from knee.phase4d_readiness import SECOND_READER_SCORE_BEARING_ALLOWED  # noqa: E402
from knee.phase4f_ssl import (  # noqa: E402
    GOLD58_SSL_OOF_EXECUTION_ALLOWED,
    UNLABELED_IMAGE_SSL_EXECUTION_ALLOWED,
    UNLABELED_IMAGE_SSL_PLANNING_ALLOWED,
    run_phase4f_ssl,
)
from knee.report_labels import (  # noqa: E402
    HARD_WEAK_LABEL_TRAINING_ALLOWED,
    SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--execute-training", action="store_true")
    parser.add_argument("--allow-tiny-encoder", action="store_true")
    parser.add_argument("--run-oof", action="store_true")
    args = parser.parse_args()
    assert HARD_WEAK_LABEL_TRAINING_ALLOWED is False
    assert SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED is False
    assert SECOND_READER_SCORE_BEARING_ALLOWED is False
    assert UNLABELED_IMAGE_SSL_PLANNING_ALLOWED is True
    assert UNLABELED_IMAGE_SSL_EXECUTION_ALLOWED is False
    assert GOLD58_SSL_OOF_EXECUTION_ALLOWED is False
    result = run_phase4f_ssl(
        data_root=args.data_root,
        output_dir=args.output_dir,
        config_path=args.config,
        execute_training=args.execute_training,
        allow_tiny_encoder_for_tests=args.allow_tiny_encoder,
        run_oof=args.run_oof,
        internet_enabled=False,
    )
    print(
        json.dumps(
            jsonable(
                {
                    "status": result.get("status"),
                    "ok": result.get("ok"),
                    "skip_reason": result.get("skip_reason"),
                    "training_run": result.get("training_run"),
                    "n_real_unlabelled_studies_processed": result.get(
                        "n_real_unlabelled_studies_processed"
                    ),
                    "OOF_run": result.get("OOF_run"),
                    "inference_run": result.get("inference_run"),
                    "competition_submission_written": result.get(
                        "competition_submission_written"
                    ),
                    "kaggle_kernel_launched": result.get("kaggle_kernel_launched"),
                }
            ),
            sort_keys=True,
        )
    )
    print("DO_NOT_SUBMIT")
    print("DO_NOT_LAUNCH_KERNEL")
    status = str(result.get("status") or "")
    if status.endswith("FAIL") or status.endswith("AUTHORIZED") or status.endswith("MISMATCH"):
        return 2
    if result.get("ok"):
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
