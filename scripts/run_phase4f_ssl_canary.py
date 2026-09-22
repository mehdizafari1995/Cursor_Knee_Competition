#!/usr/bin/env python3
"""Phase 4F.1 image-only SSL canary. 64 studies / 40 steps. Not the full SSL job."""

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
)
from knee.phase4f_ssl_canary import (  # noqa: E402
    UNLABELED_IMAGE_SSL_CANARY_EXECUTION_ALLOWED,
    run_phase4f_ssl_canary,
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
    parser.add_argument(
        "--segment",
        default="full_with_fresh_restore",
        choices=("train_to_mid", "resume_to_end", "full_with_fresh_restore"),
    )
    parser.add_argument("--allow-tiny-encoder", action="store_true")
    parser.add_argument("--synthetic-tensors", action="store_true")
    parser.add_argument("--collapse-fixture", action="store_true")
    parser.add_argument("--run-oof", action="store_true")
    parser.add_argument("--run-full-unlabeled", action="store_true")
    args = parser.parse_args()
    assert HARD_WEAK_LABEL_TRAINING_ALLOWED is False
    assert SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED is False
    assert SECOND_READER_SCORE_BEARING_ALLOWED is False
    assert UNLABELED_IMAGE_SSL_EXECUTION_ALLOWED is False
    assert GOLD58_SSL_OOF_EXECUTION_ALLOWED is False
    assert UNLABELED_IMAGE_SSL_CANARY_EXECUTION_ALLOWED is True
    result = run_phase4f_ssl_canary(
        data_root=args.data_root,
        output_dir=args.output_dir,
        config_path=args.config,
        execute_training=True,
        allow_tiny_encoder_for_tests=args.allow_tiny_encoder,
        internet_enabled=False,
        collapse_fixture=args.collapse_fixture,
        run_oof=args.run_oof,
        run_full_unlabeled=args.run_full_unlabeled,
        segment=args.segment,
        synthetic_tensors=args.synthetic_tensors,
    )
    public = {
        "status": result.get("status"),
        "ok": result.get("ok"),
        "skip_reason": result.get("skip_reason"),
        "training_run": result.get("training_run"),
        "n_real_unlabelled_studies_pixels_opened": result.get(
            "n_real_unlabelled_studies_pixels_opened"
        ),
        "n_report_text_rows_processed": result.get("n_report_text_rows_processed"),
        "n_second_reader_rows_processed": result.get("n_second_reader_rows_processed"),
        "report_text_never_read": result.get("report_text_never_read"),
        "OOF_run": result.get("OOF_run"),
        "inference_run": result.get("inference_run"),
        "competition_submission_written": result.get("competition_submission_written"),
        "full_ssl_run_authorized": result.get("full_ssl_run_authorized"),
        "completed_step": (result.get("training") or {}).get("completed_step"),
        "resumed_step": (result.get("training") or {}).get("resumed_step"),
    }
    print(json.dumps(jsonable(public), sort_keys=True))
    print("DO_NOT_SUBMIT")
    print("DO_NOT_LAUNCH_FULL_SSL")
    status = str(result.get("status") or "")
    if status == "SSL_CANARY_MID_CHECKPOINT":
        return 0
    if status == "SSL_CANARY_PASS_FULL_RUN_NOT_AUTHORIZED":
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
