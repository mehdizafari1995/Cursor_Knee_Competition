#!/usr/bin/env python3
"""Phase 4E visible-test submission rehearsal. Does not submit to Kaggle."""

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
from knee.phase4e_rehearsal import run_phase4e_rehearsal  # noqa: E402
from knee.report_labels import (  # noqa: E402
    HARD_WEAK_LABEL_TRAINING_ALLOWED,
    SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--execute-inference", action="store_true")
    parser.add_argument("--allow-tiny-encoder", action="store_true")
    parser.add_argument("--no-verify-frozen-hashes", action="store_true")
    parser.add_argument("--no-require-cuda", action="store_true")
    args = parser.parse_args()
    assert HARD_WEAK_LABEL_TRAINING_ALLOWED is False
    assert SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED is False
    assert SECOND_READER_SCORE_BEARING_ALLOWED is False
    result = run_phase4e_rehearsal(
        data_root=args.data_root,
        output_dir=args.output_dir,
        config_path=args.config,
        checkpoint_dir=args.checkpoint_dir,
        execute_inference=args.execute_inference,
        allow_tiny_encoder_for_tests=args.allow_tiny_encoder,
        verify_frozen_hashes=not args.no_verify_frozen_hashes,
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
                    "n_visible_test_studies": result.get("n_visible_test_studies"),
                    "n_report_only_studies_processed": result.get(
                        "n_report_only_studies_processed"
                    ),
                    "competition_submission_written": result.get(
                        "competition_submission_written"
                    ),
                    "consumes_submission_slot": result.get("consumes_submission_slot"),
                    "hidden_test_available": result.get("hidden_test_available"),
                }
            ),
            sort_keys=True,
        )
    )
    print("DO_NOT_SUBMIT")
    status = str(result.get("status") or "")
    if status.endswith("FAIL") or status.endswith("MISMATCH"):
        return 2
    if result.get("ok"):
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
