#!/usr/bin/env python3
"""Prepared gold-58 three-plane OOF. Execution is not authorized."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from knee.gold58_3plane_oof import (  # noqa: E402
    GOLD58_3PLANE_OOF_EXECUTION_ALLOWED,
    run_gold58_3plane_oof,
)
from knee.kaggle_audit import jsonable  # noqa: E402
from knee.phase4d_readiness import SECOND_READER_SCORE_BEARING_ALLOWED  # noqa: E402
from knee.phase4f_ssl import GOLD58_SSL_OOF_EXECUTION_ALLOWED, UNLABELED_IMAGE_SSL_EXECUTION_ALLOWED  # noqa: E402
from knee.report_labels import (  # noqa: E402
    HARD_WEAK_LABEL_TRAINING_ALLOWED,
    SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED,
)


def main() -> int:
    assert HARD_WEAK_LABEL_TRAINING_ALLOWED is False
    assert SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED is False
    assert SECOND_READER_SCORE_BEARING_ALLOWED is False
    assert UNLABELED_IMAGE_SSL_EXECUTION_ALLOWED is False
    assert GOLD58_SSL_OOF_EXECUTION_ALLOWED is False
    assert GOLD58_3PLANE_OOF_EXECUTION_ALLOWED is False
    result = run_gold58_3plane_oof(execute_training=False)
    print(json.dumps(jsonable(result), indent=2, sort_keys=True))
    print("DO_NOT_SUBMIT")
    print("DO_NOT_LAUNCH_3PLANE_OOF")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
