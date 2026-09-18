#!/usr/bin/env python3
"""Phase 4C.6 corrected Phi GPU intervention replay. Synthetic reports only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from knee.config import PROJECT_ROOT  # noqa: E402
from knee.report_labels import (  # noqa: E402
    HARD_WEAK_LABEL_TRAINING_ALLOWED,
    SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED,
)
from knee.second_reader_phase46 import (  # noqa: E402
    KAGGLE_PHASE46_OUTPUT_DIRNAME,
    run_phase46_replay,
)
from knee.second_reader_schema import DEFAULT_PILOT_CONFIG_PATH_V1_4  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs" / KAGGLE_PHASE46_OUTPUT_DIRNAME),
    )
    parser.add_argument("--config", default=str(DEFAULT_PILOT_CONFIG_PATH_V1_4))
    parser.add_argument("--weights", default=None)
    parser.add_argument("--focus-only", action="store_true")
    args = parser.parse_args()
    assert HARD_WEAK_LABEL_TRAINING_ALLOWED is False
    assert SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED is False
    result = run_phase46_replay(
        output_dir=args.output_dir,
        config_path=args.config,
        weights_path=args.weights,
        focus_only=args.focus_only,
    )
    print(f"status={result.get('status')}")
    print(f"ok={result.get('ok')}")
    print(f"skip_reason={result.get('skip_reason')}")
    print(f"gpu_replay_executed={result.get('gpu_replay_executed')}")
    print(f"n_gold_studies_processed={result.get('n_gold_studies_processed')}")
    print(f"n_report_only_studies_processed={result.get('n_report_only_studies_processed')}")
    print(f"gold_rerun_authorized={result.get('gold_rerun_authorized')}")
    print(f"semantic_gate_status={result.get('semantic_gate_status')}")
    print("gate1=" + json.dumps(result.get("gate1") or {}, sort_keys=True, default=str)[:4000])
    if result.get("gate2"):
        print("gate2_ok=" + str((result.get("gate2") or {}).get("ok")))
    if result.get("gate3"):
        print("gate3=" + json.dumps(result.get("gate3") or {}, sort_keys=True, default=str)[:8000])
    if result.get("status") == "CONSTRAINED_FORMAT_PASS_SEMANTIC_PASS":
        return 2
    if not result.get("gpu_replay_executed"):
        return 0
    if not result.get("ok"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
