#!/usr/bin/env python3
"""Constrained synthetic canary. Never processes gold or report-only studies."""

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
from knee.second_reader_canary import fixture_canary_generate_fn  # noqa: E402
from knee.second_reader_constrained import (  # noqa: E402
    KAGGLE_CONSTRAINED_OUTPUT_DIRNAME,
    run_constrained_canary,
)
from knee.second_reader_schema import DEFAULT_PILOT_CONFIG_PATH_V1_4  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs" / KAGGLE_CONSTRAINED_OUTPUT_DIRNAME),
    )
    parser.add_argument("--config", default=str(DEFAULT_PILOT_CONFIG_PATH_V1_4))
    parser.add_argument("--weights", default=None)
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    assert HARD_WEAK_LABEL_TRAINING_ALLOWED is False
    assert SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED is False
    generate_fn = fixture_canary_generate_fn if args.fixture else None
    result = run_constrained_canary(
        output_dir=args.output_dir,
        config_path=args.config,
        weights_path=args.weights,
        generate_fn=generate_fn,
        smoke=args.smoke,
        n_repeats=1 if args.smoke else 2,
    )
    print(f"status={result.get('status')}")
    print(f"ok={result.get('ok')}")
    print(f"n_examples={result.get('n_examples')}")
    print(f"n_gold_studies_processed={result.get('n_gold_studies_processed')}")
    print(f"n_report_only_studies_processed={result.get('n_report_only_studies_processed')}")
    print(f"malformed_output_rate={result.get('malformed_output_rate')}")
    print(f"n_schema_valid={result.get('n_schema_valid')}")
    print(f"stop_gold_rerun={result.get('stop_gold_rerun')}")
    print("intervention=" + json.dumps(result.get("intervention") or {}, sort_keys=True)[:4000])
    if result.get("semantic"):
        print("semantic_status=" + str(result["semantic"].get("status")))
        print(
            "synthetic_status_accuracy="
            + str((result["semantic"].get("synthetic_status") or {}).get("overall_status_accuracy"))
        )
    if not result.get("ok"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
