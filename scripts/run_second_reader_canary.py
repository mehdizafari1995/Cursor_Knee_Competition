#!/usr/bin/env python3
"""Synthetic second-reader canary. Never processes gold or report-only studies."""

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
from knee.second_reader_canary import run_synthetic_canary  # noqa: E402
from knee.second_reader_schema import DEFAULT_PILOT_CONFIG_PATH_V1_3  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "rsna_second_reader_synthetic_canary"))
    parser.add_argument("--config", default=str(DEFAULT_PILOT_CONFIG_PATH_V1_3))
    parser.add_argument("--weights", default=None)
    parser.add_argument("--fixture", action="store_true", help="Use the software generate_fn (not scientific).")
    parser.add_argument("--smoke", action="store_true", help="Three-example smoke test (one run).")
    args = parser.parse_args()
    assert HARD_WEAK_LABEL_TRAINING_ALLOWED is False
    assert SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED is False
    generate_fn = None
    if args.fixture:
        from knee.second_reader_canary import fixture_canary_generate_fn

        generate_fn = fixture_canary_generate_fn
    result = run_synthetic_canary(
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
    print(f"repeatable={result.get('repeatable')}")
    print(f"stop_gold_rerun={result.get('stop_gold_rerun')}")
    print(f"n_parsed_root_object={result.get('n_parsed_root_object')}")
    print(f"n_targets_field_present={result.get('n_targets_field_present')}")
    print(f"n_targets_is_object={result.get('n_targets_is_object')}")
    print(f"n_targets_not_object={result.get('n_targets_not_object')}")
    print(f"n_schema_valid={result.get('n_schema_valid')}")
    print(f"n_responses_with_parsed_root_object_and_targets_field={result.get('n_responses_with_parsed_root_object_and_targets_field')}")
    print(f"container_conservation_ok={result.get('container_conservation_ok')}")
    print("enum_error_table=" + json.dumps(result.get("enum_error_table") or [], sort_keys=True)[:8000])
    if not result.get("ok"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
