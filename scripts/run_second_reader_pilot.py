#!/usr/bin/env python3
"""Gold-only second-reader pilot. Never processes report-only studies."""

from __future__ import annotations

import argparse
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
from knee.second_reader_pilot import run_second_reader_gold_pilot  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=None)
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs" / "second_reader_gold_pilot"),
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--weights", default=None)
    args = parser.parse_args()

    assert HARD_WEAK_LABEL_TRAINING_ALLOWED is False
    assert SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED is False
    result = run_second_reader_gold_pilot(
        data_root=args.data_root,
        output_dir=args.output_dir,
        config_path=args.config,
        weights_path=args.weights,
    )
    print(f"status={result.get('status')}")
    print(f"scientific_evaluation={result.get('scientific_evaluation')}")
    print(f"n_gold_studies={result.get('n_gold_studies')}")
    print(f"n_report_only_studies_processed={result.get('n_report_only_studies_processed')}")
    print(f"internet_used={result.get('internet_used')}")
    print(f"download_weights={result.get('download_weights')}")
    print(f"training_run={result.get('training_run')}")
    print(f"folds_created={result.get('folds_created')}")
    print(f"malformed_output_rate={result.get('malformed_output_rate')}")
    print("private hashes recorded:", sorted((result.get("private_hashes") or {}).keys()))
    return 0 if result.get("status") in {"ok", "DATA_NOT_AVAILABLE"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
