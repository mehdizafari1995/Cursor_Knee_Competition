#!/usr/bin/env python3
"""Build submission.csv aligned to sample_submission.csv.

Validates official column spelling and StudyInstanceUID row order. Accepts
predictions keyed by official names or internal TARGET_KEYS.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from knee.paths import DataNotAvailable, get_data_root  # noqa: E402
from knee.submission import (  # noqa: E402
    load_sample_submission,
    write_competition_submission,
)


def _resolve_sample(path: str | None, data_root: str | None) -> Path:
    if path:
        return Path(path)
    try:
        root = get_data_root(data_root)
    except DataNotAvailable as exc:
        raise SystemExit(str(exc)) from exc
    sample = root / "sample_submission.csv"
    if not sample.is_file():
        raise SystemExit(f"DATA_NOT_AVAILABLE: {sample} is missing")
    return sample


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--sample-submission", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument(
        "--output",
        default=str(ROOT / "outputs" / "submission.csv"),
    )
    args = parser.parse_args()

    sample_path = _resolve_sample(args.sample_submission, args.data_root)
    sample = load_sample_submission(sample_path)
    predictions = pd.read_csv(args.predictions)
    out = write_competition_submission(predictions, sample, Path(args.output))
    print(f"wrote {args.output} ({len(out)} rows, {len(out.columns)} columns)")
    print("columns: " + ",".join(out.columns))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
