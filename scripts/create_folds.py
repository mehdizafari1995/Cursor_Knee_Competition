#!/usr/bin/env python3
"""Create deterministic, group-aware cross-validation folds.

Prefers a patient identifier when every study has one; otherwise groups by
StudyInstanceUID. Refuses to run when competition (or fixture) labels cannot
be read. Does not invent a dataset.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from knee.config import PROJECT_ROOT, STUDY_ID_COLUMN  # noqa: E402
from knee.folds import folds_from_study_table, save_folds  # noqa: E402
from knee.paths import DataNotAvailable, get_data_root  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=None)
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "outputs" / "folds.csv"),
        help="Fold assignment CSV (outputs/ is gitignored).",
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--patient-column",
        default=None,
        help="Optional patient-id column in train.csv. Ignored if incomplete.",
    )
    args = parser.parse_args()

    try:
        root = get_data_root(args.data_root)
    except DataNotAvailable as exc:
        print(exc)
        return 2

    train_csv = root / "train.csv"
    if not train_csv.is_file():
        print(f"DATA_NOT_AVAILABLE: {train_csv} is missing")
        return 2

    frame = pd.read_csv(train_csv)
    if STUDY_ID_COLUMN not in frame.columns:
        print(f"DATA_NOT_AVAILABLE: {train_csv} has no {STUDY_ID_COLUMN} column")
        return 2

    table, metadata = folds_from_study_table(
        frame,
        n_folds=args.n_folds,
        seed=args.seed,
        patient_column=args.patient_column,
    )
    save_folds(table, Path(args.output), metadata)
    print(f"wrote {args.output}")
    print(
        f"groups={metadata['n_groups']} studies={metadata['n_studies']} "
        f"group_source={metadata['group_source']}"
    )
    if metadata["impossible_full_coverage_targets"]:
        print(
            "targets that cannot appear in every fold: "
            + ", ".join(metadata["impossible_full_coverage_targets"])
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
