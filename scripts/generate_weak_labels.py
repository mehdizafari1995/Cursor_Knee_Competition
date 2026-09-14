#!/usr/bin/env python3
"""Generate weak labels for report-only studies.

Writes a gitignored row-level table (StudyInstanceUID, twelve labels,
twelve confidences, extractor version, language, quality flags). Never
commits that file. Stdout is aggregate only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from knee.config import PROJECT_ROOT  # noqa: E402
from knee.kaggle_audit import assert_no_report_text, collect_report_snippets  # noqa: E402
from knee.kaggle_bundle import sha256_file  # noqa: E402
from knee.paths import get_data_root  # noqa: E402
from knee.report_labels import (  # noqa: E402
    PrivacyLeakError,
    assert_extractor_has_no_gold_parameter,
    assert_no_identifiers,
    collect_identifiers,
    extract_frame_reports,
    jsonable,
    load_extractor_config,
    split_gold_and_report_only,
    weak_label_aggregates,
    write_private_csv,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=None)
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs" / "report_labels" / "private"),
    )
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    assert_extractor_has_no_gold_parameter()
    config = load_extractor_config(args.config)
    try:
        data_root = get_data_root(args.data_root)
    except Exception as exc:  # noqa: BLE001
        print(f"status=DATA_NOT_AVAILABLE reason={exc}")
        return 2

    train_path = Path(data_root) / "train.csv"
    if not train_path.is_file():
        print("status=DATA_NOT_AVAILABLE reason=missing train.csv")
        return 2

    import pandas as pd

    train = pd.read_csv(train_path)
    _gold, report_only = split_gold_and_report_only(train)
    print(f"n_report_only_studies={len(report_only)}")
    print("unknown_imputed_as_negative=false")
    print(f"extractor_version={config.get('extractor_version')}")
    pred = extract_frame_reports(report_only, config=config)
    drop = [c for c in pred.columns if c.endswith("_evidence") or c.endswith("_rule")]
    weak = pred.drop(columns=drop, errors="ignore")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weak_path = out_dir / f"weak_labels_{config.get('extractor_version')}.csv"
    write_private_csv(weak, weak_path)
    agg = weak_label_aggregates(pred)
    agg_path = out_dir.parent / "sanitized" / "weak_label_aggregates.json"
    agg_path.parent.mkdir(parents=True, exist_ok=True)
    agg_path.write_text(json.dumps(jsonable(agg), indent=2, sort_keys=True) + "\n")
    snippets = collect_report_snippets(train)
    identifiers = collect_identifiers(train)
    assert_no_report_text([agg_path], snippets)
    try:
        assert_no_identifiers([agg_path], identifiers)
    except PrivacyLeakError:
        raise
    print(f"weak_labels_sha256={sha256_file(weak_path)}")
    print(f"wrote_aggregates {agg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
