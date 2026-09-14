#!/usr/bin/env python3
"""Evaluate the report-label extractor against completely expert-labelled studies.

Unknown gold labels are never imputed as negative. Row-level predictions stay
under gitignored outputs. Stdout prints aggregate counts only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from knee.config import PROJECT_ROOT, REPORT_COLUMN  # noqa: E402
from knee.kaggle_audit import assert_no_report_text, collect_report_snippets  # noqa: E402
from knee.paths import get_data_root  # noqa: E402
from knee.report_labels import (  # noqa: E402
    PrivacyLeakError,
    assert_extractor_has_no_gold_parameter,
    assert_no_identifiers,
    collect_identifiers,
    evaluate_gold_predictions,
    extract_frame_reports,
    jsonable,
    load_extractor_config,
    new_evaluation_dir,
    split_gold_and_report_only,
    write_private_csv,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=None)
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs" / "report_labels"),
        help="Gitignored directory for versioned evaluation artefacts.",
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
    gold_frame, _report_only = split_gold_and_report_only(train)
    print(f"n_gold_studies={len(gold_frame)}")
    print("unknown_imputed_as_negative=false")
    print(f"extractor_version={config.get('extractor_version')}")
    pred = extract_frame_reports(gold_frame, config=config)
    metrics = evaluate_gold_predictions(gold_frame, pred, config=config)
    out_dir = new_evaluation_dir(Path(args.output_dir), config)
    private = out_dir / "private" / "gold_predictions.csv"
    write_private_csv(pred, private)
    metrics_path = out_dir / "gold_metrics.json"
    metrics_path.write_text(json.dumps(jsonable(metrics), indent=2, sort_keys=True) + "\n")
    snippets = collect_report_snippets(train)
    identifiers = collect_identifiers(train)
    assert_no_report_text([metrics_path], snippets)
    try:
        assert_no_identifiers([metrics_path], identifiers)
    except PrivacyLeakError:
        raise
    print(f"wrote {metrics_path}")
    print(f"macro_balanced_accuracy={metrics['macro'].get('balanced_accuracy')}")
    print(f"report_column_in_metrics={REPORT_COLUMN in json.dumps(jsonable(metrics))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
