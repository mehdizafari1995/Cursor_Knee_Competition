#!/usr/bin/env python3
"""Phase 4D read-only competition-readiness review. No training or inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from knee.config import PROJECT_ROOT  # noqa: E402
from knee.kaggle_audit import jsonable  # noqa: E402
from knee.phase4d_readiness import (  # noqa: E402
    PHASE4D_EXPERIMENT_EXECUTED,
    render_parked_decision_markdown,
    render_readiness_markdown,
    run_phase4d_readiness_review,
)
from knee.report_labels import (  # noqa: E402
    HARD_WEAK_LABEL_TRAINING_ALLOWED,
    SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--gold", default=None)
    parser.add_argument("--sample-submission", default=None)
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs" / "phase4d_readiness"),
    )
    args = parser.parse_args()
    assert HARD_WEAK_LABEL_TRAINING_ALLOWED is False
    assert SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED is False
    assert PHASE4D_EXPERIMENT_EXECUTED is False
    result = run_phase4d_readiness_review(
        predictions_path=args.predictions,
        gold_path=args.gold,
        sample_path=args.sample_submission,
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "phase4d_competition_readiness.json"
    json_path.write_text(json.dumps(jsonable(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    parked_path = out_dir / "phase4c_second_reader_parked.md"
    review_path = out_dir / "phase4d_competition_readiness.md"
    parked_path.write_text(render_parked_decision_markdown(result), encoding="utf-8")
    review_path.write_text(render_readiness_markdown(result), encoding="utf-8")
    print(f"status={result.get('status')}")
    print(f"training_run={result.get('training_run')}")
    print(f"inference_run={result.get('inference_run')}")
    print(f"kaggle_kernel_launched={result.get('kaggle_kernel_launched')}")
    print(f"competition_submission_written={result.get('competition_submission_written')}")
    print(f"phase4d_experiment_executed={result.get('phase4d_experiment_executed')}")
    print(f"selected_experiment={(result.get('selected_experiment') or {}).get('id')}")
    print(f"metric_reproduction_status={result.get('metric_reproduction_status')}")
    print(f"json_path={json_path}")
    print(f"parked_path={parked_path}")
    print(f"review_path={review_path}")
    if result.get("training_run") or result.get("inference_run"):
        return 2
    if result.get("phase4d_experiment_executed"):
        return 2
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
