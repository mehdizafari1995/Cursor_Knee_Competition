#!/usr/bin/env python3
"""Phase 4C.7 acceptance audit: existing sanitized metrics only. No inference."""

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
from knee.report_labels import (  # noqa: E402
    HARD_WEAK_LABEL_TRAINING_ALLOWED,
    SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED,
)
from knee.second_reader_phase47_acceptance_audit import (  # noqa: E402
    STATUS_INCOMPLETE,
    render_audit_markdown,
    run_phase47_acceptance_audit,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", default=None)
    parser.add_argument("--acceptance", default=None)
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs" / "phase47_acceptance_audit"),
    )
    args = parser.parse_args()
    assert HARD_WEAK_LABEL_TRAINING_ALLOWED is False
    assert SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED is False
    result = run_phase47_acceptance_audit(
        metrics_path=args.metrics,
        acceptance_path=args.acceptance,
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "second_reader_phase47_acceptance_audit.json"
    md_path = out_dir / "second_reader_phase47_acceptance_audit.md"
    json_path.write_text(json.dumps(jsonable(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_audit_markdown(result), encoding="utf-8")
    print(f"status={result.get('status')}")
    print(f"ok={result.get('ok')}")
    print(f"reason={result.get('reason')}")
    print(f"gold_eval_executed={result.get('gold_eval_executed')}")
    print(f"gold_rerun_executed={result.get('gold_rerun_executed')}")
    print(f"gold_rerun_authorized={result.get('gold_rerun_authorized')}")
    print(f"n_gold_studies_processed={result.get('n_gold_studies_processed')}")
    print(f"n_report_only_studies_processed={result.get('n_report_only_studies_processed')}")
    proof = result.get("medial_meniscus_ci_proof") or {}
    print("medial_meniscus_audited_status=" + str(proof.get("audited_status")))
    print("medial_meniscus_still_pass=" + str(proof.get("still_pilot_pass_after_fail_closed_ci")))
    print(f"json_path={json_path}")
    print(f"markdown_path={md_path}")
    if result.get("status") == STATUS_INCOMPLETE:
        return 2
    if not result.get("ok"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
