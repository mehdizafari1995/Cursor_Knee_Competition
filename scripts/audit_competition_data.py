#!/usr/bin/env python3
"""Read-only audit of the RSNA Knee Abnormality Detection dataset.

Exits 2 with status DATA_NOT_AVAILABLE when the real files cannot be accessed.
Does not invent counts or write anything under the data root.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from knee.audit import (  # noqa: E402
    STATUS_DATA_NOT_AVAILABLE,
    run_audit,
    write_audit_report,
)
from knee.config import PROJECT_ROOT  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default=None,
        help="Competition data root. Overrides RSNA_DATA_ROOT when given.",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "reports" / "data_audit.md"),
        help="Markdown report path (tracked reports/*.md only; do not commit DICOM).",
    )
    parser.add_argument(
        "--max-series",
        type=int,
        default=None,
        help="Optional cap on series scanned for DICOM headers.",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Label for the audit source (default: local_competition_data).",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional JSON dump (should live under outputs/, which is gitignored).",
    )
    args = parser.parse_args()

    result = run_audit(
        data_root=args.data_root,
        max_series=args.max_series,
        source=args.source,
    )
    path = write_audit_report(result, Path(args.output))
    print(f"status={result['status']}")
    print(f"wrote {path}")
    if args.json_out:
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(result)
        json_path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
        print(f"wrote {json_path}")
    return 2 if result["status"] == STATUS_DATA_NOT_AVAILABLE else 0


if __name__ == "__main__":
    raise SystemExit(main())
