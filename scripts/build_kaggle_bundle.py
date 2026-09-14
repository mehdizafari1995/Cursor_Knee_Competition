#!/usr/bin/env python3
"""Build an allowlisted ZIP of this repository for a private Kaggle dataset.

Does not use the Kaggle API and does not require a token. The ZIP is written
under outputs/ (gitignored) together with a SHA-256 manifest.
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
from knee.kaggle_bundle import BUNDLE_NAME, build_bundle  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        default=str(PROJECT_ROOT),
        help="Repository root to package (default: this checkout).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs" / "kaggle_bundle"),
        help="Directory for the zip and manifest (gitignored).",
    )
    parser.add_argument(
        "--zip-name",
        default=BUNDLE_NAME,
        help="Archive filename.",
    )
    args = parser.parse_args()
    manifest = build_bundle(
        repo_root=args.repo_root,
        output_dir=args.output_dir,
        zip_name=args.zip_name,
    )
    print(json.dumps({k: manifest[k] for k in ("zip_path", "zip_sha256", "n_files", "manifest_path")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
