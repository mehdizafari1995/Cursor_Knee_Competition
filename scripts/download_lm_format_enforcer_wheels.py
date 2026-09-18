#!/usr/bin/env python3
"""Download pinned LM Format Enforcer wheels into gitignored vendor/wheels.

This helper is for local / coordinator machines. It must not run on Kaggle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from knee.lm_format_enforcer_wheels import (  # noqa: E402
    load_wheel_manifest,
    sha256_file,
    vendored_wheel_specs,
    wheels_dir,
)

PYPI_FILES = {
    "lm_format_enforcer-0.11.3-py3-none-any.whl": (
        "https://files.pythonhosted.org/packages/a0/ef/11292bb0b85cf4c93447cab5a29f64576ed14d3ab4280e35ddd23486594a/"
        "lm_format_enforcer-0.11.3-py3-none-any.whl"
    ),
    "interegular-0.3.3-py37-none-any.whl": (
        "https://files.pythonhosted.org/packages/c4/01/72d6472f80651673716d1deda2a5bbb633e563ecf94f4479da5519d69d25/"
        "interegular-0.3.3-py37-none-any.whl"
    ),
}


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as response, dest.open("wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheels-dir", default=None)
    args = parser.parse_args()
    dest_dir = Path(args.wheels_dir) if args.wheels_dir else wheels_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_wheel_manifest()
    results = []
    for spec in vendored_wheel_specs(manifest):
        url = PYPI_FILES.get(spec["filename"])
        if not url:
            raise SystemExit(f"no download URL recorded for {spec['filename']}")
        dest = dest_dir / spec["filename"]
        if not dest.is_file() or sha256_file(dest) != spec["sha256"]:
            _download(url, dest)
        digest = sha256_file(dest)
        if digest != spec["sha256"]:
            raise SystemExit(f"downloaded hash mismatch for {dest.name}: {digest}")
        results.append({"filename": dest.name, "sha256": digest, "bytes": dest.stat().st_size})
    print(json.dumps({"ok": True, "wheels_dir": str(dest_dir), "wheels": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
