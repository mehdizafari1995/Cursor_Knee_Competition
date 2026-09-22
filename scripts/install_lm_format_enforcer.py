#!/usr/bin/env python3
"""Install the pinned LM Format Enforcer wheels from a local hashed directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from knee.lm_format_enforcer_wheels import (  # noqa: E402
    install_from_local_wheels,
    verify_vendored_wheels,
    wheels_dir,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheels-dir", default=None)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    directory = Path(args.wheels_dir) if args.wheels_dir else wheels_dir()
    if args.verify_only:
        rows = verify_vendored_wheels(directory)
        print(json.dumps({"ok": True, "verified": rows}, indent=2))
        return 0
    payload = install_from_local_wheels(directory)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
