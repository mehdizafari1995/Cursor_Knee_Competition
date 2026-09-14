"""Resolve the competition data root from the environment, never a hard-coded machine path.

The Kaggle dataset is *not* shipped with this repository. Callers must set
``RSNA_DATA_ROOT`` (see ``DATA_ROOT_ENV``) to a local copy after accepting the
competition rules. ``KNEE_DATA_ROOT`` is accepted as a fallback alias.
"""

from __future__ import annotations

import os
from pathlib import Path

from .config import DATA_ROOT_ENV, PROJECT_ROOT

FALLBACK_DATA_ROOT_ENV = "KNEE_DATA_ROOT"

# Documented filenames from the official Kaggle data page. Presence of any of
# these inside a directory is required before we treat that directory as a
# competition data root. We do not invent additional layout conventions.
DOCUMENTED_ROOT_MARKERS = (
    "sample_submission.csv",
    "train.csv",
    "train_series.csv",
    "test.csv",
    "test_series.csv",
)


class DataNotAvailable(RuntimeError):
    """Raised when the real competition files cannot be accessed."""

    status = "DATA_NOT_AVAILABLE"


def _unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for path in paths:
        resolved = path.expanduser()
        key = resolved
        if key in seen:
            continue
        seen.add(key)
        out.append(resolved)
    return out


def candidate_data_roots(explicit: str | Path | None = None) -> list[Path]:
    """Return candidate roots in priority order without checking that they exist."""
    paths: list[Path] = []
    if explicit is not None:
        paths.append(Path(explicit))
    env = os.environ.get(DATA_ROOT_ENV) or os.environ.get(FALLBACK_DATA_ROOT_ENV)
    if env:
        paths.append(Path(env))
    paths.append(PROJECT_ROOT / "data" / "rsna-knee-abnormality-detection")
    paths.append(PROJECT_ROOT / "data")
    paths.append(Path("/kaggle/input/rsna-knee-abnormality-detection"))
    paths.append(Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"))
    return _unique_paths(paths)


def is_competition_root(path: Path) -> bool:
    """True if ``path`` exists and contains at least one documented competition file."""
    if not path.is_dir():
        return False
    return any((path / marker).is_file() for marker in DOCUMENTED_ROOT_MARKERS)


def describe_search(explicit: str | Path | None = None) -> list[dict[str, str]]:
    """Record how each candidate was resolved. Used in DATA_NOT_AVAILABLE reports."""
    rows = []
    for path in candidate_data_roots(explicit):
        if not path.exists():
            status = "missing"
        elif is_competition_root(path):
            status = "competition_root"
        elif path.is_dir():
            status = "directory_without_documented_markers"
        else:
            status = "not_a_directory"
        rows.append({"path": str(path), "status": status})
    return rows


def get_data_root(explicit: str | Path | None = None) -> Path:
    """Return the competition data root.

    If ``explicit`` is given it is the only path considered (no silent fallback).
    Otherwise search ``RSNA_DATA_ROOT`` / ``KNEE_DATA_ROOT`` and documented
    default locations.

    Raises:
        DataNotAvailable: if no candidate contains documented competition files.
    """
    if explicit is not None:
        path = Path(explicit).expanduser()
        if is_competition_root(path):
            return path
        status = "missing" if not path.exists() else (
            "directory_without_documented_markers" if path.is_dir() else "not_a_directory"
        )
        raise DataNotAvailable(
            f"{DataNotAvailable.status}: explicit data root {str(path)!r} is not a "
            f"competition dataset ({status}). Looked for {', '.join(DOCUMENTED_ROOT_MARKERS)}."
        )

    searched = describe_search(None)
    for row in searched:
        if row["status"] == "competition_root":
            return Path(row["path"])
    env_value = os.environ.get(DATA_ROOT_ENV) or os.environ.get(FALLBACK_DATA_ROOT_ENV)
    env_note = (
        f"{DATA_ROOT_ENV}={env_value!r}"
        if env_value
        else f"{DATA_ROOT_ENV} is unset (optional alias {FALLBACK_DATA_ROOT_ENV} is also unset)"
    )
    details = "; ".join(f"{row['path']} ({row['status']})" for row in searched)
    raise DataNotAvailable(
        f"{DataNotAvailable.status}: no competition data root found. {env_note}. "
        f"Searched: {details}"
    )
