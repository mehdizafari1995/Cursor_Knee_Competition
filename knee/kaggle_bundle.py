"""Allowlisted source packaging for a private Kaggle dataset.

The generated ZIP is for attaching as a Kaggle input. It is never committed.
Credentials, DICOM, clinical report dumps, checkpoints, predictions, caches,
virtualenvs, and git metadata cannot enter the archive.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from .config import PROJECT_ROOT
from .kaggle_runtime import assert_not_kaggle_input

BUNDLE_NAME = "rsna_knee_source.zip"
MANIFEST_NAME = "kaggle_bundle_manifest.json"

ALLOWLIST_DIRS = (
    "knee",
    "scripts",
    "configs",
    "notebooks",
)

ALLOWLIST_FILES = (
    "pyproject.toml",
    "requirements.txt",
    "README.md",
)

EXCLUDED_DIR_NAMES = {
    ".git",
    ".github",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".cache",
    "cache",
    "caches",
    "data",
    "outputs",
    "predictions",
    "checkpoints",
    "weights",
    "runs",
    "wandb",
    "mlruns",
    ".kaggle",
    "secrets",
    "literature",
    "reports",
    ".cursor",
    "dist",
    "bundles",
    "kaggle_working",
}

EXCLUDED_SUFFIXES = {
    ".dcm",
    ".dicom",
    ".nii",
    ".nii.gz",
    ".mha",
    ".mhd",
    ".nrrd",
    ".pt",
    ".pth",
    ".ckpt",
    ".h5",
    ".hdf5",
    ".onnx",
    ".safetensors",
    ".pkl",
    ".joblib",
    ".pem",
    ".key",
    ".p12",
    ".pfx",
    ".zip",
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".tif",
    ".tiff",
    ".bmp",
    ".webp",
    ".pyc",
    ".pyo",
}

EXCLUDED_FILENAMES = {
    "kaggle.json",
    ".env",
    ".netrc",
    "weak_labels.csv",
    "gold_predictions.csv",
    "language_assignments.csv",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "credentials.json",
    "service-account.json",
    "secrets.yaml",
    "secrets.yml",
    "submission.csv",
    "predictions.csv",
}

CREDENTIAL_NAME_RE = re.compile(
    r"(^|/)(\.env(\..+)?|kaggle\.json|.*secret.*|.*credential.*|.*token.*|"
    r"id_rsa|id_dsa|id_ecdsa|id_ed25519|.*\.pem|.*\.key)$",
    re.IGNORECASE,
)

# Paths that must never be packaged even if they somehow match the allowlist.
FORBIDDEN_PATH_SUBSTRINGS = (
    "/.git/",
    ".venv/",
    "kaggle.json",
    ".kaggle/",
    "/data/",
    "/outputs/",
    "/predictions/",
    "/checkpoints/",
    "/reports/",
)


class BundleError(ValueError):
    """The source tree cannot be packaged as requested."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _posix_rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _is_excluded_dir(name: str) -> bool:
    return name in EXCLUDED_DIR_NAMES or name.endswith(".egg-info")


def _filename_forbidden(rel_posix: str) -> bool:
    name = Path(rel_posix).name
    if name in EXCLUDED_FILENAMES:
        return True
    if name.startswith(".env"):
        return True
    suffix = Path(rel_posix).suffix.lower()
    if suffix in EXCLUDED_SUFFIXES:
        return True
    if rel_posix.lower().endswith(".nii.gz"):
        return True
    if CREDENTIAL_NAME_RE.search(rel_posix.replace("\\", "/")):
        return True
    lowered = f"/{rel_posix.lower()}"
    if any(token in lowered for token in FORBIDDEN_PATH_SUBSTRINGS):
        return True
    return False


def iter_allowlisted_files(root: Path | None = None) -> list[Path]:
    """Return files that are allowed into the Kaggle source zip, in stable order."""
    root = Path(root or PROJECT_ROOT).resolve()
    selected: list[Path] = []
    for dirname in ALLOWLIST_DIRS:
        base = root / dirname
        if not base.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(name for name in dirnames if not _is_excluded_dir(name))
            for filename in sorted(filenames):
                path = Path(dirpath) / filename
                rel = _posix_rel(path, root)
                if _filename_forbidden(rel):
                    continue
                selected.append(path)
    for filename in ALLOWLIST_FILES:
        path = root / filename
        if path.is_file() and not _filename_forbidden(filename):
            selected.append(path)
    # Stable unique
    seen: set[str] = set()
    unique: list[Path] = []
    for path in selected:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def would_include(rel_posix: str) -> bool:
    """True if a relative path would be packaged. Used by tests."""
    rel = rel_posix.replace("\\", "/").lstrip("./")
    if _filename_forbidden(rel):
        return False
    if rel in ALLOWLIST_FILES:
        return True
    top = rel.split("/", 1)[0]
    return top in ALLOWLIST_DIRS


def build_bundle(
    *,
    repo_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    zip_name: str = BUNDLE_NAME,
) -> dict:
    """Write an allowlisted zip and a SHA-256 manifest. Returns manifest dict."""
    root = Path(repo_root or PROJECT_ROOT).resolve()
    dest_dir = Path(output_dir or (root / "outputs" / "kaggle_bundle")).resolve()
    assert_not_kaggle_input(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / zip_name
    files = iter_allowlisted_files(root)
    if not files:
        raise BundleError("no allowlisted files found to package")

    entries = []
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            rel = _posix_rel(path, root)
            if _filename_forbidden(rel):
                continue
            zf.write(path, arcname=rel)
            entries.append(
                {
                    "path": rel,
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
            )

    # Belt and suspenders: inspect the zip we just wrote.
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    offenders = [name for name in names if _filename_forbidden(name) or not would_include(name)]
    if offenders:
        zip_path.unlink(missing_ok=True)
        raise BundleError(f"forbidden paths entered the bundle: {offenders}")

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "repo_root": str(root),
        "zip_path": str(zip_path),
        "zip_name": zip_path.name,
        "zip_sha256": sha256_file(zip_path),
        "zip_bytes": zip_path.stat().st_size,
        "n_files": len(entries),
        "files": entries,
        "allowlist_dirs": list(ALLOWLIST_DIRS),
        "allowlist_files": list(ALLOWLIST_FILES),
        "excluded_dir_names": sorted(EXCLUDED_DIR_NAMES),
        "note": (
            "This zip is a Kaggle input artefact and must not be committed. "
            "It contains no DICOM, credentials, report text, checkpoints, or predictions."
        ),
    }
    manifest_path = dest_dir / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest
