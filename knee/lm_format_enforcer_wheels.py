"""Hashed local install of LM Format Enforcer for internet-off Kaggle runs.

Wheels are never downloaded here. The notebook and CLI must point at already
copied files under ``vendor/wheels``.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from .config import PROJECT_ROOT

MANIFEST_RELATIVE = "configs/lm_format_enforcer_wheels.yaml"
DEFAULT_WHEELS_DIRNAME = Path("vendor") / "wheels"
LMFE_PACKAGE = "lm-format-enforcer"
LMFE_VERSION = "0.11.3"
LMFE_LICENSE = "MIT"
LMFE_TAG = "v0.11.3"
LMFE_COMMIT = "05c7dc3a46305e39c1a67c8622c433d8deda48a2"
LMFE_TAG_OBJECT = "29514a40ccc64bdb7fc07b970cd31e5f53d1564e"
LMFE_WHEEL_NAME = "lm_format_enforcer-0.11.3-py3-none-any.whl"
LMFE_WHEEL_SHA256 = "cf586350875def1ae7a8fba84fcbbfc8371424b6c9d05c1fcba70aa233fbf06f"
INTEREGULAR_WHEEL_NAME = "interegular-0.3.3-py37-none-any.whl"
INTEREGULAR_WHEEL_SHA256 = "b0c07007d48c89d6d19f7204972d369b2a77222722e126b6aa63aa721dc3b19c"

NETWORK_PATTERNS = (
    "urllib",
    "requests",
    "httpx",
    "aiohttp",
    "urlopen",
    "telemetry",
    "opentelemetry",
    "sentry",
    "posthog",
    "pypi.org",
    "huggingface.co",
)


class WheelInstallError(RuntimeError):
    """Local hashed wheel install cannot proceed."""


def manifest_path(root: Path | None = None) -> Path:
    return Path(root or PROJECT_ROOT) / MANIFEST_RELATIVE


def load_wheel_manifest(root: Path | None = None) -> dict[str, Any]:
    path = manifest_path(root)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise WheelInstallError("wheel manifest must be a mapping")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def wheels_dir(root: Path | None = None) -> Path:
    env = os.environ.get("LM_FORMAT_ENFORCER_WHEELS")
    if env:
        return Path(env)
    return Path(root or PROJECT_ROOT) / DEFAULT_WHEELS_DIRNAME


def vendored_wheel_specs(manifest: dict[str, Any] | None = None) -> list[dict[str, str]]:
    payload = manifest or load_wheel_manifest()
    specs = [
        {
            "name": str(payload.get("package") or LMFE_PACKAGE),
            "filename": str((payload.get("primary_wheel") or {}).get("filename") or LMFE_WHEEL_NAME),
            "sha256": str((payload.get("primary_wheel") or {}).get("sha256") or LMFE_WHEEL_SHA256),
        }
    ]
    for row in payload.get("vendored_dependencies") or ():
        if not isinstance(row, dict):
            continue
        if not row.get("vendor_in_kaggle_dataset"):
            continue
        specs.append(
            {
                "name": str(row.get("name") or ""),
                "filename": str(row.get("filename") or ""),
                "sha256": str(row.get("sha256") or ""),
            }
        )
    return specs


def verify_vendored_wheels(directory: str | Path, *, manifest: dict[str, Any] | None = None) -> list[dict[str, str]]:
    """Hash-check every wheel that must be installed on Kaggle."""
    root = Path(directory)
    if not root.is_dir():
        raise WheelInstallError(f"wheels directory missing: {root}")
    verified: list[dict[str, str]] = []
    for spec in vendored_wheel_specs(manifest):
        path = root / spec["filename"]
        if not path.is_file():
            raise WheelInstallError(f"required wheel missing: {path}")
        digest = sha256_file(path)
        if digest != spec["sha256"]:
            raise WheelInstallError(
                f"wheel sha256 mismatch for {spec['filename']}: got {digest} expected {spec['sha256']}"
            )
        verified.append({**spec, "path": str(path), "verified_sha256": digest})
    return verified


def install_from_local_wheels(
    directory: str | Path | None = None,
    *,
    root: Path | None = None,
    python_executable: str | None = None,
) -> dict[str, Any]:
    """Install hashed local wheels with pip --no-index --no-deps.

    Never contacts PyPI. Kaggle preinstalled packages (pydantic, packaging,
    PyYAML) are left untouched.
    """
    wheels = wheels_dir(root) if directory is None else Path(directory)
    manifest = load_wheel_manifest(root)
    if manifest.get("install_from_internet_on_kaggle"):
        raise WheelInstallError("manifest forbids internet installs")
    verified = verify_vendored_wheels(wheels, manifest=manifest)
    python = python_executable or sys.executable
    installed: list[str] = []
    for spec in verified:
        cmd = [
            python,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            "--no-build-isolation",
            "--disable-pip-version-check",
            spec["path"],
        ]
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if proc.returncode != 0:
            raise WheelInstallError(
                "local wheel install failed for "
                f"{spec['filename']}: {proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
            )
        installed.append(spec["filename"])
    imported = imported_package_versions()
    if str(imported.get("lm_format_enforcer") or "").startswith("import_failed"):
        raise WheelInstallError(f"lm_format_enforcer import failed: {imported['lm_format_enforcer']}")
    if str(imported.get("interegular") or "").startswith("import_failed"):
        raise WheelInstallError(f"interegular import failed: {imported['interegular']}")
    check = pip_check(python_executable=python)
    return {
        "ok": True,
        "internet_used": False,
        "installed": installed,
        "package": LMFE_PACKAGE,
        "version": LMFE_VERSION,
        "license": LMFE_LICENSE,
        "upstream_tag": LMFE_TAG,
        "upstream_commit_sha": LMFE_COMMIT,
        "wheels": verified,
        "imported_versions": imported,
        "runtime_versions": runtime_dependency_versions(),
        "pip_check": check,
    }


def pip_check(python_executable: str | None = None) -> dict[str, Any]:
    """Run ``pip check`` after the hashed local install. Never contacts PyPI."""
    python = python_executable or sys.executable
    proc = subprocess.run(
        [python, "-m", "pip", "check", "--disable-pip-version-check"],
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": (proc.stdout or "")[-4000:],
        "stderr": (proc.stderr or "")[-2000:],
    }


def imported_package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    try:
        import lmformatenforcer

        versions["lm_format_enforcer"] = str(
            getattr(lmformatenforcer, "__version__", LMFE_VERSION)
        )
    except Exception as exc:  # pragma: no cover - environment-specific
        versions["lm_format_enforcer"] = f"import_failed:{exc}"
    try:
        import interegular

        versions["interegular"] = str(getattr(interegular, "__version__", "unknown"))
    except Exception as exc:  # pragma: no cover - environment-specific
        versions["interegular"] = f"import_failed:{exc}"
    return versions


def runtime_dependency_versions() -> dict[str, str]:
    """Python and Kaggle-preinstalled stack versions used with the local wheels."""
    versions = {
        "python": sys.version.split()[0],
        "python_full": sys.version.replace("\n", " "),
    }
    modules = (
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("tokenizers", "tokenizers"),
        ("pydantic", "pydantic"),
        ("packaging", "packaging"),
        ("pyyaml", "yaml"),
        ("lm_format_enforcer", "lmformatenforcer"),
        ("interegular", "interegular"),
    )
    for label, import_name in modules:
        try:
            mod = __import__(import_name)
            versions[label] = str(getattr(mod, "__version__", "unknown"))
        except Exception as exc:  # pragma: no cover - optional on CPU CI
            versions[label] = f"unavailable:{type(exc).__name__}"
    versions.update(imported_package_versions())
    return versions


def scan_python_sources_for_network(paths: list[Path]) -> list[dict[str, str]]:
    """Return hits for unexpected network / telemetry strings."""
    hits: list[dict[str, str]] = []
    for path in paths:
        if not path.is_file() or path.suffix != ".py":
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        lowered = text.lower()
        for pattern in NETWORK_PATTERNS:
            if pattern.lower() in lowered:
                hits.append({"path": str(path), "pattern": pattern})
    return hits
