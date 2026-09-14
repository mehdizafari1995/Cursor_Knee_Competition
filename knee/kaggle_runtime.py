"""Kaggle mount detection, preflight, source extraction, and read-only input guards.

The notebook runs with internet disabled. Competition data is attached through
the Kaggle UI at ``/kaggle/input/rsna-knee-abnormality-detection``. Repository
source is a separate private input and is copied or extracted only under
``/kaggle/working``. Nothing in this module writes under ``/kaggle/input``.
"""

from __future__ import annotations

import os
import shutil
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from .competition import DOCUMENTED_DIRECTORIES, DOCUMENTED_FILES
from .config import PROJECT_ROOT
from .paths import is_competition_root

KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
KAGGLE_COMPETITION_SLUG = "rsna-knee-abnormality-detection"
KAGGLE_OUTPUT_DIRNAME = "rsna_audit_outputs"
KAGGLE_SRC_DIRNAME = "rsna_knee_src"

DATA_SOURCE_REAL_KAGGLE = "real_kaggle_mount"
DATA_SOURCE_SOFTWARE_FIXTURE = "software_fixture"
DATA_SOURCE_LOCAL = "local_competition_data"

SOURCE_MARKERS = ("knee/config.py", "knee/kaggle_audit.py")
SOURCE_COPY_DIRS = ("knee", "scripts", "configs", "notebooks")
SOURCE_COPY_FILES = ("pyproject.toml", "requirements.txt", "README.md")
# GitHub-archive datasets nest as dataset/Repo-main/Repo-main/knee. Bound the
# walk so competition DICOM trees (skipped by slug) are never traversed.
SOURCE_SEARCH_MAX_DEPTH = 6
SOURCE_WALK_SKIP_DIRS = frozenset(
    {"train_series", "test_series", ".git", "__pycache__", ".venv", "data"}
)


class PreflightError(RuntimeError):
    """Required Kaggle/local inputs are missing. The audit must stop."""

    status = "DATA_NOT_AVAILABLE"


class ReadOnlyInputError(RuntimeError):
    """Raised when a write is aimed at a protected input path, or inputs changed."""


@dataclass(frozen=True)
class KaggleLayout:
    """Filesystem layout. Defaults match the Kaggle notebook runtime."""

    input_root: Path = KAGGLE_INPUT_ROOT
    working_root: Path = KAGGLE_WORKING_ROOT
    competition_slug: str = KAGGLE_COMPETITION_SLUG
    output_dirname: str = KAGGLE_OUTPUT_DIRNAME
    src_dirname: str = KAGGLE_SRC_DIRNAME

    @property
    def competition_dir(self) -> Path:
        return Path(self.input_root) / self.competition_slug

    @property
    def output_dir(self) -> Path:
        return Path(self.working_root) / self.output_dirname

    @property
    def src_extract_dir(self) -> Path:
        return Path(self.working_root) / self.src_dirname


def default_layout() -> KaggleLayout:
    return KaggleLayout()


def is_kaggle_runtime(layout: KaggleLayout | None = None) -> bool:
    layout = layout or default_layout()
    return layout.input_root.is_dir() and layout.working_root.is_dir()


def is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def is_kaggle_input_path(path: Path, layout: KaggleLayout | None = None) -> bool:
    layout = layout or default_layout()
    return is_under(Path(path), layout.input_root)


def assert_not_kaggle_input(path: Path, layout: KaggleLayout | None = None) -> Path:
    """Refuse any write target under the read-only Kaggle input mount."""
    layout = layout or default_layout()
    resolved = Path(path).expanduser()
    if layout.input_root.exists() and is_kaggle_input_path(resolved, layout):
        raise ReadOnlyInputError(
            f"refusing to write under Kaggle input mount: {resolved} "
            f"(protected root {layout.input_root})"
        )
    return resolved


def looks_like_source_root(path: Path) -> bool:
    return all((path / marker).is_file() for marker in SOURCE_MARKERS)


def find_source_in_tree(
    root: Path,
    max_depth: int = SOURCE_SEARCH_MAX_DEPTH,
) -> Path | None:
    """Find ``knee/config.py`` under ``root`` without walking DICOM series trees."""
    root = Path(root)
    if looks_like_source_root(root):
        return root
    if max_depth <= 0 or not root.is_dir():
        return None
    try:
        root_resolved = root.resolve()
    except OSError:
        return None
    for dirpath, dirnames, _filenames in os.walk(root):
        current = Path(dirpath)
        try:
            rel = current.resolve().relative_to(root_resolved)
            depth = 0 if str(rel) == "." else len(rel.parts)
        except (OSError, ValueError):
            dirnames[:] = []
            continue
        dirnames[:] = sorted(d for d in dirnames if d not in SOURCE_WALK_SKIP_DIRS)
        if depth > max_depth:
            dirnames[:] = []
            continue
        if looks_like_source_root(current):
            return current
    return None


def list_top_level(path: Path) -> list[str]:
    """Names of direct children only. Never recurse into DICOM series trees."""
    if not path.is_dir():
        return []
    try:
        names = sorted(child.name for child in path.iterdir())
    except OSError:
        return []
    return names


def count_child_dirs(path: Path, depth: int) -> int:
    """Count directories at ``depth`` below ``path`` without listing DICOM files.

    ``depth=1`` counts study folders; ``depth=2`` counts series folders.
    """
    if not path.is_dir() or depth < 1:
        return 0
    pattern = "/".join(["*"] * depth)
    return sum(1 for child in path.glob(pattern) if child.is_dir())


def _zip_contains_source(zip_path: Path) -> bool:
    try:
        with zipfile.ZipFile(zip_path) as zf:
            names = set(zf.namelist())
    except (OSError, zipfile.BadZipFile):
        return False
    for marker in SOURCE_MARKERS:
        if marker in names or f"./{marker}" in names:
            return True
        if any(name.endswith(marker) or name.endswith(marker.replace("/", os.sep)) for name in names):
            return True
    return False


def find_source_zip(layout: KaggleLayout) -> Path | None:
    if not layout.input_root.is_dir():
        return None
    zips: list[Path] = []
    for child in sorted(layout.input_root.iterdir()):
        if child.name == layout.competition_slug:
            continue
        if child.is_file() and child.suffix.lower() == ".zip" and _zip_contains_source(child):
            zips.append(child)
        elif child.is_dir():
            for zip_path in sorted(child.glob("*.zip")):
                if _zip_contains_source(zip_path):
                    zips.append(zip_path)
            # One extra nested folder, e.g. GitHub-archive datasets.
            for nested in sorted(p for p in child.iterdir() if p.is_dir()):
                for zip_path in sorted(nested.glob("*.zip")):
                    if _zip_contains_source(zip_path):
                        zips.append(zip_path)
    return zips[0] if zips else None


def find_source_root(
    layout: KaggleLayout,
    *,
    explicit: Path | None = None,
    search_repo: bool = True,
) -> Path | None:
    """Locate repository source. Prefers an already extracted working copy."""
    if explicit is not None and looks_like_source_root(Path(explicit)):
        return Path(explicit)
    if looks_like_source_root(layout.src_extract_dir):
        return layout.src_extract_dir
    if search_repo and looks_like_source_root(PROJECT_ROOT):
        return PROJECT_ROOT
    if not layout.input_root.is_dir():
        return None
    for child in sorted(layout.input_root.iterdir()):
        if child.name == layout.competition_slug:
            continue
        if child.is_dir():
            found = find_source_in_tree(child, max_depth=SOURCE_SEARCH_MAX_DEPTH)
            if found is not None:
                return found
    return None


def _safe_extract_zip(zip_path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    dest_resolved = dest.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = info.filename
            if not name or name.endswith("/"):
                continue
            target = (dest / name).resolve()
            if not is_under(target, dest_resolved):
                raise PreflightError(f"refusing zip path that escapes extract dir: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, target.open("wb") as out:
                shutil.copyfileobj(src, out)


def _copy_allowlisted_source(src_root: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for dirname in SOURCE_COPY_DIRS:
        src_dir = src_root / dirname
        if src_dir.is_dir():
            shutil.copytree(src_dir, dest / dirname, dirs_exist_ok=True)
    for filename in SOURCE_COPY_FILES:
        src_file = src_root / filename
        if src_file.is_file():
            shutil.copy2(src_file, dest / filename)


def extract_source_to_working(
    layout: KaggleLayout,
    *,
    explicit_source: Path | None = None,
) -> Path:
    """Copy or unzip source under ``/kaggle/working``. Never writes to input."""
    assert_not_kaggle_input(layout.src_extract_dir, layout)
    dest = layout.src_extract_dir
    if looks_like_source_root(dest):
        return dest

    located = find_source_root(layout, explicit=explicit_source, search_repo=True)
    zip_path = find_source_zip(layout)

    if zip_path is not None and (located is None or is_kaggle_input_path(zip_path, layout)):
        # Prefer a dedicated source zip when the only located tree is the git
        # checkout and we are on Kaggle (input zip is the attached dataset).
        if located is None or is_kaggle_input_path(located, layout) or (
            is_kaggle_runtime(layout) and located == PROJECT_ROOT
        ):
            _safe_extract_zip(zip_path, dest)
            found = find_source_in_tree(dest, max_depth=SOURCE_SEARCH_MAX_DEPTH)
            if found is not None:
                return found
            raise PreflightError(
                f"extracted {zip_path} under {dest} but knee/config.py was not found"
            )

    if located is None:
        raise PreflightError(
            "repository source was not found. Attach a private Kaggle dataset "
            "containing the allowlisted source zip (or an unpacked tree with "
            "knee/config.py), then re-run. Extraction writes only under "
            f"{layout.src_extract_dir}."
        )

    if is_kaggle_input_path(located, layout) or located != dest:
        _copy_allowlisted_source(located, dest)
        if looks_like_source_root(dest):
            return dest
        raise PreflightError(f"copied source from {located} but {dest} is incomplete")
    return located


def ensure_source_on_sys_path(source_root: Path) -> None:
    resolved = str(source_root.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


class ReadOnlyInputGuard:
    """Snapshot files under protected roots and later assert they are unchanged.

    ``recursive=False`` (the default) only snapshots files at the root of each
    directory so the first Kaggle audit does not ``stat`` hundreds of thousands
    of DICOM instances. Tests may enable recursion on tiny fixtures.
    """

    def __init__(self, roots: list[Path], *, recursive: bool = False):
        self.roots = [Path(root) for root in roots if Path(root).exists()]
        self.recursive = recursive
        self._snapshot = self.snapshot()

    def _iter_files(self) -> list[Path]:
        files: list[Path] = []
        for root in self.roots:
            if root.is_file():
                files.append(root)
                continue
            if not root.is_dir():
                continue
            if self.recursive:
                for path in root.rglob("*"):
                    if path.is_file():
                        files.append(path)
            else:
                for child in root.iterdir():
                    if child.is_file():
                        files.append(child)
        return files

    def snapshot(self) -> dict[str, tuple[int, int]]:
        out: dict[str, tuple[int, int]] = {}
        for path in self._iter_files():
            try:
                stat = path.stat()
            except OSError:
                continue
            out[str(path.resolve())] = (stat.st_mtime_ns, stat.st_size)
        return out

    def assert_unchanged(self) -> None:
        current = self.snapshot()
        if current != self._snapshot:
            added = sorted(set(current) - set(self._snapshot))
            removed = sorted(set(self._snapshot) - set(current))
            changed = sorted(
                key
                for key in set(current) & set(self._snapshot)
                if current[key] != self._snapshot[key]
            )
            raise ReadOnlyInputError(
                "protected input files changed during the audit "
                f"(added={added[:5]}, removed={removed[:5]}, changed={changed[:5]})"
            )


@dataclass
class PreflightResult:
    ok: bool
    status: str
    data_source: str
    scientific_evaluation: bool
    layout: KaggleLayout
    data_root: Path | None
    source_root: Path | None
    output_dir: Path
    top_level_competition: list[str] = field(default_factory=list)
    top_level_input_datasets: list[str] = field(default_factory=list)
    documented_files_present: dict[str, bool] = field(default_factory=dict)
    missing_required: list[str] = field(default_factory=list)
    message: str = ""
    internet_used: bool = False
    gpu_required: bool = False
    gpu_used: bool = False
    kaggle_runtime: bool = False

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "status": self.status,
            "data_source": self.data_source,
            "scientific_evaluation": self.scientific_evaluation,
            "data_root": str(self.data_root) if self.data_root else None,
            "source_root": str(self.source_root) if self.source_root else None,
            "output_dir": str(self.output_dir),
            "top_level_competition": self.top_level_competition,
            "top_level_input_datasets": self.top_level_input_datasets,
            "documented_files_present": self.documented_files_present,
            "missing_required": self.missing_required,
            "message": self.message,
            "internet_used": self.internet_used,
            "gpu_required": self.gpu_required,
            "gpu_used": self.gpu_used,
            "kaggle_runtime": self.kaggle_runtime,
            "layout": {
                "input_root": str(self.layout.input_root),
                "working_root": str(self.layout.working_root),
                "competition_dir": str(self.layout.competition_dir),
                "output_dir": str(self.layout.output_dir),
                "src_extract_dir": str(self.layout.src_extract_dir),
            },
        }


def infer_data_source(
    data_root: Path | None,
    *,
    explicit: str | None = None,
    layout: KaggleLayout | None = None,
) -> str:
    if explicit:
        return explicit
    layout = layout or default_layout()
    if data_root is None:
        return DATA_SOURCE_SOFTWARE_FIXTURE
    resolved = Path(data_root)
    try:
        resolved = resolved.resolve()
    except OSError:
        pass
    competition = layout.competition_dir
    try:
        competition_resolved = competition.resolve() if competition.exists() else competition
    except OSError:
        competition_resolved = competition
    if resolved == competition_resolved or is_under(resolved, competition):
        return DATA_SOURCE_REAL_KAGGLE
    text = str(resolved).lower()
    if "pytest" in text or "/tmp" in text or "fixture" in text:
        return DATA_SOURCE_SOFTWARE_FIXTURE
    if is_kaggle_runtime(layout) and is_under(resolved, layout.input_root):
        return DATA_SOURCE_REAL_KAGGLE
    return DATA_SOURCE_LOCAL


def scientific_evaluation_for(data_source: str) -> bool:
    """True only for a real Kaggle mount of competition files. Fixtures stay false."""
    return data_source == DATA_SOURCE_REAL_KAGGLE


def preflight(
    *,
    layout: KaggleLayout | None = None,
    data_root: str | Path | None = None,
    source_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    data_source: str | None = None,
    extract_source: bool = False,
) -> PreflightResult:
    """Inspect inputs before any analysis. Does not recurse into DICOM trees."""
    layout = layout or default_layout()
    kaggle = is_kaggle_runtime(layout)

    if data_root is not None:
        resolved_data = Path(data_root).expanduser()
    elif kaggle and is_competition_root(layout.competition_dir):
        resolved_data = layout.competition_dir
    else:
        resolved_data = layout.competition_dir if kaggle else None

    explicit_source = Path(source_root).expanduser() if source_root is not None else None
    located_source = find_source_root(layout, explicit=explicit_source, search_repo=True)
    if extract_source:
        try:
            located_source = extract_source_to_working(layout, explicit_source=explicit_source)
            ensure_source_on_sys_path(located_source)
        except PreflightError:
            # Source extraction is required on Kaggle; locally the git checkout is enough.
            if kaggle and located_source is None:
                raise

    if output_dir is not None:
        resolved_output = Path(output_dir).expanduser()
    else:
        resolved_output = layout.output_dir
    assert_not_kaggle_input(resolved_output, layout)

    top_datasets = list_top_level(layout.input_root) if layout.input_root.is_dir() else []
    top_competition: list[str] = []
    present: dict[str, bool] = {}
    missing: list[str] = []
    message = ""
    ok = False
    status = PreflightError.status

    if resolved_data is None or not Path(resolved_data).exists():
        missing.append("competition data root")
        message = (
            "AUDIT STOPPED: competition data were not found. On Kaggle, attach the "
            f"{KAGGLE_COMPETITION_SLUG} dataset through the Notebook Data panel "
            f"(expected at {layout.competition_dir}). This notebook does not use "
            "a Kaggle API token and does not download files."
        )
    elif not is_competition_root(Path(resolved_data)):
        missing.append("documented competition CSV files")
        message = (
            f"AUDIT STOPPED: {resolved_data} exists but does not contain any of "
            f"{', '.join(DOCUMENTED_FILES)}. Refusing to invent statistics."
        )
        top_competition = list_top_level(Path(resolved_data))
    else:
        resolved_data = Path(resolved_data)
        top_competition = list_top_level(resolved_data)
        for name in DOCUMENTED_FILES:
            found = (resolved_data / name).is_file()
            present[name] = found
            if not found:
                missing.append(name)
        for name in DOCUMENTED_DIRECTORIES:
            found = (resolved_data / name).is_dir()
            present[name] = found
        if not present.get("train.csv"):
            message = (
                f"AUDIT STOPPED: train.csv is missing under {resolved_data}. "
                "Level A/B/C cannot run."
            )
        else:
            ok = True
            status = "PREFLIGHT_OK"
            message = (
                "Preflight passed. Analysis will read CSVs and a bounded DICOM "
                "header sample only. Pixel arrays are limited to the documented "
                "Level C sample. Internet is not used. GPU is not required. "
                "Inputs are read-only."
            )

    source = infer_data_source(resolved_data if ok else None, explicit=data_source, layout=layout)
    if not ok:
        source = data_source or (
            DATA_SOURCE_REAL_KAGGLE if kaggle else DATA_SOURCE_SOFTWARE_FIXTURE
        )

    return PreflightResult(
        ok=ok,
        status=status,
        data_source=source,
        scientific_evaluation=scientific_evaluation_for(source) if ok else False,
        layout=layout,
        data_root=Path(resolved_data) if resolved_data is not None and Path(resolved_data).exists() else None,
        source_root=located_source,
        output_dir=resolved_output,
        top_level_competition=top_competition,
        top_level_input_datasets=top_datasets,
        documented_files_present=present,
        missing_required=missing,
        message=message,
        kaggle_runtime=kaggle,
    )


def raise_if_preflight_failed(result: PreflightResult) -> None:
    if not result.ok:
        raise PreflightError(result.message)


def detect_kaggle_paths(layout: KaggleLayout | None = None) -> dict:
    """Public helper used by tests and the notebook: detected paths, no DICOM walk."""
    layout = layout or default_layout()
    competition = layout.competition_dir
    return {
        "kaggle_input_root": str(layout.input_root),
        "kaggle_working_root": str(layout.working_root),
        "competition_dir": str(competition),
        "competition_dir_exists": competition.is_dir(),
        "is_competition_root": is_competition_root(competition) if competition.is_dir() else False,
        "output_dir": str(layout.output_dir),
        "src_extract_dir": str(layout.src_extract_dir),
        "top_level_input_datasets": list_top_level(layout.input_root),
        "top_level_competition": list_top_level(competition) if competition.is_dir() else [],
        "kaggle_runtime": is_kaggle_runtime(layout),
        "internet_used": False,
        "gpu_required": False,
    }
