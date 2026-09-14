"""Deterministic DICOM series loading and geometric slice ordering.

Slice order prefers the projection of ``ImagePositionPatient`` onto the slice
normal derived from ``ImageOrientationPatient``. Fallbacks, in order:

1. ImagePositionPatient along the slice normal
2. InstanceNumber
3. filename (numeric-aware)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import pydicom
    from pydicom.dataset import Dataset
except ImportError:  # pragma: no cover - exercised only when pydicom is missing
    pydicom = None  # type: ignore[assignment]
    Dataset = object  # type: ignore[misc, assignment]


class DicomReadError(RuntimeError):
    """A single instance could not be parsed. The audit records these, then continues."""


def _require_pydicom() -> None:
    if pydicom is None:  # pragma: no cover
        raise ImportError("pydicom is required to read DICOM files")


def list_dicom_paths(series_dir: Path) -> list[Path]:
    """Return DICOM-like files in a series directory, sorted by name as a last resort."""
    if not series_dir.is_dir():
        return []
    paths = [
        path
        for path in series_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".dcm", ".dicom", ""}
    ]
    # Also accept extensionless files that look like numeric instance names.
    extra = [
        path
        for path in series_dir.iterdir()
        if path.is_file() and path.suffix == "" and path.name.isdigit()
    ]
    merged = {path.resolve(): path for path in paths + extra}
    return sorted(merged.values(), key=lambda p: _filename_sort_key(p.name))


def _filename_sort_key(name: str) -> tuple:
    stem = name.rsplit(".", 1)[0]
    if stem.isdigit():
        return (0, int(stem), name)
    return (1, name)


def _as_float_tuple(value, expected: int) -> tuple[float, ...] | None:
    if value is None:
        return None
    try:
        seq = [float(x) for x in value]
    except (TypeError, ValueError):
        return None
    if len(seq) != expected:
        return None
    return tuple(seq)


def slice_sort_key(ds: Dataset, path: Path) -> tuple:
    """Return a tuple that sorts slices from first to last along the stack."""
    orientation = _as_float_tuple(getattr(ds, "ImageOrientationPatient", None), 6)
    position = _as_float_tuple(getattr(ds, "ImagePositionPatient", None), 3)
    if orientation is not None and position is not None:
        row = np.array(orientation[:3], dtype=np.float64)
        col = np.array(orientation[3:], dtype=np.float64)
        normal = np.cross(row, col)
        norm = np.linalg.norm(normal)
        if norm > 0:
            normal = normal / norm
            along = float(np.dot(np.array(position, dtype=np.float64), normal))
            return (0, along, path.name)
    instance = getattr(ds, "InstanceNumber", None)
    try:
        if instance is not None:
            return (1, int(instance), path.name)
    except (TypeError, ValueError):
        pass
    return (2, *_filename_sort_key(path.name)[:2], path.name)


def read_header(path: Path, *, stop_before_pixels: bool = True) -> Dataset:
    _require_pydicom()
    try:
        return pydicom.dcmread(path, stop_before_pixels=stop_before_pixels, force=True)
    except Exception as exc:  # noqa: BLE001 - audit must survive corrupt files
        raise DicomReadError(f"failed to read {path}: {exc}") from exc


def header_summary(ds: Dataset) -> dict:
    """Compact metadata for the audit. Identifier *values* are not included."""
    orientation = _as_float_tuple(getattr(ds, "ImageOrientationPatient", None), 6)
    position = _as_float_tuple(getattr(ds, "ImagePositionPatient", None), 3)
    spacing = _as_float_tuple(getattr(ds, "PixelSpacing", None), 2)
    thickness = getattr(ds, "SliceThickness", None)
    try:
        thickness_f = float(thickness) if thickness is not None else None
    except (TypeError, ValueError):
        thickness_f = None
    rows = getattr(ds, "Rows", None)
    cols = getattr(ds, "Columns", None)
    phi_tags = []
    for keyword in (
        "PatientName",
        "PatientID",
        "PatientBirthDate",
        "PatientAddress",
        "OtherPatientIDs",
        "OtherPatientNames",
    ):
        value = getattr(ds, keyword, None)
        if value not in (None, "", "Anon", "ANON"):
            phi_tags.append(keyword)
    def _optional_float(tag: str) -> float | None:
        raw = getattr(ds, tag, None)
        if raw is None or raw == "":
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    return {
        "modality": str(getattr(ds, "Modality", "") or ""),
        "series_description": str(getattr(ds, "SeriesDescription", "") or ""),
        "protocol_name": str(getattr(ds, "ProtocolName", "") or ""),
        "scanning_sequence": str(getattr(ds, "ScanningSequence", "") or ""),
        "sequence_variant": str(getattr(ds, "SequenceVariant", "") or ""),
        "manufacturer": str(getattr(ds, "Manufacturer", "") or ""),
        "manufacturer_model_name": str(getattr(ds, "ManufacturerModelName", "") or ""),
        "body_part_examined": str(getattr(ds, "BodyPartExamined", "") or ""),
        "magnetic_field_strength": _optional_float("MagneticFieldStrength"),
        "echo_time": _optional_float("EchoTime"),
        "repetition_time": _optional_float("RepetitionTime"),
        "rows": int(rows) if rows is not None else None,
        "columns": int(cols) if cols is not None else None,
        "pixel_spacing": spacing,
        "slice_thickness": thickness_f,
        "image_orientation_patient": orientation,
        "has_image_position_patient": position is not None,
        "instance_number": getattr(ds, "InstanceNumber", None),
        "transfer_syntax_uid": str(
            getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", "") or ""
        ),
        "phi_tag_keywords_present": phi_tags,
        "has_patient_id_tag": hasattr(ds, "PatientID") and bool(str(getattr(ds, "PatientID", "") or "")),
        "has_study_instance_uid": bool(str(getattr(ds, "StudyInstanceUID", "") or "")),
        "has_series_instance_uid": bool(str(getattr(ds, "SeriesInstanceUID", "") or "")),
    }


def load_series_volume(series_dir: Path) -> tuple[np.ndarray, list[dict]]:
    """Load a series as float32 ``(S, H, W)`` sorted by ``slice_sort_key``.

    Pixel arrays are returned un-normalised. Headers are returned in the same
    order as the stacked slices.
    """
    paths = list_dicom_paths(series_dir)
    if not paths:
        raise FileNotFoundError(f"no DICOM instances under {series_dir}")

    records: list[tuple[tuple, Path, Dataset]] = []
    for path in paths:
        ds = read_header(path, stop_before_pixels=False)
        records.append((slice_sort_key(ds, path), path, ds))
    records.sort(key=lambda item: item[0])

    slices: list[np.ndarray] = []
    headers: list[dict] = []
    expected_hw: tuple[int, int] | None = None
    for _, path, ds in records:
        try:
            arr = np.asarray(ds.pixel_array)
        except Exception as exc:  # noqa: BLE001
            raise DicomReadError(f"failed to decode pixels in {path}: {exc}") from exc
        if arr.ndim == 3:
            arr = arr[0]
        if arr.ndim != 2:
            raise DicomReadError(f"{path} has pixel_array shape {arr.shape}, expected 2-D")
        if expected_hw is None:
            expected_hw = arr.shape
        elif arr.shape != expected_hw:
            raise DicomReadError(
                f"{path} has shape {arr.shape}, expected {expected_hw} in this series"
            )
        slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
        intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
        slices.append(arr.astype(np.float32) * slope + intercept)
        meta = header_summary(ds)
        meta["path"] = str(path)
        headers.append(meta)
    volume = np.stack(slices, axis=0).astype(np.float32)
    return volume, headers


def infer_plane_from_orientation(orientation: tuple[float, ...] | None) -> str | None:
    """Map ImageOrientationPatient to Axial / Coronal / Sagittal. None if unknown."""
    if orientation is None:
        return None
    row = np.array(orientation[:3], dtype=np.float64)
    col = np.array(orientation[3:], dtype=np.float64)
    normal = np.cross(row, col)
    if np.linalg.norm(normal) == 0:
        return None
    axis = int(np.argmax(np.abs(normal)))
    return ("Sagittal", "Coronal", "Axial")[axis]
