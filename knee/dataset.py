"""Study-level dataset over the documented DICOM layout.

Used when a real (or software-fixture) competition root is present. Smoke tests
do not go through this class; they use generated tensors instead.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import NUM_TARGETS, STUDY_ID_COLUMN
from .dicom_io import load_series_volume
from .preprocess import percentile_normalize, sample_slices

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class StudySliceDataset(Dataset):
    """Load one series per study, sample a fixed number of slices, return tensors.

    ``series_lookup`` maps StudyInstanceUID -> directory of a single series. The
    caller decides which series (for example the first sagittal fluid-sensitive
    series) to keep; this class does not invent a series-selection policy beyond
    using the provided path.
    """

    def __init__(
        self,
        study_ids: list[str],
        series_dirs: list[Path],
        labels: np.ndarray | None,
        *,
        num_slices: int = 8,
        image_size: int = 224,
        percentiles: tuple[float, float] = (1.0, 99.0),
        interpolation: str = "nearest",
        imagenet_normalize: bool = False,
        imagenet_mean: tuple[float, float, float] = IMAGENET_MEAN,
        imagenet_std: tuple[float, float, float] = IMAGENET_STD,
    ):
        if len(study_ids) != len(series_dirs):
            raise ValueError("study_ids and series_dirs must be the same length")
        if labels is not None and len(labels) != len(study_ids):
            raise ValueError("labels must have one row per study")
        if labels is not None and labels.shape[1] != NUM_TARGETS:
            raise ValueError(f"labels must have {NUM_TARGETS} columns")
        if interpolation not in {"nearest", "bilinear"}:
            raise ValueError("interpolation must be 'nearest' or 'bilinear'")
        self.study_ids = list(study_ids)
        self.series_dirs = [Path(p) for p in series_dirs]
        self.labels = None if labels is None else np.asarray(labels, dtype=np.float32)
        self.num_slices = num_slices
        self.image_size = image_size
        self.percentiles = percentiles
        self.interpolation = interpolation
        self.imagenet_normalize = bool(imagenet_normalize)
        self.imagenet_mean = tuple(float(x) for x in imagenet_mean)
        self.imagenet_std = tuple(float(x) for x in imagenet_std)

    def __len__(self) -> int:
        return len(self.study_ids)

    def __getitem__(self, idx: int):
        volume, _headers = load_series_volume(self.series_dirs[idx])
        volume = percentile_normalize(volume, *self.percentiles)
        sampled = sample_slices(volume, self.num_slices)
        resized = np.stack(
            [
                _resize2d(slice_, self.image_size, interpolation=self.interpolation)
                for slice_ in sampled
            ],
            axis=0,
        )
        if self.imagenet_normalize:
            image = torch.from_numpy(
                apply_imagenet_normalization(
                    resized, mean=self.imagenet_mean, std=self.imagenet_std
                )
            )  # (S, 3, H, W)
        else:
            image = torch.from_numpy(resized[:, None, :, :])  # (S, 1, H, W)
        if self.labels is None:
            target = torch.full((NUM_TARGETS,), float("nan"))
        else:
            target = torch.from_numpy(self.labels[idx].copy())
        return {
            STUDY_ID_COLUMN: self.study_ids[idx],
            "image": image,
            "labels": target,
        }


def _resize2d(slice_: np.ndarray, size: int, interpolation: str = "nearest") -> np.ndarray:
    """Resize a 2-D slice. Default remains nearest-neighbour (v1)."""
    src = np.asarray(slice_, dtype=np.float32)
    if src.shape == (size, size):
        return src
    if interpolation == "bilinear":
        return _resize2d_bilinear(src, size)
    ys = np.linspace(0, src.shape[0] - 1, size)
    xs = np.linspace(0, src.shape[1] - 1, size)
    yi = np.round(ys).astype(np.int64)
    xi = np.round(xs).astype(np.int64)
    return src[yi][:, xi]


def _resize2d_bilinear(src: np.ndarray, size: int) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(src, dtype=np.float32)[None, None, ...])
    out = torch.nn.functional.interpolate(
        tensor, size=(size, size), mode="bilinear", align_corners=False
    )
    return out[0, 0].contiguous().numpy().astype(np.float32)


def apply_imagenet_normalization(
    slices_hw: np.ndarray,
    *,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> np.ndarray:
    """Repeat grayscale (S, H, W) in [0, 1] to RGB and apply ImageNet mean/std."""
    data = np.asarray(slices_hw, dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"expected (S, H, W), got {data.shape}")
    rgb = np.repeat(data[:, None, :, :], 3, axis=1)
    mean_arr = np.asarray(mean, dtype=np.float32).reshape(1, 3, 1, 1)
    std_arr = np.asarray(std, dtype=np.float32).reshape(1, 3, 1, 1)
    if np.any(std_arr == 0):
        raise ValueError("ImageNet std must be non-zero")
    return ((rgb - mean_arr) / std_arr).astype(np.float32)


def collate_studies(batch: list[dict]) -> dict:
    return {
        STUDY_ID_COLUMN: [item[STUDY_ID_COLUMN] for item in batch],
        "image": torch.stack([item["image"] for item in batch], dim=0),
        "labels": torch.stack([item["labels"] for item in batch], dim=0),
    }
