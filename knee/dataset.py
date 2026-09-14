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
    ):
        if len(study_ids) != len(series_dirs):
            raise ValueError("study_ids and series_dirs must be the same length")
        if labels is not None and len(labels) != len(study_ids):
            raise ValueError("labels must have one row per study")
        if labels is not None and labels.shape[1] != NUM_TARGETS:
            raise ValueError(f"labels must have {NUM_TARGETS} columns")
        self.study_ids = list(study_ids)
        self.series_dirs = [Path(p) for p in series_dirs]
        self.labels = None if labels is None else np.asarray(labels, dtype=np.float32)
        self.num_slices = num_slices
        self.image_size = image_size
        self.percentiles = percentiles

    def __len__(self) -> int:
        return len(self.study_ids)

    def __getitem__(self, idx: int):
        volume, _headers = load_series_volume(self.series_dirs[idx])
        volume = percentile_normalize(volume, *self.percentiles)
        sampled = sample_slices(volume, self.num_slices)
        resized = np.stack(
            [_resize2d(slice_, self.image_size) for slice_ in sampled], axis=0
        )
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


def _resize2d(slice_: np.ndarray, size: int) -> np.ndarray:
    """Nearest-neighbour resize so tests do not depend on torchvision I/O."""
    src = np.asarray(slice_, dtype=np.float32)
    if src.shape == (size, size):
        return src
    ys = np.linspace(0, src.shape[0] - 1, size)
    xs = np.linspace(0, src.shape[1] - 1, size)
    yi = np.round(ys).astype(np.int64)
    xi = np.round(xs).astype(np.int64)
    return src[yi][:, xi]


def collate_studies(batch: list[dict]) -> dict:
    return {
        STUDY_ID_COLUMN: [item[STUDY_ID_COLUMN] for item in batch],
        "image": torch.stack([item["image"] for item in batch], dim=0),
        "labels": torch.stack([item["labels"] for item in batch], dim=0),
    }
