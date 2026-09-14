"""Percentile intensity normalisation and deterministic slice sampling."""

from __future__ import annotations

import numpy as np


def percentile_normalize(
    volume: np.ndarray,
    low: float = 1.0,
    high: float = 99.0,
) -> np.ndarray:
    """Scale a volume to [0, 1] using the given intensity percentiles.

    Constant volumes (including empty arrays) map to zeros. NaNs are ignored
    when estimating percentiles and left as 0 after normalisation.
    """
    data = np.asarray(volume, dtype=np.float32)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return np.zeros_like(data, dtype=np.float32)
    p_lo, p_hi = np.percentile(finite, [low, high])
    if not np.isfinite(p_lo) or not np.isfinite(p_hi) or p_hi <= p_lo:
        return np.zeros_like(data, dtype=np.float32)
    scaled = (np.clip(data, p_lo, p_hi) - p_lo) / (p_hi - p_lo)
    return np.nan_to_num(scaled, nan=0.0).astype(np.float32)


def sample_slice_indices(n_available: int, n_keep: int) -> np.ndarray:
    """Evenly spaced indices covering the series, inclusive of both ends.

    If the series is shorter than ``n_keep``, indices repeat via rounding of a
    linspace so the tensor width stays fixed. ``n_available`` must be >= 1.
    """
    if n_available < 1:
        raise ValueError("n_available must be >= 1")
    if n_keep < 1:
        raise ValueError("n_keep must be >= 1")
    idx = np.linspace(0, n_available - 1, n_keep)
    return np.round(idx).astype(np.int64)


def sample_slices(volume: np.ndarray, n_keep: int) -> np.ndarray:
    """Select ``n_keep`` slices from a (S, H, W) volume along axis 0."""
    if volume.ndim != 3:
        raise ValueError(f"expected a 3-D volume, got shape {volume.shape}")
    indices = sample_slice_indices(volume.shape[0], n_keep)
    return volume[indices]
