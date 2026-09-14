"""Masked binary cross-entropy for incomplete multi-label targets."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def label_mask(targets: torch.Tensor) -> torch.Tensor:
    """True where a label is known (finite, not NaN)."""
    return torch.isfinite(targets)


def masked_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean BCE over known labels only.

    Unknown entries (NaN) are excluded from the gradient. If a batch has no
    known labels the loss is a zero that still depends on ``logits`` so that
    DDP / autograd graphs stay well-formed.
    """
    mask = label_mask(targets).to(dtype=logits.dtype)
    filled = torch.nan_to_num(targets, nan=0.0)
    per_elem = F.binary_cross_entropy_with_logits(logits, filled, reduction="none")
    denom = mask.sum().clamp_min(1.0)
    return (per_elem * mask).sum() / denom
