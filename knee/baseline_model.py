"""Study-level baseline: pretrained (or tiny) 2-D encoder + slice aggregation.

Smoke mode uses a randomly initialised tiny convnet and synthetic tensors so
software can be verified on CPU without ImageNet weights, DICOM files, or any
scientific claim. The ResNet-18 path is the modelling baseline and is not
exercised in CI.
"""

from __future__ import annotations

import torch
from torch import nn

from .config import NUM_TARGETS


class TinySliceEncoder(nn.Module):
    """CPU smoke-test encoder. Not a scientific baseline."""

    def __init__(self, in_channels: int = 1, width: int = 8):
        super().__init__()
        self.out_dim = width
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResNet18SliceEncoder(nn.Module):
    """ImageNet-pretrained ResNet-18 trunk, grayscale repeated to 3 channels."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.out_dim = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.backbone(x)


class BaselineStudyModel(nn.Module):
    """Encode each slice independently, aggregate, emit 12 logits."""

    def __init__(
        self,
        encoder: nn.Module,
        *,
        num_targets: int = NUM_TARGETS,
        aggregation: str = "mean",
    ):
        super().__init__()
        if aggregation not in {"mean", "max"}:
            raise ValueError("aggregation must be 'mean' or 'max'")
        self.encoder = encoder
        self.aggregation = aggregation
        self.head = nn.Linear(encoder.out_dim, num_targets)

    def forward(self, slices: torch.Tensor) -> torch.Tensor:
        """``slices`` has shape ``(B, S, C, H, W)``; returns ``(B, 12)`` logits."""
        if slices.ndim != 5:
            raise ValueError(f"expected (B, S, C, H, W), got {tuple(slices.shape)}")
        batch, n_slices, channels, height, width = slices.shape
        flat = slices.reshape(batch * n_slices, channels, height, width)
        features = self.encoder(flat)
        features = features.reshape(batch, n_slices, -1)
        if self.aggregation == "mean":
            pooled = features.mean(dim=1)
        else:
            pooled = features.max(dim=1).values
        return self.head(pooled)


def build_baseline_model(
    *,
    smoke: bool,
    encoder_name: str = "resnet18",
    pretrained: bool = True,
    aggregation: str = "mean",
    num_targets: int = NUM_TARGETS,
) -> BaselineStudyModel:
    if smoke or encoder_name == "tiny":
        encoder: nn.Module = TinySliceEncoder()
    elif encoder_name == "resnet18":
        encoder = ResNet18SliceEncoder(pretrained=pretrained)
    else:
        raise ValueError(f"unknown encoder {encoder_name!r}")
    return BaselineStudyModel(
        encoder, num_targets=num_targets, aggregation=aggregation
    )


def smoke_batch(
    *,
    batch_size: int = 2,
    num_slices: int = 4,
    image_size: int = 16,
    num_targets: int = NUM_TARGETS,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tiny generated tensors for software verification only."""
    rng = torch.Generator().manual_seed(seed)
    images = torch.rand(batch_size, num_slices, 1, image_size, image_size, generator=rng)
    labels = torch.randint(0, 2, (batch_size, num_targets), generator=rng).float()
    # Punch a few unknown labels so the masked loss is exercised.
    unknown = torch.zeros_like(labels)
    unknown[0, 0] = 1
    labels = labels.masked_fill(unknown.bool(), float("nan"))
    return images, labels
