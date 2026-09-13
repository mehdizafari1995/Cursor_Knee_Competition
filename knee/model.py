"""A small convolutional network for multi-label knee MRI finding detection.

The head emits twelve independent logits (one per finding); training uses
BCE-with-logits and inference applies a sigmoid per target. Kept deliberately
tiny so training converges in a few CPU seconds while still exercising the full
torch/torchvision stack (conv, pooling, batchnorm, Adam).
"""

from __future__ import annotations

import torch
from torch import nn


class KneeMRINet(nn.Module):
    def __init__(self, num_targets: int, image_size: int = 64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(64, num_targets),  # one logit per binary finding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
