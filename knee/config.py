"""Central configuration for the knee OA grading pipeline.

Values are intentionally small so the whole pipeline runs end-to-end in a few
seconds on CPU, which makes it a fast smoke test for the dev environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Kellgren-Lawrence grades (standard knee OA severity scale).
KL_GRADES = (0, 1, 2, 3, 4)
CLASS_NAMES = (
    "KL0-none",
    "KL1-doubtful",
    "KL2-minimal",
    "KL3-moderate",
    "KL4-severe",
)


@dataclass
class Config:
    seed: int = 42
    image_size: int = 64
    num_classes: int = len(KL_GRADES)

    # Dataset sizes (per split).
    n_train: int = 600
    n_val: int = 150
    n_test: int = 150

    # Training hyperparameters.
    batch_size: int = 32
    epochs: int = 6
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    output_dir: Path = field(default_factory=lambda: OUTPUT_DIR)

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
