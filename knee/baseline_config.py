"""YAML-backed configuration for the competition baseline (not the synthetic smoke CNN)."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from .config import DATA_ROOT_ENV, NUM_TARGETS, PROJECT_ROOT


@dataclass
class BaselineConfig:
    seed: int = 42
    data_root_env: str = DATA_ROOT_ENV
    n_folds: int = 5
    encoder: str = "resnet18"
    pretrained: bool = True
    image_size: int = 224
    num_slices: int = 8
    slice_sampling: str = "linspace"
    intensity_percentiles: tuple[float, float] = (1.0, 99.0)
    batch_size: int = 2
    epochs: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    aggregation: str = "mean"
    num_targets: int = NUM_TARGETS
    device: str = "cpu"
    smoke: bool = False
    allow_training: bool = False
    max_train_steps: int = 1
    output_dir: str = "outputs/baseline"

    def __post_init__(self) -> None:
        if isinstance(self.intensity_percentiles, list):
            self.intensity_percentiles = tuple(self.intensity_percentiles)
        if self.num_targets != NUM_TARGETS:
            raise ValueError(
                f"num_targets must remain {NUM_TARGETS}; refusing to change the target set"
            )
        if self.aggregation not in {"mean", "max"}:
            raise ValueError("aggregation must be 'mean' or 'max'")
        if self.slice_sampling != "linspace":
            raise ValueError("only 'linspace' slice sampling is implemented")


def load_baseline_config(path: str | Path | None = None, **overrides: Any) -> BaselineConfig:
    """Load ``configs/baseline.yaml`` and apply explicit keyword overrides."""
    cfg_path = Path(path) if path else PROJECT_ROOT / "configs" / "baseline.yaml"
    payload: dict[str, Any] = {}
    if cfg_path.is_file():
        loaded = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"{cfg_path} must contain a mapping")
        payload.update(loaded)
    known = {item.name for item in fields(BaselineConfig)}
    unknown = set(payload) - known
    if unknown:
        raise ValueError(f"unknown baseline config keys: {sorted(unknown)}")
    payload.update({key: value for key, value in overrides.items() if value is not None})
    return BaselineConfig(**payload)
