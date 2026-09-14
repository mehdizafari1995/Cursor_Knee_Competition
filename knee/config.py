"""Central configuration for the RSNA Knee Abnormality Detection pipeline.

The competition is a *study-level multi-label* problem: each knee MRI study is
scored for twelve independent binary findings, and the leaderboard metric is the
macro-averaged ROC AUC across those twelve targets. Values here are intentionally
small so the whole pipeline runs end-to-end in a few seconds on CPU, which makes
it a fast smoke test for the dev environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Environment variable that points at a local copy of the Kaggle dataset.
# Never hard-code a machine-specific path; resolve through this variable.
DATA_ROOT_ENV = "RSNA_DATA_ROOT"

# Official Kaggle identifier column (sample_submission.csv / train.csv).
STUDY_ID_COLUMN = "StudyInstanceUID"
SERIES_ID_COLUMN = "SeriesInstanceUID"
REPORT_COLUMN = "Report"

# Internal machine-friendly ids. These are the project's single source of truth
# for ontology records and must not be added to, dropped, or renamed. They are
# *not* the Kaggle submission header; see OFFICIAL_TARGET_COLUMNS.
TARGET_KEYS = (
    "acl_injury",
    "mcl_injury",
    "medial_meniscus_injury",
    "lateral_meniscus_injury",
    "medial_compartment_oa",
    "lateral_compartment_oa",
    "patellofemoral_oa",
    "joint_effusion",
    "synovitis",
    "baker_cyst",
    "bone_contusion",
    "fracture",
)

TARGET_NAMES = (
    "ACL injury",
    "MCL injury",
    "Medial meniscus injury",
    "Lateral meniscus injury",
    "Medial-compartment osteoarthritis",
    "Lateral-compartment osteoarthritis",
    "Patellofemoral osteoarthritis",
    "Joint effusion",
    "Synovitis",
    "Baker cyst",
    "Bone contusion",
    "Fracture",
)

# Verbatim Kaggle sample_submission.csv header names, in official order.
# Spelling, spacing, and the apostrophe in "Baker's" are significant.
# Source: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview
OFFICIAL_TARGET_COLUMNS = (
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
)

NUM_TARGETS = len(TARGET_KEYS)
assert NUM_TARGETS == 12, "RSNA Knee Abnormality Detection has exactly 12 targets"
assert len(OFFICIAL_TARGET_COLUMNS) == NUM_TARGETS
assert len(TARGET_NAMES) == NUM_TARGETS

TARGET_KEY_TO_OFFICIAL: dict[str, str] = dict(zip(TARGET_KEYS, OFFICIAL_TARGET_COLUMNS))
OFFICIAL_TO_TARGET_KEY: dict[str, str] = dict(zip(OFFICIAL_TARGET_COLUMNS, TARGET_KEYS))


@dataclass
class Config:
    seed: int = 42
    image_size: int = 64
    num_targets: int = NUM_TARGETS

    # Dataset sizes (per split), in number of studies.
    n_train: int = 800
    n_val: int = 200
    n_test: int = 200

    # Training hyperparameters.
    batch_size: int = 32
    epochs: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    # Weak supervision: train labels are derived from multilingual radiology
    # reports rather than expert annotations. Inference is image-only.
    use_weak_supervision: bool = True

    output_dir: Path = field(default_factory=lambda: OUTPUT_DIR)

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
