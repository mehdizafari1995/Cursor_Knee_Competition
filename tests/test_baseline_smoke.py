"""CPU smoke path: tiny generated tensors, clearly not a scientific evaluation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

from knee.baseline import run_smoke
from knee.baseline_config import load_baseline_config
from knee.baseline_model import BaselineStudyModel, TinySliceEncoder, smoke_batch
from knee.config import NUM_TARGETS
from knee.dataset import StudySliceDataset, collate_studies
from knee.labels import extract_label_matrix
from tests.competition_fixtures import make_mini_competition_root

ROOT = Path(__file__).resolve().parents[1]


def test_smoke_batch_has_unknown_label_and_twelve_targets():
    images, labels = smoke_batch(batch_size=2, num_slices=4, image_size=16)
    assert images.shape == (2, 4, 1, 16, 16)
    assert labels.shape == (2, NUM_TARGETS)
    assert torch.isnan(labels).any()


def test_tiny_model_forward_and_smoke_run():
    model = BaselineStudyModel(TinySliceEncoder(), num_targets=NUM_TARGETS)
    images, labels = smoke_batch()
    logits = model(images)
    assert logits.shape == (2, NUM_TARGETS)
    cfg = load_baseline_config(ROOT / "configs" / "baseline.yaml", smoke=True)
    metrics = run_smoke(cfg)
    assert metrics["mode"] == "software_smoke"
    assert metrics["scientific_evaluation"] is False
    assert metrics["num_targets"] == 12


def test_train_and_predict_scripts_smoke(tmp_path):
    train_out = tmp_path / "train"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "train_baseline.py"),
            "--smoke",
            "--output-dir",
            str(train_out),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads((train_out / "smoke_metrics.json").read_text())
    assert payload["scientific_evaluation"] is False

    mini = make_mini_competition_root(tmp_path / "mini")
    refuse = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "train_baseline.py"),
            "--data-root",
            str(mini),
            "--allow-training",
            "--output-dir",
            str(tmp_path / "train_refused"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert refuse.returncode == 3, refuse.stdout + refuse.stderr
    combined = refuse.stdout + refuse.stderr
    assert "NOT_READY_FOR_HARD_WEAK_LABEL_TRAINING" in combined
    assert "READY_FOR_SELECTIVE_LABEL_REFINEMENT" in combined
    pred_path = tmp_path / "preds.csv"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "predict_baseline.py"),
            "--smoke",
            "--sample-submission",
            str(mini / "sample_submission.csv"),
            "--output",
            str(pred_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    submit = tmp_path / "submission.csv"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "make_submission.py"),
            "--predictions",
            str(pred_path),
            "--sample-submission",
            str(mini / "sample_submission.csv"),
            "--output",
            str(submit),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    import pandas as pd

    from knee.competition import SAMPLE_SUBMISSION_COLUMNS

    written = pd.read_csv(submit)
    sample = pd.read_csv(mini / "sample_submission.csv")
    assert list(written.columns) == list(SAMPLE_SUBMISSION_COLUMNS)
    assert written["StudyInstanceUID"].tolist() == sample["StudyInstanceUID"].tolist()


def test_create_folds_and_audit_scripts_on_fixture(tmp_path):
    mini = make_mini_competition_root(tmp_path / "mini")
    folds = tmp_path / "folds.csv"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "create_folds.py"),
            "--data-root",
            str(mini),
            "--output",
            str(folds),
            "--n-folds",
            "2",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    import pandas as pd

    table = pd.read_csv(folds)
    assert set(table["fold"]) <= {0, 1}

    report = tmp_path / "data_audit.md"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "audit_competition_data.py"),
            "--data-root",
            str(mini),
            "--source",
            "software_fixture",
            "--output",
            str(report),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    text = report.read_text(encoding="utf-8")
    assert "AUDITED" in text
    assert "SOFTWARE FIXTURE" in text or "software_fixture" in text


def test_study_slice_dataset_loads_fixture_dicom(tmp_path):
    root = make_mini_competition_root(tmp_path / "mini")
    import pandas as pd

    train = pd.read_csv(root / "train.csv")
    series = pd.read_csv(root / "train_series.csv")
    merged = train.merge(series, on="StudyInstanceUID")
    labels, _ = extract_label_matrix(train)
    dirs = [
        root
        / "train_series"
        / row.StudyInstanceUID
        / row.SeriesInstanceUID
        for row in merged.itertuples()
    ]
    dataset = StudySliceDataset(
        merged["StudyInstanceUID"].astype(str).tolist(),
        dirs,
        labels,
        num_slices=3,
        image_size=8,
    )
    batch = collate_studies([dataset[0], dataset[1]])
    assert batch["image"].shape == (2, 3, 1, 8, 8)
    assert batch["labels"].shape == (2, NUM_TARGETS)


def test_audit_script_data_not_available(tmp_path):
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "audit_competition_data.py"),
            "--data-root",
            str(tmp_path / "empty"),
            "--output",
            str(tmp_path / "audit.md"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={**dict(**{k: v for k, v in __import__("os").environ.items()
                       if k not in {"RSNA_DATA_ROOT", "KNEE_DATA_ROOT"}})},
    )
    assert proc.returncode == 2
    assert "DATA_NOT_AVAILABLE" in (tmp_path / "audit.md").read_text()
