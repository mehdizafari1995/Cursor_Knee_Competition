"""Phase 4D read-only readiness review: no training, inference, or submission."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from knee.config import OFFICIAL_TARGET_COLUMNS, STUDY_ID_COLUMN
from knee.kaggle_bundle import sha256_file, would_include
from knee.phase4d_readiness import (
    PHASE4C_CLOSED,
    PHASE4D_EXPERIMENT_EXECUTED,
    SECOND_READER_SCORE_BEARING_ALLOWED,
    SELECTED_EXPERIMENT_ID,
    reproduce_metrics_from_predictions,
    run_phase4d_readiness_review,
)
from knee.report_labels import (
    HARD_WEAK_LABEL_TRAINING_ALLOWED,
    SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED,
)
from tests.competition_fixtures import (
    REPORT_LABEL_GOLD_UIDS,
    REPORT_LABEL_LEAK_TOKEN,
    REPORT_LABEL_PATIENTS,
    REPORT_LABEL_WEAK_UIDS,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FROZEN_V14 = "992af7a4b4483278445a6f3fb363c818611e79005618dff063460d0aed3bd7f7"


def test_gates_remain_closed_and_second_reader_is_not_score_bearing():
    assert HARD_WEAK_LABEL_TRAINING_ALLOWED is False
    assert SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED is False
    assert SECOND_READER_SCORE_BEARING_ALLOWED is False
    assert PHASE4C_CLOSED is True
    assert PHASE4D_EXPERIMENT_EXECUTED is False


def test_review_is_read_only_and_selects_gold58_image_oof():
    result = run_phase4d_readiness_review()
    assert result["status"] == "READINESS_REVIEW_COMPLETE"
    assert result["training_run"] is False
    assert result["inference_run"] is False
    assert result["kaggle_kernel_launched"] is False
    assert result["competition_submission_written"] is False
    assert result["phase4d_experiment_executed"] is False
    assert result["n_report_only_studies_processed"] == 0
    assert result["selected_experiment"]["id"] == SELECTED_EXPERIMENT_ID
    assert result["selected_experiment"]["consumes_submission_slot"] is False
    assert result["selected_experiment"]["authorized_to_run_in_phase4d"] is False
    assert result["proposed_execution"]["executed"] is False
    assert "second-reader gold error" not in json.dumps(result["selected_experiment"]).lower()
    assert result["metric_reproduction_status"] == "NO_SAVED_COMPETITION_PREDICTIONS"
    assert "not the second reader" in result["score_bearing_system"].lower()


def test_reproduce_metrics_from_synthetic_saved_tables():
    ids = ["study-a", "study-b", "study-c", "study-d"]
    gold = pd.DataFrame({STUDY_ID_COLUMN: ids})
    pred = pd.DataFrame({STUDY_ID_COLUMN: ids})
    for i, column in enumerate(OFFICIAL_TARGET_COLUMNS):
        gold[column] = [0, 1, 0, 1]
        pred[column] = [0.1, 0.9, 0.2, 0.8]
    reproduced = reproduce_metrics_from_predictions(pred, gold)
    assert reproduced["model_inference_run"] is False
    assert reproduced["n_studies"] == 4
    assert reproduced["macro_roc_auc"] == 1.0
    again = reproduce_metrics_from_predictions(pred, gold)
    assert again == reproduced


def test_review_can_recompute_from_optional_saved_tables(tmp_path):
    ids = ["s1", "s2", "s3", "s4"]
    gold = pd.DataFrame({STUDY_ID_COLUMN: ids})
    pred = pd.DataFrame({STUDY_ID_COLUMN: list(reversed(ids))})
    for column in OFFICIAL_TARGET_COLUMNS:
        gold[column] = [0, 1, 0, 1]
        lookup = dict(zip(gold[STUDY_ID_COLUMN], gold[column]))
        pred[column] = [0.1 + 0.8 * float(lookup[sid]) for sid in pred[STUDY_ID_COLUMN]]
    gold_path = tmp_path / "gold.csv"
    pred_path = tmp_path / "pred.csv"
    gold.to_csv(gold_path, index=False)
    pred.to_csv(pred_path, index=False)
    result = run_phase4d_readiness_review(predictions_path=pred_path, gold_path=gold_path)
    assert result["metric_reproduction_status"] == "REPRODUCED_FROM_SAVED_TABLES"
    assert result["reproduced_metrics"]["macro_roc_auc"] == 1.0
    assert result["inference_run"] is False


def test_phase4d_module_never_trains_or_generates():
    src = (REPO_ROOT / "knee" / "phase4d_readiness.py").read_text(encoding="utf-8")
    assert "train_model(" not in src
    assert "run_smoke(" not in src
    assert "run_phase47_gold_constrained" not in src
    assert "run_second_reader_gold_pilot" not in src
    assert "generate_fn" not in src
    assert (REPO_ROOT / "scripts" / "train_gold58_image_oof.py").is_file()
    assert PHASE4D_EXPERIMENT_EXECUTED is False


def test_phase4d_reports_and_bundle_allowlist():
    for relative in (
        "reports/phase4c_second_reader_parked.md",
        "reports/phase4d_competition_readiness.md",
    ):
        text = (REPO_ROOT / relative).read_text(encoding="utf-8")
        assert REPORT_LABEL_LEAK_TOKEN not in text
        assert "FINDINGS:" not in text
        for uid in REPORT_LABEL_GOLD_UIDS + REPORT_LABEL_WEAK_UIDS:
            assert uid not in text
        for patient in REPORT_LABEL_PATIENTS:
            assert patient not in text
        assert "kaggle.json" not in text.lower()
    parked = (REPO_ROOT / "reports" / "phase4c_second_reader_parked.md").read_text(
        encoding="utf-8"
    )
    review = (REPO_ROOT / "reports" / "phase4d_competition_readiness.md").read_text(
        encoding="utf-8"
    )
    assert "SECOND_READER_PARKED" in parked
    assert SELECTED_EXPERIMENT_ID in review
    assert "phase4d_experiment_executed: `false`" in review
    assert would_include("knee/phase4d_readiness.py")
    assert would_include("scripts/run_phase4d_readiness.py")
    assert sha256_file(REPO_ROOT / "configs" / "second_reader_pilot_v1_4.yaml") == FROZEN_V14
