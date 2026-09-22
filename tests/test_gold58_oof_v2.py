"""Phase 4D.2 gold58 image OOF v2: series ranking, ImageNet preprocess, tiny OOF."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from knee.config import OFFICIAL_TARGET_COLUMNS, PROJECT_ROOT, STUDY_ID_COLUMN
from knee.dataset import _resize2d, apply_imagenet_normalization
from knee.gold58_oof import (
    DEFAULT_CONFIG_PATH_V2,
    IMAGENET_SHA256,
    KERNEL_SLUG,
    KERNEL_SLUG_V2,
    PROTOCOL,
    PROTOCOL_V2,
    SERIES_RULE,
    SERIES_RULE_V2,
    STATUS_COMPLETE,
    STATUS_GROUPING,
    STATUS_SERIES,
    assert_train_pos_neg_every_target,
    load_gold58_config,
    render_protocol_markdown_v2,
    run_gold58_image_oof,
    select_deterministic_primary_series_v2,
    inspect_series_inventory,
)
from knee.kaggle_bundle import sha256_file, would_include
from tests.competition_fixtures import (
    REPORT_LABEL_GOLD_UIDS,
    REPORT_LABEL_LEAK_TOKEN,
    REPORT_LABEL_PATIENTS,
    REPORT_LABEL_WEAK_UIDS,
)
from tests.test_gold58_oof import FAST, _labelled_root, FROZEN_V14

REPO_ROOT = Path(__file__).resolve().parents[1]
V2 = {
    **FAST,
    "series_selection_rule": SERIES_RULE_V2,
    "interpolation": "bilinear",
    "imagenet_normalize": True,
    "imagenet_mean": [0.485, 0.456, 0.406],
    "imagenet_std": [0.229, 0.224, 0.225],
    "fold_require_pos_neg_every_target": True,
    "protocol": PROTOCOL_V2,
    "kernel_slug": KERNEL_SLUG_V2,
    "output_dirname": "rsna_gold58_image_oof_v2",
}


def test_v1_artifacts_are_unchanged():
    v1 = REPO_ROOT / "configs" / "gold58_image_resnet18_oof_v1.yaml"
    assert sha256_file(v1) == "92f9d8b52d9488646a4dd19ca9740b8ae658c5f71bf6812a393b91fa3a07af9c"
    run = (REPO_ROOT / "reports" / "gold58_image_oof_run.md").read_text(encoding="utf-8")
    assert "SERIES_SELECTION_PREFLIGHT_FAIL" in run
    assert "mahdimzafari/rsna-knee-gold58-image-oof-v1" in run
    assert SERIES_RULE in run
    assert KERNEL_SLUG != KERNEL_SLUG_V2
    assert PROTOCOL != PROTOCOL_V2


def test_v2_config_is_versioned_not_a_copy_of_v1_slug():
    cfg = load_gold58_config(DEFAULT_CONFIG_PATH_V2)
    assert cfg["protocol"] == PROTOCOL_V2
    assert cfg["kernel_slug"] == KERNEL_SLUG_V2
    assert cfg["series_selection_rule"] == SERIES_RULE_V2
    assert cfg["interpolation"] == "bilinear"
    assert cfg["imagenet_normalize"] is True
    assert cfg["epochs"] == 12
    assert cfg["seed"] == 42
    assert cfg["n_folds"] == 5
    assert cfg["num_slices"] == 8
    assert cfg["learning_rate"] == 0.0001
    assert cfg["imagenet_weights_sha256"] == IMAGENET_SHA256
    assert cfg["process_report_only"] is False
    assert cfg["use_second_reader"] is False
    assert cfg["use_weak_labels"] is False


def test_v2_ranks_ambiguous_sagittal_instead_of_failing(tmp_path):
    root = _labelled_root(tmp_path / "multi", extra_sagittal=True)
    series = pd.read_csv(root / "train_series.csv")
    inventory = inspect_series_inventory(series, root)
    ids = pd.read_csv(root / "train.csv")[STUDY_ID_COLUMN].astype(str).tolist()
    lookup, aggregates = select_deterministic_primary_series_v2(inventory, ids)
    assert len(lookup) == 10
    assert aggregates["n_selected"] == 10
    assert aggregates["n_zero_readable"] == 0
    assert aggregates["n_tier_1"] == 10
    # extra sagittal FS series have 2 slices vs 3; ranking prefers more DICOMs
    assert aggregates["n_tie_break_dicom_count"] == 10
    result = run_gold58_image_oof(
        data_root=root,
        output_dir=tmp_path / "out-v2-multi",
        config_path=DEFAULT_CONFIG_PATH_V2,
        execute_training=False,
        require_cuda=False,
        config_overrides=V2,
    )
    assert result["status"] == "PREFLIGHT_PASS_TRAINING_NOT_EXECUTED"
    assert result["training_run"] is False
    assert result["kernel_slug"] == KERNEL_SLUG_V2


def test_v2_grouping_fail_on_disagreeing_patient_ids(tmp_path):
    root = _labelled_root(tmp_path / "disagree")
    # overwrite coronal series PatientID so it disagrees with sagittal
    from tests.competition_fixtures import write_series

    write_series(
        root / "train_series" / "study-000" / "series-cor-000",
        study_uid="study-000",
        series_uid="series-cor-000",
        n_slices=2,
        patient_id="OTHER-PAT",
        scramble_filenames=False,
    )
    result = run_gold58_image_oof(
        data_root=root,
        output_dir=tmp_path / "out-disagree",
        config_path=DEFAULT_CONFIG_PATH_V2,
        execute_training=False,
        require_cuda=False,
        config_overrides=V2,
    )
    assert result["status"] == STATUS_GROUPING
    assert result["training_run"] is False


def test_v2_fold_support_fail_when_a_class_is_held_out(tmp_path):
    y = np.ones((10, 12), dtype=np.float64)
    y[0, 0] = 0.0
    fold_ids = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    with pytest.raises(Exception, match="n_failing_target_folds"):
        assert_train_pos_neg_every_target(y, fold_ids, 5)


def test_v2_tiny_oof_complete(tmp_path):
    root = _labelled_root(tmp_path / "train10")
    result = run_gold58_image_oof(
        data_root=root,
        output_dir=tmp_path / "out-v2-train",
        config_path=DEFAULT_CONFIG_PATH_V2,
        execute_training=True,
        allow_tiny_encoder_for_tests=True,
        require_cuda=False,
        config_overrides=V2,
    )
    assert result["status"] == STATUS_COMPLETE
    assert result["training_run"] is True
    assert result["protocol"] == PROTOCOL_V2
    assert result["imagenet_normalize"] is True
    assert result["interpolation"] == "bilinear"
    assert result["n_labelled_studies"] == 10
    assert result["n_report_only_studies_processed"] == 0
    assert result["round_trip"]["within_tolerance"] is True
    agg = result["preflight_aggregates"]
    assert agg["series_selection_rule"] == SERIES_RULE_V2
    assert agg["series_selection_aggregates"]["n_selected"] == 10
    assert 0.0 <= result["macro_roc_auc"] <= 1.0


def test_imagenet_normalization_and_bilinear_resize():
    gray = np.full((2, 8, 8), 0.5, dtype=np.float32)
    rgb = apply_imagenet_normalization(gray)
    assert rgb.shape == (2, 3, 8, 8)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
    expected = (0.5 - mean) / std
    np.testing.assert_allclose(rgb, np.broadcast_to(expected, rgb.shape), rtol=1e-5, atol=1e-5)
    src = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    nearest = _resize2d(src, 4, interpolation="nearest")
    bilinear = _resize2d(src, 4, interpolation="bilinear")
    assert nearest.shape == (4, 4)
    assert bilinear.shape == (4, 4)
    assert not np.allclose(nearest, bilinear)


def test_v1_still_fails_on_extra_sagittal(tmp_path):
    root = _labelled_root(tmp_path / "amb", extra_sagittal=True)
    result = run_gold58_image_oof(
        data_root=root,
        output_dir=tmp_path / "out-v1-amb",
        execute_training=False,
        require_cuda=False,
        config_overrides=FAST,
    )
    assert result["status"] == STATUS_SERIES
    assert result["kernel_slug"] == KERNEL_SLUG


def test_v2_protocol_notebook_and_allowlist():
    text = render_protocol_markdown_v2()
    assert SERIES_RULE_V2 in text
    assert KERNEL_SLUG_V2 in text
    assert REPORT_LABEL_LEAK_TOKEN not in text
    report = (REPO_ROOT / "reports" / "gold58_image_oof_v2_protocol.md").read_text(
        encoding="utf-8"
    )
    assert SERIES_RULE_V2 in report
    assert IMAGENET_SHA256 in report
    run = (REPO_ROOT / "reports" / "gold58_image_oof_v2_run.md").read_text(
        encoding="utf-8"
    )
    assert "GOLD58_OOF_COMPLETE" in run
    assert KERNEL_SLUG_V2 in run
    assert "training_run`: true" in run or "training_run: true" in run
    assert "n_report_only_studies_processed`: 0" in run or (
        "n_report_only_studies_processed: 0" in run
    )
    assert "n_tier_1" in run and "56" in run
    assert "n_tier_2" in run and "2" in run
    assert "0.6237496021463534" in run
    assert "Tesla T4" in run
    assert "notebooks 07" in run.lower() or "notebooks 07, 09" in run
    assert REPORT_LABEL_LEAK_TOKEN not in run
    assert "FINDINGS:" not in run
    assert "DummySeriesDesc" not in run
    assert would_include("configs/gold58_image_resnet18_oof_v2.yaml")
    assert would_include("notebooks/11_kaggle_gold58_image_oof_v2.ipynb")
    assert would_include("notebooks/18_kaggle_gold58_3plane_oof_v1.ipynb")
    assert would_include("scripts/train_gold58_image_oof_v2.py")
    assert not would_include("reports/gold58_image_oof_v2_protocol.md")
    assert not would_include("reports/gold58_image_oof_v2_run.md")
    assert sha256_file(REPO_ROOT / "configs" / "second_reader_pilot_v1_4.yaml") == FROZEN_V14
    import json

    nb = json.loads(
        (REPO_ROOT / "notebooks" / "11_kaggle_gold58_image_oof_v2.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join("".join(cell.get("source") or []) for cell in nb["cells"])
    assert "run_gold58_image_oof" in source
    assert "gold58_image_resnet18_oof_v2.yaml" in source
    assert "execute_training=True" in source
    assert "notebook_07_executed=false" in source
    assert "notebook_09_executed=false" in source
    assert "notebook_10_executed=false" in source
    assert "rsna-knee-gold58-image-oof-v1" in source
    assert "pip install" not in source.lower()
    meta = (nb.get("metadata") or {}).get("kaggle") or {}
    assert meta.get("isInternetEnabled") is False
    assert meta.get("gpu") is True
    for uid in REPORT_LABEL_GOLD_UIDS + REPORT_LABEL_WEAK_UIDS:
        assert uid not in report
    for patient in REPORT_LABEL_PATIENTS:
        assert patient not in report
