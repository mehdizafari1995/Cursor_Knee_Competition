"""Phase 4E visible-test submission rehearsal tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from knee.baseline import save_checkpoint
from knee.baseline_model import BaselineStudyModel, TinySliceEncoder
from knee.competition import SAMPLE_SUBMISSION_COLUMNS
from knee.config import NUM_TARGETS, OFFICIAL_TARGET_COLUMNS, PROJECT_ROOT, STUDY_ID_COLUMN
from knee.gold58_oof import SERIES_RULE, SERIES_RULE_V2
from knee.kaggle_bundle import sha256_file, would_include
from knee.phase4e_rehearsal import (
    CHECKPOINT_DATASET_SLUG,
    FROZEN_FOLD_SHA256,
    FROZEN_V1_RECIPE_SHA256,
    FROZEN_V2_CONFIG_SHA256,
    KERNEL_SLUG,
    PROTOCOL,
    STATUS_CHECKPOINT,
    STATUS_CHECKPOINT_HASH,
    STATUS_COMPLETE,
    STATUS_CONTRACT,
    STATUS_INTERNET,
    STATUS_RUNTIME,
    STATUS_SERIES,
    equal_weight_mean,
    project_runtime,
    render_protocol_markdown,
    run_phase4e_rehearsal,
    verify_fold_checkpoints,
)
from knee.submission import load_sample_submission, write_competition_submission
from tests.competition_fixtures import (
    REPORT_LABEL_GOLD_UIDS,
    REPORT_LABEL_LEAK_TOKEN,
    REPORT_LABEL_PATIENTS,
    REPORT_LABEL_WEAK_UIDS,
    write_series,
)
from tests.test_gold58_oof import FROZEN_V14

REPO_ROOT = Path(__file__).resolve().parents[1]
FAST = {
    "epochs": 1,
    "image_size": 16,
    "num_slices": 2,
    "batch_size": 2,
    "require_cuda_for_scientific_run": False,
}


def _test_root(root: Path, *, n: int = 3, extra_series: bool = False, no_readable: bool = False) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    sample_rows = []
    series_rows = []
    for i in range(n):
        study = f"test-{i:03d}"
        row = {STUDY_ID_COLUMN: study}
        for col in OFFICIAL_TARGET_COLUMNS:
            row[col] = 0.5
        sample_rows.append(row)
        sag = f"test-ser-{i:03d}"
        series_rows.append(
            {
                STUDY_ID_COLUMN: study,
                "SeriesInstanceUID": sag,
                "Fluid_Sensitive": 1,
                "Fat_Suppression": 1,
                "Anatomical_Plane": "Sagittal",
            }
        )
        if not no_readable:
            write_series(
                root / "test_series" / study / sag,
                study_uid=study,
                series_uid=sag,
                n_slices=3,
                patient_id=f"TPAT-{i:03d}",
                scramble_filenames=False,
            )
        if extra_series:
            sag2 = f"test-ser2-{i:03d}"
            series_rows.append(
                {
                    STUDY_ID_COLUMN: study,
                    "SeriesInstanceUID": sag2,
                    "Fluid_Sensitive": 1,
                    "Fat_Suppression": 1,
                    "Anatomical_Plane": "Sagittal",
                }
            )
            write_series(
                root / "test_series" / study / sag2,
                study_uid=study,
                series_uid=sag2,
                n_slices=2,
                patient_id=f"TPAT-{i:03d}",
                scramble_filenames=False,
            )
    sample = pd.DataFrame(sample_rows).loc[:, list(SAMPLE_SUBMISSION_COLUMNS)]
    sample.to_csv(root / "sample_submission.csv", index=False)
    pd.DataFrame({STUDY_ID_COLUMN: sample[STUDY_ID_COLUMN]}).to_csv(root / "test.csv", index=False)
    pd.DataFrame(series_rows).to_csv(root / "test_series.csv", index=False)
    pd.DataFrame({STUDY_ID_COLUMN: []}).to_csv(root / "train.csv", index=False)
    pd.DataFrame(
        columns=[STUDY_ID_COLUMN, "SeriesInstanceUID", "Fluid_Sensitive", "Fat_Suppression", "Anatomical_Plane"]
    ).to_csv(root / "train_series.csv", index=False)
    return root


def _tiny_checkpoints(dest: Path, *, n: int = 5, seed: int = 0) -> dict[str, str]:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for fold in range(n):
        torch.manual_seed(seed + fold)
        model = BaselineStudyModel(
            TinySliceEncoder(in_channels=3),
            num_targets=NUM_TARGETS,
            aggregation="mean",
        )
        with torch.no_grad():
            model.head.bias.copy_(torch.full((NUM_TARGETS,), 0.1 * fold))
        path = dest / f"fold{fold}_last.pt"
        save_checkpoint(model, path)
        hashes[f"fold{fold}"] = sha256_file(path)
    return hashes


def test_v1_and_v2_artifacts_remain_frozen():
    v1 = REPO_ROOT / "configs" / "gold58_image_resnet18_oof_v1.yaml"
    v2 = REPO_ROOT / "configs" / "gold58_image_resnet18_oof_v2.yaml"
    assert sha256_file(v1) == FROZEN_V1_RECIPE_SHA256
    assert sha256_file(v2) == FROZEN_V2_CONFIG_SHA256
    run = (REPO_ROOT / "reports" / "gold58_image_oof_v2_run.md").read_text(encoding="utf-8")
    assert "GOLD58_OOF_COMPLETE" in run
    assert "mahdimzafari/rsna-knee-gold58-image-oof-v2" in run
    v1_run = (REPO_ROOT / "reports" / "gold58_image_oof_run.md").read_text(encoding="utf-8")
    assert "SERIES_SELECTION_PREFLIGHT_FAIL" in v1_run
    assert SERIES_RULE in v1_run
    assert SERIES_RULE_V2 != SERIES_RULE
    assert sha256_file(REPO_ROOT / "configs" / "second_reader_pilot_v1_4.yaml") == FROZEN_V14
    for fold, digest in FROZEN_FOLD_SHA256.items():
        assert digest in run


def test_equal_weight_mean_is_unweighted_arithmetic_mean():
    arrays = [np.full((2, NUM_TARGETS), float(i), dtype=np.float64) for i in range(5)]
    mean = equal_weight_mean(arrays)
    np.testing.assert_allclose(mean, np.full((2, NUM_TARGETS), 2.0))
    with pytest.raises(Exception):
        equal_weight_mean(arrays[:4])


def test_runtime_guard_fails_before_limit_breach():
    info = project_runtime(seconds_per_study=1.0, n_studies=3)
    assert info["this_run_fits_limit"] is True
    assert info["n_studies_9h_capacity"] >= 3
    with pytest.raises(Exception) as exc:
        project_runtime(seconds_per_study=20000.0, n_studies=3, limit_seconds=100.0)
    assert getattr(exc.value, "status", "") == STATUS_RUNTIME


def test_checkpoint_hash_mismatch_and_missing(tmp_path):
    hashes = _tiny_checkpoints(tmp_path / "ckpts")
    got = verify_fold_checkpoints(tmp_path / "ckpts", expected=hashes)
    assert got == hashes
    bad = dict(hashes)
    bad["fold0"] = "0" * 64
    with pytest.raises(Exception) as exc:
        verify_fold_checkpoints(tmp_path / "ckpts", expected=bad)
    assert getattr(exc.value, "status", "") == STATUS_CHECKPOINT_HASH
    (tmp_path / "ckpts" / "fold4_last.pt").unlink()
    with pytest.raises(Exception) as exc:
        verify_fold_checkpoints(tmp_path / "ckpts", expected=hashes)
    assert getattr(exc.value, "status", "") == STATUS_CHECKPOINT


def test_ensemble_averages_five_tiny_checkpoints(tmp_path):
    root = _test_root(tmp_path / "data")
    hashes = _tiny_checkpoints(tmp_path / "ckpts")
    result = run_phase4e_rehearsal(
        data_root=root,
        output_dir=tmp_path / "out",
        checkpoint_dir=tmp_path / "ckpts",
        execute_inference=True,
        allow_tiny_encoder_for_tests=True,
        verify_frozen_hashes=False,
        expected_fold_sha256=hashes,
        require_cuda=False,
        config_overrides=FAST,
    )
    assert result["status"] == STATUS_COMPLETE
    assert result["n_models_loaded"] == 5
    assert result["competition_submission_written"] is False
    assert result["consumes_submission_slot"] is False
    assert result["n_report_only_studies_processed"] == 0
    assert result["hidden_test_available"] is False
    private = tmp_path / "out" / "private" / "rehearsal_submission.csv"
    assert private.is_file()
    frame = pd.read_csv(private)
    sample = load_sample_submission(root / "sample_submission.csv")
    assert list(frame.columns) == list(sample.columns)
    assert frame[STUDY_ID_COLUMN].astype(str).tolist() == sample[STUDY_ID_COLUMN].astype(str).tolist()
    values = frame.loc[:, list(OFFICIAL_TARGET_COLUMNS)].to_numpy()
    assert np.isfinite(values).all()
    assert (values >= 0).all() and (values <= 1).all()
    assert result["deterministic_repeat"]["within_tolerance"] is True
    assert result["series_selection_aggregates"]["n_selected"] == 3
    assert not (tmp_path / "out" / "submission.csv").exists()


def test_exact_column_order_enforced(tmp_path):
    sample = pd.DataFrame({STUDY_ID_COLUMN: ["a", "b"]})
    for col in OFFICIAL_TARGET_COLUMNS:
        sample[col] = 0.5
    sample = sample.loc[:, list(SAMPLE_SUBMISSION_COLUMNS)]
    preds = sample.copy()
    preds["ACL"] = 0.1
    out = write_competition_submission(preds, sample, tmp_path / "sub.csv")
    assert list(out.columns) == list(SAMPLE_SUBMISSION_COLUMNS)
    shuffled = preds[[STUDY_ID_COLUMN, *reversed(OFFICIAL_TARGET_COLUMNS)]]
    out2 = write_competition_submission(shuffled, sample, tmp_path / "sub2.csv")
    assert list(out2.columns) == list(SAMPLE_SUBMISSION_COLUMNS)


def test_missing_and_duplicate_studies(tmp_path):
    root = _test_root(tmp_path / "data")
    hashes = _tiny_checkpoints(tmp_path / "ckpts")
    sample = pd.read_csv(root / "sample_submission.csv")
    sample = pd.concat([sample, sample.iloc[[0]]], ignore_index=True)
    sample.to_csv(root / "sample_submission.csv", index=False)
    result = run_phase4e_rehearsal(
        data_root=root,
        output_dir=tmp_path / "out-dup",
        checkpoint_dir=tmp_path / "ckpts",
        execute_inference=False,
        allow_tiny_encoder_for_tests=True,
        verify_frozen_hashes=False,
        expected_fold_sha256=hashes,
        require_cuda=False,
        config_overrides=FAST,
    )
    assert result["status"] == STATUS_CONTRACT

    root2 = _test_root(tmp_path / "data2")
    test = pd.read_csv(root2 / "test.csv")
    test = test.iloc[:-1]
    test.to_csv(root2 / "test.csv", index=False)
    result2 = run_phase4e_rehearsal(
        data_root=root2,
        output_dir=tmp_path / "out-miss",
        checkpoint_dir=tmp_path / "ckpts",
        execute_inference=False,
        allow_tiny_encoder_for_tests=True,
        verify_frozen_hashes=False,
        expected_fold_sha256=hashes,
        require_cuda=False,
        config_overrides=FAST,
    )
    assert result2["status"] == STATUS_CONTRACT


def test_series_selection_failure(tmp_path):
    root = _test_root(tmp_path / "empty", no_readable=True)
    hashes = _tiny_checkpoints(tmp_path / "ckpts")
    result = run_phase4e_rehearsal(
        data_root=root,
        output_dir=tmp_path / "out-series",
        checkpoint_dir=tmp_path / "ckpts",
        execute_inference=True,
        allow_tiny_encoder_for_tests=True,
        verify_frozen_hashes=False,
        expected_fold_sha256=hashes,
        require_cuda=False,
        config_overrides=FAST,
    )
    assert result["status"] == STATUS_SERIES
    assert not (tmp_path / "out-series" / "private" / "rehearsal_submission.csv").exists()


def test_internet_guard_and_protocol_allowlist():
    result = run_phase4e_rehearsal(
        execute_inference=False,
        internet_enabled=True,
        verify_frozen_hashes=False,
        require_cuda=False,
    )
    assert result["status"] == STATUS_INTERNET
    text = render_protocol_markdown()
    assert KERNEL_SLUG in text
    assert CHECKPOINT_DATASET_SLUG in text
    assert "write_competition_submission" in text
    report = (REPO_ROOT / "reports" / "phase4e_submission_rehearsal_protocol.md").read_text(
        encoding="utf-8"
    )
    assert KERNEL_SLUG in report
    assert REPORT_LABEL_LEAK_TOKEN not in report
    assert would_include("knee/phase4e_rehearsal.py")
    assert would_include("scripts/run_phase4e_rehearsal.py")
    assert would_include("notebooks/12_kaggle_gold58_submission_rehearsal.ipynb")
    assert not would_include("reports/phase4e_submission_rehearsal_protocol.md")
    for uid in REPORT_LABEL_GOLD_UIDS + REPORT_LABEL_WEAK_UIDS:
        assert uid not in report
    for patient in REPORT_LABEL_PATIENTS:
        assert patient not in report


def test_notebook_12_is_offline_gpu_and_does_not_submit():
    import json

    nb = json.loads(
        (REPO_ROOT / "notebooks" / "12_kaggle_gold58_submission_rehearsal.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join("".join(cell.get("source") or []) for cell in nb["cells"])
    lowered = source.lower()
    assert "pip install" not in lowered
    assert "competitions submit" not in lowered
    assert "run_phase4e_rehearsal" in source
    assert "execute_inference=True" in source
    assert "notebook_07_executed=false" in source
    assert "notebook_11_executed=false" in source
    assert "DO_NOT_SUBMIT" in source
    assert KERNEL_SLUG in source
    meta = (nb.get("metadata") or {}).get("kaggle") or {}
    assert meta.get("isInternetEnabled") is False
    assert meta.get("gpu") is True
