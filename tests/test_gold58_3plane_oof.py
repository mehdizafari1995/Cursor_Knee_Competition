"""Prepared gold-58 three-plane OOF protocol: selection, fusion, fail-closed execution."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from knee.baseline_model import BaselineStudyModel, TinySliceEncoder
from knee.config import NUM_TARGETS, OFFICIAL_TARGET_COLUMNS, STUDY_ID_COLUMN
from knee.gold58_3plane_oof import (
    FROZEN_V2_MACRO_AUC,
    GOLD58_3PLANE_OOF_EXECUTION_ALLOWED,
    KERNEL_SLUG,
    PLANES,
    PROTOCOL,
    SERIES_RULE_3PLANE,
    SSL_ENGINEERING_PATH_PARKED,
    STATUS_COMPLETE,
    STATUS_PREPARED,
    STATUS_RUNTIME,
    ThreePlaneFusionModel,
    classifier_has_batchnorm1d,
    collate_three_plane,
    estimate_3plane_resources,
    fold_assignment_hash,
    load_3plane_config,
    mean_fuse_encoder_embeddings,
    render_3plane_protocol_markdown,
    run_gold58_3plane_oof,
    select_one_series_per_plane,
    train_loader_drop_last,
    validate_oof_table,
)
from knee.gold58_oof import SERIES_RULE_V2, Gold58PreflightError, inspect_series_inventory
from knee.kaggle_bundle import sha256_file, would_include
from knee.phase4f_ssl import FROZEN_V2_CONFIG_SHA256, UNLABELED_IMAGE_SSL_EXECUTION_ALLOWED
from tests.competition_fixtures import (
    REPORT_LABEL_GOLD_UIDS,
    REPORT_LABEL_LEAK_TOKEN,
    REPORT_LABEL_PATIENTS,
    REPORT_LABEL_WEAK_UIDS,
    write_series,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _three_plane_root(root: Path, *, n: int = 4, drop_axial_last: bool = False) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    train_rows = []
    series_rows = []
    for i in range(n):
        study = f"study-{i:03d}"
        row = {STUDY_ID_COLUMN: study}
        for t, col in enumerate(OFFICIAL_TARGET_COLUMNS):
            row[col] = (i + t) % 2
        train_rows.append(row)
        for plane, suffix in (("Sagittal", "sag"), ("Coronal", "cor"), ("Axial", "ax")):
            if drop_axial_last and i == n - 1 and plane == "Axial":
                continue
            sid = f"series-{suffix}-{i:03d}"
            series_rows.append(
                {
                    STUDY_ID_COLUMN: study,
                    "SeriesInstanceUID": sid,
                    "Fluid_Sensitive": 1 if plane == "Sagittal" else 0,
                    "Fat_Suppression": 1 if plane == "Sagittal" else 0,
                    "Anatomical_Plane": plane,
                }
            )
            write_series(
                root / "train_series" / study / sid,
                n_slices=2,
                study_uid=study,
                series_uid=sid,
                patient_id=f"PAT-{i:03d}",
                scramble_filenames=False,
            )
    pd.DataFrame(train_rows).to_csv(root / "train.csv", index=False)
    pd.DataFrame(series_rows).to_csv(root / "train_series.csv", index=False)
    return root


def test_pins_and_execution_gate():
    cfg = load_3plane_config()
    assert cfg["protocol"] == PROTOCOL == "gold58-image-resnet18-3plane-oof-v1"
    assert cfg["kernel_slug"] == KERNEL_SLUG == "mahdimzafari/rsna-knee-gold58-3plane-oof-v1"
    assert cfg["series_selection_rule"] == SERIES_RULE_3PLANE
    assert cfg["planes"] == list(PLANES)
    assert cfg["fusion"] == "mean_encoder_embedding"
    assert cfg["expected_n_labelled_real"] == 58
    assert cfg["use_second_reader"] is False
    assert cfg["use_weak_labels"] is False
    assert cfg["process_report_only"] is False
    assert cfg["consumes_submission_slot"] is False
    assert float(cfg["comparator_macro_roc_auc"]) == FROZEN_V2_MACRO_AUC
    assert (
        sha256_file(REPO_ROOT / "configs" / "gold58_image_resnet18_3plane_oof_v1.yaml")
        == "57c8c6f3b8d6f951b9a00ee4e90028fbd1d61abcd68bc3e6c4009b1af5123ec8"
    )
    assert GOLD58_3PLANE_OOF_EXECUTION_ALLOWED is True
    assert UNLABELED_IMAGE_SSL_EXECUTION_ALLOWED is False
    assert SSL_ENGINEERING_PATH_PARKED is True
    assert (
        sha256_file(REPO_ROOT / "configs" / "gold58_image_resnet18_oof_v2.yaml")
        == FROZEN_V2_CONFIG_SHA256
    )


def test_one_series_per_plane_is_deterministic(tmp_path):
    root = _three_plane_root(tmp_path / "data", n=3)
    series = pd.read_csv(root / "train_series.csv")
    inventory = inspect_series_inventory(series, root, split="train")
    studies = [f"study-{i:03d}" for i in range(3)]
    lookup, aggregates = select_one_series_per_plane(inventory, studies)
    assert aggregates["n_selected"] == 3
    assert aggregates["n_below_min_planes"] == 0
    assert aggregates["selected_plane_counts"]["sagittal"] == 3
    assert aggregates["selected_plane_counts"]["coronal"] == 3
    assert aggregates["selected_plane_counts"]["axial"] == 3
    again, _ = select_one_series_per_plane(inventory, studies)
    assert lookup == again
    for study, planes in lookup.items():
        assert set(planes) == set(PLANES)
        assert planes["sagittal"].startswith("series-sag-")


def test_missing_plane_averages_available(tmp_path):
    root = _three_plane_root(tmp_path / "data", n=2, drop_axial_last=True)
    series = pd.read_csv(root / "train_series.csv")
    inventory = inspect_series_inventory(series, root, split="train")
    studies = ["study-000", "study-001"]
    lookup, aggregates = select_one_series_per_plane(inventory, studies, min_planes_required=1)
    assert "axial" not in lookup["study-001"]
    assert set(lookup["study-001"]) == {"sagittal", "coronal"}
    assert aggregates["n_selected"] == 2


def test_mean_fusion_and_resource_estimate():
    a = torch.ones(2, 4)
    b = torch.ones(2, 4) * 3
    fused = mean_fuse_encoder_embeddings([a, b])
    assert torch.allclose(fused, torch.ones(2, 4) * 2)
    estimate = estimate_3plane_resources()
    assert estimate["plane_factor"] == 3.0
    assert estimate["fits_private_wall"] is True
    assert estimate["execution_authorized"] is True
    assert estimate["n_labelled_studies"] == 58
    assert estimate["projected_seconds"] == pytest.approx(8168.106 * 3.0)


def test_execution_without_data_stays_prepared_or_runtime_fail():
    prepared = run_gold58_3plane_oof(execute_training=False)
    assert prepared["status"] == STATUS_PREPARED
    assert prepared["kaggle_kernel_launched"] is False
    assert prepared["n_report_text_rows_processed"] == 0
    missing = run_gold58_3plane_oof(execute_training=True, require_cuda=False)
    assert missing["training_run"] is False
    assert missing["OOF_run"] is False
    assert missing["status"] == STATUS_RUNTIME


def test_tiny_three_plane_oof_complete(tmp_path):
    root = _three_plane_root(tmp_path / "data", n=10)
    result = run_gold58_3plane_oof(
        data_root=root,
        output_dir=tmp_path / "out-3plane",
        execute_training=True,
        allow_tiny_encoder_for_tests=True,
        require_cuda=False,
        config_overrides={
            "epochs": 1,
            "image_size": 16,
            "num_slices": 2,
            "batch_size": 2,
            "require_cuda_for_scientific_run": False,
        },
    )
    assert result["status"] == STATUS_COMPLETE
    assert result["training_run"] is True
    assert result["OOF_run"] is True
    assert result["n_labelled_studies"] == 10
    assert result["n_report_only_studies_processed"] == 0
    assert result["competition_submission_written"] is False
    assert result["preflight_aggregates"]["fold_assignment_matches_v2"] is True
    assert result["round_trip"]["within_tolerance"] is True
    assert 0.0 <= float(result["macro_roc_auc"]) <= 1.0
    assert result["preflight_aggregates"]["series_selection_aggregates"]["n_selected"] == 10


def test_missing_plane_is_not_a_zero_embedding(tmp_path):
    root = _three_plane_root(tmp_path / "data", n=10, drop_axial_last=True)
    result = run_gold58_3plane_oof(
        data_root=root,
        output_dir=tmp_path / "out-missing",
        execute_training=True,
        allow_tiny_encoder_for_tests=True,
        require_cuda=False,
        config_overrides={
            "epochs": 1,
            "image_size": 16,
            "num_slices": 2,
            "batch_size": 2,
            "require_cuda_for_scientific_run": False,
        },
    )
    assert result["status"] == STATUS_COMPLETE
    hist = result["preflight_aggregates"]["series_selection_aggregates"]["n_planes_histogram"]
    assert hist.get("2") == 1 or "2" in hist


def test_protocol_privacy_and_allowlist():
    cfg = load_3plane_config()
    text = render_3plane_protocol_markdown(cfg)
    protocol = (REPO_ROOT / "reports" / "gold58_3plane_oof_protocol.md").read_text(encoding="utf-8")
    run = (REPO_ROOT / "reports" / "gold58_3plane_oof_run.md").read_text(encoding="utf-8")
    parked = (REPO_ROOT / "reports" / "phase4f_ssl_engineering_parked.md").read_text(encoding="utf-8")
    for blob in (text, protocol, run, parked):
        assert KERNEL_SLUG in blob or "SSL_ENGINEERING_PATH_PARKED" in blob
        assert "UNLABELED_IMAGE_SSL_EXECUTION_ALLOWED=false" in blob
        assert REPORT_LABEL_LEAK_TOKEN not in blob
        assert "FINDINGS:" not in blob
        for uid in REPORT_LABEL_GOLD_UIDS + REPORT_LABEL_WEAK_UIDS:
            assert uid not in blob
        for patient in REPORT_LABEL_PATIENTS:
            assert patient not in blob
    assert SERIES_RULE_V2 in protocol or "deterministic-primary-series-v2" in protocol
    assert "kaggle_kernel_launched: true" in run
    assert "KernelWorkerStatus.ERROR" in run
    assert "GOLD58_3PLANE_TRAINING_FAIL" in run
    assert "SSL_ENGINEERING_PATH_PARKED" in parked
    assert would_include("knee/gold58_3plane_oof.py")
    assert would_include("scripts/run_gold58_3plane_oof.py")
    assert would_include("configs/gold58_image_resnet18_3plane_oof_v1.yaml")
    assert would_include("notebooks/18_kaggle_gold58_3plane_oof_v1.ipynb")
    assert not would_include("reports/gold58_3plane_oof_protocol.md")
    assert not would_include("reports/gold58_3plane_oof_run.md")


def _tiny_model() -> ThreePlaneFusionModel:
    return ThreePlaneFusionModel(BaselineStudyModel(TinySliceEncoder(in_channels=1), num_targets=NUM_TARGETS))


def _plane_tensor(value: float, batch: int = 2) -> torch.Tensor:
    return torch.full((batch, 2, 1, 8, 8), float(value))


def test_classifier_path_has_no_batchnorm1d_and_odd_batch_policy():
    model = _tiny_model()
    assert classifier_has_batchnorm1d(model) is False
    assert train_loader_drop_last(n_items=5, batch_size=2, has_bn1d=False) is False
    assert train_loader_drop_last(n_items=5, batch_size=2, has_bn1d=True) is True


def test_one_two_three_planes_and_mixed_batch_and_all_missing():
    model = _tiny_model()
    model.eval()
    one = {
        "sagittal": _plane_tensor(1.0),
        "coronal": _plane_tensor(0.0),
        "axial": _plane_tensor(0.0),
    }
    mask_one = {
        "sagittal": torch.ones(2, dtype=torch.bool),
        "coronal": torch.zeros(2, dtype=torch.bool),
        "axial": torch.zeros(2, dtype=torch.bool),
    }
    two = dict(one)
    two["coronal"] = _plane_tensor(3.0)
    mask_two = dict(mask_one)
    mask_two["coronal"] = torch.ones(2, dtype=torch.bool)
    three = dict(two)
    three["axial"] = _plane_tensor(5.0)
    mask_three = {plane: torch.ones(2, dtype=torch.bool) for plane in PLANES}
    with torch.no_grad():
        e1 = model.fuse_embeddings(one, mask_one)
        e2 = model.fuse_embeddings(two, mask_two)
        e3 = model.fuse_embeddings(three, mask_three)
        solo = model.encode_volume(one["sagittal"])
    assert torch.allclose(e1, solo)
    assert e2.shape == (2, model.backbone.encoder.out_dim)
    assert e3.shape == e2.shape
    mixed_images = {
        "sagittal": _plane_tensor(1.0),
        "coronal": _plane_tensor(2.0),
        "axial": _plane_tensor(3.0),
    }
    mixed_mask = {
        "sagittal": torch.tensor([True, True]),
        "coronal": torch.tensor([True, False]),
        "axial": torch.tensor([True, False]),
    }
    fused = model.fuse_embeddings(mixed_images, mixed_mask)
    assert fused.shape[0] == 2
    empty_mask = {plane: torch.zeros(2, dtype=torch.bool) for plane in PLANES}
    with pytest.raises(Gold58PreflightError) as exc:
        model.fuse_embeddings(mixed_images, empty_mask)
    assert "zero_planes" in exc.value.reason
    logits = model(one, mask_one)
    assert logits.shape == (2, 12)
    assert torch.isfinite(logits).all()
    probs = torch.sigmoid(logits)
    assert ((probs >= 0) & (probs <= 1)).all()


def test_eval_mode_supports_batch_size_one():
    model = _tiny_model()
    model.eval()
    images = {plane: _plane_tensor(1.0, batch=1) for plane in PLANES}
    mask = {plane: torch.ones(1, dtype=torch.bool) for plane in PLANES}
    with torch.no_grad():
        logits = model(images, mask)
    assert logits.shape == (1, 12)
    assert torch.isfinite(logits).all()


def test_collate_mixed_availability_does_not_fuse_dummy_zeros():
    sag = _plane_tensor(1.0, batch=1)[0]
    cor = _plane_tensor(2.0, batch=1)[0]
    batch = [
        {
            STUDY_ID_COLUMN: "s0",
            "planes": {"sagittal": sag, "coronal": cor, "axial": None},
            "labels": torch.zeros(12),
        },
        {
            STUDY_ID_COLUMN: "s1",
            "planes": {"sagittal": sag, "coronal": None, "axial": None},
            "labels": torch.zeros(12),
        },
    ]
    collated = collate_three_plane(batch)
    assert collated["plane_mask"]["sagittal"].tolist() == [True, True]
    assert collated["plane_mask"]["coronal"].tolist() == [True, False]
    assert collated["plane_mask"]["axial"].tolist() == [False, False]
    model = _tiny_model()
    fused = model.fuse_embeddings(collated["plane_images"], collated["plane_mask"])
    only_sag = model.encode_volume(collated["plane_images"]["sagittal"][1:2])
    assert torch.allclose(fused[1], only_sag[0])


def test_synthetic_58_row_oof_table_validators():
    ids = [f"synth-study-{i:03d}" for i in range(58)]
    frame = pd.DataFrame({STUDY_ID_COLUMN: ids})
    rng = np.random.default_rng(0)
    for column in OFFICIAL_TARGET_COLUMNS:
        frame[column] = rng.random(58)
    ok = validate_oof_table(frame, expected_n=58, expected_ids=ids)
    assert ok["ok"] is True
    assert ok["shape"] == [58, 12]
    dup = frame.copy()
    dup.loc[1, STUDY_ID_COLUMN] = dup.loc[0, STUDY_ID_COLUMN]
    with pytest.raises(Gold58PreflightError):
        validate_oof_table(dup, expected_n=58)
    bad = frame.copy()
    bad.iloc[0, 1] = 1.5
    with pytest.raises(Gold58PreflightError):
        validate_oof_table(bad, expected_n=58)


def test_miniature_one_epoch_public_entry_and_odd_train_size(tmp_path):
    root = _three_plane_root(tmp_path / "data", n=10)
    overrides = {
        "epochs": 1,
        "n_folds": 5,
        "image_size": 16,
        "num_slices": 2,
        "batch_size": 7,
        "require_cuda_for_scientific_run": False,
    }
    result = run_gold58_3plane_oof(
        data_root=root,
        output_dir=tmp_path / "out-mini",
        execute_training=True,
        allow_tiny_encoder_for_tests=True,
        require_cuda=False,
        config_overrides=overrides,
    )
    assert result["status"] == STATUS_COMPLETE
    assert result["n_labelled_studies"] == 10
    assert result["round_trip"]["max_abs_diff"] == 0.0
    assert result["oof_table_validation"]["ok"] is True
    assert result["runtime_preflight"]["preflight_round_trip_max_abs_diff"] == 0.0
    assert result["runtime_preflight"]["classifier_has_batchnorm1d"] is False
    again = run_gold58_3plane_oof(
        data_root=root,
        output_dir=tmp_path / "out-mini-2",
        execute_training=True,
        allow_tiny_encoder_for_tests=True,
        require_cuda=False,
        config_overrides=overrides,
    )
    assert again["oof_predictions_sha256"] == result["oof_predictions_sha256"]


def test_notebook_18_compiles_in_clean_process():
    script = (
        "import json, pathlib\n"
        "nb = json.loads(pathlib.Path('notebooks/18_kaggle_gold58_3plane_oof_v1.ipynb').read_text())\n"
        "src = '\\n'.join(''.join(c.get('source') or []) for c in nb['cells'] if c.get('cell_type')=='code')\n"
        "compile(src, 'notebook18', 'exec')\n"
        "print('NOTEBOOK18_COMPILE_OK')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
        env={**dict(**{k: v for k, v in __import__("os").environ.items()}), "PYTHONPATH": str(REPO_ROOT)},
    )
    assert proc.returncode == 0, proc.stderr
    assert "NOTEBOOK18_COMPILE_OK" in proc.stdout
    nb = json.loads((REPO_ROOT / "notebooks" / "18_kaggle_gold58_3plane_oof_v1.ipynb").read_text())
    source = "\n".join("".join(cell.get("source") or []) for cell in nb["cells"])
    assert "run_gold58_3plane_oof" in source
    assert "execute_training=True" in source
    assert "DO_NOT_SUBMIT" in source
    assert "competitions submit" not in source.lower()
