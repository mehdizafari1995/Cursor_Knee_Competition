"""Prepared gold-58 three-plane OOF protocol: selection, fusion, fail-closed execution."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

from knee.config import OFFICIAL_TARGET_COLUMNS, STUDY_ID_COLUMN
from knee.gold58_3plane_oof import (
    FROZEN_V2_MACRO_AUC,
    GOLD58_3PLANE_OOF_EXECUTION_ALLOWED,
    KERNEL_SLUG,
    PLANES,
    PROTOCOL,
    SERIES_RULE_3PLANE,
    SSL_ENGINEERING_PATH_PARKED,
    STATUS_NOT_AUTHORIZED,
    STATUS_PREPARED,
    estimate_3plane_resources,
    load_3plane_config,
    mean_fuse_encoder_embeddings,
    render_3plane_protocol_markdown,
    run_gold58_3plane_oof,
    select_one_series_per_plane,
)
from knee.gold58_oof import SERIES_RULE_V2, inspect_series_inventory
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
    assert GOLD58_3PLANE_OOF_EXECUTION_ALLOWED is False
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
    assert estimate["execution_authorized"] is False
    assert estimate["n_labelled_studies"] == 58
    assert estimate["projected_seconds"] == pytest.approx(8168.106 * 3.0)


def test_execution_is_refused():
    prepared = run_gold58_3plane_oof(execute_training=False)
    assert prepared["status"] == STATUS_PREPARED
    assert prepared["kaggle_kernel_launched"] is False
    assert prepared["n_report_text_rows_processed"] == 0
    refused = run_gold58_3plane_oof(execute_training=True)
    assert refused["status"] == STATUS_NOT_AUTHORIZED
    assert refused["training_run"] is False
    assert refused["OOF_run"] is False


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
    assert "kaggle_kernel_launched: false" in run
    assert "SSL_ENGINEERING_PATH_PARKED" in parked
    assert would_include("knee/gold58_3plane_oof.py")
    assert would_include("scripts/run_gold58_3plane_oof.py")
    assert would_include("configs/gold58_image_resnet18_3plane_oof_v1.yaml")
    assert not would_include("reports/gold58_3plane_oof_protocol.md")
    assert not would_include("reports/gold58_3plane_oof_run.md")
