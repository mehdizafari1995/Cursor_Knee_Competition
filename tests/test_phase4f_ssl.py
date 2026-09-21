"""Phase 4F image-only SimSiam SSL: synthetic smoke, guards, frozen pins."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from knee.config import OFFICIAL_TARGET_COLUMNS, PROJECT_ROOT, REPORT_COLUMN, STUDY_ID_COLUMN
from knee.gold58_oof import IMAGENET_SHA256, SERIES_RULE, SERIES_RULE_V2
from knee.kaggle_bundle import sha256_file, would_include
from knee.phase4f_ssl import (
    EXPECTED_UNLABELED_REAL,
    FROZEN_V1_RECIPE_SHA256,
    FROZEN_V14,
    FROZEN_V2_CONFIG_SHA256,
    GOLD58_SSL_OOF_EXECUTION_ALLOWED,
    KERNEL_SLUG,
    PROTOCOL,
    STATUS_COLLAPSE,
    STATUS_EXECUTION,
    STATUS_GOLD,
    STATUS_HIDDEN,
    STATUS_INTERNET,
    STATUS_LABELS,
    STATUS_OOF,
    STATUS_PLANNING,
    STATUS_SERIES,
    STATUS_SMOKE,
    UNLABELED_IMAGE_SSL_EXECUTION_ALLOWED,
    UNLABELED_IMAGE_SSL_PLANNING_ALLOWED,
    UnlabeledStudySliceDataset,
    detect_collapse,
    load_ssl_config,
    plan_unlabeled_ssl_cohort,
    render_protocol_markdown,
    representation_std,
    run_phase4f_ssl,
    select_unlabeled_study_ids,
    simsiam_loss,
)
from tests.competition_fixtures import (
    REPORT_LABEL_GOLD_UIDS,
    REPORT_LABEL_LEAK_TOKEN,
    REPORT_LABEL_PATIENTS,
    REPORT_LABEL_WEAK_UIDS,
    write_series,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _mixed_root(root: Path, *, n_gold: int = 2, n_unlabeled: int = 4, extra_sagittal: bool = False) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    train_rows = []
    series_rows = []
    for i in range(n_gold + n_unlabeled):
        study = f"study-{i:03d}"
        row: dict[str, object] = {STUDY_ID_COLUMN: study, REPORT_COLUMN: "FINDINGS: synthetic"}
        for t, col in enumerate(OFFICIAL_TARGET_COLUMNS):
            row[col] = (i + t) % 2 if i < n_gold else np.nan
        train_rows.append(row)
        sag = f"series-sag-{i:03d}"
        series_rows.append(
            {
                STUDY_ID_COLUMN: study,
                "SeriesInstanceUID": sag,
                "Fluid_Sensitive": 1,
                "Fat_Suppression": 1,
                "Anatomical_Plane": "Sagittal",
            }
        )
        write_series(
            root / "train_series" / study / sag,
            study_uid=study,
            series_uid=sag,
            n_slices=3,
            patient_id=f"PAT-{i:03d}",
            scramble_filenames=False,
        )
        if extra_sagittal:
            sag2 = f"series-sag2-{i:03d}"
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
                root / "train_series" / study / sag2,
                study_uid=study,
                series_uid=sag2,
                n_slices=2,
                patient_id=f"PAT-{i:03d}",
                scramble_filenames=False,
            )
    pd.DataFrame(train_rows).to_csv(root / "train.csv", index=False)
    pd.DataFrame(series_rows).to_csv(root / "train_series.csv", index=False)
    pd.DataFrame({STUDY_ID_COLUMN: ["hidden-001"]}).to_csv(root / "test.csv", index=False)
    return root


def test_frozen_pins_and_gates_unchanged():
    assert sha256_file(REPO_ROOT / "configs" / "gold58_image_resnet18_oof_v1.yaml") == FROZEN_V1_RECIPE_SHA256
    assert sha256_file(REPO_ROOT / "configs" / "gold58_image_resnet18_oof_v2.yaml") == FROZEN_V2_CONFIG_SHA256
    assert sha256_file(REPO_ROOT / "configs" / "second_reader_pilot_v1_4.yaml") == FROZEN_V14
    cfg = load_ssl_config()
    assert cfg["protocol"] == PROTOCOL
    assert cfg["series_selection_rule"] == SERIES_RULE_V2
    assert cfg["imagenet_weights_sha256"] == IMAGENET_SHA256
    assert cfg["execute_training"] is False
    assert cfg["open_unlabeled_pixels"] is False
    assert UNLABELED_IMAGE_SSL_PLANNING_ALLOWED is True
    assert UNLABELED_IMAGE_SSL_EXECUTION_ALLOWED is False
    assert GOLD58_SSL_OOF_EXECUTION_ALLOWED is False
    assert SERIES_RULE_V2 != SERIES_RULE
    v2_run = (REPO_ROOT / "reports" / "gold58_image_oof_v2_run.md").read_text(encoding="utf-8")
    assert "GOLD58_OOF_COMPLETE" in v2_run
    rehearsal = (REPO_ROOT / "reports" / "phase4e_submission_rehearsal_run.md").read_text(
        encoding="utf-8"
    )
    assert "SUBMISSION_REHEARSAL_COMPLETE" in rehearsal


def test_simsiam_loss_is_symmetric_stopgrad():
    p1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    z1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    p2 = torch.tensor([[0.0, 1.0], [1.0, 0.0]], requires_grad=True)
    z2 = torch.tensor([[0.0, 1.0], [1.0, 0.0]], requires_grad=True)
    loss = simsiam_loss(p1, z2, p2, z1)
    loss.backward()
    assert z1.grad is None or float(z1.grad.abs().sum()) == 0.0
    assert z2.grad is None or float(z2.grad.abs().sum()) == 0.0
    assert p1.grad is not None and float(p1.grad.abs().sum()) > 0
    assert p2.grad is not None and float(p2.grad.abs().sum()) > 0


def test_collapse_detector_flags_constant_representation():
    collapsed = torch.ones(8, 16)
    diverse = torch.randn(8, 16)
    assert detect_collapse(collapsed, 0.001) is True
    assert detect_collapse(diverse, 0.001) is False
    assert representation_std(collapsed) < 1e-6


def test_dataset_rejects_labels_and_reports(tmp_path):
    root = _mixed_root(tmp_path / "ds")
    sag = root / "train_series" / "study-002" / "series-sag-002"
    with pytest.raises(Exception) as exc:
        UnlabeledStudySliceDataset(
            ["study-002"],
            [sag],
            labels=np.zeros((1, 12)),
        )
    assert getattr(exc.value, "status", "") == STATUS_LABELS
    with pytest.raises(Exception) as exc2:
        UnlabeledStudySliceDataset(["study-002"], [sag], report="FINDINGS: x")
    assert getattr(exc2.value, "status", "") == STATUS_LABELS
    item = UnlabeledStudySliceDataset(
        ["study-002"], [sag], num_slices=2, image_size=8, imagenet_normalize=True
    )[0]
    assert set(item) == {"image"}
    assert item["image"].ndim == 4


def test_gold_intersection_is_empty_and_labels_dropped(tmp_path):
    root = _mixed_root(tmp_path / "mix")
    train = pd.read_csv(root / "train.csv")
    ssl_ids, gold_ids, stats = select_unlabeled_study_ids(train)
    assert stats["n_gold_intersection"] == 0
    assert stats["n_gold_excluded"] == 2
    assert stats["n_unlabeled_selected"] == 4
    assert set(ssl_ids).isdisjoint(gold_ids)
    series = pd.read_csv(root / "train_series.csv")
    planned = plan_unlabeled_ssl_cohort(train, series, root)
    assert planned["n_image_frame_columns"] == 1
    assert planned["n_selected"] == 4
    assert planned["series_selection_aggregates"]["series_selection_rule"] == SERIES_RULE_V2


def test_duplicate_series_fail_closed(tmp_path):
    root = _mixed_root(tmp_path / "amb", extra_sagittal=True)
    train = pd.read_csv(root / "train.csv")
    series = pd.read_csv(root / "train_series.csv")
    planned = plan_unlabeled_ssl_cohort(train, series, root)
    assert planned["n_selected"] == 4
    empty = root / "empty"
    empty.mkdir()
    pd.DataFrame({STUDY_ID_COLUMN: ["study-000"]}).assign(**{col: np.nan for col in OFFICIAL_TARGET_COLUMNS}).to_csv(
        empty / "train.csv", index=False
    )
    pd.DataFrame(
        columns=[STUDY_ID_COLUMN, "SeriesInstanceUID", "Fluid_Sensitive", "Anatomical_Plane"]
    ).to_csv(empty / "train_series.csv", index=False)
    with pytest.raises(Exception) as exc:
        plan_unlabeled_ssl_cohort(pd.read_csv(empty / "train.csv"), pd.read_csv(empty / "train_series.csv"), empty)
    assert getattr(exc.value, "status", "") in {STATUS_SERIES, STATUS_GOLD}


def test_hidden_test_and_real_execution_refused(tmp_path):
    result = run_phase4f_ssl(
        data_root=tmp_path / "unit" / "test_series" / "hidden",
        output_dir=tmp_path / "out-hidden",
        execute_training=False,
        internet_enabled=False,
    )
    assert result["status"] == STATUS_HIDDEN
    exec_result = run_phase4f_ssl(
        output_dir=tmp_path / "out-exec",
        execute_training=True,
        allow_tiny_encoder_for_tests=False,
    )
    assert exec_result["status"] == STATUS_EXECUTION
    assert exec_result["training_run"] is False
    assert exec_result["n_real_unlabelled_studies_processed"] == 0
    oof = run_phase4f_ssl(output_dir=tmp_path / "out-oof", run_oof=True)
    assert oof["status"] == STATUS_OOF
    net = run_phase4f_ssl(output_dir=tmp_path / "out-net", internet_enabled=True)
    assert net["status"] == STATUS_INTERNET


def test_synthetic_smoke_backprop_resume_and_collapse(tmp_path):
    result = run_phase4f_ssl(
        output_dir=tmp_path / "smoke",
        execute_training=True,
        allow_tiny_encoder_for_tests=True,
    )
    assert result["status"] == STATUS_SMOKE
    assert result["training_run"] is False
    assert result["n_real_unlabelled_studies_processed"] == 0
    assert result["OOF_run"] is False
    assert result["inference_run"] is False
    assert result["competition_submission_written"] is False
    smoke = result["smoke"]
    assert smoke["backprop_ran"] is True
    assert smoke["n_steps"] == 3
    assert all(np.isfinite(v) for v in smoke["losses"])
    assert smoke["encoder_round_trip_max_abs"] == 0.0
    assert smoke["resume_encoder_match_max_abs"] == 0.0
    collapsed = run_phase4f_ssl(
        output_dir=tmp_path / "collapse",
        execute_training=True,
        allow_tiny_encoder_for_tests=True,
        collapse_fixture=True,
    )
    assert collapsed["status"] == STATUS_COLLAPSE


def test_planning_status_without_training(tmp_path):
    result = run_phase4f_ssl(output_dir=tmp_path / "plan", execute_training=False)
    assert result["status"] == STATUS_PLANNING
    assert result["kaggle_kernel_launched"] is False
    assert result["resource_estimate_n4349"]["n_studies"] == EXPECTED_UNLABELED_REAL


def test_protocol_notebook_allowlist_and_privacy():
    cfg = load_ssl_config()
    text = render_protocol_markdown(cfg)
    assert "SimSiam" in text
    assert KERNEL_SLUG in text
    assert SERIES_RULE_V2 in text
    assert "UNLABELED_IMAGE_SSL_EXECUTION_ALLOWED=false" in text
    assert REPORT_LABEL_LEAK_TOKEN not in text
    assert "FINDINGS:" not in text
    assert "DummySeriesDesc" not in text
    protocol = (REPO_ROOT / "reports" / "phase4f_ssl_protocol.md").read_text(encoding="utf-8")
    readiness = (REPO_ROOT / "reports" / "phase4f_ssl_readiness.md").read_text(encoding="utf-8")
    for blob in (protocol, readiness, text):
        assert REPORT_LABEL_LEAK_TOKEN not in blob
        assert "FINDINGS:" not in blob
        for uid in REPORT_LABEL_GOLD_UIDS + REPORT_LABEL_WEAK_UIDS:
            assert uid not in blob
        for patient in REPORT_LABEL_PATIENTS:
            assert patient not in blob
    assert "SSL_SMOKE_COMPLETE" in readiness or "SSL_PLANNING_COMPLETE" in readiness
    assert "n_real_unlabelled_studies_processed: 0" in readiness
    assert "training_run: false" in readiness
    assert would_include("knee/phase4f_ssl.py")
    assert would_include("scripts/run_phase4f_ssl.py")
    assert would_include("configs/unlabeled_image_simsiam_ssl_v1.yaml")
    assert would_include("notebooks/13_kaggle_unlabeled_image_ssl_v1.ipynb")
    assert would_include("notebooks/14_kaggle_unlabeled_image_ssl_canary.ipynb")
    assert would_include("notebooks/15_kaggle_unlabeled_image_ssl_canary_v2.ipynb")
    assert would_include("notebooks/16_kaggle_unlabeled_image_ssl_canary_v3.ipynb")
    assert would_include("knee/phase4f_ssl_canary.py")
    assert would_include("knee/phase4f_ssl_canary_v2.py")
    assert would_include("knee/phase4f_ssl_canary_v3.py")
    assert would_include("knee/gold58_3plane_oof.py")
    assert not would_include("reports/phase4f_ssl_protocol.md")
    assert not would_include("reports/phase4f_ssl_readiness.md")
    assert not would_include("reports/phase4f_ssl_canary_protocol.md")
    import json

    nb = json.loads(
        (REPO_ROOT / "notebooks" / "13_kaggle_unlabeled_image_ssl_v1.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join("".join(cell.get("source") or []) for cell in nb["cells"])
    lowered = source.lower()
    assert "pip install" not in lowered
    assert "competitions submit" not in lowered
    assert "execute_training=False" in source
    assert "notebook_07_executed=false" in source
    assert "UNLABELED_IMAGE_SSL_EXECUTION_ALLOWED" in source
    assert KERNEL_SLUG in source
    assert "DO_NOT_SUBMIT" in source
    meta = (nb.get("metadata") or {}).get("kaggle") or {}
    assert meta.get("isInternetEnabled") is False
    assert meta.get("gpu") is True
