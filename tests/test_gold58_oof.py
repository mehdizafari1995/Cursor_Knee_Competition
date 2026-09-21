"""Phase 4D.1 gold58 image OOF: fail-closed preflight and tiny training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from knee.competition import SAMPLE_SUBMISSION_COLUMNS
from knee.config import OFFICIAL_TARGET_COLUMNS, PROJECT_ROOT, STUDY_ID_COLUMN
from knee.gold58_oof import (
    EXPECTED_LABELLED_REAL,
    IMAGENET_SHA256,
    KERNEL_SLUG,
    PROTOCOL,
    SERIES_RULE,
    STATUS_COMPLETE,
    STATUS_FOLDS,
    STATUS_GROUPING,
    STATUS_SERIES,
    STATUS_WEIGHTS,
    locate_imagenet_weights,
    patient_bootstrap_aucs,
    render_protocol_markdown,
    round_trip_state_dict,
    run_gold58_image_oof,
    select_sagittal_fluid_sensitive_series,
    submission_contract_from_oof,
    unique_series_or_fail,
    verify_imagenet_weights,
)
from knee.baseline_model import build_baseline_model
from knee.kaggle_bundle import sha256_file, would_include
from knee.phase4d_readiness import SECOND_READER_SCORE_BEARING_ALLOWED
from knee.report_labels import (
    HARD_WEAK_LABEL_TRAINING_ALLOWED,
    SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED,
)
from tests.competition_fixtures import (
    REPORT_LABEL_GOLD_UIDS,
    REPORT_LABEL_LEAK_TOKEN,
    REPORT_LABEL_PATIENTS,
    REPORT_LABEL_WEAK_UIDS,
    write_series,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FROZEN_V14 = "992af7a4b4483278445a6f3fb363c818611e79005618dff063460d0aed3bd7f7"
FAST = {
    "epochs": 1,
    "image_size": 16,
    "num_slices": 2,
    "batch_size": 2,
    "require_cuda_for_scientific_run": False,
}


def _labelled_root(root: Path, *, n: int = 10, extra_sagittal: bool = False, blank_patient: bool = False) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    train_rows = []
    series_rows = []
    for i in range(n):
        study = f"study-{i:03d}"
        patient = "" if blank_patient else f"PAT-{i:03d}"
        row = {STUDY_ID_COLUMN: study}
        for t, col in enumerate(OFFICIAL_TARGET_COLUMNS):
            row[col] = (i + t) % 2
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
            patient_id=patient or "MISSING",
            scramble_filenames=False,
        )
        if blank_patient:
            # overwrite with empty PatientID
            write_series(
                root / "train_series" / study / sag,
                study_uid=study,
                series_uid=sag,
                n_slices=3,
                patient_id="",
                scramble_filenames=False,
            )
        cor = f"series-cor-{i:03d}"
        series_rows.append(
            {
                STUDY_ID_COLUMN: study,
                "SeriesInstanceUID": cor,
                "Fluid_Sensitive": 0,
                "Fat_Suppression": 0,
                "Anatomical_Plane": "Coronal",
            }
        )
        write_series(
            root / "train_series" / study / cor,
            study_uid=study,
            series_uid=cor,
            n_slices=2,
            patient_id=patient or "MISSING",
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
                patient_id=patient or "MISSING",
                scramble_filenames=False,
            )
    pd.DataFrame(train_rows).to_csv(root / "train.csv", index=False)
    pd.DataFrame(series_rows).to_csv(root / "train_series.csv", index=False)
    sample = pd.DataFrame({STUDY_ID_COLUMN: [f"test-{j}" for j in range(2)]})
    for col in OFFICIAL_TARGET_COLUMNS:
        sample[col] = 0.5
    sample.loc[:, list(SAMPLE_SUBMISSION_COLUMNS)].to_csv(root / "sample_submission.csv", index=False)
    pd.DataFrame({STUDY_ID_COLUMN: sample[STUDY_ID_COLUMN]}).to_csv(root / "test.csv", index=False)
    pd.DataFrame(
        [
            {
                STUDY_ID_COLUMN: f"test-{j}",
                "SeriesInstanceUID": f"test-ser-{j}",
                "Fluid_Sensitive": 1,
                "Fat_Suppression": 1,
                "Anatomical_Plane": "Sagittal",
            }
            for j in range(2)
        ]
    ).to_csv(root / "test_series.csv", index=False)
    return root


def test_gates_remain_closed():
    assert HARD_WEAK_LABEL_TRAINING_ALLOWED is False
    assert SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED is False
    assert SECOND_READER_SCORE_BEARING_ALLOWED is False


def test_train_baseline_guard_is_unchanged():
    text = (REPO_ROOT / "scripts" / "train_baseline.py").read_text(encoding="utf-8")
    assert "Refusing to start full training in this phase even with" in text
    assert "assert_hard_weak_label_training_forbidden" in text


def test_series_rule_unique_or_fail(tmp_path):
    root = _labelled_root(tmp_path / "ok")
    series = pd.read_csv(root / "train_series.csv")
    candidates = select_sagittal_fluid_sensitive_series(series)
    lookup = unique_series_or_fail(
        candidates, pd.read_csv(root / "train.csv")[STUDY_ID_COLUMN].astype(str).tolist()
    )
    assert len(lookup) == 10
    bad = _labelled_root(tmp_path / "multi", extra_sagittal=True)
    with pytest.raises(Exception, match="sagittal-fluid-sensitive"):
        unique_series_or_fail(
            select_sagittal_fluid_sensitive_series(pd.read_csv(bad / "train_series.csv")),
            pd.read_csv(bad / "train.csv")[STUDY_ID_COLUMN].astype(str).tolist(),
        )


def test_grouping_preflight_fail_on_blank_patient(tmp_path):
    root = _labelled_root(tmp_path / "blank", blank_patient=True)
    result = run_gold58_image_oof(
        data_root=root,
        output_dir=tmp_path / "out-blank",
        execute_training=False,
        require_cuda=False,
        config_overrides=FAST,
    )
    assert result["status"] == STATUS_GROUPING
    assert result["training_run"] is False


def test_series_preflight_fail_on_ambiguous_selection(tmp_path):
    root = _labelled_root(tmp_path / "amb", extra_sagittal=True)
    result = run_gold58_image_oof(
        data_root=root,
        output_dir=tmp_path / "out-amb",
        execute_training=False,
        require_cuda=False,
        config_overrides=FAST,
    )
    assert result["status"] == STATUS_SERIES
    assert result["training_run"] is False


def test_fold_support_fail_when_too_few_patients(tmp_path):
    root = _labelled_root(tmp_path / "tiny", n=3)
    result = run_gold58_image_oof(
        data_root=root,
        output_dir=tmp_path / "out-tiny",
        execute_training=False,
        require_cuda=False,
        config_overrides=FAST,
    )
    assert result["status"] == STATUS_FOLDS
    assert result["training_run"] is False


def test_preflight_pass_then_tiny_oof(tmp_path):
    root = _labelled_root(tmp_path / "train10")
    result = run_gold58_image_oof(
        data_root=root,
        output_dir=tmp_path / "out-train",
        execute_training=True,
        allow_tiny_encoder_for_tests=True,
        require_cuda=False,
        config_overrides=FAST,
    )
    assert result["status"] == STATUS_COMPLETE
    assert result["training_run"] is True
    assert result["n_labelled_studies"] == 10
    assert result["n_report_only_studies_processed"] == 0
    assert result["competition_submission_written"] is False
    assert result["use_second_reader"] is False
    assert result["scientific_evaluation"] is False
    assert result["round_trip"]["within_tolerance"] is True
    assert result["n_oof_rows"] == 10
    assert 0.0 <= result["macro_roc_auc"] <= 1.0
    agg = result["preflight_aggregates"]
    assert agg["n_unique_patients"] == 10
    assert agg["n_chosen_series"] == 10
    for name in OFFICIAL_TARGET_COLUMNS:
        counts = agg["per_target_counts"][name]
        assert counts["n_positive"] + counts["n_negative"] == 10
        assert counts["n_unknown"] == 0


def test_imagenet_hash_mismatch(tmp_path):
    fake = tmp_path / "resnet18-f37072fd.pth"
    fake.write_bytes(b"not-imagenet")
    with pytest.raises(Exception, match="SHA-256 mismatch"):
        verify_imagenet_weights(fake, IMAGENET_SHA256)


def test_scientific_run_without_weights_fails(tmp_path, monkeypatch):
    root = _labelled_root(tmp_path / "noweights")
    monkeypatch.delenv("GOLD58_IMAGENET_WEIGHTS", raising=False)
    monkeypatch.setattr(
        "knee.gold58_oof.locate_imagenet_weights", lambda explicit=None: None
    )
    result = run_gold58_image_oof(
        data_root=root,
        output_dir=tmp_path / "out-nw",
        execute_training=True,
        allow_tiny_encoder_for_tests=False,
        require_cuda=False,
        config_overrides=FAST,
    )
    assert result["status"] == STATUS_WEIGHTS
    assert result["training_run"] is False


def test_round_trip_tiny(tmp_path):
    model = build_baseline_model(smoke=True, encoder_name="tiny", pretrained=False)
    images = torch.rand(2, 2, 1, 16, 16)
    result = round_trip_state_dict(
        model,
        images,
        tmp_path / "rt.pt",
        torch.device("cpu"),
        lambda: build_baseline_model(smoke=True, encoder_name="tiny", pretrained=False),
    )
    assert result["within_tolerance"] is True


def test_bootstrap_and_submission_contract(tmp_path):
    n = 8
    y_true = np.zeros((n, 12), dtype=np.float64)
    for t in range(12):
        y_true[:, t] = (np.arange(n) + t) % 2
    y_prob = y_true * 0.8 + 0.1
    groups = np.array([f"g{i}" for i in range(n)], dtype=object)
    boot = patient_bootstrap_aucs(
        y_true, y_prob, groups, n_resamples=20, seed=42, ci_level=0.95
    )
    assert boot["macro_roc_auc_ci_low"] == 1.0
    oof = pd.DataFrame({STUDY_ID_COLUMN: [f"s{i}" for i in range(n)]})
    for col in OFFICIAL_TARGET_COLUMNS:
        oof[col] = 0.2
    contract = submission_contract_from_oof(oof, tmp_path)
    assert contract["ok"] is True
    assert contract["competition_submission_written"] is False
    assert not (tmp_path / "contract_check_not_a_submission.csv").exists()


def test_module_never_calls_second_reader_generate():
    src = (REPO_ROOT / "knee" / "gold58_oof.py").read_text(encoding="utf-8")
    assert "run_phase47_gold_constrained" not in src
    assert "generate_fn" not in src
    assert "model.generate" not in src
    assert KERNEL_SLUG in src
    assert PROTOCOL in src
    assert SERIES_RULE in src
    assert EXPECTED_LABELLED_REAL == 58
    assert would_include("knee/gold58_oof.py")
    assert would_include("scripts/train_gold58_image_oof.py")
    assert would_include("configs/gold58_image_resnet18_oof_v1.yaml")
    assert would_include("notebooks/10_kaggle_gold58_image_oof.ipynb")
    assert would_include("notebooks/11_kaggle_gold58_image_oof_v2.ipynb")
    assert would_include("notebooks/12_kaggle_gold58_submission_rehearsal.ipynb")
    assert would_include("configs/gold58_image_resnet18_oof_v2.yaml")
    assert would_include("scripts/train_gold58_image_oof_v2.py")
    assert would_include("knee/phase4e_rehearsal.py")
    assert would_include("scripts/run_phase4e_rehearsal.py")
    assert would_include("knee/phase4f_ssl.py")
    assert would_include("scripts/run_phase4f_ssl.py")
    assert would_include("notebooks/13_kaggle_unlabeled_image_ssl_v1.ipynb")
    assert would_include("configs/unlabeled_image_simsiam_ssl_v1.yaml")
    assert would_include("notebooks/14_kaggle_unlabeled_image_ssl_canary.ipynb")
    assert would_include("notebooks/15_kaggle_unlabeled_image_ssl_canary_v2.ipynb")
    assert would_include("knee/phase4f_ssl_canary.py")
    assert would_include("knee/phase4f_ssl_canary_v2.py")
    assert would_include("scripts/run_phase4f_ssl_canary.py")
    assert would_include("scripts/run_phase4f_ssl_canary_v2.py")
    assert would_include("configs/unlabeled_image_simsiam_ssl_canary_v1.yaml")
    assert would_include("configs/unlabeled_image_simsiam_ssl_canary_v2.yaml")
    assert sha256_file(REPO_ROOT / "configs" / "second_reader_pilot_v1_4.yaml") == FROZEN_V14


def test_protocol_and_reports_are_sanitized():
    text = render_protocol_markdown()
    assert REPORT_LABEL_LEAK_TOKEN not in text
    assert "FINDINGS:" not in text
    for uid in REPORT_LABEL_GOLD_UIDS + REPORT_LABEL_WEAK_UIDS:
        assert uid not in text
    for patient in REPORT_LABEL_PATIENTS:
        assert patient not in text
    assert "kaggle.json" not in text.lower()
    report = (REPO_ROOT / "reports" / "gold58_image_oof_protocol.md").read_text(encoding="utf-8")
    assert SERIES_RULE in report
    assert IMAGENET_SHA256 in report
    assert "PatientID" in report
    run = (REPO_ROOT / "reports" / "gold58_image_oof_run.md").read_text(encoding="utf-8")
    assert "SERIES_SELECTION_PREFLIGHT_FAIL" in run
    assert "training_run" in run
    assert REPORT_LABEL_LEAK_TOKEN not in run
    assert "FINDINGS:" not in run


def test_notebook_10_is_offline_gpu_and_image_only():
    import json

    nb_path = REPO_ROOT / "notebooks" / "10_kaggle_gold58_image_oof.ipynb"
    payload = json.loads(nb_path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source") or []) for cell in payload["cells"])
    lowered = source.lower()
    assert "kaggle.json" not in lowered
    assert "pip install" not in lowered
    assert "process_report_only" not in lowered or "false" in lowered
    assert "run_gold58_image_oof" in source
    assert "execute_training=True" in source
    assert "notebook_07_executed=false" in source
    assert "notebook_09_executed=false" in source
    meta = payload.get("metadata") or {}
    kaggle_meta = meta.get("kaggle") or {}
    assert kaggle_meta.get("isInternetEnabled") is False
    assert kaggle_meta.get("gpu") is True
