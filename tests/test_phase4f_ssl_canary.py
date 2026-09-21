"""Phase 4F.1 SSL canary: preprocess order, resume, guards, frozen v1 pins."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from knee.config import OFFICIAL_TARGET_COLUMNS, PROJECT_ROOT, REPORT_COLUMN, STUDY_ID_COLUMN
from knee.gold58_oof import IMAGENET_SHA256, SERIES_RULE_V2
from knee.kaggle_bundle import sha256_file, would_include
from knee.phase4f_ssl import (
    FROZEN_V1_RECIPE_SHA256,
    FROZEN_V14,
    FROZEN_V2_CONFIG_SHA256,
    GOLD58_SSL_OOF_EXECUTION_ALLOWED,
    UNLABELED_IMAGE_SSL_EXECUTION_ALLOWED,
)
from knee.phase4f_ssl_canary import (
    AUTHORIZED_BATCH,
    AUTHORIZED_CHECKPOINT_STEP,
    AUTHORIZED_N_STUDIES,
    AUTHORIZED_STEPS,
    FROZEN_SSL_V1_CONFIG_SHA256,
    KERNEL_SLUG,
    PROTOCOL,
    REQUIRED_CHECKPOINT_KEYS,
    STATUS_COLLAPSE,
    STATUS_EMPTY,
    STATUS_FAIL,
    STATUS_FULL,
    STATUS_GOLD,
    STATUS_HIDDEN,
    STATUS_INTERNET,
    STATUS_LABELS,
    STATUS_OOF,
    STATUS_PASS,
    STATUS_PREPROCESS,
    STATUS_RESUME,
    UNLABELED_IMAGE_SSL_CANARY_EXECUTION_ALLOWED,
    CanaryUnlabeledStudySliceDataset,
    Phase4FCanaryError,
    STATUS_GPU_PREFLIGHT,
    apply_imagenet_last,
    assert_checkpoint_complete,
    assert_not_empty_batch,
    assert_resume_hashes,
    assert_unit_interval_for_intensity_jitter,
    assert_zero_gold_overlap,
    build_canary_model,
    compute_collapse_diagnostics,
    effective_rank,
    load_canary_config,
    render_canary_protocol_markdown,
    run_canary_amp_software_preflight,
    run_canary_gpu_preflight,
    run_interrupted_synthetic,
    run_phase4f_ssl_canary,
    run_uninterrupted_synthetic,
    select_canary_study_ids,
    std_only_diagnostics,
    two_views_canary,
    validate_collapse_fixtures,
    _make_optimizer_bundle,
    _one_optimizer_step,
)
from tests.competition_fixtures import (
    REPORT_LABEL_GOLD_UIDS,
    REPORT_LABEL_LEAK_TOKEN,
    REPORT_LABEL_PATIENTS,
    REPORT_LABEL_WEAK_UIDS,
    write_series,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _mixed_root(root: Path, *, n_gold: int = 2, n_unlabeled: int = 8) -> Path:
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
    pd.DataFrame(train_rows).to_csv(root / "train.csv", index=False)
    pd.DataFrame(series_rows).to_csv(root / "train_series.csv", index=False)
    pd.DataFrame({STUDY_ID_COLUMN: ["hidden-001"]}).to_csv(root / "test.csv", index=False)
    return root


def test_v1_and_canary_pins_are_frozen():
    assert sha256_file(REPO_ROOT / "configs" / "unlabeled_image_simsiam_ssl_v1.yaml") == FROZEN_SSL_V1_CONFIG_SHA256
    assert sha256_file(REPO_ROOT / "configs" / "gold58_image_resnet18_oof_v1.yaml") == FROZEN_V1_RECIPE_SHA256
    assert sha256_file(REPO_ROOT / "configs" / "gold58_image_resnet18_oof_v2.yaml") == FROZEN_V2_CONFIG_SHA256
    assert sha256_file(REPO_ROOT / "configs" / "second_reader_pilot_v1_4.yaml") == FROZEN_V14
    cfg = load_canary_config()
    assert cfg["protocol"] == PROTOCOL
    assert cfg["kernel_slug"] == KERNEL_SLUG
    assert cfg["series_selection_rule"] == SERIES_RULE_V2
    assert cfg["imagenet_weights_sha256"] == IMAGENET_SHA256
    assert cfg["n_canary_studies"] == AUTHORIZED_N_STUDIES
    assert cfg["max_optimizer_steps"] == AUTHORIZED_STEPS
    assert cfg["checkpoint_at_step"] == AUTHORIZED_CHECKPOINT_STEP
    assert cfg["batch_size"] == AUTHORIZED_BATCH
    assert cfg["safety_factor"] == 1.25
    assert cfg["shutdown_reserve_seconds"] == 3600
    assert cfg["imagenet_normalize_in_dataset"] is False
    assert cfg["imagenet_last"] is True
    assert UNLABELED_IMAGE_SSL_EXECUTION_ALLOWED is False
    assert GOLD58_SSL_OOF_EXECUTION_ALLOWED is False
    assert UNLABELED_IMAGE_SSL_CANARY_EXECUTION_ALLOWED is True
    v1 = (REPO_ROOT / "reports" / "phase4f_ssl_readiness.md").read_text(encoding="utf-8")
    assert "SSL_SMOKE_COMPLETE" in v1
    assert "n_real_unlabelled_studies_processed: 0" in v1


def test_intensity_jitter_never_runs_on_imagenet_tensors():
    gray = torch.rand(2, 1, 8, 8)
    assert_unit_interval_for_intensity_jitter(gray)
    imagenet = apply_imagenet_last(gray)
    assert imagenet.size(1) == 3
    with pytest.raises(Phase4FCanaryError) as exc:
        assert_unit_interval_for_intensity_jitter(imagenet)
    assert exc.value.status == STATUS_PREPROCESS
    with pytest.raises(Phase4FCanaryError) as exc2:
        two_views_canary(imagenet.unsqueeze(0), generator=torch.Generator().manual_seed(0), cfg=load_canary_config())
    assert exc2.value.status == STATUS_PREPROCESS
    v1, v2 = two_views_canary(
        gray.unsqueeze(0),
        generator=torch.Generator().manual_seed(0),
        cfg=load_canary_config(),
    )
    assert v1.shape[1:] == imagenet.shape
    with pytest.raises(Phase4FCanaryError):
        assert_unit_interval_for_intensity_jitter(v1[0])


def test_canary_dataset_rejects_imagenet_normalize_and_labels(tmp_path):
    root = _mixed_root(tmp_path / "ds")
    sag = root / "train_series" / "study-002" / "series-sag-002"
    with pytest.raises(Phase4FCanaryError) as exc:
        CanaryUnlabeledStudySliceDataset(["study-002"], [sag], imagenet_normalize=True)
    assert exc.value.status == STATUS_PREPROCESS
    with pytest.raises(Phase4FCanaryError) as exc2:
        CanaryUnlabeledStudySliceDataset(["study-002"], [sag], labels=np.zeros((1, 12)))
    assert exc2.value.status == STATUS_LABELS
    item = CanaryUnlabeledStudySliceDataset(
        ["study-002"], [sag], num_slices=2, image_size=8, imagenet_normalize=False
    )[0]
    assert item["image"].shape[1] == 1
    assert float(item["image"].min()) >= 0.0
    assert float(item["image"].max()) <= 1.0


def test_sha256_cohort_is_deterministic_and_gold_disjoint():
    unlabeled = [f"u-{i:03d}" for i in range(20)]
    gold = [f"g-{i:03d}" for i in range(5)]
    a = select_canary_study_ids(unlabeled, n=8, seed=42)
    b = select_canary_study_ids(unlabeled, n=8, seed=42)
    c = select_canary_study_ids(unlabeled, n=8, seed=7)
    assert a == b
    assert a != c
    assert len(a) == 8
    expected = sorted(
        unlabeled,
        key=lambda uid: hashlib.sha256(f"42|{uid}".encode("utf-8")).hexdigest(),
    )[:8]
    assert a == expected
    assert assert_zero_gold_overlap(a, gold) == 0
    with pytest.raises(Phase4FCanaryError) as exc:
        assert_zero_gold_overlap(gold[:1] + a, gold)
    assert exc.value.status == STATUS_GOLD
    assert "n_overlap=1" in exc.value.reason


def test_collapse_diagnostics_accept_float16():
    cfg = load_canary_config()
    diverse = torch.randn(16, 32, dtype=torch.float16)
    collapsed = torch.ones(16, 32, dtype=torch.float16)
    ok = compute_collapse_diagnostics(diverse, diverse, diverse, cfg=cfg)
    bad = compute_collapse_diagnostics(collapsed, collapsed, collapsed, cfg=cfg)
    assert ok["collapsed"] is False
    assert math.isfinite(ok["effective_rank"])
    assert bad["collapsed"] is True
    cfg = load_canary_config()
    report = validate_collapse_fixtures(cfg)
    assert report["non_collapsed_detected"] is False
    assert report["collapsed_detected"] is True
    diverse = torch.randn(16, 32)
    collapsed = torch.ones(16, 32)
    assert compute_collapse_diagnostics(diverse, diverse, diverse, cfg=cfg)["collapsed"] is False
    assert compute_collapse_diagnostics(collapsed, collapsed, collapsed, cfg=cfg)["collapsed"] is True


def test_optimizer_step_std_only_does_not_call_eigvalsh(monkeypatch):
    cfg = load_canary_config()
    device = torch.device("cpu")
    model = build_canary_model(cfg, tiny=True, weights_path=None).to(device)
    optimizer, scheduler, scaler, _use_amp = _make_optimizer_bundle(model, cfg, device=device)
    generator = torch.Generator().manual_seed(0)
    batch = torch.rand(4, 2, 1, 16, 16)

    def boom(*_args, **_kwargs):
        raise AssertionError("eigvalsh_called_during_optimizer_step")

    monkeypatch.setattr(torch.linalg, "eigvalsh", boom)
    loss, grad, diag = _one_optimizer_step(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        batch=batch,
        generator=generator,
        cfg=cfg,
        device=device,
        use_amp=False,
        collapse_fixture=False,
    )
    assert math.isfinite(loss)
    assert math.isfinite(grad)
    assert diag["std_only"] is True
    assert "encoder_std" in diag
    assert "effective_rank" not in diag
    assert "mean_abs_pairwise_cosine" not in diag
    std = std_only_diagnostics(
        torch.randn(8, 32), torch.randn(8, 32), torch.randn(8, 32), cfg=cfg
    )
    assert std["std_only"] is True
    assert std["collapsed"] is False


def test_probe_eigvalsh_float32_at_most_16x16_rejects_collapse(monkeypatch):
    calls = []
    orig = torch.linalg.eigvalsh

    def wrapped(matrix, *args, **kwargs):
        calls.append({"dtype": matrix.dtype, "shape": tuple(matrix.shape)})
        return orig(matrix, *args, **kwargs)

    monkeypatch.setattr(torch.linalg, "eigvalsh", wrapped)
    cfg = load_canary_config()
    collapsed = torch.ones(16, 2048, dtype=torch.float16)
    bad = compute_collapse_diagnostics(collapsed, collapsed, collapsed, cfg=cfg)
    assert bad["collapsed"] is True
    assert calls
    for call in calls:
        assert call["dtype"] == torch.float32
        assert call["shape"] == (16, 16)
        assert max(call["shape"]) <= 16
    diverse = torch.randn(16, 2048, dtype=torch.float16)
    ok = compute_collapse_diagnostics(diverse, diverse, diverse, cfg=cfg)
    assert ok["collapsed"] is False
    assert math.isfinite(ok["effective_rank"])
    rank = effective_rank(torch.randn(16, 2048, dtype=torch.float16))
    assert math.isfinite(rank)
    assert calls[-1]["dtype"] == torch.float32
    assert calls[-1]["shape"] == (16, 16)


def test_gpu_preflight_fails_closed_without_cuda_and_local_amp_software_passes():
    cfg = load_canary_config()
    if not torch.cuda.is_available():
        gpu = run_canary_gpu_preflight(cfg=cfg)
        assert gpu["status"] == STATUS_GPU_PREFLIGHT
        assert gpu["ok"] is False
        assert gpu["skip_reason"] == "cuda_required"
        assert gpu["n_real_unlabelled_studies_pixels_opened"] == 0
        assert gpu["n_report_text_rows_processed"] == 0
        assert gpu["n_second_reader_rows_processed"] == 0
    model = build_canary_model(cfg, tiny=True, weights_path=None)
    software = run_canary_amp_software_preflight(
        cfg=cfg, model=model, device=torch.device("cpu"), use_amp=False
    )
    assert software["ok"] is True
    assert software["eigvalsh_calls_during_optimizer_step"] == 0
    assert software["eigvalsh"]["is_float32"] is True
    assert int(software["eigvalsh"]["max_dim"]) <= 16
    assert software["n_real_unlabelled_studies_pixels_opened"] == 0


def test_empty_batch_and_missing_checkpoint_keys_fail_closed():
    with pytest.raises(Phase4FCanaryError) as exc:
        assert_not_empty_batch(torch.zeros(0, 2, 1, 4, 4))
    assert exc.value.status == STATUS_EMPTY
    with pytest.raises(Phase4FCanaryError) as exc2:
        assert_checkpoint_complete({"encoder": {}})
    assert exc2.value.status == STATUS_RESUME
    cfg = load_canary_config()
    with pytest.raises(Phase4FCanaryError) as exc3:
        assert_resume_hashes(
            {key: 0 for key in REQUIRED_CHECKPOINT_KEYS}
            | {"protocol": PROTOCOL, "recipe_sha256": "dead", "config_sha256": "dead", "source_sha256": "dead"},
            cfg,
        )
    assert exc3.value.status == STATUS_RESUME


def test_interrupted_matches_uninterrupted_synthetic(tmp_path):
    cfg = load_canary_config()
    unint = run_uninterrupted_synthetic(
        cfg,
        output_dir=tmp_path / "unint",
        steps=4,
        n_studies=8,
        batch_size=4,
    )
    interrupted = run_interrupted_synthetic(
        cfg,
        output_dir=tmp_path / "int",
        steps=4,
        checkpoint_at=2,
        n_studies=8,
        batch_size=4,
    )
    end = interrupted["end"]
    assert unint["completed_step"] == 4
    assert interrupted["mid"]["completed_step"] == 2
    assert end["completed_step"] == 4
    assert end["resumed_step"] == 2
    assert end["checkpoint_round_trip"]["encoder_max_abs"] == 0.0
    assert len(unint["losses"]) == 4
    assert len(end["losses"]) == 4
    for left, right in zip(unint["losses"], end["losses"]):
        assert abs(left - right) < 1e-5


def test_subprocess_resume_is_fresh_process(tmp_path):
    cfg = load_canary_config()
    overrides = {
        "n_canary_studies": 8,
        "max_optimizer_steps": 4,
        "checkpoint_at_step": 2,
        "batch_size": 4,
        "tiny_image_size": 16,
        "tiny_num_slices": 2,
        "require_cuda_for_scientific_run": False,
    }
    mid = run_phase4f_ssl_canary(
        output_dir=tmp_path / "sub",
        execute_training=True,
        allow_tiny_encoder_for_tests=True,
        synthetic_tensors=True,
        segment="train_to_mid",
        config_overrides=overrides,
    )
    assert mid["status"] == "SSL_CANARY_MID_CHECKPOINT"
    script = tmp_path / "resume.py"
    script.write_text(
        "from pathlib import Path\n"
        "from knee.phase4f_ssl_canary import run_phase4f_ssl_canary\n"
        f"out = Path({str(tmp_path / 'sub')!r})\n"
        "result = run_phase4f_ssl_canary("
        "output_dir=out, execute_training=True, allow_tiny_encoder_for_tests=True, "
        "synthetic_tensors=True, segment='resume_to_end', "
        f"config_overrides={overrides!r})\n"
        "print(result['status'])\n"
        "print(result['training']['resumed_step'])\n"
        "print(result['training']['completed_step'])\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
        env={**dict(**{k: v for k, v in __import__("os").environ.items()}), "PYTHONPATH": str(REPO_ROOT)},
    )
    assert proc.returncode == 0, proc.stderr
    lines = [line.strip() for line in proc.stdout.strip().splitlines() if line.strip()]
    assert STATUS_PASS in lines[0]
    assert lines[1] == "2"
    assert lines[2] == "4"


def test_hidden_full_oof_internet_and_collapse(tmp_path):
    hidden = run_phase4f_ssl_canary(
        data_root=tmp_path / "unit" / "test_series" / "hidden",
        output_dir=tmp_path / "out-hidden",
        allow_tiny_encoder_for_tests=True,
        synthetic_tensors=False,
    )
    assert hidden["status"] == STATUS_HIDDEN
    full = run_phase4f_ssl_canary(
        output_dir=tmp_path / "out-full",
        run_full_unlabeled=True,
        allow_tiny_encoder_for_tests=True,
        synthetic_tensors=True,
    )
    assert full["status"] == STATUS_FULL
    oof = run_phase4f_ssl_canary(output_dir=tmp_path / "out-oof", run_oof=True)
    assert oof["status"] == STATUS_OOF
    net = run_phase4f_ssl_canary(output_dir=tmp_path / "out-net", internet_enabled=True)
    assert net["status"] == STATUS_INTERNET
    collapsed = run_phase4f_ssl_canary(
        output_dir=tmp_path / "collapse",
        allow_tiny_encoder_for_tests=True,
        synthetic_tensors=True,
        collapse_fixture=True,
        config_overrides={
            "n_canary_studies": 8,
            "max_optimizer_steps": 4,
            "checkpoint_at_step": 2,
            "batch_size": 4,
        },
    )
    assert collapsed["status"] == STATUS_COLLAPSE
    assert collapsed["full_ssl_run_authorized"] is False
    assert collapsed["n_real_unlabelled_studies_pixels_opened"] == 0


def test_fixture_dicom_canary_does_not_log_ids(tmp_path):
    root = _mixed_root(tmp_path / "mix")
    result = run_phase4f_ssl_canary(
        data_root=root,
        output_dir=tmp_path / "dicom",
        allow_tiny_encoder_for_tests=True,
        synthetic_tensors=False,
        config_overrides={
            "n_canary_studies": 4,
            "max_optimizer_steps": 4,
            "checkpoint_at_step": 2,
            "batch_size": 2,
            "num_slices": 2,
            "image_size": 16,
        },
    )
    assert result["status"] == STATUS_PASS
    assert result["training_run"] is True
    assert result["n_real_unlabelled_studies_pixels_opened"] == 0
    assert result["cohort"]["n_selected"] == 4
    assert result["cohort"]["n_gold_intersection"] == 0
    blob = json.dumps(result)
    for i in range(10):
        assert f"study-{i:03d}" not in blob
        assert f"series-sag-{i:03d}" not in blob
        assert f"PAT-{i:03d}" not in blob
    assert REPORT_LABEL_LEAK_TOKEN not in blob
    assert "FINDINGS:" not in blob
    assert result["UNLABELED_IMAGE_SSL_EXECUTION_ALLOWED"] is False
    assert result["full_ssl_run_authorized"] is False
    assert result["training"]["resumed_step"] == 2
    assert result["training"]["completed_step"] == 4


def test_protocol_notebook_allowlist_and_privacy():
    cfg = load_canary_config()
    text = render_canary_protocol_markdown(cfg)
    protocol = (REPO_ROOT / "reports" / "phase4f_ssl_canary_protocol.md").read_text(encoding="utf-8")
    run = (REPO_ROOT / "reports" / "phase4f_ssl_canary_run.md").read_text(encoding="utf-8")
    for blob in (text, protocol, run):
        assert KERNEL_SLUG in blob
        assert PROTOCOL in blob
        assert "UNLABELED_IMAGE_SSL_EXECUTION_ALLOWED=false" in blob
        assert REPORT_LABEL_LEAK_TOKEN not in blob
        assert "FINDINGS:" not in blob
        assert "DummySeriesDesc" not in blob
        for uid in REPORT_LABEL_GOLD_UIDS + REPORT_LABEL_WEAK_UIDS:
            assert uid not in blob
        for patient in REPORT_LABEL_PATIENTS:
            assert patient not in blob
    assert "SSL_CANARY_FAIL" in run
    assert "n_real_unlabelled_studies_pixels_opened: 64" in run
    assert "SSL_CANARY_GPU_PREFLIGHT_FAIL" in run
    assert "nonfinite_gradient" in run
    assert would_include("knee/phase4f_ssl_canary.py")
    assert would_include("scripts/run_phase4f_ssl_canary.py")
    assert would_include("configs/unlabeled_image_simsiam_ssl_canary_v1.yaml")
    assert would_include("notebooks/14_kaggle_unlabeled_image_ssl_canary.ipynb")
    assert not would_include("reports/phase4f_ssl_canary_protocol.md")
    assert not would_include("reports/phase4f_ssl_canary_run.md")
    nb = json.loads(
        (REPO_ROOT / "notebooks" / "14_kaggle_unlabeled_image_ssl_canary.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join("".join(cell.get("source") or []) for cell in nb["cells"])
    lowered = source.lower()
    assert "pip install" not in lowered
    assert "competitions submit" not in lowered
    assert "segment=\"train_to_mid\"" in source
    assert "segment=\"resume_to_end\"" in source
    assert "run_canary_gpu_preflight" in source
    assert source.index("run_canary_gpu_preflight") < source.index('segment="train_to_mid"')
    assert "SSL_CANARY_GPU_PREFLIGHT_FAIL" in source
    assert "n_report_text_rows_processed" in source
    assert "n_second_reader_rows_processed" in source
    assert "notebook_07_executed=false" in source
    assert "notebook_13_executed=false" in source
    assert "UNLABELED_IMAGE_SSL_EXECUTION_ALLOWED" in source
    assert KERNEL_SLUG in source
    assert "DO_NOT_SUBMIT" in source
    assert "DO_NOT_LAUNCH_FULL_SSL" in source
    meta = (nb.get("metadata") or {}).get("kaggle") or {}
    assert meta.get("isInternetEnabled") is False
    assert meta.get("gpu") is True
    v1_nb = (REPO_ROOT / "notebooks" / "13_kaggle_unlabeled_image_ssl_v1.ipynb").read_text(
        encoding="utf-8"
    )
    assert "execute_training=False" in v1_nb
