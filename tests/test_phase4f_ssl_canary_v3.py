"""Phase 4F.3 documented-AMP SSL canary: frozen 4F.1 pins, AMP-safe loss, applied steps."""

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

from knee.config import OFFICIAL_TARGET_COLUMNS, REPORT_COLUMN, STUDY_ID_COLUMN
from knee.gold58_oof import IMAGENET_SHA256, SERIES_RULE_V2, set_deterministic_seeds
from knee.kaggle_bundle import sha256_file, would_include
from knee.phase4f_ssl import (
    FROZEN_V1_RECIPE_SHA256,
    FROZEN_V14,
    FROZEN_V2_CONFIG_SHA256,
    GOLD58_SSL_OOF_EXECUTION_ALLOWED,
    UNLABELED_IMAGE_SSL_EXECUTION_ALLOWED,
)
from knee.phase4f_ssl_canary_v3 import (
    AUTHORIZED_BATCH,
    AUTHORIZED_CHECKPOINT_STEP,
    AUTHORIZED_N_STUDIES,
    AUTHORIZED_STEPS,
    FROZEN_CANARY_V1_CONFIG_SHA256,
    FROZEN_CANARY_V2_CONFIG_SHA256,
    FROZEN_SSL_V1_CONFIG_SHA256,
    KERNEL_SLUG,
    PROTOCOL,
    REQUIRED_CHECKPOINT_KEYS,
    STATUS_COLLAPSE,
    STATUS_EMPTY,
    STATUS_FAIL,
    STATUS_FULL,
    STATUS_GOLD,
    STATUS_GPU_PREFLIGHT,
    STATUS_HIDDEN,
    STATUS_INTERNET,
    STATUS_LABELS,
    STATUS_OOF,
    STATUS_PASS,
    STATUS_PREPROCESS,
    STATUS_RESUME,
    UNLABELED_IMAGE_SSL_CANARY_V3_EXECUTION_ALLOWED,
    CanaryUnlabeledStudySliceDataset,
    DocumentedAmpScaler,
    Phase4FCanaryError,
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
    project_full_run_from_canary,
    render_canary_protocol_markdown,
    run_canary_amp_software_preflight,
    run_canary_gpu_preflight,
    run_interrupted_synthetic,
    run_phase4f_ssl_canary_v3,
    run_uninterrupted_synthetic,
    select_canary_study_ids,
    simsiam_loss_amp_safe,
    two_views_canary,
    _make_optimizer_bundle,
    _one_logical_optimizer_step,
    _one_optimizer_step,
    exception_diagnostics,
    is_documented_amp_order,
    skip_reason_from_diagnostics,
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


def test_v1_and_4f1_pins_are_frozen_and_v2_is_versioned():
    assert sha256_file(REPO_ROOT / "configs" / "unlabeled_image_simsiam_ssl_v1.yaml") == FROZEN_SSL_V1_CONFIG_SHA256
    assert (
        sha256_file(REPO_ROOT / "configs" / "unlabeled_image_simsiam_ssl_canary_v1.yaml")
        == FROZEN_CANARY_V1_CONFIG_SHA256
    )
    assert (
        sha256_file(REPO_ROOT / "configs" / "unlabeled_image_simsiam_ssl_canary_v2.yaml")
        == FROZEN_CANARY_V2_CONFIG_SHA256
    )
    assert (
        sha256_file(REPO_ROOT / "configs" / "unlabeled_image_simsiam_ssl_canary_v3.yaml")
        == "7f8eb3b0143441d96bf9cb445ecf12a2d22bfdae6f1d9a49f3603ab41374cb4d"
    )
    assert sha256_file(REPO_ROOT / "configs" / "gold58_image_resnet18_oof_v1.yaml") == FROZEN_V1_RECIPE_SHA256
    assert sha256_file(REPO_ROOT / "configs" / "gold58_image_resnet18_oof_v2.yaml") == FROZEN_V2_CONFIG_SHA256
    assert sha256_file(REPO_ROOT / "configs" / "second_reader_pilot_v1_4.yaml") == FROZEN_V14
    cfg = load_canary_config()
    assert cfg["protocol"] == PROTOCOL == "unlabeled-image-simsiam-ssl-canary-v3"
    assert cfg["kernel_slug"] == KERNEL_SLUG == "mahdimzafari/rsna-knee-unlabeled-image-ssl-canary-v3"
    assert cfg["series_selection_rule"] == SERIES_RULE_V2
    assert cfg["imagenet_weights_sha256"] == IMAGENET_SHA256
    assert cfg["n_canary_studies"] == AUTHORIZED_N_STUDIES
    assert cfg["max_optimizer_steps"] == AUTHORIZED_STEPS
    assert cfg["checkpoint_at_step"] == AUTHORIZED_CHECKPOINT_STEP
    assert cfg["batch_size"] == AUTHORIZED_BATCH
    assert cfg["amp_scaler_init_scale"] == 1024.0
    assert cfg["max_consecutive_amp_backoffs"] == 3
    assert float(cfg["amp_loss_eps"]) == 1e-6
    assert cfg["gpu_preflight_batch_size"] == 8
    assert cfg["gpu_preflight_num_slices"] == 8
    assert cfg["gpu_preflight_image_size"] == 224
    assert cfg["collapse_encoder_std_min"] == 0.001
    assert cfg["safety_factor"] == 1.25
    assert cfg["shutdown_reserve_seconds"] == 3600
    assert cfg["allow_fp32_fallback"] is True
    assert UNLABELED_IMAGE_SSL_EXECUTION_ALLOWED is False
    assert GOLD58_SSL_OOF_EXECUTION_ALLOWED is False
    assert UNLABELED_IMAGE_SSL_CANARY_V3_EXECUTION_ALLOWED is True
    v1 = (REPO_ROOT / "reports" / "phase4f_ssl_canary_run.md").read_text(encoding="utf-8")
    assert "SSL_CANARY_FAIL" in v1
    assert "SSL_CANARY_GPU_PREFLIGHT_FAIL" in v1
    assert "nonfinite_gradient" in v1
    assert "n_real_unlabelled_studies_pixels_opened: 64" in v1


def test_amp_safe_loss_is_float32_and_does_not_call_frozen_v1(monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("frozen_v1_simsiam_loss_called")

    monkeypatch.setattr("knee.phase4f_ssl_canary_v3.simsiam_loss", boom)
    p1 = torch.randn(4, 8, dtype=torch.float16, requires_grad=True)
    z1 = torch.randn(4, 8, dtype=torch.float16, requires_grad=True)
    p2 = torch.randn(4, 8, dtype=torch.float16, requires_grad=True)
    z2 = torch.randn(4, 8, dtype=torch.float16, requires_grad=True)
    loss = simsiam_loss_amp_safe(p1, z2, p2, z1, eps=1e-6)
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    loss.backward()
    assert p1.grad is not None and torch.isfinite(p1.grad).all()
    assert z1.grad is None
    assert z2.grad is None

    cfg = load_canary_config()
    device = torch.device("cpu")
    model = build_canary_model(cfg, tiny=True, weights_path=None).to(device)
    optimizer, scheduler, scaler, _use_amp = _make_optimizer_bundle(model, cfg, device=device)
    assert float(cfg["amp_scaler_init_scale"]) == 1024.0
    if scaler.is_enabled():
        assert float(scaler.get_scale()) == 1024.0
    generator = torch.Generator().manual_seed(0)
    batch = torch.rand(4, 2, 1, 16, 16)
    loss_v, grad, diag = _one_optimizer_step(
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
    assert math.isfinite(loss_v)
    assert math.isfinite(grad)
    assert str(diag["amp"]["loss_dtype"]) in {"torch.float32", "float32"}
    assert diag["amp"]["applied"] is True
    assert diag["amp"]["n_backoffs"] == 0


def test_applied_step_accounts_overflow_retry_without_advancing_scheduler():
    cfg = load_canary_config()
    device = torch.device("cpu")
    model = build_canary_model(cfg, tiny=True, weights_path=None).to(device)
    init_state = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
    optimizer, scheduler, scaler, _use_amp = _make_optimizer_bundle(model, cfg, device=device)
    generator = torch.Generator().manual_seed(1)
    batch = torch.rand(4, 2, 1, 16, 16)
    lr_before = float(optimizer.param_groups[0]["lr"])
    result = _one_logical_optimizer_step(
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
        force_overflow_attempts=1,
        fail_status=STATUS_FAIL,
    )
    assert result["applied"] is True
    assert result["n_backoffs"] == 1
    assert result["n_attempts"] == 2
    assert float(optimizer.param_groups[0]["lr"]) != lr_before
    after_retry = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
    changed = any(not torch.equal(after_retry[key], init_state[key]) for key in init_state)
    assert changed is True

    model.load_state_dict(init_state)
    model.to(device)
    optimizer2, scheduler2, scaler2, _ = _make_optimizer_bundle(model, cfg, device=device)
    generator2 = torch.Generator().manual_seed(1)
    clean = _one_logical_optimizer_step(
        model=model,
        optimizer=optimizer2,
        scheduler=scheduler2,
        scaler=scaler2,
        batch=batch,
        generator=generator2,
        cfg=cfg,
        device=device,
        use_amp=False,
        collapse_fixture=False,
        force_overflow_attempts=0,
        fail_status=STATUS_FAIL,
    )
    assert clean["applied"] is True
    assert clean["n_backoffs"] == 0
    for key, tensor in model.state_dict().items():
        assert torch.allclose(tensor.cpu(), after_retry[key], atol=1e-6)


def test_retry_exhaustion_fails_closed_without_optimizer_or_scheduler_step():
    cfg = load_canary_config()
    device = torch.device("cpu")
    model = build_canary_model(cfg, tiny=True, weights_path=None).to(device)
    optimizer, scheduler, scaler, _ = _make_optimizer_bundle(model, cfg, device=device)
    generator = torch.Generator().manual_seed(2)
    batch = torch.rand(4, 2, 1, 16, 16)
    lr_before = float(optimizer.param_groups[0]["lr"])
    snapshot = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
    with pytest.raises(Phase4FCanaryError) as exc:
        _one_logical_optimizer_step(
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
            force_overflow_attempts=4,
            fail_status=STATUS_GPU_PREFLIGHT,
        )
    assert exc.value.status == STATUS_GPU_PREFLIGHT
    assert "amp_backoff_exhausted:4" in exc.value.reason
    assert float(optimizer.param_groups[0]["lr"]) == lr_before
    for key, tensor in model.state_dict().items():
        assert torch.equal(tensor.cpu(), snapshot[key].cpu())


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
    assert "effective_rank" not in diag
    assert "mean_abs_pairwise_cosine" not in diag


def test_probe_eigvalsh_float32_at_most_16x16(monkeypatch):
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
    for call in calls:
        assert call["dtype"] == torch.float32
        assert call["shape"] == (16, 16)
    rank = effective_rank(torch.randn(16, 2048, dtype=torch.float16))
    assert math.isfinite(rank)


def test_gpu_preflight_fails_closed_without_cuda_and_local_amp_software_passes():
    cfg = load_canary_config()
    if not torch.cuda.is_available():
        gpu = run_canary_gpu_preflight(cfg=cfg)
        assert gpu["status"] == STATUS_GPU_PREFLIGHT
        assert gpu["ok"] is False
        assert gpu["skip_reason"] == "cuda_required"
        diag = gpu.get("exception_diagnostics") or {}
        assert diag.get("exception_class") == "Phase4FCanaryError"
        assert "cuda_required" in str(diag.get("exception_message_sanitized") or "")
        assert diag.get("traceback_fingerprint")
        assert diag.get("torch_version")
        assert gpu["n_real_unlabelled_studies_pixels_opened"] == 0
        assert gpu["n_report_text_rows_processed"] == 0
        assert gpu["n_second_reader_rows_processed"] == 0
    set_deterministic_seeds(int(cfg.get("seed") or 42))
    model = build_canary_model(cfg, tiny=True, weights_path=None)
    software = run_canary_amp_software_preflight(
        cfg=cfg,
        model=model,
        device=torch.device("cpu"),
        use_amp=False,
        scientific_shape=True,
    )
    assert software["ok"] is True
    assert software["preflight_batch_shape"] == [8, 8, 1, 224, 224]
    assert software["eigvalsh_calls_during_optimizer_step"] == 0
    assert software["eigvalsh"]["is_float32"] is True
    assert int(software["eigvalsh"]["max_dim"]) <= 16
    assert software["applied_optimizer_steps"] == 1
    assert str(software["loss_dtype"]) in {"torch.float32", "float32"}
    assert software["amp_scaler_init_scale"] == 1024.0
    assert software["scaler_initial_scale"] == 1024.0
    assert software["n_real_unlabelled_studies_pixels_opened"] == 0


def test_projection_includes_amp_retry():
    cfg = load_canary_config()
    projection = project_full_run_from_canary(
        {
            "dataset_series_setup_seconds": 10.0,
            "dicom_seconds_per_study": 1.0,
            "seconds_per_optimizer_step": 2.0,
            "amp_retry_seconds_per_applied_step": 0.5,
            "checkpoint_save_seconds": 1.0,
            "checkpoint_load_seconds": 1.0,
        },
        cfg,
    )
    assert "amp_retry" in projection["formula"]
    assert projection["amp_retry_seconds_per_applied_step"] == 0.5
    without_retry = project_full_run_from_canary(
        {
            "dataset_series_setup_seconds": 10.0,
            "dicom_seconds_per_study": 1.0,
            "seconds_per_optimizer_step": 2.0,
            "amp_retry_seconds_per_applied_step": 0.0,
            "checkpoint_save_seconds": 1.0,
            "checkpoint_load_seconds": 1.0,
        },
        cfg,
    )
    assert projection["projected_seconds"] > without_retry["projected_seconds"]


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
    assert end["applied_optimizer_steps"] == 2
    assert abs(end["checkpoint_round_trip"]["encoder_max_abs"]) == 0.0
    for left, right in zip(unint["losses"], end["losses"]):
        assert abs(left - right) < 1e-5


def test_subprocess_resume_is_fresh_process(tmp_path):
    overrides = {
        "n_canary_studies": 8,
        "max_optimizer_steps": 4,
        "checkpoint_at_step": 2,
        "batch_size": 4,
        "tiny_image_size": 16,
        "tiny_num_slices": 2,
        "require_cuda_for_scientific_run": False,
    }
    mid = run_phase4f_ssl_canary_v3(
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
        "from knee.phase4f_ssl_canary_v3 import run_phase4f_ssl_canary_v3\n"
        f"out = Path({str(tmp_path / 'sub')!r})\n"
        "result = run_phase4f_ssl_canary_v3("
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
    hidden = run_phase4f_ssl_canary_v3(
        data_root=tmp_path / "unit" / "test_series" / "hidden",
        output_dir=tmp_path / "out-hidden",
        allow_tiny_encoder_for_tests=True,
        synthetic_tensors=False,
    )
    assert hidden["status"] == STATUS_HIDDEN
    full = run_phase4f_ssl_canary_v3(
        output_dir=tmp_path / "out-full",
        run_full_unlabeled=True,
        allow_tiny_encoder_for_tests=True,
        synthetic_tensors=True,
    )
    assert full["status"] == STATUS_FULL
    oof = run_phase4f_ssl_canary_v3(output_dir=tmp_path / "out-oof", run_oof=True)
    assert oof["status"] == STATUS_OOF
    net = run_phase4f_ssl_canary_v3(output_dir=tmp_path / "out-net", internet_enabled=True)
    assert net["status"] == STATUS_INTERNET
    collapsed = run_phase4f_ssl_canary_v3(
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
    assert collapsed["notebook_14_executed"] is False
    assert collapsed["notebook_15_executed"] is False


def test_fixture_dicom_canary_does_not_log_ids(tmp_path):
    root = _mixed_root(tmp_path / "mix")
    result = run_phase4f_ssl_canary_v3(
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
    assert result["training"]["applied_optimizer_steps"] == 4
    assert result["training"]["completed_step"] == 4
    assert result["training"]["resumed_step"] == 2
    blob = json.dumps(result)
    for i in range(10):
        assert f"study-{i:03d}" not in blob
        assert f"series-sag-{i:03d}" not in blob
        assert f"PAT-{i:03d}" not in blob
    assert REPORT_LABEL_LEAK_TOKEN not in blob
    assert "FINDINGS:" not in blob
    public = (tmp_path / "dicom" / "sanitized" / "phase4f_ssl_canary_v3.json").read_text(encoding="utf-8")
    assert PROTOCOL in public
    assert "FINDINGS:" not in public


def test_intensity_jitter_never_runs_on_imagenet_tensors():
    gray = torch.rand(2, 1, 8, 8)
    assert_unit_interval_for_intensity_jitter(gray)
    imagenet = apply_imagenet_last(gray)
    with pytest.raises(Phase4FCanaryError) as exc:
        assert_unit_interval_for_intensity_jitter(imagenet)
    assert exc.value.status == STATUS_PREPROCESS
    with pytest.raises(Phase4FCanaryError) as exc2:
        two_views_canary(imagenet.unsqueeze(0), generator=torch.Generator().manual_seed(0), cfg=load_canary_config())
    assert exc2.value.status == STATUS_PREPROCESS


def test_canary_dataset_rejects_imagenet_normalize_and_labels(tmp_path):
    root = _mixed_root(tmp_path / "ds")
    sag = root / "train_series" / "study-002" / "series-sag-002"
    with pytest.raises(Phase4FCanaryError) as exc:
        CanaryUnlabeledStudySliceDataset(["study-002"], [sag], imagenet_normalize=True)
    assert exc.value.status == STATUS_PREPROCESS
    with pytest.raises(Phase4FCanaryError) as exc2:
        CanaryUnlabeledStudySliceDataset(["study-002"], [sag], labels=np.zeros((1, 12)))
    assert exc2.value.status == STATUS_LABELS


def test_sha256_cohort_is_deterministic_and_gold_disjoint():
    unlabeled = [f"u-{i:03d}" for i in range(20)]
    gold = [f"g-{i:03d}" for i in range(5)]
    a = select_canary_study_ids(unlabeled, n=8, seed=42)
    b = select_canary_study_ids(unlabeled, n=8, seed=42)
    assert a == b
    expected = sorted(
        unlabeled,
        key=lambda uid: hashlib.sha256(f"42|{uid}".encode("utf-8")).hexdigest(),
    )[:8]
    assert a == expected
    assert assert_zero_gold_overlap(a, gold) == 0
    with pytest.raises(Phase4FCanaryError) as exc:
        assert_zero_gold_overlap(gold[:1] + a, gold)
    assert exc.value.status == STATUS_GOLD


def test_protocol_notebook_allowlist_and_privacy():
    cfg = load_canary_config()
    text = render_canary_protocol_markdown(cfg)
    protocol = (REPO_ROOT / "reports" / "phase4f_ssl_canary_v3_protocol.md").read_text(encoding="utf-8")
    run = (REPO_ROOT / "reports" / "phase4f_ssl_canary_v3_run.md").read_text(encoding="utf-8")
    for blob in (text, protocol, run):
        assert KERNEL_SLUG in blob
        assert PROTOCOL in blob
        assert "UNLABELED_IMAGE_SSL_EXECUTION_ALLOWED=false" in blob
        assert REPORT_LABEL_LEAK_TOKEN not in blob
        assert "FINDINGS:" not in blob
        for uid in REPORT_LABEL_GOLD_UIDS + REPORT_LABEL_WEAK_UIDS:
            assert uid not in blob
        for patient in REPORT_LABEL_PATIENTS:
            assert patient not in blob
    assert "SSL_CANARY_FAIL" in run
    assert "SSL_CANARY_GPU_PREFLIGHT_FAIL" in run
    assert "SSL_ENGINEERING_PATH_PARKED" in run
    assert "nonfinite_gradient" in run
    assert "mahdimzafari/rsna-knee-unlabeled-image-ssl-canary-v1" in run
    assert would_include("knee/phase4f_ssl_canary_v3.py")
    assert would_include("scripts/run_phase4f_ssl_canary_v3.py")
    assert would_include("configs/unlabeled_image_simsiam_ssl_canary_v3.yaml")
    assert would_include("notebooks/16_kaggle_unlabeled_image_ssl_canary_v3.ipynb")
    assert not would_include("reports/phase4f_ssl_canary_v3_protocol.md")
    assert not would_include("reports/phase4f_ssl_canary_v3_run.md")
    nb = json.loads(
        (REPO_ROOT / "notebooks" / "16_kaggle_unlabeled_image_ssl_canary_v3.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join("".join(cell.get("source") or []) for cell in nb["cells"])
    lowered = source.lower()
    assert "pip install" not in lowered
    assert "competitions submit" not in lowered
    assert 'segment="train_to_mid"' in source
    assert 'segment="resume_to_end"' in source
    assert "run_canary_gpu_preflight" in source
    assert source.index("run_canary_gpu_preflight") < source.index('segment="train_to_mid"')
    assert "SSL_CANARY_GPU_PREFLIGHT_FAIL" in source
    assert "exception_diagnostics" in source
    assert "fp32_fallback" in source
    assert "n_report_text_rows_processed" in source
    assert "n_second_reader_rows_processed" in source
    assert "notebook_07_executed=false" in source
    assert "notebook_15_executed=false" in source
    assert "UNLABELED_IMAGE_SSL_EXECUTION_ALLOWED" in source
    assert KERNEL_SLUG in source
    assert "knee/phase4f_ssl_canary_v3.py" in source
    assert "DO_NOT_SUBMIT" in source
    assert "DO_NOT_LAUNCH_FULL_SSL" in source
    meta = (nb.get("metadata") or {}).get("kaggle") or {}
    assert meta.get("isInternetEnabled") is False
    assert meta.get("gpu") is True
    prior_nb = (REPO_ROOT / "notebooks" / "15_kaggle_unlabeled_image_ssl_canary_v2.ipynb").read_text(
        encoding="utf-8"
    )
    assert "run_phase4f_ssl_canary_v2" in prior_nb
    assert "run_phase4f_ssl_canary_v3" not in prior_nb


def test_documented_amp_order_rejects_update_before_step():
    assert is_documented_amp_order(["scale", "unscale_", "step", "update"]) is True
    assert is_documented_amp_order(["scale", "unscale_", "update"]) is False
    assert is_documented_amp_order(["scale", "unscale_", "update", "step"]) is False
    assert is_documented_amp_order(["scale", "unscale_", "unscale_", "step", "update"]) is False


def test_documented_scaler_raises_on_invalid_order_and_double_unscale():
    """Reproduce the 4F.2 invalid GradScaler transition: update() before step()."""
    param = torch.nn.Parameter(torch.ones(2, requires_grad=True))
    optimizer = torch.optim.SGD([param], lr=0.1)
    inner = torch.amp.GradScaler("cuda", enabled=False, init_scale=1024.0)
    scaler = DocumentedAmpScaler(
        inner, emulate_inf_skip=True, backoff_factor=0.5, init_scale=1024.0
    )
    assert float(scaler.get_scale()) == 1024.0
    loss = param.sum()
    scaler.begin_attempt()
    scaler.scale(loss).backward()
    with pytest.raises(RuntimeError, match="scaler_update_before_step"):
        scaler.update()
    assert "step" not in scaler.attempt_calls()
    assert scaler.attempt_calls()[-1] == "update"

    scaler.begin_attempt()
    scaler.scale(loss.detach().requires_grad_(True))
    scaler.unscale_(optimizer)
    with pytest.raises(RuntimeError, match="scaler_unscale_more_than_once"):
        scaler.unscale_(optimizer)


def test_recording_scaler_and_optimizer_prove_documented_order(monkeypatch):
    recorded: list[str] = []
    cfg = load_canary_config()
    device = torch.device("cpu")
    model = build_canary_model(cfg, tiny=True, weights_path=None).to(device)
    optimizer, scheduler, scaler, _ = _make_optimizer_bundle(
        model, cfg, device=device, precision_mode="amp"
    )
    orig_step = optimizer.step

    def _record_opt_step(*args, **kwargs):
        recorded.append("optimizer.step")
        return orig_step(*args, **kwargs)

    monkeypatch.setattr(optimizer, "step", _record_opt_step)
    orig_scale = scaler.scale
    orig_unscale = scaler.unscale_
    orig_scaler_step = scaler.step
    orig_update = scaler.update

    def _scale(loss):
        recorded.append("scale")
        return orig_scale(loss)

    def _unscale(opt):
        recorded.append("unscale_")
        return orig_unscale(opt)

    def _step(opt):
        recorded.append("step")
        return orig_scaler_step(opt)

    def _update():
        recorded.append("update")
        return orig_update()

    monkeypatch.setattr(scaler, "scale", _scale)
    monkeypatch.setattr(scaler, "unscale_", _unscale)
    monkeypatch.setattr(scaler, "step", _step)
    monkeypatch.setattr(scaler, "update", _update)
    generator = torch.Generator().manual_seed(4)
    batch = torch.rand(4, 2, 1, 16, 16)
    result = _one_logical_optimizer_step(
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
        force_overflow_attempts=0,
        fail_status=STATUS_FAIL,
        precision_mode="amp",
    )
    assert result["applied"] is True
    amp_calls = [name for name in recorded if name in {"scale", "unscale_", "step", "update"}]
    assert amp_calls == ["scale", "unscale_", "step", "update"]
    assert recorded.count("unscale_") == 1
    assert recorded.index("update") > recorded.index("step")
    assert recorded.index("optimizer.step") > recorded.index("unscale_")
    assert recorded.index("optimizer.step") < recorded.index("update")


def test_exception_diagnostics_never_collapse_to_uncaught_valueerror():
    try:
        raise ValueError("/kaggle/input/secret/train.csv uid=1.2.840.113619.2.55.3 failed")
    except ValueError as exc:
        diag = exception_diagnostics(
            exc, stage="scaler_update", last_completed_stage="unscale"
        )
    assert diag["exception_class"] == "ValueError"
    assert diag["stage"] == "scaler_update"
    assert "<path>" in diag["exception_message_sanitized"]
    assert "<uid>" in diag["exception_message_sanitized"]
    assert "/kaggle/" not in diag["exception_message_sanitized"]
    assert "1.2.840." not in diag["exception_message_sanitized"]
    reason = skip_reason_from_diagnostics(diag, "uncaught_ValueError")
    assert reason != "uncaught_ValueError"
    assert "ValueError" in reason
    assert "scaler_update" in reason


def test_logical_step_records_documented_scaler_order_and_never_updates_before_step():
    cfg = load_canary_config()
    device = torch.device("cpu")
    model = build_canary_model(cfg, tiny=True, weights_path=None).to(device)
    optimizer, scheduler, scaler, _ = _make_optimizer_bundle(
        model, cfg, device=device, precision_mode="amp"
    )
    generator = torch.Generator().manual_seed(3)
    batch = torch.rand(4, 2, 1, 16, 16)
    result = _one_logical_optimizer_step(
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
        force_overflow_attempts=1,
        fail_status=STATUS_FAIL,
        precision_mode="amp",
    )
    assert result["applied"] is True
    assert result["n_backoffs"] == 1
    assert is_documented_amp_order(result["amp_call_order"]) is True
    assert result["amp_call_order"] == ["scale", "unscale_", "step", "update"]
    assert scaler.calls.count("update") == scaler.calls.count("step")
    for i, name in enumerate(scaler.calls):
        if name == "update":
            assert "step" in scaler.calls[:i]
