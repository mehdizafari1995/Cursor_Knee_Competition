"""Phase 4E visible-test submission-pipeline rehearsal.

Loads the five frozen gold58 v2 fold checkpoints, selects one series per
visible test study with ``deterministic-primary-series-v2``, averages five
fold sigmoid probabilities with equal weights, and writes a private
rehearsal CSV through ``write_competition_submission``.

Never submits to Kaggle. Never trains. Never uses reports, report-only
rows, second-reader outputs, or hidden-test labels. Never retunes v2.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .baseline import load_checkpoint
from .baseline_model import BaselineStudyModel, TinySliceEncoder
from .competition import SAMPLE_SUBMISSION_COLUMNS
from .config import NUM_TARGETS, OFFICIAL_TARGET_COLUMNS, PROJECT_ROOT, STUDY_ID_COLUMN
from .dataset import IMAGENET_MEAN, IMAGENET_STD, StudySliceDataset, collate_studies
from .gold58_oof import (
    DEFAULT_CONFIG_PATH_V2,
    FORBIDDEN_SEMANTIC_PASS,
    IMAGENET_SHA256,
    SERIES_RULE_V2,
    Gold58PreflightError,
    STATUS_PREFLIGHT,
    STATUS_SERIES,
    assert_probabilities_ok,
    inspect_series_inventory,
    load_gold58_config,
    select_deterministic_primary_series_v2,
    set_deterministic_seeds,
    _series_dir,
)
from .kaggle_audit import jsonable
from .kaggle_bundle import sha256_file
from .kaggle_runtime import (
    DATA_SOURCE_REAL_KAGGLE,
    KaggleLayout,
    infer_data_source,
    is_kaggle_runtime,
    preflight,
)
from .phase4d_readiness import SECOND_READER_SCORE_BEARING_ALLOWED
from .report_labels import (
    HARD_WEAK_LABEL_TRAINING_ALLOWED,
    SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED,
)
from .second_reader_phase46 import git_head_sha
from .submission import (
    SubmissionError,
    load_sample_submission,
    write_competition_submission,
)

PROTOCOL = "gold58-image-resnet18-submission-rehearsal-v1"
KERNEL_SLUG = "mahdimzafari/rsna-knee-gold58-submission-rehearsal-v1"
CHECKPOINT_DATASET_SLUG = "mahdimzafari/gold58-oof-v2-fold-checkpoints"
FROZEN_V2_CONFIG_SHA256 = "4b65bc0d4d56a3dd36b6898c71d630a113c99f6e4b986e51a15ae1b66e1ea093"
FROZEN_V14_SHA256 = "992af7a4b4483278445a6f3fb363c818611e79005618dff063460d0aed3bd7f7"
FROZEN_V1_RECIPE_SHA256 = "92f9d8b52d9488646a4dd19ca9740b8ae658c5f71bf6812a393b91fa3a07af9c"
FROZEN_FOLD_SHA256 = {
    "fold0": "1ea83eb228c4986655edc068c1df72b5cb38baa53d51259a7f53cc046b297dce",
    "fold1": "508f887a790e56d4eae6e9aada25757af59a99f4dee36814fcd7e54f22406a8f",
    "fold2": "ea9caa34a62434cb77741d8f437cfe15c96e4ad6fd39565e5fcdf92bf6b85e40",
    "fold3": "c5f8fb85ecab87a5c64a44d5ed73ddcacae2b1cf6671bb0a8fe3e2de44bbb2bf",
    "fold4": "4ec6e488a6801e2b3c175aec397fa6c431804f020f77eede99b34833deecad6b",
}
FOLD_FILENAMES = tuple(f"fold{i}_last.pt" for i in range(5))
N_FOLDS = 5
COMPETITION_RUNTIME_LIMIT_SECONDS = 9 * 3600
HIDDEN_TEST_N_DOCUMENTED = 1300
RUNTIME_SAFETY_FACTOR = 2.0
ROUND_TRIP_ATOL = 1.0e-5
EXPECTED_VISIBLE_TEST_N = 3
VISIBLE_TEST_N_DOCUMENTED = 3

STATUS_COMPLETE = "SUBMISSION_REHEARSAL_COMPLETE"
STATUS_CHECKPOINT = "CHECKPOINT_PREFLIGHT_FAIL"
STATUS_CHECKPOINT_HASH = "CHECKPOINT_HASH_MISMATCH"
STATUS_CONFIG = "CONFIG_HASH_MISMATCH"
STATUS_CONTRACT = "SUBMISSION_CONTRACT_FAIL"
STATUS_RUNTIME = "RUNTIME_GUARD_FAIL"
STATUS_INTERNET = "INTERNET_PREFLIGHT_FAIL"

SKIP_WALK = {
    "train_series",
    "test_series",
    "competitions",
    "rsna-knee-abnormality-detection",
    ".git",
    "__pycache__",
}


class RehearsalError(RuntimeError):
    def __init__(self, status: str, reason: str):
        super().__init__(reason)
        self.status = status
        self.reason = reason


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def equal_weight_mean(fold_probs: list[np.ndarray]) -> np.ndarray:
    """Arithmetic mean of fold sigmoid probabilities. No target weights."""
    if len(fold_probs) != N_FOLDS:
        raise RehearsalError(STATUS_CHECKPOINT, f"expected {N_FOLDS} fold probability arrays")
    stacked = np.stack([np.asarray(p, dtype=np.float64) for p in fold_probs], axis=0)
    return stacked.mean(axis=0).astype(np.float64)


def build_resnet18_architecture() -> BaselineStudyModel:
    """Uninitialized ResNet-18 matching the v2 fold checkpoint layout."""
    from torch import nn
    from torchvision.models import resnet18

    backbone = resnet18(weights=None)
    backbone.fc = nn.Identity()

    class _VerifiedEncoder(nn.Module):
        def __init__(self, net: nn.Module):
            super().__init__()
            self.backbone = net
            self.out_dim = 512

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.size(1) == 1:
                x = x.repeat(1, 3, 1, 1)
            return self.backbone(x)

    return BaselineStudyModel(_VerifiedEncoder(backbone), num_targets=NUM_TARGETS, aggregation="mean")


def locate_fold_checkpoint_dir(explicit: str | Path | None = None) -> Path | None:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    env = os.environ.get("GOLD58_V2_CHECKPOINTS")
    if env:
        candidates.append(Path(env))
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        for top in sorted(kaggle_input.iterdir()):
            if not top.is_dir() or top.name in SKIP_WALK:
                continue
            for dirpath, dirnames, filenames in os.walk(top):
                dirnames[:] = [name for name in dirnames if name not in SKIP_WALK]
                if all(name in filenames for name in FOLD_FILENAMES):
                    candidates.append(Path(dirpath))
    local = (
        PROJECT_ROOT
        / "outputs"
        / "kaggle_gold58_v2"
        / "rsna_gold58_image_oof_v2"
        / "private"
    )
    candidates.append(local)
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.is_dir() and all((path / name).is_file() for name in FOLD_FILENAMES):
            return path
    return None


def verify_fold_checkpoints(
    checkpoint_dir: Path, expected: dict[str, str] | None = None
) -> dict[str, str]:
    expected = expected if expected is not None else dict(FROZEN_FOLD_SHA256)
    got: dict[str, str] = {}
    missing: list[str] = []
    mismatched: list[str] = []
    for fold in range(N_FOLDS):
        name = f"fold{fold}_last.pt"
        path = Path(checkpoint_dir) / name
        if not path.is_file():
            missing.append(name)
            continue
        digest = sha256_file(path)
        got[f"fold{fold}"] = digest
        want = expected.get(f"fold{fold}")
        if want is not None and digest != want:
            mismatched.append(f"fold{fold}")
    if missing:
        raise RehearsalError(STATUS_CHECKPOINT, f"missing_checkpoints:{','.join(missing)}")
    if mismatched:
        raise RehearsalError(
            STATUS_CHECKPOINT_HASH, f"checkpoint_sha256_mismatch:{','.join(mismatched)}"
        )
    return got


def assert_config_pins(cfg: dict[str, Any]) -> None:
    digest = str(cfg.get("_config_sha256") or "")
    if digest != FROZEN_V2_CONFIG_SHA256:
        raise RehearsalError(
            STATUS_CONFIG,
            f"v2_config_sha256_mismatch got={digest} expected={FROZEN_V2_CONFIG_SHA256}",
        )
    if str(cfg.get("series_selection_rule")) != SERIES_RULE_V2:
        raise RehearsalError(STATUS_CONFIG, "series_selection_rule is not frozen v2")
    if str(cfg.get("interpolation")) != "bilinear":
        raise RehearsalError(STATUS_CONFIG, "interpolation is not bilinear")
    if not bool(cfg.get("imagenet_normalize")):
        raise RehearsalError(STATUS_CONFIG, "imagenet_normalize must be true")
    if int(cfg.get("num_slices") or 0) != 8:
        raise RehearsalError(STATUS_CONFIG, "num_slices is not 8")
    if str(cfg.get("aggregation") or "mean") != "mean":
        raise RehearsalError(STATUS_CONFIG, "aggregation is not mean")


def project_runtime(
    *,
    seconds_per_study: float,
    n_studies: int,
    limit_seconds: float = COMPETITION_RUNTIME_LIMIT_SECONDS,
    n_hidden_documented: int = HIDDEN_TEST_N_DOCUMENTED,
    safety_factor: float = RUNTIME_SAFETY_FACTOR,
) -> dict[str, Any]:
    if seconds_per_study <= 0 or not np.isfinite(seconds_per_study):
        raise RehearsalError(STATUS_RUNTIME, "seconds_per_study is not a finite positive value")
    projected_this_run = float(seconds_per_study) * int(n_studies) * float(safety_factor)
    projected_hidden = float(seconds_per_study) * int(n_hidden_documented) * float(safety_factor)
    capacity = int(limit_seconds / (float(seconds_per_study) * float(safety_factor)))
    payload = {
        "seconds_per_study": float(seconds_per_study),
        "n_studies_this_run": int(n_studies),
        "safety_factor": float(safety_factor),
        "limit_seconds": float(limit_seconds),
        "projected_this_run_seconds": projected_this_run,
        "projected_hidden_seconds": projected_hidden,
        "n_hidden_documented": int(n_hidden_documented),
        "n_studies_9h_capacity": capacity,
        "hidden_scoring_capacity_ok": bool(capacity >= n_hidden_documented),
        "this_run_fits_limit": bool(projected_this_run <= limit_seconds),
    }
    if projected_this_run > limit_seconds:
        raise RehearsalError(
            STATUS_RUNTIME,
            (
                f"projected_this_run_seconds={projected_this_run:.1f} exceeds "
                f"limit_seconds={limit_seconds}"
            ),
        )
    return payload


def _peak_gpu_bytes() -> int:
    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.max_memory_allocated())


def _gpu_name() -> str | None:
    if not torch.cuda.is_available():
        return None
    return str(torch.cuda.get_device_name(0))


@torch.no_grad()
def _predict_one_model(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> tuple[list[str], np.ndarray]:
    model.eval()
    ids: list[str] = []
    probs: list[np.ndarray] = []
    for batch in loader:
        images = batch["image"].to(device)
        logits = model(images)
        p = torch.sigmoid(logits).cpu().numpy()
        ids.extend(batch[STUDY_ID_COLUMN])
        probs.append(p)
    if not ids:
        return [], np.zeros((0, NUM_TARGETS), dtype=np.float64)
    return ids, np.concatenate(probs, axis=0)


def _load_fold_model(
    path: Path,
    device: torch.device,
    *,
    tiny: bool,
    in_channels: int,
) -> torch.nn.Module:
    if tiny:
        model = BaselineStudyModel(
            TinySliceEncoder(in_channels=in_channels),
            num_targets=NUM_TARGETS,
            aggregation="mean",
        )
    else:
        model = build_resnet18_architecture()
    load_checkpoint(model, path, device)
    return model.to(device)


def _public_strip(payload: dict[str, Any]) -> dict[str, Any]:
    banned = {
        "predictions",
        "study_ids",
        "series_lookup",
        "row_predictions",
        "submission_rows",
    }
    return {key: value for key, value in payload.items() if key not in banned}


def _base_payload(status: str, reason: str | None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "protocol": PROTOCOL,
        "status": status,
        "ok": status == STATUS_COMPLETE,
        "skip_reason": reason,
        "kernel_slug": KERNEL_SLUG,
        "checkpoint_dataset_slug": CHECKPOINT_DATASET_SLUG,
        "HARD_WEAK_LABEL_TRAINING_ALLOWED": HARD_WEAK_LABEL_TRAINING_ALLOWED,
        "SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED": SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED,
        "SECOND_READER_SCORE_BEARING_ALLOWED": SECOND_READER_SCORE_BEARING_ALLOWED,
        "gold_second_reader_rerun_authorized": False,
        "forbidden_outcome_not_used": FORBIDDEN_SEMANTIC_PASS,
        "training_run": False,
        "inference_on_hidden_test": False,
        "hidden_test_available": False,
        "visible_test_only": True,
        "kaggle_kernel_launched": False,
        "competition_submission_written": False,
        "consumes_submission_slot": False,
        "notebook_07_executed": False,
        "notebook_09_executed": False,
        "notebook_10_executed": False,
        "notebook_11_executed": False,
        "n_report_only_studies_processed": 0,
        "use_second_reader": False,
        "use_weak_labels": False,
        "target_specific_weights": False,
        "oof_derived_calibration": False,
        "series_selection_rule": SERIES_RULE_V2,
        "utc": _utc(),
        "git_head_sha": git_head_sha(),
    }
    if extra:
        payload.update(extra)
    return jsonable(_public_strip(payload))


def _write_public(output_dir: Path, payload: dict[str, Any]) -> None:
    public_dir = output_dir / "sanitized"
    public_dir.mkdir(parents=True, exist_ok=True)
    path = public_dir / "phase4e_rehearsal.json"
    path.write_text(json.dumps(jsonable(_public_strip(payload)), indent=2, sort_keys=True) + "\n")


def render_protocol_markdown() -> str:
    return "\n".join(
        [
            "# Phase 4E gold58 submission-pipeline rehearsal",
            "",
            "Predeclared before execution. No hyperparameter search. No v2 retuning.",
            "Does not edit Phase 4D.1/v1 or Phase 4D.2/v2 history.",
            "Does not create a Kaggle competition submission.",
            "",
            f"- protocol: `{PROTOCOL}`",
            f"- kernel: `{KERNEL_SLUG}`",
            f"- checkpoints: five frozen v2 fold last-epoch files; dataset `{CHECKPOINT_DATASET_SLUG}`",
            f"- series rule: `{SERIES_RULE_V2}` on test metadata/files before predictions",
            "- preprocess: bilinear 224, percentiles [1.0, 99.0], ImageNet mean/std, 8 linspace slices, mean aggregation",
            "- ensemble: equal-weight arithmetic mean of five fold sigmoid probabilities",
            "- writer: `knee.submission.write_competition_submission`",
            "- visible/example test only; hidden test is unavailable outside official scoring",
            f"- runtime guard: fail-closed if projected this-run time exceeds {COMPETITION_RUNTIME_LIMIT_SECONDS} s",
            "- no report text, report-only labels, second reader, target weights, sign flips, or OOF calibration",
            "",
        ]
    )


def run_phase4e_rehearsal(
    *,
    data_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    config_path: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
    execute_inference: bool = False,
    allow_tiny_encoder_for_tests: bool = False,
    verify_frozen_hashes: bool = True,
    expected_fold_sha256: dict[str, str] | None = None,
    require_cuda: bool | None = None,
    layout: KaggleLayout | None = None,
    internet_enabled: bool | None = None,
    config_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    assert HARD_WEAK_LABEL_TRAINING_ALLOWED is False
    assert SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED is False
    assert SECOND_READER_SCORE_BEARING_ALLOWED is False

    cfg = load_gold58_config(config_path or DEFAULT_CONFIG_PATH_V2)
    if config_overrides:
        cfg.update(config_overrides)
    out_dir = Path(output_dir or (PROJECT_ROOT / "outputs" / "rsna_gold58_submission_rehearsal"))
    out_dir.mkdir(parents=True, exist_ok=True)
    layout = layout or KaggleLayout(output_dirname="rsna_gold58_submission_rehearsal")
    pre = preflight(layout=layout, data_root=data_root, output_dir=out_dir)

    def _payload(status: str, reason: str | None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        result = _base_payload(status, reason, extra)
        _write_public(out_dir, result)
        return result

    if internet_enabled is True:
        return _payload(STATUS_INTERNET, "internet_must_be_off")

    try:
        if verify_frozen_hashes:
            assert_config_pins(cfg)
    except RehearsalError as exc:
        return _payload(exc.status, exc.reason)

    if not pre.ok or pre.data_root is None:
        return _payload(STATUS_PREFLIGHT, pre.message or "DATA_NOT_AVAILABLE")

    root = Path(pre.data_root)
    data_source = infer_data_source(root, layout=layout)
    sample_path = root / "sample_submission.csv"
    test_path = root / "test.csv"
    series_path = root / "test_series.csv"
    if not sample_path.is_file() or not test_path.is_file() or not series_path.is_file():
        return _payload(STATUS_PREFLIGHT, "visible_test_tables_missing", extra={"data_source": data_source})

    try:
        sample = load_sample_submission(sample_path)
    except SubmissionError as exc:
        return _payload(STATUS_CONTRACT, str(exc), extra={"data_source": data_source})

    test = pd.read_csv(test_path)
    if STUDY_ID_COLUMN not in test.columns:
        return _payload(STATUS_CONTRACT, "test.csv missing StudyInstanceUID")
    sample_ids = sample[STUDY_ID_COLUMN].astype(str).tolist()
    test_ids = test[STUDY_ID_COLUMN].astype(str).tolist()
    if sample_ids != test_ids:
        return _payload(
            STATUS_CONTRACT,
            "test.csv StudyInstanceUID order does not match sample_submission.csv",
            extra={"n_sample_rows": len(sample_ids), "n_test_rows": len(test_ids), "data_source": data_source},
        )
    if len(set(sample_ids)) != len(sample_ids):
        return _payload(STATUS_CONTRACT, "duplicate StudyInstanceUID in sample_submission.csv")

    n_visible = len(sample_ids)
    hidden_available = data_source == DATA_SOURCE_REAL_KAGGLE and n_visible > VISIBLE_TEST_N_DOCUMENTED
    if hidden_available:
        # Official scoring replaces the example fixture. This rehearsal must not run that path.
        return _payload(
            STATUS_PREFLIGHT,
            "hidden_test_inferred_from_row_count; rehearsal is visible-fixture only",
            extra={"n_sample_rows": n_visible, "data_source": data_source},
        )

    series = pd.read_csv(series_path)
    try:
        inventory = inspect_series_inventory(series, root, split="test")
        series_lookup, series_aggregates = select_deterministic_primary_series_v2(
            inventory, sample_ids
        )
    except Gold58PreflightError as exc:
        extra = {
            "n_visible_test_studies": n_visible,
            "data_source": data_source,
            "hidden_test_available": False,
        }
        return _payload(exc.status, exc.reason, extra=extra)

    ckpt_dir = locate_fold_checkpoint_dir(checkpoint_dir)
    if ckpt_dir is None:
        return _payload(STATUS_CHECKPOINT, "fold_checkpoints_not_found")
    try:
        expected = None if not verify_frozen_hashes else (expected_fold_sha256 or dict(FROZEN_FOLD_SHA256))
        if not verify_frozen_hashes:
            expected = expected_fold_sha256
        fold_hashes = verify_fold_checkpoints(ckpt_dir, expected=expected)
    except RehearsalError as exc:
        return _payload(exc.status, exc.reason, extra={"data_source": data_source})

    if not execute_inference:
        payload = _payload(
            "PREFLIGHT_PASS_INFERENCE_NOT_EXECUTED",
            "execute_inference=false",
            extra={
                "ok": True,
                "n_visible_test_studies": n_visible,
                "n_models_loaded": 0,
                "series_selection_aggregates": series_aggregates,
                "checkpoint_sha256": fold_hashes,
                "config_sha256": cfg["_config_sha256"],
                "data_source": data_source,
                "imagenet_weights_sha256": IMAGENET_SHA256,
            },
        )
        payload["ok"] = True
        _write_public(out_dir, payload)
        return payload

    cuda_needed = True if require_cuda is None else require_cuda
    scientific = not allow_tiny_encoder_for_tests
    if scientific and cuda_needed and not torch.cuda.is_available():
        return _payload(STATUS_PREFLIGHT, "cuda_required", extra={"data_source": data_source})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    set_deterministic_seeds(int(cfg["seed"]))
    in_channels = 3 if bool(cfg.get("imagenet_normalize")) else 1
    ds_kwargs = {
        "num_slices": int(cfg["num_slices"]),
        "image_size": int(cfg["image_size"]),
        "percentiles": tuple(cfg["intensity_percentiles"]),
        "interpolation": str(cfg.get("interpolation") or "bilinear"),
        "imagenet_normalize": bool(cfg.get("imagenet_normalize")),
        "imagenet_mean": tuple(cfg.get("imagenet_mean") or IMAGENET_MEAN),
        "imagenet_std": tuple(cfg.get("imagenet_std") or IMAGENET_STD),
    }
    series_dirs = [_series_dir(root, sid, series_lookup[sid], split="test") for sid in sample_ids]
    dataset = StudySliceDataset(sample_ids, series_dirs, labels=None, **ds_kwargs)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        collate_fn=collate_studies,
        num_workers=0,
    )

    import time

    t0 = time.perf_counter()
    models_loaded = 0
    fold_prob_runs: list[list[np.ndarray]] = []
    id_runs: list[list[str]] = []
    try:
        for repeat in range(2):
            fold_probs: list[np.ndarray] = []
            ids_this: list[str] | None = None
            for fold in range(N_FOLDS):
                path = ckpt_dir / f"fold{fold}_last.pt"
                model = _load_fold_model(
                    path, device, tiny=allow_tiny_encoder_for_tests, in_channels=in_channels
                )
                models_loaded += 1
                ids, probs = _predict_one_model(model, loader, device)
                if ids != sample_ids:
                    raise RehearsalError(
                        STATUS_CONTRACT,
                        "prediction StudyInstanceUID order does not match sample_submission.csv",
                    )
                assert_probabilities_ok(probs)
                fold_probs.append(probs)
                ids_this = ids
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            fold_prob_runs.append(fold_probs)
            id_runs.append(ids_this or [])
            if repeat == 0:
                elapsed_first = time.perf_counter() - t0
                seconds_per_study = float(elapsed_first) / max(n_visible, 1)
                runtime_info = project_runtime(
                    seconds_per_study=seconds_per_study, n_studies=n_visible
                )
    except (RehearsalError, Gold58PreflightError) as exc:
        status = getattr(exc, "status", STATUS_PREFLIGHT)
        reason = getattr(exc, "reason", str(exc))
        return _payload(
            status,
            reason,
            extra={
                "n_visible_test_studies": n_visible,
                "n_models_loaded": models_loaded,
                "data_source": data_source,
                "series_selection_aggregates": series_aggregates,
            },
        )

    wall = float(time.perf_counter() - t0)
    mean_a = equal_weight_mean(fold_prob_runs[0])
    mean_b = equal_weight_mean(fold_prob_runs[1])
    assert_probabilities_ok(mean_a)
    assert_probabilities_ok(mean_b)
    max_abs = float(np.max(np.abs(mean_a - mean_b))) if mean_a.size else 0.0
    repeat_ok = bool(max_abs <= ROUND_TRIP_ATOL)

    if not repeat_ok:
        return _payload(
            STATUS_CONTRACT,
            f"deterministic_repeat_max_abs_diff={max_abs} exceeds atol={ROUND_TRIP_ATOL}",
            extra={"n_visible_test_studies": n_visible, "data_source": data_source},
        )

    predictions = pd.DataFrame({STUDY_ID_COLUMN: sample_ids})
    for i, column in enumerate(OFFICIAL_TARGET_COLUMNS):
        predictions[column] = mean_a[:, i]
    private_dir = out_dir / "private"
    private_dir.mkdir(parents=True, exist_ok=True)
    dest = private_dir / "rehearsal_submission.csv"
    try:
        aligned = write_competition_submission(predictions, sample, dest)
    except SubmissionError as exc:
        if dest.exists():
            dest.unlink()
        return _payload(STATUS_CONTRACT, str(exc), extra={"data_source": data_source})

    working_root_submission = Path("/kaggle/working/submission.csv") if is_kaggle_runtime() else None
    if working_root_submission is not None and working_root_submission.exists():
        return _payload(
            STATUS_CONTRACT,
            "refusing_kaggle_working_submission_csv",
            extra={"data_source": data_source},
        )

    values = aligned.loc[:, list(OFFICIAL_TARGET_COLUMNS)].to_numpy(dtype=np.float64)
    extra = {
        "n_visible_test_studies": n_visible,
        "n_oof_rows": n_visible,
        "n_models_loaded": N_FOLDS,
        "n_models_expected": N_FOLDS,
        "series_selection_aggregates": series_aggregates,
        "checkpoint_sha256": fold_hashes,
        "config_sha256": cfg["_config_sha256"],
        "imagenet_weights_sha256": IMAGENET_SHA256,
        "data_source": data_source,
        "device": str(device),
        "gpu_name": _gpu_name(),
        "kaggle_runtime": is_kaggle_runtime(),
        "wall_seconds": wall,
        "peak_gpu_memory_bytes": _peak_gpu_bytes(),
        "runtime_guard": runtime_info,
        "deterministic_repeat": {
            "max_abs_diff": max_abs,
            "within_tolerance": repeat_ok,
            "atol": ROUND_TRIP_ATOL,
            "n_rows": n_visible,
        },
        "rehearsal_submission": {
            "n_rows": int(len(aligned)),
            "n_columns": int(len(aligned.columns)),
            "columns": list(aligned.columns),
            "sha256": sha256_file(dest),
            "probability_min": float(np.min(values)),
            "probability_max": float(np.max(values)),
            "path_basename": dest.name,
        },
        "interpolation": "bilinear",
        "imagenet_normalize": True,
        "ensemble": "equal_weight_mean_of_five_fold_sigmoid",
        "scientific_evaluation": True,
        "training_run": False,
        "competition_submission_written": False,
        "consumes_submission_slot": False,
        "hidden_test_n_documented": HIDDEN_TEST_N_DOCUMENTED,
        "hidden_test_available": False,
        "visible_test_only": True,
        "visible_test_n_documented": VISIBLE_TEST_N_DOCUMENTED,
    }
    payload = _payload(STATUS_COMPLETE, None, extra=extra)
    payload["ok"] = True
    _write_public(out_dir, payload)
    return payload
