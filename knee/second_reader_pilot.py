"""Gold-only second-reader pilot. Never processes report-only studies."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .config import (
    OFFICIAL_TARGET_COLUMNS,
    PROJECT_ROOT,
    REPORT_COLUMN,
    STUDY_ID_COLUMN,
    TARGET_KEY_TO_OFFICIAL,
    TARGET_KEYS,
)
from .kaggle_audit import assert_no_report_text, collect_report_snippets, jsonable
from .kaggle_bundle import sha256_file
from .kaggle_runtime import (
    DATA_SOURCE_REAL_KAGGLE,
    DATA_SOURCE_SOFTWARE_FIXTURE,
    KaggleLayout,
    assert_not_kaggle_input,
    preflight,
    scientific_evaluation_for,
)
from .labels import extract_label_matrix
from .report_labels import (
    LABEL_NEGATIVE,
    LABEL_POSITIVE,
    LABEL_UNKNOWN,
    PrivacyLeakError,
    SANITIZED_HASHES_FILENAME,
    assert_no_identifiers,
    collect_identifiers,
    extract_frame_reports,
    label_to_float,
    load_extractor_config,
    split_gold_and_report_only,
    write_private_csv,
)
from .report_second_reader import (
    GenerateFn,
    assert_second_reader_has_no_gold_parameter,
    digest_map_from_config,
    load_prompt_bundle,
    locate_weights_directory,
    run_second_reader,
    verify_attached_snapshot,
)
from .second_reader_acceptance import decide_all_pilot_targets
from .second_reader_consensus import strict_agreement
from .second_reader_eval import evaluate_system_against_gold
from .second_reader_schema import (
    DEFAULT_PILOT_CONFIG_PATH,
    EVAL_PROTOCOL_VERSION,
    ONTOLOGY_VERSION,
    PILOT_VERSION,
    PROMPT_VERSION,
    SCHEMA_VERSION,
    official_name,
)

KAGGLE_PILOT_OUTPUT_DIRNAME = "rsna_second_reader_gold_pilot"
PRIVATE_DET_FILENAME = "gold_deterministic_predictions.csv"
PRIVATE_READER_FILENAME = "gold_second_reader_predictions.csv"
PRIVATE_CONSENSUS_FILENAME = "gold_consensus_predictions.csv"
PRIVATE_STATUS_FILENAME = "gold_second_reader_status.csv"
SANITIZED_SUMMARY_FILENAME = "second_reader_gold_pilot_summary.md"
SANITIZED_METRICS_FILENAME = "second_reader_gold_metrics.json"
SANITIZED_FREEZE_FILENAME = "freeze_manifest.json"
SANITIZED_RUNTIME_FILENAME = "runtime_metrics.json"
SANITIZED_ACCEPTANCE_FILENAME = "pilot_acceptance.json"


class ReportOnlyScaleError(RuntimeError):
    """The pilot attempted to score report-only studies."""


def load_pilot_config(path: str | Path | None = None) -> dict:
    config_path = Path(path or DEFAULT_PILOT_CONFIG_PATH)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("pilot config must be a mapping")
    payload["_config_path"] = str(config_path)
    payload["_config_sha256"] = sha256_file(config_path)
    return payload


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _resolve_pilot_file(path_value: str | Path | None, default: Path) -> Path:
    """Resolve freeze-file paths against the package root when relative."""
    path = Path(path_value) if path_value else Path(default)
    if path.is_file():
        return path
    if not path.is_absolute():
        candidate = PROJECT_ROOT / path
        if candidate.is_file():
            return candidate
    return path


def freeze_manifest(config: dict, *, prompt: dict) -> dict:
    """Record versions and hashes before metrics are computed."""
    prompt_path = _resolve_pilot_file(
        config.get("prompt_path"), PROJECT_ROOT / "configs" / "second_reader_prompt.yaml"
    )
    schema_path = _resolve_pilot_file(
        config.get("schema_path"), PROJECT_ROOT / "configs" / "second_reader_schema.json"
    )
    ontology_path = _resolve_pilot_file(
        config.get("ontology_path"), PROJECT_ROOT / "literature" / "ontology.yaml"
    )
    extractor_cfg_path = PROJECT_ROOT / "configs" / "report_labeler.yaml"
    files = {
        "prompt": str(prompt_path),
        "schema": str(schema_path),
        "ontology": str(ontology_path),
        "pilot_config": str(config.get("_config_path") or DEFAULT_PILOT_CONFIG_PATH),
        "extractor_config": str(extractor_cfg_path),
    }
    missing = [name for name, path in files.items() if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"freeze files missing: {missing}")
    hashes = {name: sha256_file(Path(path)) for name, path in files.items()}
    selected = config.get("selected_model") or {}
    decoding = config.get("decoding") or {}
    return {
        "frozen_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "metrics_computed": False,
        "pilot_version": config.get("pilot_version") or PILOT_VERSION,
        "eval_protocol_version": config.get("eval_protocol_version") or EVAL_PROTOCOL_VERSION,
        "extractor_version": config.get("extractor_version"),
        "prompt_version": prompt.get("prompt_version") or PROMPT_VERSION,
        "schema_version": prompt.get("schema_version") or SCHEMA_VERSION,
        "ontology_version": config.get("ontology_version") or ONTOLOGY_VERSION,
        "model_id": selected.get("id"),
        "model_revision": selected.get("revision"),
        "quantization": selected.get("quantization"),
        "decoding": decoding,
        "file_sha256": hashes,
        "prompt_sha256": _sha256_text(json.dumps(prompt, sort_keys=True)),
        "config_sha256": config.get("_config_sha256"),
        "gold_only": True,
        "process_report_only_studies": False,
        "download_weights": False,
        "internet_required": False,
    }


def _empty_pred_row(study_id: str, *, system: str, version: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        STUDY_ID_COLUMN: study_id,
        "system": system,
        "version": version,
    }
    for key in TARGET_KEYS:
        official = TARGET_KEY_TO_OFFICIAL[key]
        row[official] = float("nan")
        row[f"{key}_status"] = "not_mentioned"
        row[f"{key}_confidence"] = "low"
        row[f"{key}_evidence"] = "not_mentioned"
        row[f"{key}_uncertainty"] = "not_mentioned"
        row[f"{key}_reason"] = ""
    return row


def _reader_row(study_id: str, result) -> dict[str, Any]:
    row = _empty_pred_row(
        study_id, system="second_reader", version=str(result.prompt_version or PROMPT_VERSION)
    )
    row["malformed"] = bool(result.malformed)
    row["malformed_reason"] = result.malformed_reason
    row["inference_backend"] = result.inference_backend
    row["skipped_reason"] = result.skipped_reason
    for key, pred in result.predictions.items():
        official = TARGET_KEY_TO_OFFICIAL[key]
        row[official] = label_to_float(pred.label)
        row[f"{key}_status"] = pred.status
        row[f"{key}_confidence"] = pred.confidence_category
        row[f"{key}_evidence"] = pred.evidence_category
        row[f"{key}_uncertainty"] = pred.uncertainty_or_negation_category
        row[f"{key}_reason"] = pred.evidence_class
    return row


def _consensus_row(study_id: str, det_row: pd.Series, reader_row: dict[str, Any], *, global_normal: bool) -> dict[str, Any]:
    row = _empty_pred_row(study_id, system="strict_agreement", version="strict_agreement-v1.0.0")
    row["malformed"] = bool(reader_row.get("malformed"))
    for key in TARGET_KEYS:
        official = TARGET_KEY_TO_OFFICIAL[key]
        det_value = det_row.get(official)
        if det_value == 1:
            det_label = LABEL_POSITIVE
        elif det_value == 0:
            det_label = LABEL_NEGATIVE
        else:
            det_label = LABEL_UNKNOWN
        call = strict_agreement(
            det_label,
            str(reader_row.get(f"{key}_status") or "not_mentioned"),
            deterministic_evidence=str(det_row.get(f"{key}_evidence") or ""),
            global_normal=global_normal,
            reader_enabled=True,
            reader_malformed=bool(reader_row.get("malformed")),
        )
        row[official] = label_to_float(call.label)
        row[f"{key}_status"] = call.label
        row[f"{key}_reason"] = call.reason
        row[f"{key}_evidence"] = call.reason
        row[f"{key}_uncertainty"] = "none" if call.label != LABEL_UNKNOWN else "not_mentioned"
        row[f"{key}_confidence"] = "high" if call.source == "strict_agreement" else "low"
    return row


def _gpu_memory_bytes() -> int | None:
    try:
        import torch
    except ImportError:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        return int(torch.cuda.max_memory_allocated())
    except Exception:
        return None


def _gpu_runtime_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "gpu_memory_bytes": _gpu_memory_bytes(),
        "gpu_name": None,
        "cuda_capability": None,
        "cuda_kernels_usable": False,
    }
    try:
        import torch
    except ImportError:
        return info
    try:
        if not torch.cuda.is_available():
            return info
        info["gpu_name"] = str(torch.cuda.get_device_name(0))
        major, minor = torch.cuda.get_device_capability(0)
        info["cuda_capability"] = f"{major}.{minor}"
        info["cuda_kernels_usable"] = major >= 7
    except Exception:
        return info
    return info


def render_pilot_summary(
    *,
    freeze: dict,
    comparisons: dict[str, dict],
    acceptance: dict[str, dict],
    runtime: dict,
    data_source: str,
    scientific_evaluation: bool,
    warnings: list[str],
    hashes: dict,
    n_gold: int,
    n_report_only_refused: int,
) -> str:
    lines = [
        "# Second-reader gold-pilot comparison",
        "",
        "Sanitized aggregates only. Report text, StudyInstanceUID values, and",
        "PatientID values are not included. Chain-of-thought is not stored.",
        "",
        f"- pilot_version: `{freeze.get('pilot_version')}`",
        f"- eval_protocol_version: `{freeze.get('eval_protocol_version')}`",
        f"- extractor_version: `{freeze.get('extractor_version')}`",
        f"- prompt_version: `{freeze.get('prompt_version')}`",
        f"- schema_version: `{freeze.get('schema_version')}`",
        f"- ontology_version: `{freeze.get('ontology_version')}`",
        f"- model_id: `{freeze.get('model_id')}`",
        f"- model_revision: `{freeze.get('model_revision')}`",
        f"- quantization: `{freeze.get('quantization')}`",
        f"- data_source: `{data_source}`",
        f"- scientific_evaluation: `{str(scientific_evaluation).lower()}`",
        f"- gold_only: `true`",
        f"- n_gold_studies: {n_gold}",
        f"- n_report_only_studies_processed: 0",
        f"- n_report_only_studies_refused: {n_report_only_refused}",
        f"- unknown_imputed_as_negative: `false`",
        f"- internet_used: `false`",
        f"- download_weights: `false`",
        f"- training_run: `false`",
        f"- folds_created: `false`",
        f"- freeze_before_metrics: `{str(not freeze.get('metrics_computed', True)).lower()}`",
        "",
    ]
    if not scientific_evaluation:
        lines.extend(
            [
                "## SOFTWARE FIXTURE / PENDING REAL KAGGLE GPU RUN",
                "",
                "These numbers are software verification or a pending operator run.",
                "They must not be quoted as scientific gold-label performance on the",
                "58 expert-labelled studies.",
                "",
            ]
        )
    lines.extend(
        [
            "## Runtime",
            "",
            f"- n_studies_scored: {runtime.get('n_studies_scored')}",
            f"- wall_seconds: {runtime.get('wall_seconds')}",
            f"- mean_seconds_per_study: {runtime.get('mean_seconds_per_study')}",
            f"- gpu_memory_bytes: {runtime.get('gpu_memory_bytes')}",
            f"- gpu_name: `{runtime.get('gpu_name')}`",
            f"- cuda_capability: `{runtime.get('cuda_capability')}`",
            f"- cuda_kernels_usable: `{runtime.get('cuda_kernels_usable')}`",
            f"- inference_backend: `{runtime.get('inference_backend')}`",
            f"- malformed_output_rate: {runtime.get('malformed_output_rate')}",
            "",
        ]
    )
    for system_name, payload in comparisons.items():
        lines.extend([f"## System `{system_name}`", ""])
        lines.append(
            "| Official column | coverage | pos prec | NPV | pos capture | neg capture | selective error | abstentions | malformed |"
        )
        lines.append(
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        )
        for official in OFFICIAL_TARGET_COLUMNS:
            stats = (payload.get("per_target") or {}).get(official) or {}

            def fmt(key: str) -> str:
                value = stats.get(key)
                if value is None or value != value:
                    return "n/a"
                if isinstance(value, float):
                    return f"{value:.3f}"
                return str(value)

            lines.append(
                f"| {official} | {fmt('coverage')} | {fmt('positive_precision') if stats.get('positive_precision') is not None else fmt('precision')} "
                f"| {fmt('npv')} | {fmt('positive_capture')} | {fmt('negative_capture')} "
                f"| {fmt('selective_error')} | {stats.get('n_abstentions', 'n/a')} | {fmt('malformed_output_rate')} |"
            )
        macro = payload.get("macro") or {}
        lines.extend(
            [
                "",
                f"Macro coverage: {macro.get('coverage')}",
                f"Macro precision: {macro.get('precision')}",
                f"Macro NPV: {macro.get('npv')}",
                "",
            ]
        )
    lines.extend(["## Acceptance (strict agreement)", "", "| Official column | status | precision | NPV | coverage | abstentions | reasons |", "| --- | --- | ---: | ---: | ---: | ---: | --- |"])
    for official, row in acceptance.items():
        reasons = ", ".join(row.get("reasons") or [])
        lines.append(
            f"| {official} | `{row.get('status')}` | {row.get('precision')} | {row.get('npv')} "
            f"| {row.get('coverage')} | {row.get('n_abstentions')} | {reasons} |"
        )
    lines.extend(["", "## Warnings", ""])
    if warnings:
        for item in warnings:
            lines.append(f"- {item}")
    else:
        lines.append("- none")
    lines.extend(["", "## Private output hashes (contents not committed)", "", "| file | sha256 | bytes |", "| --- | --- | ---: |"])
    for name, meta in hashes.items():
        lines.append(f"| `{name}` | `{meta.get('sha256')}` | {meta.get('bytes')} |")
    lines.extend(
        [
            "",
            "## Coordinator gates",
            "",
            "- `NOT_READY_FOR_HARD_WEAK_LABEL_TRAINING`",
            "- `READY_FOR_SELECTIVE_LABEL_REFINEMENT`",
            "- `NOT_READY_TO_SCALE_SECOND_READER_TO_REPORT_ONLY`",
            "",
            "Do not accept model-only predictions as training labels. Do not process",
            "the 4,349 report-only studies until a scientific gold pilot passes.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def run_second_reader_gold_pilot(
    *,
    data_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    layout: KaggleLayout | None = None,
    data_source: str | None = None,
    config_path: str | Path | None = None,
    extractor_config_path: str | Path | None = None,
    generate_fn: GenerateFn | None = None,
    weights_path: str | Path | None = None,
) -> dict:
    """Score the completely labelled gold studies only.

    Report-only rows are counted and refused. Gold labels never enter the reader.
    """
    assert_second_reader_has_no_gold_parameter()
    layout = layout or KaggleLayout(output_dirname=KAGGLE_PILOT_OUTPUT_DIRNAME)
    pre = preflight(
        layout=layout,
        data_root=data_root,
        output_dir=output_dir,
        data_source=data_source,
    )
    out_dir = Path(output_dir) if output_dir is not None else pre.output_dir
    assert_not_kaggle_input(out_dir, pre.layout)
    private_dir = out_dir / "private"
    sanitized_dir = out_dir / "sanitized"
    private_dir.mkdir(parents=True, exist_ok=True)
    sanitized_dir.mkdir(parents=True, exist_ok=True)

    config = load_pilot_config(config_path)
    if config.get("process_report_only_studies"):
        raise ReportOnlyScaleError("pilot config must keep process_report_only_studies false")
    prompt = load_prompt_bundle(
        _resolve_pilot_file(config.get("prompt_path"), PROJECT_ROOT / "configs" / "second_reader_prompt.yaml")
    )
    freeze = freeze_manifest(config, prompt=prompt)
    (sanitized_dir / SANITIZED_FREEZE_FILENAME).write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    source = data_source or pre.data_source
    scientific = scientific_evaluation_for(source) and source == DATA_SOURCE_REAL_KAGGLE
    if source == DATA_SOURCE_SOFTWARE_FIXTURE or generate_fn is not None:
        scientific = False

    if not pre.ok or pre.data_root is None:
        summary = render_pilot_summary(
            freeze=freeze,
            comparisons={},
            acceptance={},
            runtime={"n_studies_scored": 0, "inference_backend": "skipped"},
            data_source=source,
            scientific_evaluation=False,
            warnings=["PILOT STOPPED: required inputs missing."],
            hashes={},
            n_gold=0,
            n_report_only_refused=0,
        )
        (sanitized_dir / SANITIZED_SUMMARY_FILENAME).write_text(summary, encoding="utf-8")
        return {
            "status": "DATA_NOT_AVAILABLE",
            "scientific_evaluation": False,
            "claim_scientific_performance": False,
            "data_source": source,
            "output_dir": str(out_dir),
            "n_gold_studies": 0,
            "n_report_only_studies_processed": 0,
            "internet_used": False,
            "gpu_used": False,
            "training_run": False,
            "folds_created": False,
            "freeze": freeze,
        }

    train = pd.read_csv(Path(pre.data_root) / "train.csv")
    snippets = collect_report_snippets(train)
    identifiers = collect_identifiers(train)
    gold_frame, report_only = split_gold_and_report_only(train)
    n_report_only = int(len(report_only))
    if n_report_only and config.get("process_report_only_studies"):
        raise ReportOnlyScaleError("pilot config must keep process_report_only_studies false")

    digest_map = digest_map_from_config(config)
    if generate_fn is None:
        directory = locate_weights_directory(config.get("second_reader") or {}, weights_path)
        if source == DATA_SOURCE_REAL_KAGGLE:
            if directory is None:
                raise RuntimeError(
                    "PILOT STOPPED: checksummed Phi-3.5 snapshot is not attached"
                )
            verified = verify_attached_snapshot(directory, digest_map)
            freeze["verified_file_sha256"] = verified
            freeze["weights_directory_name"] = directory.name
            (sanitized_dir / SANITIZED_FREEZE_FILENAME).write_text(
                json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            weights_path = directory
        elif directory is not None and digest_map:
            verify_attached_snapshot(directory, digest_map)
            weights_path = directory

    extractor_cfg = load_extractor_config(extractor_config_path)
    det_pred = extract_frame_reports(gold_frame, config=extractor_cfg)

    reader_rows: list[dict[str, Any]] = []
    consensus_rows: list[dict[str, Any]] = []
    malformed_flags: list[bool] = []
    per_study_seconds: list[float] = []
    backend = "skipped"
    started = time.perf_counter()
    gold_ids = gold_frame[STUDY_ID_COLUMN].astype(str).tolist()
    reports = gold_frame[REPORT_COLUMN].tolist()
    det_indexed = det_pred.set_index(det_pred[STUDY_ID_COLUMN].astype(str))
    n_gold_loop = len(gold_ids)
    for index, (study_id, report) in enumerate(zip(gold_ids, reports), start=1):
        t0 = time.perf_counter()
        result = run_second_reader(
            report,
            config=config,
            language_supported=True,
            weights_path=weights_path,
            generate_fn=generate_fn,
            prompt=prompt,
        )
        per_study_seconds.append(time.perf_counter() - t0)
        backend = result.inference_backend
        if index == 1 or index % 5 == 0 or index == n_gold_loop:
            print(
                f"second_reader_progress={index}/{n_gold_loop} "
                f"backend={backend} last_seconds={per_study_seconds[-1]:.1f}",
                flush=True,
            )
        row = _reader_row(study_id, result)
        reader_rows.append(row)
        malformed_flags.append(bool(result.malformed))
        det_row = det_indexed.loc[study_id]
        global_normal = str(det_row.get("acl_injury_evidence") or "") == "global_normal" or any(
            str(det_row.get(f"{key}_evidence") or "") == "global_normal" for key in TARGET_KEYS
        )
        consensus_rows.append(
            _consensus_row(study_id, det_row, row, global_normal=global_normal)
        )
    wall = time.perf_counter() - started
    reader_frame = pd.DataFrame(reader_rows)
    consensus_frame = pd.DataFrame(consensus_rows)
    boot = config.get("bootstrap") or {}
    comparisons = {
        "deterministic_v2": evaluate_system_against_gold(
            gold_frame,
            det_pred,
            n_resamples=int(boot.get("n_resamples") or 1000),
            seed=int(boot.get("seed") or 42),
            ci_level=float(boot.get("ci_level") or 0.95),
            system_name="deterministic_v2",
        ),
        "second_reader": evaluate_system_against_gold(
            gold_frame,
            reader_frame,
            n_resamples=int(boot.get("n_resamples") or 1000),
            seed=int(boot.get("seed") or 42),
            ci_level=float(boot.get("ci_level") or 0.95),
            system_name="second_reader",
            malformed_by_study=malformed_flags,
        ),
        "strict_agreement": evaluate_system_against_gold(
            gold_frame,
            consensus_frame,
            n_resamples=int(boot.get("n_resamples") or 1000),
            seed=int(boot.get("seed") or 42),
            ci_level=float(boot.get("ci_level") or 0.95),
            system_name="strict_agreement",
            malformed_by_study=malformed_flags,
        ),
    }
    if generate_fn is not None:
        scientific = False
    if backend != "local_transformers":
        scientific = False
    freeze_after = dict(freeze)
    freeze_after["metrics_computed"] = True
    freeze_after["metrics_computed_after_freeze"] = True
    acceptance = decide_all_pilot_targets(
        comparisons["strict_agreement"].get("per_target") or {},
        malformed_output_rate=float(np.mean(malformed_flags) if malformed_flags else 0.0),
        config=config,
    )
    gpu_info = _gpu_runtime_info()
    runtime = {
        "n_studies_scored": len(gold_ids),
        "wall_seconds": wall,
        "mean_seconds_per_study": float(np.mean(per_study_seconds) if per_study_seconds else 0.0),
        "gpu_memory_bytes": gpu_info.get("gpu_memory_bytes"),
        "gpu_name": gpu_info.get("gpu_name"),
        "cuda_capability": gpu_info.get("cuda_capability"),
        "cuda_kernels_usable": gpu_info.get("cuda_kernels_usable"),
        "inference_backend": backend,
        "malformed_output_rate": float(np.mean(malformed_flags) if malformed_flags else 0.0),
        "n_malformed_studies": int(sum(malformed_flags)),
    }
    write_private_csv(det_pred, private_dir / PRIVATE_DET_FILENAME)
    write_private_csv(reader_frame, private_dir / PRIVATE_READER_FILENAME)
    write_private_csv(consensus_frame, private_dir / PRIVATE_CONSENSUS_FILENAME)
    status_cols = [STUDY_ID_COLUMN, "malformed", "malformed_reason", "inference_backend", "skipped_reason"]
    write_private_csv(reader_frame.loc[:, [c for c in status_cols if c in reader_frame.columns]], private_dir / PRIVATE_STATUS_FILENAME)

    hashes = {}
    for name in (
        PRIVATE_DET_FILENAME,
        PRIVATE_READER_FILENAME,
        PRIVATE_CONSENSUS_FILENAME,
        PRIVATE_STATUS_FILENAME,
    ):
        path = private_dir / name
        hashes[name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}

    warnings = [
        "Report-only studies were not processed.",
        "Model-only predictions are not training labels.",
        "conditional_sensitivity is not full-set sensitivity; use positive_capture.",
    ]
    if not scientific:
        warnings.append("scientific_evaluation=false; do not quote as 58-study performance.")
    summary = render_pilot_summary(
        freeze=freeze,
        comparisons=comparisons,
        acceptance=acceptance,
        runtime=runtime,
        data_source=source,
        scientific_evaluation=scientific,
        warnings=warnings,
        hashes=hashes,
        n_gold=len(gold_ids),
        n_report_only_refused=n_report_only,
    )
    summary_path = sanitized_dir / SANITIZED_SUMMARY_FILENAME
    metrics_path = sanitized_dir / SANITIZED_METRICS_FILENAME
    acceptance_path = sanitized_dir / SANITIZED_ACCEPTANCE_FILENAME
    runtime_path = sanitized_dir / SANITIZED_RUNTIME_FILENAME
    hashes_path = sanitized_dir / SANITIZED_HASHES_FILENAME
    freeze_path = sanitized_dir / SANITIZED_FREEZE_FILENAME
    summary_path.write_text(summary, encoding="utf-8")
    metrics_path.write_text(json.dumps(jsonable(comparisons), indent=2, sort_keys=True) + "\n")
    acceptance_path.write_text(json.dumps(jsonable(acceptance), indent=2, sort_keys=True) + "\n")
    runtime_path.write_text(json.dumps(jsonable(runtime), indent=2, sort_keys=True) + "\n")
    hashes_path.write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n")
    freeze_path.write_text(json.dumps(freeze_after, indent=2, sort_keys=True) + "\n")

    sanitized_files = [
        summary_path,
        metrics_path,
        acceptance_path,
        runtime_path,
        hashes_path,
        freeze_path,
    ]
    assert_no_report_text(sanitized_files, snippets)
    try:
        assert_no_identifiers(sanitized_files, identifiers)
    except PrivacyLeakError:
        raise

    return {
        "status": "ok",
        "scientific_evaluation": scientific,
        "claim_scientific_performance": scientific,
        "data_source": source,
        "output_dir": str(out_dir),
        "n_gold_studies": len(gold_ids),
        "n_report_only_studies_processed": 0,
        "n_report_only_studies_refused": n_report_only,
        "internet_used": False,
        "gpu_used": backend == "local_transformers",
        "gpu_required": True,
        "training_run": False,
        "folds_created": False,
        "download_weights": False,
        "inference_backend": backend,
        "malformed_output_rate": runtime["malformed_output_rate"],
        "pilot_version": freeze["pilot_version"],
        "eval_protocol_version": freeze["eval_protocol_version"],
        "extractor_version": freeze["extractor_version"],
        "freeze": freeze_after,
        "acceptance": acceptance,
        "runtime": runtime,
        "private_hashes": hashes,
        "comparisons": comparisons,
        "unknown_imputed_as_negative": False,
        "scale_to_report_only_allowed": False,
    }
