"""Synthetic internet-off second-reader canary. Never reads competition reports."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import PROJECT_ROOT, TARGET_KEYS
from .kaggle_audit import jsonable
from .report_second_reader import (
    GenerateFn,
    load_prompt_bundle,
    locate_weights_directory,
    run_second_reader,
)
from .second_reader_parser import MALFORMED_STAGES
from .second_reader_pilot import freeze_manifest, load_pilot_config
from .second_reader_schema import (
    DEFAULT_PILOT_CONFIG_PATH_V1_2,
    SCHEMA_VERSION,
    compact_output_skeleton,
)
from .second_reader_json_stop import PINNED_PHI35_TERMINATION

CANARY_VERSION = "second-reader-synthetic-canary-v1.2"
CANARY_VERSION_V1_1 = "second-reader-synthetic-canary-v1.1"
KAGGLE_CANARY_OUTPUT_DIRNAME = "rsna_second_reader_synthetic_canary"
SANITIZED_SUMMARY_FILENAME = "second_reader_synthetic_canary_summary.md"
SANITIZED_METRICS_FILENAME = "second_reader_synthetic_canary.json"
SANITIZED_FREEZE_FILENAME = "freeze_manifest.json"
SANITIZED_TERMINATION_FILENAME = "termination_inspection.json"

# Hand-written synthetic reports. Not competition gold or report-only rows.
SYNTHETIC_CANARY_EXAMPLES: tuple[dict[str, str], ...] = (
    {
        "example_id": "CANARY_SYNTHETIC_01",
        "language": "en",
        "coverage": "positive",
        "report": (
            "CANARY_SYNTHETIC_01. FINDINGS: Complete acute tear of the ACL. "
            "MCL intact. Menisci, cartilage, effusion, synovium, popliteal fossa, "
            "marrow and bone are not described further."
        ),
    },
    {
        "example_id": "CANARY_SYNTHETIC_02",
        "language": "en",
        "coverage": "negated",
        "report": (
            "CANARY_SYNTHETIC_02. FINDINGS: No ACL tear. No MCL tear. "
            "No meniscal tear. No osteoarthritis. No joint effusion. No synovitis. "
            "No Baker cyst. No bone contusion. No fracture."
        ),
    },
    {
        "example_id": "CANARY_SYNTHETIC_03",
        "language": "en",
        "coverage": "uncertain",
        "report": (
            "CANARY_SYNTHETIC_03. FINDINGS: Possible partial ACL sprain cannot be "
            "excluded. Other ligaments, menisci, cartilage and bone are not mentioned."
        ),
    },
    {
        "example_id": "CANARY_SYNTHETIC_04",
        "language": "en",
        "coverage": "not_mentioned",
        "report": (
            "CANARY_SYNTHETIC_04. TECHNIQUE: Routine knee MRI. "
            "FINDINGS: Skin marker at the lateral joint line. No other description."
        ),
    },
    {
        "example_id": "CANARY_SYNTHETIC_05",
        "language": "fr",
        "coverage": "positive",
        "report": (
            "CANARY_SYNTHETIC_05. RESULTATS: Rupture complete du LCA. "
            "Le LCM, les menisques, le cartilage et l'os ne sont pas detailles."
        ),
    },
    {
        "example_id": "CANARY_SYNTHETIC_06",
        "language": "es",
        "coverage": "negated",
        "report": (
            "CANARY_SYNTHETIC_06. HALLAZGOS: No hay rotura del menisco medial. "
            "No hay rotura del menisco lateral. El resto de estructuras no se menciona."
        ),
    },
    {
        "example_id": "CANARY_SYNTHETIC_07",
        "language": "de",
        "coverage": "positive",
        "report": (
            "CANARY_SYNTHETIC_07. BEFUND: Deutlicher Gelenkerguss. "
            "Keine Synovitis beschrieben. Baender und Menisken nicht erwaehnt."
        ),
    },
    {
        "example_id": "CANARY_SYNTHETIC_08",
        "language": "it",
        "coverage": "uncertain",
        "report": (
            "CANARY_SYNTHETIC_08. REPERTO: Possibile condropatia femoro-rotulea, "
            "non confermata. Comparti mediale e laterale non descritti."
        ),
    },
    {
        "example_id": "CANARY_SYNTHETIC_09",
        "language": "pt",
        "coverage": "positive",
        "report": (
            "CANARY_SYNTHETIC_09. ACHADOS: Cisto de Baker popliteo. "
            "Sem derrame articular. Ligamentos e meniscos nao descritos."
        ),
    },
    {
        "example_id": "CANARY_SYNTHETIC_10",
        "language": "en",
        "coverage": "uncertain",
        "report": (
            "CANARY_SYNTHETIC_10. FINDINGS: Bone marrow oedema in the lateral "
            "femoral condyle without stated contusion or fracture. Soft tissues "
            "otherwise not mentioned."
        ),
    },
    {
        "example_id": "CANARY_SYNTHETIC_11",
        "language": "nl",
        "coverage": "negated",
        "report": (
            "CANARY_SYNTHETIC_11. BEVINDINGEN: Geen fractuur. Geen botcontusie. "
            "Overige kniestructuren niet beschreven."
        ),
    },
    {
        "example_id": "CANARY_SYNTHETIC_12",
        "language": "en",
        "coverage": "positive",
        "report": (
            "CANARY_SYNTHETIC_12. FINDINGS: Confirmed medial femoral condyle "
            "fracture. Synovitis with synovial thickening. No joint effusion. "
            "Ligaments and menisci not mentioned."
        ),
    },
    {
        "example_id": "CANARY_SYNTHETIC_13",
        "language": "en",
        "coverage": "uncertain",
        "report": (
            "CANARY_SYNTHETIC_13. FINDINGS: Intact ACL graft after reconstruction, "
            "no new tear. Medial meniscus degeneration without tear. "
            "Unspecified-compartment osteoarthritis mentioned only as degeneration."
        ),
    },
    {
        "example_id": "CANARY_SYNTHETIC_14",
        "language": "pl",
        "coverage": "positive",
        "report": (
            "CANARY_SYNTHETIC_14. OPIS: Zerwanie wiezadla pobocznego przysrodkowego. "
            "Chondropatia przedzialu przysrodkowego. Pozostale struktury nieopisane."
        ),
    },
)


def _record(status: str, evidence: str, uncertainty: str, confidence: str) -> dict[str, str]:
    return {
        "status": status,
        "confidence": confidence,
        "evidence_category": evidence,
        "uncertainty_or_negation_category": uncertainty,
    }


def fixture_canary_generate_fn(report: str) -> str:
    """Deterministic compact JSON for software tests. Not a scientific model."""
    payload = json.loads(compact_output_skeleton())
    targets = payload["targets"]
    text = report or ""
    if "CANARY_SYNTHETIC_01" in text or "CANARY_SYNTHETIC_05" in text:
        targets["acl_injury"] = _record("positive", "explicit_positive", "none", "high")
    elif "CANARY_SYNTHETIC_02" in text:
        for key in TARGET_KEYS:
            targets[key] = _record("negative", "explicit_negative", "negated", "high")
        targets["fracture"]["evidence_category"] = "fracture_excluded"
    elif "CANARY_SYNTHETIC_03" in text:
        targets["acl_injury"] = _record("uncertain", "uncertain_finding", "uncertain_hedged", "low")
    elif "CANARY_SYNTHETIC_06" in text:
        for key in ("medial_meniscus_injury", "lateral_meniscus_injury"):
            targets[key] = _record("negative", "explicit_negative", "negated", "high")
    elif "CANARY_SYNTHETIC_07" in text:
        targets["joint_effusion"] = _record("positive", "effusion_not_synovitis", "none", "high")
        targets["synovitis"] = _record("negative", "explicit_negative", "negated", "medium")
    elif "CANARY_SYNTHETIC_08" in text:
        targets["patellofemoral_oa"] = _record("uncertain", "uncertain_finding", "uncertain_hedged", "low")
    elif "CANARY_SYNTHETIC_09" in text:
        targets["baker_cyst"] = _record("positive", "baker_cyst", "none", "high")
        targets["joint_effusion"] = _record("negative", "explicit_negative", "negated", "medium")
    elif "CANARY_SYNTHETIC_10" in text:
        targets["bone_contusion"] = _record(
            "uncertain", "marrow_signal_not_contusion", "uncertain_hedged", "low"
        )
    elif "CANARY_SYNTHETIC_11" in text:
        targets["fracture"] = _record("negative", "fracture_excluded", "negated", "high")
        targets["bone_contusion"] = _record("negative", "explicit_negative", "negated", "high")
    elif "CANARY_SYNTHETIC_12" in text:
        targets["fracture"] = _record("positive", "confirmed_fracture", "none", "high")
        targets["synovitis"] = _record("positive", "synovitis_not_effusion", "none", "high")
        targets["joint_effusion"] = _record("negative", "explicit_negative", "negated", "medium")
    elif "CANARY_SYNTHETIC_13" in text:
        targets["acl_injury"] = _record(
            "uncertain", "postoperative_or_reconstructed", "postoperative", "medium"
        )
        targets["medial_meniscus_injury"] = _record(
            "uncertain", "degeneration_not_meniscal_injury", "none", "medium"
        )
        for key in ("medial_compartment_oa", "lateral_compartment_oa", "patellofemoral_oa"):
            targets[key] = _record("uncertain", "compartment_unspecified_oa", "none", "low")
    elif "CANARY_SYNTHETIC_14" in text:
        targets["mcl_injury"] = _record("positive", "explicit_positive", "none", "high")
        targets["medial_compartment_oa"] = _record(
            "positive", "compartment_specific_oa", "none", "medium"
        )
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=True)


def _normalized_hash(result) -> str:
    if result.malformed:
        return hashlib.sha256(b"").hexdigest()
    payload = {
        "schema_version": result.schema_version or SCHEMA_VERSION,
        "targets": {
            key: {
                "status": pred.status,
                "confidence": pred.confidence_category,
                "evidence_category": pred.evidence_category,
                "uncertainty_or_negation_category": pred.uncertainty_or_negation_category,
            }
            for key, pred in result.predictions.items()
        },
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _score_example(example: dict[str, str], *, config: dict, prompt: dict, generate_fn, weights_path) -> dict[str, Any]:
    result = run_second_reader(
        example["report"],
        config=config,
        language_supported=True,
        weights_path=weights_path,
        generate_fn=generate_fn,
        prompt=prompt,
    )
    parsed_ok = bool(result.enabled and not result.malformed)
    target_keys = list(result.predictions)
    gen = result.generation
    return {
        "example_id": example["example_id"],
        "language": example["language"],
        "coverage": example["coverage"],
        "ok": parsed_ok,
        "malformed": bool(result.malformed),
        "malformed_reason": result.malformed_reason,
        "malformed_stage": result.malformed_stage or ("ok" if parsed_ok else "other"),
        "n_targets": len(target_keys),
        "duplicate_targets": len(target_keys) != len(set(target_keys)),
        "missing_targets": [key for key in TARGET_KEYS if key not in result.predictions],
        "unexpected_targets": [key for key in target_keys if key not in TARGET_KEYS],
        "normalized_json_sha256": _normalized_hash(result),
        "completion_sha256": gen.completion_sha256,
        "n_generated_tokens": gen.n_generated_tokens,
        "reached_max_new_tokens": gen.reached_max_new_tokens,
        "ended_with_eos": gen.ended_with_eos,
        "stopped_on_json_close": gen.stopped_on_json_close,
        "stopped_on_eos": gen.stopped_on_eos,
        "begins_with_brace": gen.begins_with_brace,
        "contains_balanced_json": gen.contains_balanced_json,
        "markdown_json_fence_only": gen.markdown_json_fence_only,
        "targets_is_object": parsed_ok,
        "targets_not_object": result.malformed_reason == "targets_not_object",
        "forbidden_extra_text": result.malformed_stage == "forbidden_extra_text",
        "prompt_included": result.malformed_stage == "prompt_included_in_decoded_output",
        "inference_backend": result.inference_backend,
        "raw_was_stored": False,
    }


def aggregate_generation_diagnostics(rows: list[dict[str, Any]], *, max_new_tokens: int) -> dict[str, Any]:
    tokens = [int(row.get("n_generated_tokens") or 0) for row in rows]
    stages = {stage: 0 for stage in MALFORMED_STAGES}
    for row in rows:
        if row.get("ok"):
            continue
        stage = row.get("malformed_stage") or "other"
        if stage in stages:
            stages[stage] += 1
        else:
            stages["other"] += 1
    return {
        "n_examples": len(rows),
        "n_generated_tokens_min": int(min(tokens) if tokens else 0),
        "n_generated_tokens_median": float(np.median(tokens) if tokens else 0.0),
        "n_generated_tokens_max": int(max(tokens) if tokens else 0),
        "configured_max_new_tokens": int(max_new_tokens),
        "n_reaching_max_new_tokens": int(sum(bool(row.get("reached_max_new_tokens")) for row in rows)),
        "n_ending_with_eos": int(sum(bool(row.get("ended_with_eos")) for row in rows)),
        "n_beginning_with_brace": int(sum(bool(row.get("begins_with_brace")) for row in rows)),
        "n_containing_balanced_json": int(sum(bool(row.get("contains_balanced_json")) for row in rows)),
        "n_markdown_json_fence_only": int(sum(bool(row.get("markdown_json_fence_only")) for row in rows)),
        "n_stopped_on_json_close": int(sum(bool(row.get("stopped_on_json_close")) for row in rows)),
        "n_stopped_on_eos": int(sum(bool(row.get("stopped_on_eos")) for row in rows)),
        "n_targets_not_object": int(sum(bool(row.get("targets_not_object")) for row in rows)),
        "n_forbidden_extra_text": int(sum(bool(row.get("forbidden_extra_text")) for row in rows)),
        "n_prompt_included": int(sum(bool(row.get("prompt_included")) for row in rows)),
        "n_targets_is_object": int(sum(bool(row.get("targets_is_object") and row.get("ok")) for row in rows)),
        "malformed_stage_counts": stages,
        "completion_sha256_sorted": sorted(str(row.get("completion_sha256") or "") for row in rows),
        "raw_completions_stored": False,
    }


def render_canary_summary(result: dict[str, Any]) -> str:
    diag = result.get("generation_diagnostics") or {}
    stages = diag.get("malformed_stage_counts") or {}
    lines = [
        "# Second-reader synthetic canary",
        "",
        "Synthetic reports only. No gold studies. No report-only studies.",
        "Report text and model completions are not included.",
        "",
        f"- canary_version: `{result.get('canary_version')}`",
        f"- transport_protocol_version: `{result.get('transport_protocol_version')}`",
        f"- prompt_version: `{result.get('prompt_version')}`",
        f"- schema_version: `{result.get('schema_version')}`",
        f"- status: `{result.get('status')}`",
        f"- ok: `{str(result.get('ok')).lower()}`",
        f"- n_examples: {result.get('n_examples')}",
        f"- n_gold_studies_processed: {result.get('n_gold_studies_processed')}",
        f"- n_report_only_studies_processed: {result.get('n_report_only_studies_processed')}",
        f"- malformed_output_rate: {result.get('malformed_output_rate')}",
        f"- all_parseable: `{str(result.get('all_parseable')).lower()}`",
        f"- repeatable: `{str(result.get('repeatable')).lower()}`",
        "- scientific_evaluation: `false`",
        f"- stop_gold_rerun: `{str(result.get('stop_gold_rerun')).lower()}`",
        f"- inference_backend: `{result.get('inference_backend')}`",
        "",
        "## Generation diagnostics (aggregates)",
        "",
        f"- generated-token count min/median/max: {diag.get('n_generated_tokens_min')}/"
        f"{diag.get('n_generated_tokens_median')}/{diag.get('n_generated_tokens_max')}",
        f"- configured max_new_tokens: {diag.get('configured_max_new_tokens')}",
        f"- n reaching max_new_tokens: {diag.get('n_reaching_max_new_tokens')}",
        f"- n ending with EOS: {diag.get('n_ending_with_eos')}",
        f"- n beginning with `{{`: {diag.get('n_beginning_with_brace')}",
        f"- n containing a balanced JSON object: {diag.get('n_containing_balanced_json')}",
        f"- n wrapped only in a Markdown JSON fence: {diag.get('n_markdown_json_fence_only')}",
        f"- n stopped on JSON close: {diag.get('n_stopped_on_json_close')}",
        f"- n stopped on EOS: {diag.get('n_stopped_on_eos')}",
        f"- n targets_not_object: {diag.get('n_targets_not_object')}",
        f"- n forbidden_extra_text: {diag.get('n_forbidden_extra_text')}",
        f"- n prompt included: {diag.get('n_prompt_included')}",
        f"- raw completions stored: `{str(diag.get('raw_completions_stored')).lower()}`",
        "",
        "## Malformed-stage counts",
        "",
    ]
    for stage in MALFORMED_STAGES:
        lines.append(f"- `{stage}`: {stages.get(stage, 0)}")
    lines.extend(
        [
            "",
            "## Coordinator",
            "",
            "- Do not rerun the 58-study gold pilot if this canary is not `ok`.",
            "- Do not process the 4,349 report-only studies.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def run_synthetic_canary(
    *,
    output_dir: str | Path | None = None,
    config_path: str | Path | None = None,
    generate_fn: GenerateFn | None = None,
    weights_path: str | Path | None = None,
    n_repeats: int = 2,
    smoke: bool = False,
    examples: tuple[dict[str, str], ...] | None = None,
) -> dict[str, Any]:
    """Score synthetic reports only. Refuses competition gold and report-only rows."""
    selected = examples or (
        SYNTHETIC_CANARY_EXAMPLES[:3] if smoke else SYNTHETIC_CANARY_EXAMPLES
    )
    if smoke:
        if n_repeats < 1:
            raise ValueError("smoke canary requires at least one run")
        if len(selected) != 3:
            raise RuntimeError("smoke canary requires exactly three synthetic examples")
    else:
        if n_repeats < 2:
            raise ValueError("canary requires two runs for repeatability")
        if len(selected) < 12:
            raise RuntimeError("canary requires at least 12 synthetic examples")
    out_dir = Path(output_dir or (PROJECT_ROOT / "outputs" / KAGGLE_CANARY_OUTPUT_DIRNAME))
    sanitized_dir = out_dir / "sanitized"
    sanitized_dir.mkdir(parents=True, exist_ok=True)
    config = load_pilot_config(config_path or DEFAULT_PILOT_CONFIG_PATH_V1_2)
    prompt = load_prompt_bundle(config.get("prompt_path"))
    freeze = freeze_manifest(config, prompt=prompt)
    freeze["canary"] = True
    freeze["smoke"] = bool(smoke)
    freeze["n_gold_studies_processed"] = 0
    freeze["n_report_only_studies_processed"] = 0
    freeze["pinned_termination"] = dict(PINNED_PHI35_TERMINATION)
    (sanitized_dir / SANITIZED_FREEZE_FILENAME).write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (sanitized_dir / SANITIZED_TERMINATION_FILENAME).write_text(
        json.dumps(PINNED_PHI35_TERMINATION, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if generate_fn is None:
        directory = locate_weights_directory(config.get("second_reader") or {}, weights_path)
        weights_path = directory
        if directory is None:
            payload = {
                "status": "WEIGHTS_NOT_ATTACHED",
                "ok": False,
                "malformed_output_rate": 1.0,
                "n_gold_studies_processed": 0,
                "n_report_only_studies_processed": 0,
                "scientific_evaluation": False,
                "stop_gold_rerun": True,
                "smoke": bool(smoke),
            }
            (sanitized_dir / SANITIZED_METRICS_FILENAME).write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            return payload

    run_rows: list[list[dict[str, Any]]] = []
    for _repeat in range(n_repeats):
        rows = [
            _score_example(
                example,
                config=config,
                prompt=prompt,
                generate_fn=generate_fn,
                weights_path=weights_path,
            )
            for example in selected
        ]
        run_rows.append(rows)

    first = run_rows[0]
    second = run_rows[1] if len(run_rows) > 1 else first
    n_malformed = sum(1 for row in first if row["malformed"] or not row["ok"])
    malformed_rate = float(n_malformed) / float(len(first))
    hashes_match = [a["normalized_json_sha256"] for a in first] == [
        b["normalized_json_sha256"] for b in second
    ] and [a["completion_sha256"] for a in first] == [b["completion_sha256"] for b in second]
    target_ok = all(
        (not row["missing_targets"])
        and (not row["unexpected_targets"])
        and (not row["duplicate_targets"])
        and row["n_targets"] == 12
        for row in first
        if row["ok"]
    )
    all_parseable = n_malformed == 0 and target_ok
    decoding = config.get("decoding") or {}
    diagnostics = aggregate_generation_diagnostics(
        first, max_new_tokens=int(decoding.get("max_new_tokens") or 0)
    )
    n_ex = len(first)
    transport_ok = (
        diagnostics.get("n_beginning_with_brace") == n_ex
        and diagnostics.get("n_containing_balanced_json") == n_ex
        and diagnostics.get("n_reaching_max_new_tokens") == 0
        and diagnostics.get("n_forbidden_extra_text") == 0
        and diagnostics.get("n_prompt_included") == 0
        and diagnostics.get("n_targets_not_object") == 0
        and diagnostics.get("n_targets_is_object") == n_ex
    )
    repeatable_ok = hashes_match if (not smoke or n_repeats >= 2) else True
    ok = all_parseable and repeatable_ok and malformed_rate == 0.0 and transport_ok
    fail_status = "SMOKE_FAIL" if smoke else "CANARY_FAIL"
    result = {
        "status": "ok" if ok else fail_status,
        "ok": ok,
        "smoke": bool(smoke),
        "canary_version": CANARY_VERSION,
        "transport_protocol_version": config.get("transport_protocol_version"),
        "prompt_version": prompt.get("prompt_version"),
        "schema_version": SCHEMA_VERSION,
        "n_examples": n_ex,
        "n_gold_studies_processed": 0,
        "n_report_only_studies_processed": 0,
        "malformed_output_rate": malformed_rate,
        "n_malformed": n_malformed,
        "all_parseable": all_parseable,
        "repeatable": hashes_match if n_repeats >= 2 else None,
        "transport_ok": transport_ok,
        "scientific_evaluation": False,
        "claim_scientific_performance": False,
        "stop_gold_rerun": not ok,
        "inference_backend": first[0]["inference_backend"] if first else "skipped",
        "generation_diagnostics": diagnostics,
        "example_ids": [row["example_id"] for row in first],
        "termination_inspection": dict(PINNED_PHI35_TERMINATION),
        "freeze": freeze,
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    summary = render_canary_summary(result)
    (sanitized_dir / SANITIZED_SUMMARY_FILENAME).write_text(summary, encoding="utf-8")
    public = dict(result)
    public.pop("freeze", None)
    (sanitized_dir / SANITIZED_METRICS_FILENAME).write_text(
        json.dumps(jsonable(public), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    freeze["metrics_computed"] = True
    (sanitized_dir / SANITIZED_FREEZE_FILENAME).write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    result["output_dir"] = str(out_dir)
    return result
