"""Frozen clinical extraction schema for the Phase 4C second-reader pilot.

The twelve ``TARGET_KEYS`` are the only targets. Status, confidence, evidence,
and uncertainty/negation categories are closed vocabularies. Chain-of-thought
and report quotations are forbidden.
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import PROJECT_ROOT, TARGET_KEYS, TARGET_NAMES, TARGET_KEY_TO_OFFICIAL

SCHEMA_VERSION = "second-reader-schema-v1.0.0"
PROMPT_VERSION = "second-reader-prompt-v1.0.0"
PROMPT_VERSION_V1_1 = "second-reader-prompt-v1.1.0"
PILOT_VERSION = "second-reader-pilot-v1.0.0"
PILOT_VERSION_V1_1 = "second-reader-pilot-v1.1.0"
PILOT_VERSION_V1_2 = "second-reader-pilot-v1.2.0"
EVAL_PROTOCOL_VERSION = "gold-pilot-v1.0.0"
EVAL_PROTOCOL_VERSION_V1_1 = "gold-pilot-v1.1.0"
EVAL_PROTOCOL_VERSION_V1_2 = "gold-pilot-v1.2.0"
TRANSPORT_PROTOCOL_VERSION_V1 = "second-reader-transport-v1.0"
TRANSPORT_PROTOCOL_VERSION_V1_1 = "second-reader-transport-v1.1"
TRANSPORT_PROTOCOL_VERSION_V1_2 = "second-reader-transport-v1.2"
ONTOLOGY_VERSION = "pilot-1.0"
DEFAULT_SCHEMA_PATH = PROJECT_ROOT / "configs" / "second_reader_schema.json"
DEFAULT_PROMPT_PATH = PROJECT_ROOT / "configs" / "second_reader_prompt.yaml"
DEFAULT_PROMPT_PATH_V1_1 = PROJECT_ROOT / "configs" / "second_reader_prompt_v1_1.yaml"
DEFAULT_PILOT_CONFIG_PATH = PROJECT_ROOT / "configs" / "second_reader_pilot.yaml"
DEFAULT_PILOT_CONFIG_PATH_V1_1 = PROJECT_ROOT / "configs" / "second_reader_pilot_v1_1.yaml"
DEFAULT_PILOT_CONFIG_PATH_V1_2 = PROJECT_ROOT / "configs" / "second_reader_pilot_v1_2.yaml"

STATUSES = ("positive", "negative", "uncertain", "not_mentioned")
CONFIDENCE_CATEGORIES = ("high", "medium", "low")

EVIDENCE_CATEGORIES = (
    "explicit_positive",
    "explicit_negative",
    "uncertain_finding",
    "not_mentioned",
    "historical_finding",
    "postoperative_or_reconstructed",
    "reconstruction_not_acute",
    "degeneration_not_meniscal_injury",
    "compartment_specific_oa",
    "compartment_unspecified_oa",
    "generic_degeneration_not_oa",
    "effusion_not_synovitis",
    "synovitis_not_effusion",
    "baker_cyst",
    "other_popliteal_finding",
    "marrow_signal_not_contusion",
    "bone_contusion",
    "suspected_fracture",
    "confirmed_fracture",
    "fracture_excluded",
    "insufficient_evidence",
    "malformed_output",
)

UNCERTAINTY_OR_NEGATION_CATEGORIES = (
    "none",
    "negated",
    "uncertain_hedged",
    "differential_diagnosis",
    "history_only",
    "postoperative",
    "suspected_not_confirmed",
    "not_mentioned",
)

FORBIDDEN_OUTPUT_KEYS = (
    "chain_of_thought",
    "cot",
    "reasoning",
    "explanation",
    "rationale",
    "quote",
    "quotation",
    "excerpt",
    "span",
    "evidence_span",
    "report_text",
    "report",
)

STATUS_TO_LABEL = {
    "positive": "positive",
    "negative": "negative",
    "uncertain": "unknown",
    "not_mentioned": "unknown",
}

CONFIDENCE_TO_FLOAT = {
    "high": 0.90,
    "medium": 0.60,
    "low": 0.30,
}

TARGET_FIELD_KEYS = (
    "status",
    "confidence",
    "evidence_category",
    "uncertainty_or_negation_category",
)

CLINICAL_DISTINCTIONS = (
    "present_versus_absent",
    "uncertain_versus_confirmed",
    "historical_and_postoperative",
    "reconstruction_versus_acute_injury",
    "degeneration_versus_meniscal_injury",
    "compartment_specific_oa",
    "effusion_versus_synovitis",
    "baker_cyst_versus_other_popliteal",
    "marrow_signal_versus_bone_contusion",
    "suspected_versus_confirmed_fracture",
)


def target_record_schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(TARGET_FIELD_KEYS),
        "properties": {
            "status": {"type": "string", "enum": list(STATUSES)},
            "confidence": {"type": "string", "enum": list(CONFIDENCE_CATEGORIES)},
            "evidence_category": {"type": "string", "enum": list(EVIDENCE_CATEGORIES)},
            "uncertainty_or_negation_category": {
                "type": "string",
                "enum": list(UNCERTAINTY_OR_NEGATION_CATEGORIES),
            },
        },
    }


def json_schema() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "RSNA knee second-reader extraction",
        "version": SCHEMA_VERSION,
        "ontology_version": ONTOLOGY_VERSION,
        "source_of_truth_targets": "knee/config.py::TARGET_KEYS",
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "targets"],
        "properties": {
            "schema_version": {"type": "string", "const": SCHEMA_VERSION},
            "targets": {
                "type": "object",
                "additionalProperties": False,
                "required": list(TARGET_KEYS),
                "properties": {key: target_record_schema() for key in TARGET_KEYS},
            },
        },
    }


def compact_target_record() -> dict[str, str]:
    """Format-only skeleton. Not a clinical default and not a negative label."""
    return {
        "status": "not_mentioned",
        "confidence": "low",
        "evidence_category": "not_mentioned",
        "uncertainty_or_negation_category": "not_mentioned",
    }


def compact_output_skeleton() -> str:
    """One-line JSON showing the required keys. Values are format placeholders."""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "targets": {key: compact_target_record() for key in TARGET_KEYS},
    }
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=True)


def dump_schema(path: Path | None = None) -> dict:
    payload = json_schema()
    dest = Path(path or DEFAULT_SCHEMA_PATH)
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def official_name(target_key: str) -> str:
    return TARGET_KEY_TO_OFFICIAL[target_key]


def target_display() -> list[dict[str, str]]:
    return [
        {
            "target_key": key,
            "official_column": TARGET_KEY_TO_OFFICIAL[key],
            "label": name,
        }
        for key, name in zip(TARGET_KEYS, TARGET_NAMES)
    ]
