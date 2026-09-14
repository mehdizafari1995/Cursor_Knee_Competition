"""Parse second-reader JSON. Never store chain-of-thought or report quotations."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any

from .config import TARGET_KEYS
from .second_reader_schema import (
    CONFIDENCE_CATEGORIES,
    CONFIDENCE_TO_FLOAT,
    EVIDENCE_CATEGORIES,
    FORBIDDEN_OUTPUT_KEYS,
    SCHEMA_VERSION,
    STATUSES,
    STATUS_TO_LABEL,
    TARGET_FIELD_KEYS,
    UNCERTAINTY_OR_NEGATION_CATEGORIES,
)

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_QUOTATION_RE = re.compile(r"[\"«»„“”].{20,}[\"«»„“”]")


class MalformedSecondReaderOutput(ValueError):
    """The model output could not be parsed into the frozen schema."""


@dataclass(frozen=True)
class ParsedTarget:
    target_key: str
    status: str
    confidence: str
    evidence_category: str
    uncertainty_or_negation_category: str
    label: str
    confidence_value: float
    malformed: bool
    drop_reasons: tuple[str, ...] = ()


@dataclass
class ParsedSecondReaderOutput:
    ok: bool
    malformed: bool
    malformed_reason: str
    schema_version: str | None
    dropped_forbidden_fields: tuple[str, ...]
    targets: dict[str, ParsedTarget]
    raw_was_stored: bool = False


def _unknown_target(key: str, *reasons: str) -> ParsedTarget:
    return ParsedTarget(
        target_key=key,
        status="not_mentioned",
        confidence="low",
        evidence_category="malformed_output",
        uncertainty_or_negation_category="not_mentioned",
        label="unknown",
        confidence_value=0.0,
        malformed=True,
        drop_reasons=tuple(reasons),
    )


def strip_fences(text: str) -> str:
    return _FENCE_RE.sub("", text or "").strip()


def extract_json_object(text: str) -> str:
    cleaned = strip_fences(text)
    if cleaned.startswith("{") and cleaned.endswith("}"):
        return cleaned
    match = _JSON_OBJECT_RE.search(cleaned)
    if not match:
        raise MalformedSecondReaderOutput("no_json_object")
    return match.group(0)


def _looks_like_quotation(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    if len(value) > 80:
        return True
    if _QUOTATION_RE.search(value):
        return True
    return False


def _collect_forbidden(payload: Any, found: list[str]) -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            lowered = str(key).lower()
            if lowered in FORBIDDEN_OUTPUT_KEYS or lowered.replace("-", "_") in FORBIDDEN_OUTPUT_KEYS:
                found.append(str(key))
            if _looks_like_quotation(value) and lowered not in {"schema_version"}:
                found.append(f"quotation:{key}")
            _collect_forbidden(value, found)
    elif isinstance(payload, list):
        for item in payload:
            _collect_forbidden(item, found)


def _validate_target(key: str, record: Any) -> ParsedTarget:
    if not isinstance(record, dict):
        return _unknown_target(key, "target_not_object")
    extra = sorted(set(record) - set(TARGET_FIELD_KEYS))
    missing = [field for field in TARGET_FIELD_KEYS if field not in record]
    reasons: list[str] = []
    if extra:
        reasons.append("unexpected_fields:" + ",".join(extra))
    if missing:
        reasons.append("missing_fields:" + ",".join(missing))
    status = record.get("status")
    confidence = record.get("confidence")
    evidence = record.get("evidence_category")
    uncertainty = record.get("uncertainty_or_negation_category")
    if status not in STATUSES:
        reasons.append("invalid_status")
    if confidence not in CONFIDENCE_CATEGORIES:
        reasons.append("invalid_confidence")
    if evidence not in EVIDENCE_CATEGORIES:
        reasons.append("invalid_evidence_category")
    if uncertainty not in UNCERTAINTY_OR_NEGATION_CATEGORIES:
        reasons.append("invalid_uncertainty_or_negation_category")
    for field_name in TARGET_FIELD_KEYS:
        if _looks_like_quotation(record.get(field_name)):
            reasons.append(f"quotation:{field_name}")
    if reasons:
        return _unknown_target(key, *reasons)
    return ParsedTarget(
        target_key=key,
        status=str(status),
        confidence=str(confidence),
        evidence_category=str(evidence),
        uncertainty_or_negation_category=str(uncertainty),
        label=STATUS_TO_LABEL[str(status)],
        confidence_value=CONFIDENCE_TO_FLOAT[str(confidence)],
        malformed=False,
    )


def parse_second_reader_output(text: str) -> ParsedSecondReaderOutput:
    """Parse model text into closed-vocabulary targets. Raw text is not retained."""
    forbidden: list[str] = []
    try:
        blob = extract_json_object(text)
        payload = json.loads(blob)
    except (MalformedSecondReaderOutput, json.JSONDecodeError, TypeError) as exc:
        reason = "json_decode_error" if isinstance(exc, json.JSONDecodeError) else str(exc) or "malformed"
        return ParsedSecondReaderOutput(
            ok=False,
            malformed=True,
            malformed_reason=reason,
            schema_version=None,
            dropped_forbidden_fields=(),
            targets={key: _unknown_target(key, reason) for key in TARGET_KEYS},
        )
    if not isinstance(payload, dict):
        return ParsedSecondReaderOutput(
            ok=False,
            malformed=True,
            malformed_reason="root_not_object",
            schema_version=None,
            dropped_forbidden_fields=(),
            targets={key: _unknown_target(key, "root_not_object") for key in TARGET_KEYS},
        )
    _collect_forbidden(payload, forbidden)
    schema_version = payload.get("schema_version")
    targets_payload = payload.get("targets")
    if schema_version != SCHEMA_VERSION:
        forbidden.append("schema_version_mismatch")
    if not isinstance(targets_payload, dict):
        return ParsedSecondReaderOutput(
            ok=False,
            malformed=True,
            malformed_reason="targets_not_object",
            schema_version=str(schema_version) if schema_version is not None else None,
            dropped_forbidden_fields=tuple(dict.fromkeys(forbidden)),
            targets={key: _unknown_target(key, "targets_not_object") for key in TARGET_KEYS},
        )
    parsed_targets: dict[str, ParsedTarget] = {}
    any_malformed = schema_version != SCHEMA_VERSION
    for key in TARGET_KEYS:
        parsed_targets[key] = _validate_target(key, targets_payload.get(key))
        if parsed_targets[key].malformed:
            any_malformed = True
    extra_targets = sorted(set(targets_payload) - set(TARGET_KEYS))
    if extra_targets:
        forbidden.append("extra_targets:" + ",".join(extra_targets))
        any_malformed = True
    return ParsedSecondReaderOutput(
        ok=not any_malformed and not any(
            name.startswith("quotation:") for name in forbidden
        ),
        malformed=any_malformed or any(name.startswith("quotation:") for name in forbidden),
        malformed_reason="ok" if not any_malformed else "schema_or_target_invalid",
        schema_version=str(schema_version) if schema_version is not None else None,
        dropped_forbidden_fields=tuple(dict.fromkeys(forbidden)),
        targets=parsed_targets,
        raw_was_stored=False,
    )
