"""Parse second-reader JSON. Never store chain-of-thought or report quotations.

Transport protocol v1.1 extracts one balanced JSON object after at most one
outer Markdown JSON fence. v1.0 used a greedy ``{.*}`` match that could pick
an inner object from truncated output and report ``targets_not_object``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
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
    TRANSPORT_PROTOCOL_VERSION_V1_1,
    UNCERTAINTY_OR_NEGATION_CATEGORIES,
)

MALFORMED_STAGES = (
    "generation_truncated",
    "prompt_included_in_decoded_output",
    "empty_generation",
    "missing_JSON_object",
    "unbalanced_braces",
    "Markdown_code_fence",
    "JSON_decode_error",
    "schema_validation_error",
    "missing_target",
    "unexpected_target",
    "invalid_enum",
    "forbidden_extra_text",
    "other",
)

# Distinctive frozen-prompt fragments. Used only as boolean leak flags.
_PROMPT_LEAK_MARKERS = (
    "You are an offline clinical NLP extractor",
    "Extract the twelve targets from this radiology report",
    "REPORT:",
)

_OUTER_FENCE_RE = re.compile(
    r"^```(?:json)?\s*\r?\n?(.*?)\r?\n?```\s*$",
    re.IGNORECASE | re.DOTALL,
)
_FENCE_RE_V1 = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)
_JSON_OBJECT_RE_V1 = re.compile(r"\{.*\}", re.DOTALL)
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


@dataclass(frozen=True)
class GenerationDiagnostics:
    n_generated_tokens: int = 0
    max_new_tokens: int = 0
    reached_max_new_tokens: bool = False
    ended_with_eos: bool = False
    begins_with_brace: bool = False
    contains_balanced_json: bool = False
    markdown_json_fence_only: bool = False
    completion_sha256: str = ""
    decode_slice: str = "generated_ids[:, input_ids.shape[1]:]"
    raw_was_stored: bool = False
    prompt_included_in_decoded_output: bool = False


@dataclass
class ParsedSecondReaderOutput:
    ok: bool
    malformed: bool
    malformed_reason: str
    malformed_stage: str
    schema_version: str | None
    dropped_forbidden_fields: tuple[str, ...]
    targets: dict[str, ParsedTarget]
    raw_was_stored: bool = False
    transport_protocol: str = TRANSPORT_PROTOCOL_VERSION_V1_1
    generation: GenerationDiagnostics = field(default_factory=GenerationDiagnostics)
    used_root_level_targets: bool = False


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


def _malformed_result(
    reason: str,
    stage: str,
    *,
    generation: GenerationDiagnostics | None = None,
    schema_version: str | None = None,
    forbidden: tuple[str, ...] = (),
    used_root_level_targets: bool = False,
) -> ParsedSecondReaderOutput:
    gen = generation or GenerationDiagnostics()
    if gen.prompt_included_in_decoded_output and stage not in {
        "prompt_included_in_decoded_output",
        "empty_generation",
    }:
        stage = "prompt_included_in_decoded_output"
        reason = "prompt_included_in_decoded_output"
    if (
        gen.reached_max_new_tokens
        and stage in {"unbalanced_braces", "JSON_decode_error", "missing_JSON_object"}
    ):
        stage = "generation_truncated"
        reason = "generation_truncated"
    return ParsedSecondReaderOutput(
        ok=False,
        malformed=True,
        malformed_reason=reason,
        malformed_stage=stage if stage in MALFORMED_STAGES else "other",
        schema_version=schema_version,
        dropped_forbidden_fields=forbidden,
        targets={key: _unknown_target(key, reason) for key in TARGET_KEYS},
        raw_was_stored=False,
        generation=gen,
        used_root_level_targets=used_root_level_targets,
    )


def completion_sha256(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def text_has_prompt_leak(text: str) -> bool:
    blob = text or ""
    return any(marker in blob for marker in _PROMPT_LEAK_MARKERS)


def strip_one_outer_markdown_json_fence(text: str) -> tuple[str, bool]:
    """Strip exactly one outer ```json ... ``` wrapper. Inner fences are not stripped."""
    stripped = (text or "").strip()
    match = _OUTER_FENCE_RE.fullmatch(stripped)
    if match:
        return match.group(1).strip(), True
    return stripped, False


def extract_balanced_json_object(text: str) -> tuple[str | None, str, str]:
    """Return ``(object, remainder, reason)``. Reason is empty on success."""
    if not text or not str(text).strip():
        return None, "", "empty_generation"
    start = text.find("{")
    if start < 0:
        return None, text, "missing_JSON_object"
    in_string = False
    escape = False
    depth = 0
    for index, char in enumerate(text[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                blob = text[start : index + 1]
                remainder = text[:start] + text[index + 1 :]
                return blob, remainder, ""
    return None, text, "unbalanced_braces"


def remainder_is_allowed_transport_wrapper(remainder: str, *, fence_stripped: bool) -> bool:
    leftover = (remainder or "").strip()
    if not leftover:
        return True
    if fence_stripped:
        return False
    match = _OUTER_FENCE_RE.fullmatch(leftover)
    return match is not None and not match.group(1).strip()


def diagnostics_from_text(
    text: str,
    *,
    max_new_tokens: int = 0,
    n_generated_tokens: int = 0,
    reached_max_new_tokens: bool = False,
    ended_with_eos: bool = False,
) -> GenerationDiagnostics:
    stripped, fenced = strip_one_outer_markdown_json_fence(text)
    blob, remainder, reason = extract_balanced_json_object(stripped)
    balanced = bool(blob) and reason == "" and remainder_is_allowed_transport_wrapper(
        remainder, fence_stripped=fenced
    )
    fence_only = fenced and balanced and not (remainder or "").strip()
    return GenerationDiagnostics(
        n_generated_tokens=int(n_generated_tokens),
        max_new_tokens=int(max_new_tokens),
        reached_max_new_tokens=bool(reached_max_new_tokens),
        ended_with_eos=bool(ended_with_eos),
        begins_with_brace=(text or "").lstrip().startswith("{"),
        contains_balanced_json=balanced,
        markdown_json_fence_only=fence_only,
        completion_sha256=completion_sha256(text),
        prompt_included_in_decoded_output=text_has_prompt_leak(text),
        raw_was_stored=False,
    )


def strip_fences(text: str) -> str:
    """v1.0 fence strip. Kept so the failed protocol remains reproducible."""
    return _FENCE_RE_V1.sub("", text or "").strip()


def extract_json_object_v1(text: str) -> str:
    """v1.0 greedy extractor. Can return an inner object from truncated JSON."""
    cleaned = strip_fences(text)
    if cleaned.startswith("{") and cleaned.endswith("}"):
        return cleaned
    match = _JSON_OBJECT_RE_V1.search(cleaned)
    if not match:
        raise MalformedSecondReaderOutput("no_json_object")
    return match.group(0)


def extract_json_object(text: str) -> str:
    """v1.1 balanced extractor. Raises ``MalformedSecondReaderOutput`` with a stage."""
    stripped, fenced = strip_one_outer_markdown_json_fence(text)
    if not stripped:
        raise MalformedSecondReaderOutput("empty_generation")
    blob, remainder, reason = extract_balanced_json_object(stripped)
    if reason:
        if fenced and reason == "missing_JSON_object":
            raise MalformedSecondReaderOutput("Markdown_code_fence")
        raise MalformedSecondReaderOutput(reason)
    if not remainder_is_allowed_transport_wrapper(remainder, fence_stripped=fenced):
        raise MalformedSecondReaderOutput("forbidden_extra_text")
    assert blob is not None
    return blob


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


def _coerce_targets_payload(payload: dict) -> tuple[Any, bool]:
    """Accept root-level TARGET_KEYS as a mechanical layout unwrap. No fuzzy names."""
    targets_payload = payload.get("targets")
    if isinstance(targets_payload, dict):
        return targets_payload, False
    if all(key in payload and isinstance(payload.get(key), dict) for key in TARGET_KEYS):
        return {key: payload[key] for key in TARGET_KEYS}, True
    return targets_payload, False


def _stage_for_extract_error(reason: str) -> str:
    mapping = {
        "empty_generation": "empty_generation",
        "missing_JSON_object": "missing_JSON_object",
        "no_json_object": "missing_JSON_object",
        "unbalanced_braces": "unbalanced_braces",
        "Markdown_code_fence": "Markdown_code_fence",
        "forbidden_extra_text": "forbidden_extra_text",
    }
    return mapping.get(reason, "other")


def parse_second_reader_output(
    text: str,
    *,
    protocol: str = "v1.1",
    generation: GenerationDiagnostics | None = None,
) -> ParsedSecondReaderOutput:
    """Parse model text into closed-vocabulary targets. Raw text is not retained."""
    gen = generation or diagnostics_from_text(text)
    if gen.prompt_included_in_decoded_output:
        return _malformed_result(
            "prompt_included_in_decoded_output",
            "prompt_included_in_decoded_output",
            generation=gen,
        )
    if not (text or "").strip():
        return _malformed_result("empty_generation", "empty_generation", generation=gen)

    forbidden: list[str] = []
    try:
        if protocol == "v1.0":
            blob = extract_json_object_v1(text)
        else:
            blob = extract_json_object(text)
        payload = json.loads(blob)
    except MalformedSecondReaderOutput as exc:
        reason = str(exc) or "malformed"
        return _malformed_result(reason, _stage_for_extract_error(reason), generation=gen)
    except (json.JSONDecodeError, TypeError):
        return _malformed_result("json_decode_error", "JSON_decode_error", generation=gen)

    if not isinstance(payload, dict):
        return _malformed_result("root_not_object", "schema_validation_error", generation=gen)

    _collect_forbidden(payload, forbidden)
    schema_version = payload.get("schema_version")
    targets_payload, used_root = _coerce_targets_payload(payload)
    if schema_version != SCHEMA_VERSION:
        forbidden.append("schema_version_mismatch")
    if not isinstance(targets_payload, dict):
        return _malformed_result(
            "targets_not_object",
            "schema_validation_error",
            generation=gen,
            schema_version=str(schema_version) if schema_version is not None else None,
            forbidden=tuple(dict.fromkeys(forbidden)),
        )

    parsed_targets: dict[str, ParsedTarget] = {}
    any_malformed = schema_version != SCHEMA_VERSION
    stage = "ok"
    reason = "ok"
    missing_keys = [key for key in TARGET_KEYS if key not in targets_payload]
    extra_targets = sorted(set(targets_payload) - set(TARGET_KEYS))
    if missing_keys:
        any_malformed = True
        stage = "missing_target"
        reason = "missing_target:" + ",".join(missing_keys)
    if extra_targets:
        forbidden.append("extra_targets:" + ",".join(extra_targets))
        any_malformed = True
        if stage == "ok":
            stage = "unexpected_target"
            reason = "unexpected_target:" + ",".join(extra_targets)
    for key in TARGET_KEYS:
        parsed_targets[key] = _validate_target(key, targets_payload.get(key))
        if parsed_targets[key].malformed:
            any_malformed = True
            drop = parsed_targets[key].drop_reasons
            if any(item.startswith("invalid_") for item in drop) and stage == "ok":
                stage = "invalid_enum"
                reason = "invalid_enum"
            elif stage == "ok":
                stage = "schema_validation_error"
                reason = "schema_or_target_invalid"
    if any(name.startswith("quotation:") for name in forbidden):
        any_malformed = True
        if stage == "ok":
            stage = "forbidden_extra_text"
            reason = "quotation"
    if "schema_version_mismatch" in forbidden and stage == "ok":
        any_malformed = True
        stage = "schema_validation_error"
        reason = "schema_or_target_invalid"
    if any_malformed and stage == "ok":
        stage = "schema_validation_error"
        reason = "schema_or_target_invalid"

    ok = not any_malformed
    return ParsedSecondReaderOutput(
        ok=ok,
        malformed=not ok,
        malformed_reason="ok" if ok else reason,
        malformed_stage="ok" if ok else stage,
        schema_version=str(schema_version) if schema_version is not None else None,
        dropped_forbidden_fields=tuple(dict.fromkeys(forbidden)),
        targets=parsed_targets,
        raw_was_stored=False,
        generation=gen,
        used_root_level_targets=used_root,
    )


def normalized_payload_json(parsed: ParsedSecondReaderOutput) -> str:
    """Deterministic compact JSON of accepted fields only. Empty if malformed."""
    if parsed.malformed or not parsed.ok:
        return ""
    payload = {
        "schema_version": parsed.schema_version,
        "targets": {
            key: {
                "status": item.status,
                "confidence": item.confidence,
                "evidence_category": item.evidence_category,
                "uncertainty_or_negation_category": item.uncertainty_or_negation_category,
            }
            for key, item in parsed.targets.items()
        },
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
