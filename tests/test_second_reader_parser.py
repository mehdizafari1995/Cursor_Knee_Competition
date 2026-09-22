"""Structured-output parser: schema, malformed JSON, no quotations or CoT."""

from __future__ import annotations

import json

from knee.config import TARGET_KEYS
from knee.second_reader_parser import parse_second_reader_output
from knee.second_reader_schema import SCHEMA_VERSION, json_schema


def _valid_record(**overrides):
    rec = {
        "status": "not_mentioned",
        "confidence": "low",
        "evidence_category": "not_mentioned",
        "uncertainty_or_negation_category": "not_mentioned",
    }
    rec.update(overrides)
    return rec


def valid_payload(**target_overrides) -> dict:
    targets = {key: _valid_record() for key in TARGET_KEYS}
    for key, rec in target_overrides.items():
        targets[key] = _valid_record(**rec)
    return {"schema_version": SCHEMA_VERSION, "targets": targets}


def test_json_schema_covers_all_twelve_targets():
    schema = json_schema()
    required = schema["properties"]["targets"]["required"]
    assert required == list(TARGET_KEYS)
    assert set(schema["properties"]["targets"]["properties"]) == set(TARGET_KEYS)


def test_parse_accepts_fenced_valid_json():
    payload = valid_payload(
        acl_injury={
            "status": "positive",
            "confidence": "high",
            "evidence_category": "explicit_positive",
            "uncertainty_or_negation_category": "none",
        }
    )
    text = "```json\n" + json.dumps(payload) + "\n```"
    parsed = parse_second_reader_output(text)
    assert parsed.ok
    assert not parsed.malformed
    assert parsed.raw_was_stored is False
    assert parsed.targets["acl_injury"].label == "positive"
    assert parsed.targets["mcl_injury"].label == "unknown"


def test_parse_rejects_non_json_as_malformed():
    parsed = parse_second_reader_output("The ACL is torn because I thought step by step.")
    assert parsed.malformed
    assert not parsed.ok
    for key in TARGET_KEYS:
        assert parsed.targets[key].malformed
        assert parsed.targets[key].label == "unknown"


def test_parse_drops_chain_of_thought_and_quotes():
    payload = valid_payload()
    payload["reasoning"] = "Let me think about the report."
    payload["targets"]["acl_injury"]["quote"] = "Complete ACL tear in the midsubstance."
    parsed = parse_second_reader_output(json.dumps(payload))
    assert "reasoning" in parsed.dropped_forbidden_fields
    assert parsed.targets["acl_injury"].malformed
    assert parsed.raw_was_stored is False


def test_parse_rejects_invalid_status():
    payload = valid_payload(acl_injury={"status": "maybe"})
    parsed = parse_second_reader_output(json.dumps(payload))
    assert parsed.malformed
    assert parsed.targets["acl_injury"].malformed
    assert parsed.targets["acl_injury"].label == "unknown"


def test_uncertain_and_not_mentioned_map_to_unknown():
    payload = valid_payload(
        acl_injury={"status": "uncertain", "evidence_category": "uncertain_finding", "uncertainty_or_negation_category": "uncertain_hedged"},
        mcl_injury={"status": "not_mentioned"},
    )
    parsed = parse_second_reader_output(json.dumps(payload))
    assert parsed.targets["acl_injury"].label == "unknown"
    assert parsed.targets["mcl_injury"].label == "unknown"
    assert not parsed.targets["acl_injury"].malformed
