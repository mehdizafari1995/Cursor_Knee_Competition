"""v1.3 prompt-only enum compliance. Prior hashes stay frozen."""

from __future__ import annotations

import json
from pathlib import Path

from knee.config import TARGET_KEYS
from knee.kaggle_bundle import sha256_file
from knee.report_second_reader import load_prompt_bundle, render_user_prompt
from knee.second_reader_canary import (
    fixture_canary_generate_fn,
    fixture_expected_enum_coverage,
    run_synthetic_canary,
)
from knee.second_reader_parser import (
    collect_enum_violations,
    container_conservation_holds,
    parse_second_reader_output,
    target_container_flags,
)
from knee.second_reader_schema import (
    SCHEMA_VERSION,
    allowed_enum_values,
    compact_output_skeleton,
)
from tests.test_second_reader_parser import valid_payload

REPO_ROOT = Path(__file__).resolve().parents[1]

FROZEN_HASHES = {
    "configs/second_reader_pilot.yaml": "82c257f2e8cf14cb5476e1a949ed9acbe0693308e9fdf2a8470a9ab58d5a42a5",
    "configs/second_reader_prompt.yaml": "e5bbfd9ce21bc8c5b6c25ca19177e3ef96b5cfcd9ccde5da61f5e2f1f27645d5",
    "configs/second_reader_schema.json": "639c41cc4f9ba1621cc1dd45404ca82ab0a6595c3e7494dca2ef2be1888f9b20",
    "configs/second_reader_pilot_v1_1.yaml": "1ab11bbda8c6c54cbad10298669ac8cba3971f060007b31eaa8106869994abb1",
    "configs/second_reader_prompt_v1_1.yaml": "3244aa15f0f7fbfcd913e961f36e02dcb3ddaaf35b5823446431ce6ec5e31b1a",
    "configs/second_reader_pilot_v1_2.yaml": "d6adcac36b91e9796d0da99f048248b2d727024658c0aa642658e43ce99bdce9",
}

V1_3_HASHES = {
    "configs/second_reader_prompt_v1_2.yaml": "5f7842b00137d0a84ea75ece5503ef31a0101255eb9ab6fcd117f9efb6723289",
    "configs/second_reader_pilot_v1_3.yaml": "ffb67e4e7ad7ae84cf30660084e19c4c5982ef8a39a3f031cb2622f2fd1851e3",
}


def test_parser_accepts_every_allowed_enum():
    for field, values in allowed_enum_values().items():
        for value in values:
            payload = valid_payload()
            payload["targets"]["acl_injury"][field] = value
            parsed = parse_second_reader_output(json.dumps(payload), protocol="v1.2")
            assert parsed.ok, (field, value, parsed.malformed_reason, parsed.enum_violations)
            assert parsed.enum_violations == ()


def test_prior_hashes_including_v12_are_unchanged():
    for rel, digest in FROZEN_HASHES.items():
        assert sha256_file(REPO_ROOT / rel) == digest


def test_v13_config_and_prompt_hashes_are_frozen():
    for rel, digest in V1_3_HASHES.items():
        assert sha256_file(REPO_ROOT / rel) == digest
    assert sha256_file(REPO_ROOT / "configs" / "second_reader_schema.json") == (
        FROZEN_HASHES["configs/second_reader_schema.json"]
    )


def test_v12_prompt_lists_every_enum_and_twelve_keys():
    text = (REPO_ROOT / "configs" / "second_reader_prompt_v1_2.yaml").read_text(encoding="utf-8")
    assert "second-reader-prompt-v1.2.0" in text
    assert "case-sensitive" in text
    assert "copied exactly" in text or "copy exactly" in text.lower()
    for values in allowed_enum_values().values():
        for item in values:
            assert item in text
    for key in TARGET_KEYS:
        assert key in text
    prompt = load_prompt_bundle(REPO_ROOT / "configs" / "second_reader_prompt_v1_2.yaml")
    rendered = render_user_prompt("SYNTHETIC ONLY.", prompt=prompt)
    assert compact_output_skeleton() in rendered
    assert '"targets":{' in compact_output_skeleton().replace(" ", "")
    assert "chain-of-thought" in prompt["system"] or prompt["forbid_chain_of_thought"] is True
    assert prompt["forbid_report_quotations"] is True
    enums_at = text.index("Allowed status values")
    report_at = text.index("REPORT:")
    assert enums_at < report_at
    assert text.index("{compact_json_skeleton}") > enums_at


def test_parser_reports_all_enum_fields_not_only_the_first():
    payload = valid_payload()
    payload["targets"]["acl_injury"] = {
        "status": "Positive",
        "confidence": "HIGH",
        "evidence_category": "complete_tear",
        "uncertainty_or_negation_category": "no",
    }
    violations = collect_enum_violations(payload["targets"])
    fields = {item["field"] for item in violations}
    assert fields == {
        "status",
        "confidence",
        "evidence_category",
        "uncertainty_or_negation_category",
    }
    emitted = {item["field"]: item["emitted"] for item in violations}
    assert emitted["status"] == "Positive"
    assert emitted["confidence"] == "HIGH"
    assert emitted["evidence_category"] == "complete_tear"
    assert emitted["uncertainty_or_negation_category"] == "no"
    parsed = parse_second_reader_output(json.dumps(payload), protocol="v1.2")
    assert parsed.malformed
    assert parsed.malformed_stage == "invalid_enum"
    assert len(parsed.enum_violations) == 4
    assert parsed.targets_is_object is True
    assert parsed.targets_not_object is False
    assert parsed.parsed_root_object is True
    assert parsed.targets_field_present is True


def test_parser_does_not_fuzzy_match_or_lowercase_enums():
    payload = valid_payload(acl_injury={"status": "POSITIVE"})
    parsed = parse_second_reader_output(json.dumps(payload), protocol="v1.2")
    assert parsed.malformed
    assert parsed.enum_violations[0]["emitted"] == "POSITIVE"
    assert "positive" in parsed.enum_violations[0]["allowed"]


def test_container_conservation_for_dict_and_list_targets():
    flags_ok = target_container_flags(valid_payload())
    assert flags_ok["parsed_root_object"] is True
    assert flags_ok["targets_field_present"] is True
    assert flags_ok["targets_is_object"] is True
    assert flags_ok["targets_not_object"] is False
    listed = valid_payload()
    listed["targets"] = [listed["targets"][key] for key in TARGET_KEYS]
    flags_list = target_container_flags(listed)
    assert flags_list["targets_field_present"] is True
    assert flags_list["targets_is_object"] is False
    assert flags_list["targets_not_object"] is True
    missing = {"schema_version": SCHEMA_VERSION}
    flags_missing = target_container_flags(missing)
    assert flags_missing["parsed_root_object"] is True
    assert flags_missing["targets_field_present"] is False
    assert flags_missing["targets_is_object"] is False
    assert flags_missing["targets_not_object"] is False
    for flags in (flags_ok, flags_list, flags_missing):
        assert container_conservation_holds(
            n_targets_is_object=int(flags["targets_is_object"]),
            n_targets_not_object=int(flags["targets_not_object"]),
            n_responses_with_parsed_root_object_and_targets_field=int(
                flags["parsed_root_object"] and flags["targets_field_present"]
            ),
        )


def test_parse_missing_targets_increments_neither_container_counter():
    parsed = parse_second_reader_output(
        json.dumps({"schema_version": SCHEMA_VERSION}), protocol="v1.2"
    )
    assert parsed.parsed_root_object is True
    assert parsed.targets_field_present is False
    assert parsed.targets_is_object is False
    assert parsed.targets_not_object is False
    listed = valid_payload()
    listed["targets"] = list(listed["targets"].values())
    parsed_list = parse_second_reader_output(json.dumps(listed), protocol="v1.2")
    assert parsed_list.targets_field_present is True
    assert parsed_list.targets_is_object is False
    assert parsed_list.targets_not_object is True


def test_v13_fixture_canary_conservation_and_zero_enum_errors(tmp_path):
    smoke = run_synthetic_canary(
        output_dir=tmp_path / "smoke",
        generate_fn=fixture_canary_generate_fn,
        smoke=True,
        n_repeats=1,
    )
    assert smoke["ok"] is True
    assert smoke["container_conservation_ok"] is True
    assert smoke["n_targets_is_object"] == 3
    assert smoke["n_targets_not_object"] == 0
    assert smoke["n_parsed_root_object"] == 3
    assert smoke["n_targets_field_present"] == 3
    assert smoke["n_schema_valid"] == 3
    assert smoke["n_targets_is_object"] + smoke["n_targets_not_object"] == smoke[
        "n_responses_with_parsed_root_object_and_targets_field"
    ]
    assert smoke["enum_error_table"] == []
    assert smoke["prompt_version"] == "second-reader-prompt-v1.2.0"
    full = run_synthetic_canary(
        output_dir=tmp_path / "full",
        generate_fn=fixture_canary_generate_fn,
    )
    assert full["ok"] is True
    assert full["n_examples"] == 14
    assert full["repeatable"] is True
    assert full["n_schema_valid"] == 14
    assert full["container_conservation_ok"] is True
    expected = fixture_expected_enum_coverage()
    assert expected["status"]
    table_path = tmp_path / "full" / "sanitized" / "enum_error_table.json"
    assert table_path.is_file()
    coverage_path = tmp_path / "full" / "sanitized" / "enum_coverage.json"
    assert coverage_path.is_file()


def test_fixture_exercises_every_clinical_enum():
    expected = fixture_expected_enum_coverage()
    allowed = allowed_enum_values()
    reserved = {"evidence_category": {"malformed_output"}}
    for field, values in allowed.items():
        skip = reserved.get(field, set())
        missing = set(values) - set(expected[field]) - skip
        assert not missing, (field, sorted(missing))


def test_invalid_enum_generate_fn_increments_object_not_nonobject(tmp_path):
    def bad_fn(report: str) -> str:
        payload = json.loads(compact_output_skeleton())
        payload["targets"]["acl_injury"]["status"] = "torn"
        payload["targets"]["acl_injury"]["evidence_category"] = "complete_tear"
        return json.dumps(payload)

    result = run_synthetic_canary(
        output_dir=tmp_path / "bad",
        generate_fn=bad_fn,
        smoke=True,
        n_repeats=1,
    )
    assert result["ok"] is False
    assert result["n_targets_is_object"] == 3
    assert result["n_targets_not_object"] == 0
    assert result["n_parsed_root_object"] == 3
    assert result["n_targets_field_present"] == 3
    assert result["n_schema_valid"] == 0
    assert result["container_conservation_ok"] is True
    assert result["enum_error_table"]
    fields = {row["field"] for row in result["enum_error_table"]}
    assert "status" in fields
    assert "evidence_category" in fields
    emitted = {row["emitted"] for row in result["enum_error_table"]}
    assert "torn" in emitted
    assert "complete_tear" in emitted


def test_canary_notebook_runs_v13_smoke_after_v12_diagnosis():
    nb_path = REPO_ROOT / "notebooks" / "05_kaggle_second_reader_synthetic_canary.ipynb"
    payload = json.loads(nb_path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source") or []) for cell in payload["cells"])
    assert "second_reader_pilot_v1_2.yaml" in source
    assert "second_reader_pilot_v1_3.yaml" in source
    assert "enum_error_table" in source
    assert "run_second_reader_gold_pilot" not in source
    assert "kaggle.json" not in source.lower()
    meta = payload.get("metadata") or {}
    assert (meta.get("kaggle") or {}).get("isInternetEnabled") is False
    assert (meta.get("kaggle") or {}).get("gpu") is True


def test_v13_canary_fail_report_is_recorded():
    text = (REPO_ROOT / "reports" / "second_reader_synthetic_canary_v13.md").read_text(
        encoding="utf-8"
    )
    assert "CANARY_FAIL" in text
    assert "suspected_not_confirmed" in text
    assert "possible" in text
    assert "cannot_exclude" in text
    assert "ffb67e4e7ad7ae84cf30660084e19c4c5982ef8a39a3f031cb2622f2fd1851e3" in text
    assert "5f7842b00137d0a84ea75ece5503ef31a0101255eb9ab6fcd117f9efb6723289" in text
    assert "n_gold_studies_processed: 0" in text
    assert "gold_rerun_launched: false" in text


def test_v12_smoke_fail_report_is_preserved():
    text = (REPO_ROOT / "reports" / "second_reader_synthetic_canary_v12.md").read_text(
        encoding="utf-8"
    )
    assert "SMOKE_FAIL" in text
    assert "invalid_enum" in text
    assert "d6adcac36b91e9796d0da99f048248b2d727024658c0aa642658e43ce99bdce9" in text
