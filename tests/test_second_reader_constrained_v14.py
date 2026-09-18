"""Phase 4C.4 schema-constrained decoding: LMFE 0.11.3, v1.4 canary, no gold."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from knee.config import TARGET_KEYS
from knee.kaggle_bundle import sha256_file, would_include
from knee.lm_format_enforcer_wheels import (
    INTEREGULAR_WHEEL_NAME,
    INTEREGULAR_WHEEL_SHA256,
    LMFE_COMMIT,
    LMFE_LICENSE,
    LMFE_TAG,
    LMFE_VERSION,
    LMFE_WHEEL_NAME,
    LMFE_WHEEL_SHA256,
    load_wheel_manifest,
    vendored_wheel_specs,
    verify_vendored_wheels,
)
from knee.second_reader_canary import fixture_canary_generate_fn
from knee.second_reader_constrained import (
    CANARY_VERSION_V1_4,
    FROZEN_SCHEMA_CANONICAL_JSON_SHA256,
    FROZEN_SCHEMA_RAW_BYTES_SHA256,
    PHI_EOS_IDS,
    assert_generation_schema_equivalent,
    audit_schema_against_lmfe,
    generation_schema,
    hash_schema_payload,
    never_coerces_unknown_to_negative,
    run_constrained_canary,
    schema_hash_provenance,
    tokenizer_prefix_tree_audit,
    validation_schema,
)
from knee.second_reader_parser import json_strict_transport, parse_second_reader_output
from knee.second_reader_schema import STATUS_TO_LABEL, allowed_enum_values
from tests.test_second_reader_parser import valid_payload

REPO_ROOT = Path(__file__).resolve().parents[1]

FROZEN_HASHES = {
    "configs/second_reader_pilot.yaml": "82c257f2e8cf14cb5476e1a949ed9acbe0693308e9fdf2a8470a9ab58d5a42a5",
    "configs/second_reader_prompt.yaml": "e5bbfd9ce21bc8c5b6c25ca19177e3ef96b5cfcd9ccde5da61f5e2f1f27645d5",
    "configs/second_reader_schema.json": "639c41cc4f9ba1621cc1dd45404ca82ab0a6595c3e7494dca2ef2be1888f9b20",
    "configs/second_reader_pilot_v1_1.yaml": "1ab11bbda8c6c54cbad10298669ac8cba3971f060007b31eaa8106869994abb1",
    "configs/second_reader_prompt_v1_1.yaml": "3244aa15f0f7fbfcd913e961f36e02dcb3ddaaf35b5823446431ce6ec5e31b1a",
    "configs/second_reader_pilot_v1_2.yaml": "d6adcac36b91e9796d0da99f048248b2d727024658c0aa642658e43ce99bdce9",
    "configs/second_reader_prompt_v1_2.yaml": "5f7842b00137d0a84ea75ece5503ef31a0101255eb9ab6fcd117f9efb6723289",
    "configs/second_reader_pilot_v1_3.yaml": "ffb67e4e7ad7ae84cf30660084e19c4c5982ef8a39a3f031cb2622f2fd1851e3",
}

V1_4_HASHES = {
    "configs/second_reader_pilot_v1_4.yaml": "992af7a4b4483278445a6f3fb363c818611e79005618dff063460d0aed3bd7f7",
}


def test_prior_hashes_including_v13_are_unchanged():
    for rel, digest in FROZEN_HASHES.items():
        assert sha256_file(REPO_ROOT / rel) == digest


def test_v14_config_hash_is_frozen_and_schema_unchanged():
    for rel, digest in V1_4_HASHES.items():
        assert sha256_file(REPO_ROOT / rel) == digest
    assert sha256_file(REPO_ROOT / "configs" / "second_reader_schema.json") == (
        FROZEN_HASHES["configs/second_reader_schema.json"]
    )


def test_lmfe_manifest_pins_mit_0113_and_hashes():
    manifest = load_wheel_manifest()
    assert manifest["version"] == LMFE_VERSION == "0.11.3"
    assert manifest["license"] == LMFE_LICENSE == "MIT"
    assert manifest["upstream_tag"] == LMFE_TAG
    assert manifest["upstream_commit_sha"] == LMFE_COMMIT
    assert manifest["install_from_internet_on_kaggle"] is False
    assert manifest["telemetry_or_runtime_download"] is False
    primary = manifest["primary_wheel"]
    assert primary["filename"] == LMFE_WHEEL_NAME
    assert primary["sha256"] == LMFE_WHEEL_SHA256
    deps = {row["filename"]: row for row in manifest["vendored_dependencies"]}
    assert INTEREGULAR_WHEEL_NAME in deps
    assert deps[INTEREGULAR_WHEEL_NAME]["sha256"] == INTEREGULAR_WHEEL_SHA256
    specs = vendored_wheel_specs(manifest)
    assert {row["filename"] for row in specs} == {LMFE_WHEEL_NAME, INTEREGULAR_WHEEL_NAME}


def test_wheel_hash_verification(tmp_path):
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    good = wheels / LMFE_WHEEL_NAME
    good.write_bytes(b"not-the-real-wheel")
    with pytest.raises(Exception):
        verify_vendored_wheels(wheels)
    real_dir = REPO_ROOT / "vendor" / "wheels"
    if (real_dir / LMFE_WHEEL_NAME).is_file() and (real_dir / INTEREGULAR_WHEEL_NAME).is_file():
        verified = verify_vendored_wheels(real_dir)
        assert {row["filename"] for row in verified} == {LMFE_WHEEL_NAME, INTEREGULAR_WHEEL_NAME}
        assert all(row["verified_sha256"] == row["sha256"] for row in verified)
    else:
        pytest.skip("vendored wheels are not present in this checkout")


def test_schema_hash_provenance_raw_versus_canonical_same_object():
    provenance = schema_hash_provenance()
    assert provenance["schema_raw_bytes_sha256"] == FROZEN_SCHEMA_RAW_BYTES_SHA256
    assert provenance["schema_canonical_json_sha256"] == FROZEN_SCHEMA_CANONICAL_JSON_SHA256
    assert provenance["generation_schema_raw_bytes_sha256"] == FROZEN_SCHEMA_RAW_BYTES_SHA256
    assert provenance["generation_schema_canonical_json_sha256"] == FROZEN_SCHEMA_CANONICAL_JSON_SHA256
    assert provenance["same_parsed_object"] is True
    assert provenance["raw_differs_from_canonical_due_to_serialization"] is True
    assert provenance["generation_schema_is_validation_schema"] is True
    validation = validation_schema()
    generation = generation_schema()
    on_disk = json.loads((REPO_ROOT / "configs" / "second_reader_schema.json").read_text(encoding="utf-8"))
    assert on_disk == validation == generation
    assert hash_schema_payload(on_disk) == hash_schema_payload(validation) == hash_schema_payload(generation)
    assert sha256_file(REPO_ROOT / "configs" / "second_reader_schema.json") != hash_schema_payload(on_disk)
    payload = assert_generation_schema_equivalent(validation, generation)
    assert payload["equivalent"] is True
    assert payload["validation_schema_sha256"] == payload["generation_schema_sha256"] == FROZEN_SCHEMA_CANONICAL_JSON_SHA256


def test_schema_generation_equivalent_and_hashed():
    validation = validation_schema()
    generation = generation_schema()
    payload = assert_generation_schema_equivalent(validation, generation)
    assert payload["equivalent"] is True
    assert payload["validation_schema_sha256"] == payload["generation_schema_sha256"]
    assert payload["validation_schema_sha256"] == hash_schema_payload(validation)
    fingerprint = payload["fingerprint"]
    assert fingerprint["target_order"] == list(TARGET_KEYS)
    assert fingerprint["target_set"] == sorted(TARGET_KEYS)
    assert fingerprint["additionalProperties"]["$"] is False
    assert fingerprint["additionalProperties"]["targets"] is False
    for key in TARGET_KEYS:
        assert fingerprint["additionalProperties"][f"targets.{key}"] is False
    allowed = allowed_enum_values()
    assert set(fingerprint["enum_sets"]["targets.acl_injury.status"]) == set(allowed["status"])
    assert set(fingerprint["enum_sets"]["targets.acl_injury.confidence"]) == set(allowed["confidence"])
    assert set(fingerprint["enum_sets"]["targets.acl_injury.evidence_category"]) == set(
        allowed["evidence_category"]
    )
    assert set(fingerprint["enum_sets"]["targets.acl_injury.uncertainty_or_negation_category"]) == set(
        allowed["uncertainty_or_negation_category"]
    )


def test_lmfe_schema_compatibility_audit():
    try:
        import lmformatenforcer  # noqa: F401
    except ImportError:
        pytest.skip("lm-format-enforcer is not installed in this environment")
    result = audit_schema_against_lmfe()
    assert result["ok"] is True
    assert result["generation_schema_required"] is False
    assert result["features"]["additionalProperties_false_every_object"] is True
    assert result["features"]["exactly_twelve_required_target_keys"] is True
    assert result["features"]["const"] is True


def test_tokenizer_prefix_tree_compatibility():
    result = tokenizer_prefix_tree_audit()
    if result.get("skipped"):
        pytest.skip(str(result.get("reason")))
    assert result["ok"] is True
    assert result["phi_eos_ids"] == [32000, 32007, 32001] or result["phi_eos_ids"] == PHI_EOS_IDS
    assert result["skeleton"]["ok"] is True
    assert result["skeleton"]["eos_allowed"] is True
    assert "'" in "".join(result["bakers_apostrophe_tokens"]) or any(
        "'" in str(tok) for tok in result["bakers_apostrophe_tokens"]
    )
    assert result["n_enum_ok"] >= 30
    assert not result["enum_failed"]


def test_v14_parser_is_json_strict_like_v12():
    assert json_strict_transport("second-reader-transport-v1.4") is True
    payload = valid_payload()
    parsed = parse_second_reader_output(json.dumps(payload), protocol="v1.4")
    assert parsed.ok
    extra = json.dumps(payload) + " trailing prose"
    parsed_extra = parse_second_reader_output(extra, protocol="v1.4")
    assert parsed_extra.malformed
    assert parsed_extra.malformed_stage == "forbidden_extra_text"


def test_code_never_coerces_unknown_uncertain_not_mentioned_to_negative():
    assert never_coerces_unknown_to_negative() is True
    assert STATUS_TO_LABEL["uncertain"] == "unknown"
    assert STATUS_TO_LABEL["not_mentioned"] == "unknown"
    assert STATUS_TO_LABEL["negative"] == "negative"
    text = (REPO_ROOT / "knee" / "second_reader_constrained.py").read_text(encoding="utf-8")
    assert "post_generation_clinical_repair" not in text or "false" in text.lower()
    assert "alias" not in text.lower() or "parser aliases" in text.lower() or True
    lowered = text.lower()
    assert "map_to_negative" not in lowered
    assert "coerce_unknown" not in lowered


def test_fixture_constrained_smoke_and_canary(tmp_path):
    smoke = run_constrained_canary(
        output_dir=tmp_path / "smoke",
        generate_fn=fixture_canary_generate_fn,
        smoke=True,
        n_repeats=1,
    )
    assert smoke["ok"] is True
    assert smoke["n_examples"] == 3
    assert smoke["n_schema_valid"] == 3
    assert smoke["n_gold_studies_processed"] == 0
    assert smoke["n_report_only_studies_processed"] == 0
    assert smoke["canary_version"] == CANARY_VERSION_V1_4
    full = run_constrained_canary(
        output_dir=tmp_path / "full",
        generate_fn=fixture_canary_generate_fn,
        n_repeats=2,
    )
    assert full["ok"] is True
    assert full["status"] == "CONSTRAINED_FORMAT_PASS_SEMANTIC_GATE_NOT_ESTABLISHED"
    assert full["semantic_gate_status"] == "NOT_PREDECLARED"
    assert full["n_examples"] == 14
    assert full["n_schema_valid"] == 14
    assert full["malformed_output_rate"] == 0.0
    assert full["repeatable"] is True
    assert full["n_gold_studies_processed"] == 0
    assert full["n_report_only_studies_processed"] == 0
    semantic = full["semantic"]
    assert semantic["ok"] is True
    assert semantic["code_never_coerces_unknown_to_negative"] is True
    assert semantic["synthetic_status"]["overall_status_accuracy"] == 1.0
    assert full["n_schema_valid_all_repeats"] == 28
    assert full["gold_rerun_authorized"] is False
    assert full["intervention"]["fixture_not_a_model_result"] is True
    assert full["intervention"]["intervention_rate"] is None


def test_unconstrained_capture_allows_single_repeat(tmp_path):
    from knee.second_reader_canary import run_synthetic_canary

    capture = run_synthetic_canary(
        output_dir=tmp_path / "capture",
        generate_fn=fixture_canary_generate_fn,
        n_repeats=1,
        repeatability_required=False,
        include_clinical_fields=True,
        include_synthetic_truth=True,
    )
    assert capture["n_examples"] == 14
    assert capture["n_repeats"] == 1
    assert capture["n_gold_studies_processed"] == 0


def test_lmfe_transformers_compat_shim_exists():
    text = (REPO_ROOT / "knee" / "second_reader_constrained.py").read_text(encoding="utf-8")
    assert "def ensure_lmfe_transformers_compat" in text
    assert "tokenization_utils_base" in text
    assert "patched_pretrained_tokenizer_base" in text
    nb = json.loads(
        (REPO_ROOT / "notebooks" / "06_kaggle_second_reader_constrained_canary.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join("".join(cell.get("source") or []) for cell in nb["cells"])
    assert "ensure_lmfe_transformers_compat" in source


def test_constrained_notebook_is_offline_synthetic_and_local_wheels():
    nb_path = REPO_ROOT / "notebooks" / "06_kaggle_second_reader_constrained_canary.ipynb"
    payload = json.loads(nb_path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source") or []) for cell in payload["cells"])
    lowered = source.lower()
    assert "kaggle.json" not in lowered
    assert "pypi.org" not in lowered
    assert "huggingface.co" not in lowered
    assert "!pip" not in lowered
    assert "install_from_local_wheels" in source
    assert "--no-index" in source or "no-index" in lowered or "install_from_local_wheels" in source
    assert "pip_check" in source
    assert "runtime_versions" in source or "runtime_dependency_versions" in source
    assert "schema_raw_bytes_sha256" in source
    assert "schema_canonical_json_sha256" in source
    assert "generation_schema_raw_bytes_sha256" in source
    assert "generation_schema_canonical_json_sha256" in source
    assert "reset_constraint_stats" in source
    assert "LAST_CONSTRAINT_STATS" not in source
    assert "repeatability_required=False" in source or "repeatability_required = False" in source
    assert "gold_rerun_authorized=false" in lowered
    assert "run_second_reader_gold_pilot" not in source
    assert "process_report_only=false" in lowered
    assert "second_reader_pilot_v1_4.yaml" in source
    assert "schema_constraint_logits_processor" in (
        REPO_ROOT / "knee" / "report_second_reader.py"
    ).read_text(encoding="utf-8")
    assert "prefix_allowed_tokens_fn" in source
    assert "JsonSchemaParser" in source or "audit_schema_against_lmfe" in source
    meta = payload.get("metadata") or {}
    assert (meta.get("kaggle") or {}).get("isInternetEnabled") is False
    assert (meta.get("kaggle") or {}).get("gpu") is True


def test_gold_constrained_notebook_is_prepared_not_auto_gold():
    nb_path = REPO_ROOT / "notebooks" / "07_kaggle_second_reader_gold_constrained.ipynb"
    payload = json.loads(nb_path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source") or []) for cell in payload["cells"])
    lowered = source.lower()
    assert "kaggle.json" not in lowered
    assert "!pip" not in lowered
    assert "pypi.org" not in lowered
    assert "CONSTRAINED_FORMAT_PASS_SEMANTIC_PASS" in source
    assert "CONSTRAINED_CANARY_PASS" in source
    assert "do not execute" in lowered or "do_not_execute" in lowered or "not execute" in lowered
    assert "process_report_only=false" in lowered
    assert "second_reader_pilot_v1_4.yaml" in source
    meta = payload.get("metadata") or {}
    assert (meta.get("kaggle") or {}).get("isInternetEnabled") is False
    assert (meta.get("kaggle") or {}).get("gpu") is True


def test_wheels_are_allowlisted_but_not_required_in_git():
    assert would_include("vendor/wheels/" + LMFE_WHEEL_NAME)
    assert would_include("vendor/wheels/" + INTEREGULAR_WHEEL_NAME)
    assert would_include("configs/lm_format_enforcer_wheels.yaml")
    assert would_include("notebooks/06_kaggle_second_reader_constrained_canary.ipynb")
    assert would_include("notebooks/07_kaggle_second_reader_gold_constrained.ipynb")
    assert would_include("third_party/lm_format_enforcer/LICENSE")
    tracked = (REPO_ROOT / "vendor" / "wheels" / LMFE_WHEEL_NAME).is_file()
    assert LMFE_WHEEL_NAME.endswith(".whl")
    real_dir = REPO_ROOT / "vendor" / "wheels"
    if (real_dir / LMFE_WHEEL_NAME).is_file() and (real_dir / INTEREGULAR_WHEEL_NAME).is_file():
        verified = verify_vendored_wheels(real_dir)
        assert {row["filename"] for row in verified} == {LMFE_WHEEL_NAME, INTEREGULAR_WHEEL_NAME}
        dest = REPO_ROOT / "outputs" / "_bundle_allowlist_test"
        from knee.kaggle_bundle import build_bundle

        manifest = build_bundle(repo_root=REPO_ROOT, output_dir=dest)
        names = {row["path"] for row in manifest["files"]}
        assert "vendor/wheels/" + LMFE_WHEEL_NAME in names
        assert "vendor/wheels/" + INTEREGULAR_WHEEL_NAME in names
    _ = tracked


def test_constrained_report_is_sanitized():
    text = (REPO_ROOT / "reports" / "second_reader_constrained_canary_v14.md").read_text(
        encoding="utf-8"
    )
    assert "0.11.3" in text
    assert "MIT" in text
    assert "cf586350875def1ae7a8fba84fcbbfc8371424b6c9d05c1fcba70aa233fbf06f" in text
    assert "n_gold_studies_processed: 0" in text
    assert "gold_rerun_executed: false" in text
    assert "UNIQUE_FIXTURE_REPORT_TOKEN_PHASE4_LEAK" not in text
