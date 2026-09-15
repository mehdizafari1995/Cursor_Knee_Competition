"""v1.1 transport/parser: balanced JSON, compact skeleton, synthetic canary."""

from __future__ import annotations

import json
from pathlib import Path

from knee.config import TARGET_KEYS
from knee.kaggle_bundle import sha256_file
from knee.report_second_reader import load_prompt_bundle, render_user_prompt
from knee.second_reader_acceptance import STATUS_EXECUTION_FAIL, decide_execution_status
from knee.second_reader_canary import (
    SYNTHETIC_CANARY_EXAMPLES,
    fixture_canary_generate_fn,
    run_synthetic_canary,
)
from knee.second_reader_parser import parse_second_reader_output
from knee.second_reader_schema import SCHEMA_VERSION, compact_output_skeleton
from tests.test_second_reader_parser import valid_payload

REPO_ROOT = Path(__file__).resolve().parents[1]

V1_0_HASHES = {
    "configs/second_reader_pilot.yaml": "82c257f2e8cf14cb5476e1a949ed9acbe0693308e9fdf2a8470a9ab58d5a42a5",
    "configs/second_reader_prompt.yaml": "e5bbfd9ce21bc8c5b6c25ca19177e3ef96b5cfcd9ccde5da61f5e2f1f27645d5",
    "configs/second_reader_schema.json": "639c41cc4f9ba1621cc1dd45404ca82ab0a6595c3e7494dca2ef2be1888f9b20",
}


def test_v1_0_prompt_schema_config_hashes_are_unchanged():
    for rel, digest in V1_0_HASHES.items():
        assert sha256_file(REPO_ROOT / rel) == digest
    assert sha256_file(REPO_ROOT / "configs" / "second_reader_pilot_v1_1.yaml") == (
        "1ab11bbda8c6c54cbad10298669ac8cba3971f060007b31eaa8106869994abb1"
    )
    assert sha256_file(REPO_ROOT / "configs" / "second_reader_prompt_v1_1.yaml") == (
        "3244aa15f0f7fbfcd913e961f36e02dcb3ddaaf35b5823446431ce6ec5e31b1a"
    )


def test_compact_skeleton_parses_and_has_all_targets():
    parsed = parse_second_reader_output(compact_output_skeleton())
    assert parsed.ok
    assert list(parsed.targets) == list(TARGET_KEYS)
    assert parsed.schema_version == SCHEMA_VERSION


def test_v1_1_prompt_injects_compact_skeleton_not_report_as_json():
    prompt = load_prompt_bundle(REPO_ROOT / "configs" / "second_reader_prompt_v1_1.yaml")
    assert prompt["prompt_version"] == "second-reader-prompt-v1.1.0"
    assert prompt["forbid_chain_of_thought"] is True
    rendered = render_user_prompt("SYNTHETIC ONLY.", prompt=prompt)
    assert compact_output_skeleton() in rendered
    assert "SYNTHETIC ONLY." in rendered
    assert "{compact_json_skeleton}" not in rendered
    assert "{report}" not in rendered


def test_balanced_extractor_does_not_take_inner_object_from_truncated_json():
    inner = json.dumps(
        {
            "status": "positive",
            "confidence": "high",
            "evidence_category": "explicit_positive",
            "uncertainty_or_negation_category": "none",
        }
    )
    truncated = '{"schema_version":"second-reader-schema-v1.0.0","targets":{"acl_injury":' + inner + ',"mcl_injury":{"status":"neg'
    v1 = parse_second_reader_output(truncated, protocol="v1.0")
    assert v1.malformed
    v11 = parse_second_reader_output(truncated, protocol="v1.1")
    assert v11.malformed
    assert v11.malformed_stage in {"unbalanced_braces", "generation_truncated"}


def test_one_outer_markdown_fence_is_stripped():
    payload = valid_payload()
    text = "```json\n" + json.dumps(payload, separators=(",", ":")) + "\n```"
    parsed = parse_second_reader_output(text)
    assert parsed.ok
    assert parsed.generation.markdown_json_fence_only is True


def test_root_level_target_keys_are_unwrapped():
    payload = {key: valid_payload()["targets"][key] for key in TARGET_KEYS}
    payload["schema_version"] = SCHEMA_VERSION
    parsed = parse_second_reader_output(json.dumps(payload))
    assert parsed.ok
    assert parsed.used_root_level_targets is True


def test_prose_around_json_is_forbidden_extra_text():
    text = "Here is the extraction:\n" + json.dumps(valid_payload())
    parsed = parse_second_reader_output(text)
    assert parsed.malformed
    assert parsed.malformed_stage == "forbidden_extra_text"


def test_prompt_echo_is_prompt_included_stage():
    text = "You are an offline clinical NLP extractor\n" + json.dumps(valid_payload())
    parsed = parse_second_reader_output(text)
    assert parsed.malformed
    assert parsed.malformed_stage == "prompt_included_in_decoded_output"


def test_generate_decodes_new_tokens_only():
    src = (REPO_ROOT / "knee" / "report_second_reader.py").read_text(encoding="utf-8")
    assert "generated_ids = output_ids[:, input_ids.shape[1]:]" in src
    assert "add_generation_prompt=True" in src
    cfg = (REPO_ROOT / "configs" / "second_reader_pilot_v1_1.yaml").read_text(encoding="utf-8")
    assert "generated_ids[:, input_ids.shape[1]:]" in cfg
    assert "max_new_tokens: 2048" in cfg
    assert "c5214cdb995ed3dd716add8d9efbfe016b76bb2f1c4c1e6c1c6a95497d7a8837" in cfg


def test_execution_fail_is_format_failure_not_clinical():
    row = decide_execution_status(
        n_gold_studies_processed=58,
        n_valid_second_reader_outputs=0,
        n_report_only_studies_processed=0,
        malformed_output_rate=1.0,
        real_data_execution=True,
    )
    assert row["status"] == STATUS_EXECUTION_FAIL
    assert row["real_data_execution"] is True
    assert row["scientific_performance_evaluable"] is False
    assert row["reason"] == "all second-reader outputs malformed"


def test_synthetic_canary_fixture_passes(tmp_path):
    assert len(SYNTHETIC_CANARY_EXAMPLES) >= 12
    languages = {row["language"] for row in SYNTHETIC_CANARY_EXAMPLES}
    coverages = {row["coverage"] for row in SYNTHETIC_CANARY_EXAMPLES}
    assert "en" in languages
    assert languages - {"en"}
    assert {"positive", "negated", "uncertain", "not_mentioned"} <= coverages
    result = run_synthetic_canary(
        output_dir=tmp_path / "canary",
        generate_fn=fixture_canary_generate_fn,
    )
    assert result["ok"] is True
    assert result["malformed_output_rate"] == 0.0
    assert result["n_gold_studies_processed"] == 0
    assert result["n_report_only_studies_processed"] == 0
    assert result["repeatable"] is True
    assert result["all_parseable"] is True
    assert result["scientific_evaluation"] is False
    assert result["n_examples"] >= 12
    summary = (tmp_path / "canary" / "sanitized" / "second_reader_synthetic_canary_summary.md").read_text()
    assert "n_gold_studies_processed: 0" in summary
    assert "CANARY_FAIL" not in summary or result["status"] == "ok"


def test_canary_notebook_is_offline_gpu_and_synthetic_only():
    nb_path = REPO_ROOT / "notebooks" / "05_kaggle_second_reader_synthetic_canary.ipynb"
    payload = json.loads(nb_path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source") or []) for cell in payload["cells"])
    lowered = source.lower()
    assert "kaggle.json" not in lowered
    assert "pip install" not in lowered
    assert "run_synthetic_canary" in source
    assert "run_second_reader_gold_pilot" not in source
    assert "train_baseline" not in lowered
    assert "create_folds" not in lowered
    meta = payload.get("metadata") or {}
    kaggle_meta = meta.get("kaggle") or {}
    assert kaggle_meta.get("isInternetEnabled") is False
    assert kaggle_meta.get("gpu") is True


def test_committed_canary_report_stops_gold_rerun():
    text = (REPO_ROOT / "reports" / "second_reader_synthetic_canary.md").read_text(encoding="utf-8")
    assert "CANARY_FAIL" in text
    assert "stop_gold_rerun: `true`" in text
    assert "n_gold_studies_processed: 0" in text
    assert "n_report_only_studies_processed: 0" in text
    assert "malformed_output_rate: 1.0" in text
    assert "Do not rerun the 58-study gold pilot" in text
    assert "PILOT_EXECUTION_FAIL" in text
