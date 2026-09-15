"""v1.2 JSON-aware stopping and strict parser. v1.0/v1.1 hashes stay frozen."""

from __future__ import annotations

import json
from pathlib import Path

from knee.config import TARGET_KEYS
from knee.kaggle_bundle import sha256_file
from knee.second_reader_canary import (
    SYNTHETIC_CANARY_EXAMPLES,
    fixture_canary_generate_fn,
    run_synthetic_canary,
)
from knee.second_reader_json_stop import (
    JsonObjectStoppingCriteria,
    PINNED_PHI35_TERMINATION,
    collect_eos_token_ids,
    inspect_termination_from_snapshot,
    json_object_should_stop,
    scan_json_object_progress,
    stopping_cause,
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
V1_1_HASHES = {
    "configs/second_reader_pilot_v1_1.yaml": "1ab11bbda8c6c54cbad10298669ac8cba3971f060007b31eaa8106869994abb1",
    "configs/second_reader_prompt_v1_1.yaml": "3244aa15f0f7fbfcd913e961f36e02dcb3ddaaf35b5823446431ce6ec5e31b1a",
}


class _CharTokenizer:
    """Maps one integer to one character. Used only in software tests."""

    def __init__(self, table: dict[int, str]):
        self.table = table

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self.table[int(i)] for i in ids)


def _ids_for(text: str, start: int = 1) -> tuple[list[int], dict[int, str]]:
    table = {start + i: ch for i, ch in enumerate(text)}
    return list(table), table


def test_v1_0_and_v1_1_hashes_are_unchanged():
    for rel, digest in {**V1_0_HASHES, **V1_1_HASHES}.items():
        assert sha256_file(REPO_ROOT / rel) == digest


def test_v1_2_config_keeps_weights_and_enables_json_stop():
    cfg = (REPO_ROOT / "configs" / "second_reader_pilot_v1_2.yaml").read_text(encoding="utf-8")
    assert "second-reader-transport-v1.2" in cfg
    assert "json_object_stop: true" in cfg
    assert "do_sample: false" in cfg
    assert "max_new_tokens: 2048" in cfg
    assert "generated_ids[:, input_ids.shape[1]:]" in cfg
    assert "c5214cdb995ed3dd716add8d9efbfe016b76bb2f1c4c1e6c1c6a95497d7a8837" in cfg
    assert "41246eed2b75b66526339c5d32d6f7acdefe0bd24180f97c74303f4656877344" in cfg
    assert "92a6e3f894075fda19f47d8755c6ce65b2a974716307d13ae5d79196011ebb44" in cfg
    assert sha256_file(REPO_ROOT / "configs" / "second_reader_pilot_v1_2.yaml") == (
        "d6adcac36b91e9796d0da99f048248b2d727024658c0aa642658e43ce99bdce9"
    )
    assert "process_report_only_studies: false" in cfg


def test_pinned_termination_does_not_treat_tokenizer_eos_as_phi_end():
    pinned = PINNED_PHI35_TERMINATION
    assert pinned["tokenizer_eos_token"] == "<|endoftext|>"
    assert pinned["tokenizer_eos_token_id"] == 32000
    assert pinned["end_token_id"] == 32007
    assert pinned["generation_config_eos_token_id"] == [32007, 32001, 32000]
    assert pinned["tokenizer_eos_token_id"] != pinned["end_token_id"]
    assert pinned["v1_1_stopping_criteria"] == []


def test_stopper_nested_dicts_and_arrays():
    text = '{"a":{"b":[1,{"c":2}]},"d":[]}'
    state = scan_json_object_progress(text)
    assert state.closed is True
    assert json_object_should_stop(text) is True


def test_stopper_ignores_braces_inside_quoted_strings():
    text = '{"note":"not a brace } or { here"}'
    assert json_object_should_stop(text) is True
    assert json_object_should_stop('{"note":"still open }') is False


def test_stopper_escaped_quotes():
    text = '{"note":"say \\"hi\\"} still"}'
    assert json_object_should_stop(text) is True


def test_stopper_escaped_backslashes():
    text = '{"path":"C:\\\\"}'
    assert json_object_should_stop(text) is True
    assert json_object_should_stop('{"path":"C:\\\\') is False


def test_stopper_unicode_text():
    text = '{"note":"日本語 café {not a close}"}'
    assert json_object_should_stop(text) is True


def test_stopper_incomplete_json_does_not_stop():
    assert json_object_should_stop('{"a": 1') is False
    assert json_object_should_stop("{") is False
    assert json_object_should_stop('{"a": [1, 2]') is False


def test_stopper_stops_before_trailing_prose():
    text = '{"a":1} then extra prose'
    state = scan_json_object_progress(text)
    assert state.closed is True
    assert state.close_index == text.index("}")


def test_stopper_stops_at_first_of_two_concatenated_objects():
    text = '{"a":1}{"b":2}'
    state = scan_json_object_progress(text)
    assert state.closed is True
    assert state.close_index == 6


def test_stopper_ignores_prompt_tokens_preceding_generated_json():
    prompt = "PROMPT TOKENS "
    json_text = '{"schema_version":"second-reader-schema-v1.0.0"}'
    prompt_ids, prompt_table = _ids_for(prompt, start=1)
    json_ids, json_table = _ids_for(json_text, start=100)
    table = {**prompt_table, **json_table}
    tok = _CharTokenizer(table)
    stopper = JsonObjectStoppingCriteria(tok, prompt_length=len(prompt_ids))
    assert stopper([prompt_ids + json_ids[:3]]) is False
    assert stopper([prompt_ids + json_ids]) is True
    assert stopper.stopped_on_json_close is True
    assert stopper.n_generated_tokens == len(json_ids)


def test_stopper_top_level_array_does_not_stop():
    assert json_object_should_stop("[1, 2, 3]") is False
    assert json_object_should_stop('[{"a":1}]') is False


def test_parser_rejects_valid_json_followed_by_extra_prose():
    text = json.dumps(valid_payload()) + " trailing prose"
    parsed = parse_second_reader_output(text, protocol="v1.2")
    assert parsed.malformed
    assert parsed.malformed_stage == "forbidden_extra_text"
    assert parsed.ok is False


def test_parser_rejects_two_concatenated_json_objects():
    blob = json.dumps(valid_payload(), separators=(",", ":"))
    parsed = parse_second_reader_output(blob + blob, protocol="v1.2")
    assert parsed.malformed
    assert parsed.malformed_stage == "forbidden_extra_text"


def test_parser_rejects_top_level_array():
    parsed = parse_second_reader_output("[1, 2, 3]", protocol="v1.2")
    assert parsed.malformed
    assert parsed.malformed_reason == "missing_JSON_object"


def test_parser_does_not_convert_targets_list():
    payload = valid_payload()
    payload["targets"] = [payload["targets"][key] for key in TARGET_KEYS]
    parsed = parse_second_reader_output(json.dumps(payload), protocol="v1.2")
    assert parsed.malformed
    assert parsed.malformed_reason == "targets_not_object"
    assert parsed.used_root_level_targets is False


def test_v12_does_not_unwrap_root_level_targets():
    payload = {key: valid_payload()["targets"][key] for key in TARGET_KEYS}
    payload["schema_version"] = SCHEMA_VERSION
    parsed = parse_second_reader_output(json.dumps(payload), protocol="v1.2")
    assert parsed.malformed
    assert parsed.malformed_reason == "targets_not_object"


def test_v12_accepts_all_twelve_valid_target_objects():
    parsed = parse_second_reader_output(compact_output_skeleton(), protocol="v1.2")
    assert parsed.ok
    assert list(parsed.targets) == list(TARGET_KEYS)
    assert parsed.schema_version == SCHEMA_VERSION
    assert parsed.transport_protocol == "second-reader-transport-v1.2"


def test_v12_prompt_tokens_before_json_are_not_salvaged():
    text = "Extract the twelve targets from this radiology report.\n" + json.dumps(
        valid_payload()
    )
    parsed = parse_second_reader_output(text, protocol="v1.2")
    assert parsed.malformed
    assert parsed.malformed_stage == "prompt_included_in_decoded_output"


def test_parser_rejects_non_prompt_prefix_before_json():
    text = "hello\n" + json.dumps(valid_payload())
    parsed = parse_second_reader_output(text, protocol="v1.2")
    assert parsed.malformed
    assert parsed.malformed_stage == "forbidden_extra_text"


def test_collect_eos_token_ids_unions_generation_config_and_phi_end():
    class _Tok:
        eos_token = "<|endoftext|>"
        eos_token_id = 32000
        pad_token_id = 32000
        unk_token_id = 0
        added_tokens_decoder = {
            32000: type("T", (), {"content": "<|endoftext|>"})(),
            32001: type("T", (), {"content": "<|assistant|>"})(),
            32007: type("T", (), {"content": "<|end|>"})(),
        }

        def convert_tokens_to_ids(self, name):
            return {"<|endoftext|>": 32000, "<|end|>": 32007, "<|assistant|>": 32001}[name]

        def convert_ids_to_tokens(self, tid):
            return {32000: "<|endoftext|>", 32007: "<|end|>", 32001: "<|assistant|>"}[tid]

    class _Model:
        generation_config = type("G", (), {"eos_token_id": [32007, 32001, 32000]})()

    ids = collect_eos_token_ids(_Tok(), _Model())
    assert ids == [32000, 32007, 32001] or set(ids) == {32000, 32007, 32001}
    assert 32007 in ids
    assert 32000 in ids


def test_stopping_cause_json_close_wins_over_max_tokens():
    causes = stopping_cause(
        stopped_on_json_close=True,
        last_token_id=7,
        eos_token_ids=[32000, 32007],
        n_generated_tokens=2048,
        max_new_tokens=2048,
    )
    assert causes["stopped_on_json_close"] is True
    assert causes["reached_max_new_tokens"] is False
    assert causes["stopped_on_eos"] is False


def test_inspect_termination_from_snapshot_omits_chat_template(tmp_path):
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"eos_token_id": [32007, 32001, 32000], "pad_token_id": 32000}),
        encoding="utf-8",
    )
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "eos_token": "<|endoftext|>",
                "pad_token": "<|endoftext|>",
                "chat_template": "MUST_NOT_LEAK",
                "added_tokens_decoder": {
                    "32000": {"content": "<|endoftext|>", "special": True},
                    "32007": {"content": "<|end|>", "special": True},
                },
            }
        ),
        encoding="utf-8",
    )
    payload = inspect_termination_from_snapshot(tmp_path)
    dumped = json.dumps(payload)
    assert "MUST_NOT_LEAK" not in dumped
    assert payload["snapshot"]["end_token_id"] == 32007
    assert payload["snapshot"]["generation_config_eos_token_id"] == [32007, 32001, 32000]
    assert payload["chat_template_recorded"] is False


def test_v12_fixture_canary_and_smoke_pass(tmp_path):
    smoke = run_synthetic_canary(
        output_dir=tmp_path / "smoke",
        generate_fn=fixture_canary_generate_fn,
        smoke=True,
        n_repeats=1,
    )
    assert smoke["ok"] is True
    assert smoke["n_examples"] == 3
    assert smoke["n_gold_studies_processed"] == 0
    assert smoke["n_report_only_studies_processed"] == 0
    assert smoke["generation_diagnostics"]["n_reaching_max_new_tokens"] == 0
    full = run_synthetic_canary(
        output_dir=tmp_path / "full",
        generate_fn=fixture_canary_generate_fn,
    )
    assert full["ok"] is True
    assert full["n_examples"] == len(SYNTHETIC_CANARY_EXAMPLES) == 14
    assert full["malformed_output_rate"] == 0.0
    assert full["repeatable"] is True
    assert full["transport_protocol_version"] == "second-reader-transport-v1.2"
    assert full["generation_diagnostics"]["n_beginning_with_brace"] == 14
    assert full["generation_diagnostics"]["n_containing_balanced_json"] == 14
    assert full["generation_diagnostics"]["n_targets_not_object"] == 0


def test_canary_notebook_is_v12_staged_and_synthetic_only():
    nb_path = REPO_ROOT / "notebooks" / "05_kaggle_second_reader_synthetic_canary.ipynb"
    payload = json.loads(nb_path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source") or []) for cell in payload["cells"])
    lowered = source.lower()
    assert "kaggle.json" not in lowered
    assert "pip install" not in lowered
    assert "run_synthetic_canary" in source
    assert "smoke=True" in source or "smoke=true" in source
    assert "second_reader_pilot_v1_2.yaml" in source
    assert "run_second_reader_gold_pilot" not in source
    assert "train_baseline" not in lowered
    meta = payload.get("metadata") or {}
    kaggle_meta = meta.get("kaggle") or {}
    assert kaggle_meta.get("isInternetEnabled") is False
    assert kaggle_meta.get("gpu") is True


def test_v1_canary_fail_report_is_preserved():
    text = (REPO_ROOT / "reports" / "second_reader_synthetic_canary.md").read_text(
        encoding="utf-8"
    )
    assert "CANARY_FAIL" in text
    assert "second-reader-synthetic-canary-v1.1" in text
    assert "malformed_output_rate: 1.0" in text
    assert "n reaching max_new_tokens: 10" in text
