# Second-reader synthetic canary v1.3

Synthetic reports only. No gold studies. No report-only studies.
Report text and model completions are not included.

Kaggle kernel: https://www.kaggle.com/code/mahdimzafari/rsna-knee-second-reader-synthetic-canary
Kernel version 3, `KernelWorkerStatus.ERROR` after `CANARY_FAIL` (`SystemExit`).
Internet disabled. GPU: Tesla T4 requested. Competition dataset was not attached.
`n_gold_studies_processed: 0`. `n_report_only_studies_processed: 0`.
`gold_rerun_launched: false`.

Immutable prior records:

- gold kernel v7: `PILOT_EXECUTION_FAIL`
- synthetic canary v1: `CANARY_FAIL`
- synthetic smoke v2: `SMOKE_FAIL`

Protocol: `second-reader-transport-v1.2` with prompt
`second-reader-prompt-v1.2.0`. Schema remains
`second-reader-schema-v1.0.0` (hash unchanged). Canary version
`second-reader-synthetic-canary-v1.3`. Pilot config
`second-reader-pilot-v1.3.0`. The production parser does not alias,
lowercase, translate, or repair enum values.

## Phase A — frozen v1.2 enum diagnosis (same three synthetic examples)

Re-run of prompt `second-reader-prompt-v1.1.0` / config
`second_reader_pilot_v1_2.yaml`. Completion SHA-256 values match kernel
version 2 exactly, so this is the same three generations.

| Counter | Value |
| --- | --- |
| parsed root-object count | 3 |
| targets-field-present count | 3 |
| targets-object count | 3 |
| targets-nonobject count | 0 |
| schema-valid count | 0 |
| n_responses_with_parsed_root_object_and_targets_field | 3 |
| container conservation | `3 + 0 == 3` (`true`) |

v2 recorded `n_targets_is_object=0` and `n_targets_not_object=0` because
object counters required a schema-valid row. That diagnostic bug is repaired:
responses that reach enum validation now increment exactly one container
counter.

Malformed stage: `invalid_enum` 3/3. JSON close 3/3. Zero `max_new_tokens`.
Generated-token min/median/max: 248/258.0/571.

### Sanitized enum-error table (all collected field violations)

| JSON path | Field | Emitted | Count |
| --- | --- | --- | --- |
| `targets.acl_injury.evidence_category` | `evidence_category` | `possible` | 1 |
| `targets.acl_injury.uncertainty_or_negation_category` | `uncertainty_or_negation_category` | `cannot_exclude` | 1 |

Allowed values are the frozen schema enums (`status`, `confidence`,
`evidence_category`, `uncertainty_or_negation_category`). Completions were
not stored. Two emitted-string rows were collected; three responses were
still staged `invalid_enum`, so at least one additional invalid-enum failure
did not produce a distinct emitted string in this table (missing field or
`None` compared with a closed enum). Status and confidence had no collected
invalid emitted strings on these three examples.

## Phase B — prompt v1.2.0

New immutable prompt. v1.1 is unchanged. Schema v1.0.0 hash is unchanged.
Transport v1.2 JSON-aware stopping and pinned Phi EOS IDs are unchanged.
No parser aliases.

## Phase C — three-example smoke (prompt v1.2.0)

Status: `ok`.

| Gate | Result |
| --- | --- |
| valid JSON | 3/3 |
| `targets` dictionaries | 3/3 |
| exactly 12 targets | 3/3 |
| invalid enums | 0 |
| missing/unexpected targets | 0 |
| extra properties | 0 |
| reaching `max_new_tokens` | 0 |
| forbidden extra text | 0 |
| prompt inclusion | 0 |
| n_gold_studies_processed | 0 |
| n_report_only_studies_processed | 0 |
| n_schema_valid | 3 |
| conservation | `3 + 0 == 3` |
| generated-token min/median/max | 553/555.0/575 |

## Phase D — 14-example repeated canary

Status: `CANARY_FAIL`. Smoke passed, so this run was required. Gold was not
launched.

| Gate | Result |
| --- | --- |
| schema-valid across both runs | 11/14 per run (not 28/28) |
| malformed_output_rate | 0.21428571428571427 |
| repeatable normalized hashes | `true` |
| invalid enums (fields) | 1 |
| target-container failures | 0 (`n_targets_not_object=0`) |
| truncation / max_new_tokens | 0 |
| forbidden extra text | 0 |
| prompt inclusion | 0 |
| n_gold_studies_processed | 0 |
| n_report_only_studies_processed | 0 |
| n_parsed_root_object | 13 |
| n_targets_field_present | 13 |
| n_targets_is_object | 13 |
| n_targets_not_object | 0 |
| n_schema_valid | 11 |
| conservation | `13 + 0 == 13` |
| generated-token min/median/max | 104/575.0/583 |
| malformed stages | `invalid_enum` 1, `missing_target` 1, `unbalanced_braces` 1 |

### Sanitized enum-error table (14-example canary)

| JSON path | Field | Emitted | Count |
| --- | --- | --- | --- |
| `targets.fracture.evidence_category` | `evidence_category` | `suspected_not_confirmed` | 1 |

`suspected_not_confirmed` is an allowed `uncertainty_or_negation_category`
value used in the wrong field. The parser did not remap it.

### Enum coverage (valid emitted vs allowed)

Status emitted: `positive`, `negative`, `uncertain`, `not_mentioned`.
Confidence emitted: `high`, `medium`, `low`.
Evidence emitted valid: `explicit_negative`, `explicit_positive`,
`not_mentioned`, `other_popliteal_finding`, `uncertain_finding`.
Evidence emitted invalid: `suspected_not_confirmed`.
Uncertainty emitted valid: `negated`, `none`, `not_mentioned`,
`uncertain_hedged`.
`malformed_output` is parser-reserved and was not expected from a valid
fixture response.

## Phase E — grammar-constrained decoder (not implemented)

Prompt-only correction left one invalid enum on the 14-example canary.
No semantic aliases were added.

A grammar-constrained decoder **can** enforce the existing JSON Schema
against the pinned Phi-3.5 tokenizer locally, without changing clinical
meaning or model weights, if it is wired as a deterministic logits mask on
the already-loaded snapshot:

- Internet off: vendor the library in the allowlisted source zip. Do not
  download weights or extra tokenizers.
- No clinical-meaning change: constrain to `second-reader-schema-v1.0.0`.
- No model-weight change: mask logits only.
- Deterministic decoding: keep `do_sample=false` / temperature 0; the
  allowed-token set must be a pure function of schema plus prefix.
- Compatible integration: a `LogitsProcessor` beside the existing
  `JsonObjectStoppingCriteria`, using the checksummed `AutoTokenizer`
  (SentencePiece, EOS IDs `[32000, 32007, 32001]`).
- Candidate families (not installed here): schema-token prefix matchers
  such as `lm-format-enforcer` `JsonSchemaParser`, or an Outlines JSON-schema
  FSM, both of which operate on Hugging Face tokenizer IDs.
- Risks to validate before any implementation: SentencePiece leading-space
  and quote splits can reject a legal JSON prefix; Phi special tokens must
  remain stop tokens; `missing_target` and `unbalanced_braces` need required
  properties / balanced-object constraints, not enum aliases.

Not implemented in this version.

## Frozen hashes

| File | SHA-256 |
| --- | --- |
| `configs/second_reader_pilot_v1_3.yaml` | `ffb67e4e7ad7ae84cf30660084e19c4c5982ef8a39a3f031cb2622f2fd1851e3` |
| `configs/second_reader_prompt_v1_2.yaml` | `5f7842b00137d0a84ea75ece5503ef31a0101255eb9ab6fcd117f9efb6723289` |
| `configs/second_reader_pilot_v1_2.yaml` | `d6adcac36b91e9796d0da99f048248b2d727024658c0aa642658e43ce99bdce9` |
| `configs/second_reader_prompt_v1_1.yaml` | `3244aa15f0f7fbfcd913e961f36e02dcb3ddaaf35b5823446431ce6ec5e31b1a` |
| `configs/second_reader_schema.json` | `639c41cc4f9ba1621cc1dd45404ca82ab0a6595c3e7494dca2ef2be1888f9b20` |
| `configs/second_reader_pilot.yaml` (v1.0) | `82c257f2e8cf14cb5476e1a949ed9acbe0693308e9fdf2a8470a9ab58d5a42a5` |
| `configs/second_reader_prompt.yaml` (v1.0) | `e5bbfd9ce21bc8c5b6c25ca19177e3ef96b5cfcd9ccde5da61f5e2f1f27645d5` |
| `configs/second_reader_pilot_v1_1.yaml` | `1ab11bbda8c6c54cbad10298669ac8cba3971f060007b31eaa8106869994abb1` |

Weight hashes are unchanged (shard-1 `c5214cdb…`, shard-2 `41246eed…`,
`config.json` `224de4f6…`, `tokenizer_config.json` `92a6e3f8…`,
`tokenizer.model` `9e556afd…`).

## Coordinator

- Do not rerun the 58-study gold pilot. This canary is not `ok`.
- Do not process the 4,349 report-only studies.
- Do not add semantic aliases to the parser.
