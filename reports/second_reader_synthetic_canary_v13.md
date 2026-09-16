# Second-reader synthetic canary v1.3

Synthetic reports only. No gold studies. No report-only studies.
Report text and model completions are not included.

Protocol: `second-reader-transport-v1.2` with prompt
`second-reader-prompt-v1.2.0`. Schema remains
`second-reader-schema-v1.0.0` (hash unchanged). Canary version
`second-reader-synthetic-canary-v1.3`. Pilot config
`second-reader-pilot-v1.3.0`.

Immutable prior records:

- gold kernel v7: `PILOT_EXECUTION_FAIL`
- synthetic canary v1: `CANARY_FAIL`
- synthetic smoke v2: `SMOKE_FAIL`

The production parser does not alias, lowercase, translate, or repair enum
values. Completions are not stored.

GPU diagnosis / smoke / 14-example numbers are filled after the staged
Kaggle run. Until then, software fixtures are not scientific.

- n_gold_studies_processed: 0
- n_report_only_studies_processed: 0
- scientific_evaluation: `false`
- gold_rerun_launched: `false`

## Phase A — frozen v1.2 enum diagnosis (same three synthetic examples)

Re-run of prompt `second-reader-prompt-v1.1.0` / config
`second_reader_pilot_v1_2.yaml` with the v1.3 diagnostic counters. Exact
emitted strings are recorded only in the sanitized enum-error table after the
Kaggle diagnosis cell.

Until that GPU run, the v2 smoke record is:

- n beginning with `{`: 3
- n containing a balanced JSON object: 3
- n stopped on JSON close: 3
- n reaching max_new_tokens: 0
- malformed stage: `invalid_enum` 3/3
- n_targets_is_object / n_targets_not_object in v2: 0 / 0 (counter bug;
  conservation is repaired in this version)

## Phase B — prompt v1.2.0

New immutable prompt. v1.1 is unchanged. Schema v1.0.0 hash is unchanged.
Transport v1.2 JSON-aware stopping and pinned Phi EOS IDs are unchanged.

## Phase C — three-example smoke

Not yet run on GPU.

## Phase D — 14-example repeated canary

Not run until smoke is `ok`.

## Phase E — grammar-constrained decoder

Not evaluated until the prompt-only canary result is reported. No decoder is
implemented in this version.

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

- Do not rerun the 58-study gold pilot until this canary is `ok`.
- Do not process the 4,349 report-only studies.
- Do not add semantic aliases to the parser.
