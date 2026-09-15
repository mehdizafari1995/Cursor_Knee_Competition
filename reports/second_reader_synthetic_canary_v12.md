# Second-reader synthetic canary v1.2

Synthetic reports only. No gold studies. No report-only studies.
Report text and model completions are not included.

Kaggle kernel: https://www.kaggle.com/code/mahdimzafari/rsna-knee-second-reader-synthetic-canary
Kernel version 2, `KernelWorkerStatus.ERROR` after `SMOKE_FAIL` (SystemExit).
Internet disabled. GPU: Tesla T4 requested. Competition dataset was not attached.
`n_gold_studies_processed: 0`. `n_report_only_studies_processed: 0`.

Kernel version 1 remains the immutable v1.1 `CANARY_FAIL` record in
`reports/second_reader_synthetic_canary.md`. Kernel version 7 remains the
immutable gold `PILOT_EXECUTION_FAIL` run. The 14-example canary was not started.
The 58-study gold rerun was not launched.

- canary_version: `second-reader-synthetic-canary-v1.2`
- transport_protocol_version: `second-reader-transport-v1.2`
- prompt_version: `second-reader-prompt-v1.1.0`
- schema_version: `second-reader-schema-v1.0.0`
- status: `SMOKE_FAIL`
- ok: `false`
- n_examples: 3
- n_gold_studies_processed: 0
- n_report_only_studies_processed: 0
- malformed_output_rate: 1.0
- all_parseable: `false`
- repeatable: not evaluated (smoke is one run)
- scientific_evaluation: `false`
- stop_gold_rerun: `true`
- inference_backend: `local_transformers`
- gold_rerun_launched: `false`

## Live termination inspection (attached snapshot)

Confirmed on the Kaggle-attached Phi snapshot. Prompts, reports, and
completions were not printed.

| Field | Live value |
| --- | --- |
| `tokenizer.eos_token` | `<\|endoftext\|>` |
| `tokenizer.eos_token_id` | `32000` |
| `pad_token_id` | `32000` |
| `generation_config.eos_token_id` | `[32007, 32001, 32000]` |
| `<\|end\|>` | `32007` |
| `<\|endoftext\|>` | `32000` |
| EOS IDs passed to `generate()` | `[32000, 32007, 32001]` |
| Stopping criteria | `JsonObjectStoppingCriteria` (batch size 1) |

## Three-example smoke (GPU)

JSON-aware stopping fired. Schema enums did not.

- generated-token count min/median/max: 248/258.0/571
- configured max_new_tokens: 2048
- n reaching max_new_tokens: 0
- n ending with EOS: 0
- n beginning with `{`: 3
- n containing a balanced JSON object: 3
- n stopped on JSON close: 3
- n stopped on EOS: 0
- n `targets_not_object`: 0
- n forbidden extra text: 0
- n prompt included: 0
- malformed stage: `invalid_enum` 3/3
- raw completions stored: `false`

## Full 14-example canary

Not run. Smoke is not `ok`.

## Frozen hashes (v1.2 config; v1.0/v1.1 unchanged)

| File | SHA-256 |
| --- | --- |
| `configs/second_reader_pilot_v1_2.yaml` | `d6adcac36b91e9796d0da99f048248b2d727024658c0aa642658e43ce99bdce9` |
| `configs/second_reader_prompt_v1_1.yaml` | `3244aa15f0f7fbfcd913e961f36e02dcb3ddaaf35b5823446431ce6ec5e31b1a` |
| `configs/second_reader_schema.json` | `639c41cc4f9ba1621cc1dd45404ca82ab0a6595c3e7494dca2ef2be1888f9b20` |
| `configs/second_reader_pilot.yaml` (v1.0) | `82c257f2e8cf14cb5476e1a949ed9acbe0693308e9fdf2a8470a9ab58d5a42a5` |
| `configs/second_reader_prompt.yaml` (v1.0) | `e5bbfd9ce21bc8c5b6c25ca19177e3ef96b5cfcd9ccde5da61f5e2f1f27645d5` |
| `configs/second_reader_pilot_v1_1.yaml` | `1ab11bbda8c6c54cbad10298669ac8cba3971f060007b31eaa8106869994abb1` |

Weight shard hashes are unchanged from v1.0.

## Coordinator

- Do not rerun the 58-study gold pilot. This smoke is not `ok`.
- Do not start the 14-example canary from this smoke.
- Do not process the 4,349 report-only studies.
- Termination is no longer the v1.1 max-token continuation defect.
  Remaining smoke failure is `invalid_enum` (closed-vocabulary values),
  not `targets_not_object` and not extra trailing prose.
