# Kernel version 7 — PILOT_EXECUTION_FAIL

Immutable technical record of the Phase 4C gold-only second-reader run.
This is an **output-format failure**, not a clinical-performance conclusion.
Kernel version 7 must not be overwritten or removed.

| Field | Value |
| --- | --- |
| status | `PILOT_EXECUTION_FAIL` |
| real_data_execution | `true` |
| scientific_performance_evaluable | `false` |
| reason | all second-reader outputs malformed |
| n_gold_studies_processed | 58 |
| n_valid_second_reader_outputs | 0 |
| n_report_only_studies_processed | 0 |
| kernel | `mahdimzafari/rsna-knee-second-reader-gold-pilot` version 7 |
| inference_backend | `local_transformers` |
| GPU | Tesla T4, CUDA 7.5 |
| transport_protocol | `second-reader-transport-v1.0` |
| prompt / schema | `second-reader-prompt-v1.0.0` / `second-reader-schema-v1.0.0` |
| decoding `max_new_tokens` | 1024 |
| decode path | `output_ids[0][prompt_len:]` with `prompt_len = input_ids.shape[-1]` (equivalent to `generated_ids[:, input_ids.shape[1]:]`) |
| chat template | `add_generation_prompt=True` |
| raw completions stored | `false` |

v1.0 prompt, schema, config and hashes remain frozen:

| file | sha256 |
| --- | --- |
| `configs/second_reader_pilot.yaml` | `82c257f2e8cf14cb5476e1a949ed9acbe0693308e9fdf2a8470a9ab58d5a42a5` |
| `configs/second_reader_prompt.yaml` | `e5bbfd9ce21bc8c5b6c25ca19177e3ef96b5cfcd9ccde5da61f5e2f1f27645d5` |
| `configs/second_reader_schema.json` | `639c41cc4f9ba1621cc1dd45404ca82ab0a6595c3e7494dca2ef2be1888f9b20` |

## Phase A failure taxonomy (aggregates only)

Private status CSV `gold_second_reader_status.csv` (58 rows, identifiers not printed)
recorded parser reason `targets_not_object` for every study. JSON was extracted
and decoded; the root was an object; `targets` was not a dict.

| malformed stage | count |
| --- | ---: |
| generation_truncated | 0 (not stored; cannot confirm) |
| prompt_included_in_decoded_output | 0 (not stored; cannot confirm) |
| empty_generation | 0 |
| missing_JSON_object | 0 |
| unbalanced_braces | 0 |
| Markdown_code_fence | 0 (not stored; cannot confirm) |
| JSON_decode_error | 0 |
| schema_validation_error | 58 (`targets_not_object`) |
| missing_target | 0 |
| unexpected_target | 0 |
| invalid_enum | 0 |
| forbidden_extra_text | 0 |
| other | 0 |

Generation token diagnostics were **not stored** in protocol v1.0
(`raw_was_stored=false`). They cannot be reconstructed without the private
completions, which this environment does not retain.

| diagnostic | v7 value |
| --- | --- |
| generated-token count min / median / max | not stored |
| configured max_new_tokens | 1024 |
| n reaching max_new_tokens | not stored |
| n ending with EOS | not stored |
| n beginning with `{` | not stored |
| n containing a balanced JSON object | 58 produced a JSON object that `json.loads` accepted |
| n wrapped only in a Markdown JSON fence | not stored |
| parser stage | `schema_validation_error` / `targets_not_object` for 58/58 |
| SHA-256 of each raw private completion | not stored |

Wall time was 47.2 seconds/study on T4 with `max_new_tokens=1024`. That is
compatible with filling the token budget, but the recorded parser stage is
**not** `JSON_decode_error` or `unbalanced_braces`, so truncation is not
established. The v1.0 greedy `{.*}` extractor can also map a complete
non-schema object onto `targets_not_object`.

## Mechanical repair (protocol v1.1)

Supported by this diagnosis, without clinical imputation:

- keep decoding `generated_ids[:, input_ids.shape[1]:]` only
- keep `add_generation_prompt=True`
- compact JSON skeleton in prompt v1.1
- `max_new_tokens=2048` as a truncation safety margin
- strip exactly one outer Markdown JSON fence
- extract one balanced JSON object; leftover must be an allowed wrapper
- after parse, unwrap root-level `TARGET_KEYS` when `targets` is absent
- enforce the twelve frozen keys and enums
- store sanitized generation diagnostics and completion SHA-256 values, never text

v1.1 transport files (new; v1.0 files above are unchanged):

| file | sha256 |
| --- | --- |
| `configs/second_reader_pilot_v1_1.yaml` | `1ab11bbda8c6c54cbad10298669ac8cba3971f060007b31eaa8106869994abb1` |
| `configs/second_reader_prompt_v1_1.yaml` | `3244aa15f0f7fbfcd913e961f36e02dcb3ddaaf35b5823446431ce6ec5e31b1a` |
