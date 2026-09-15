# Second-reader synthetic canary

Synthetic reports only. No gold studies. No report-only studies.
Report text and model completions are not included.

Kaggle kernel: https://www.kaggle.com/code/mahdimzafari/rsna-knee-second-reader-synthetic-canary
Kernel version 1, `KernelWorkerStatus.ERROR` after `CANARY_FAIL` (SystemExit).
Internet disabled. GPU: Tesla T4 requested. Competition dataset was not attached.
`n_gold_studies_processed: 0`. `n_report_only_studies_processed: 0`.

- canary_version: `second-reader-synthetic-canary-v1.1`
- transport_protocol_version: `second-reader-transport-v1.1`
- prompt_version: `second-reader-prompt-v1.1.0`
- schema_version: `second-reader-schema-v1.0.0`
- status: `CANARY_FAIL`
- ok: `false`
- n_examples: 14
- n_gold_studies_processed: 0
- n_report_only_studies_processed: 0
- malformed_output_rate: 1.0
- all_parseable: `false`
- repeatable: `true`
- scientific_evaluation: `false`
- stop_gold_rerun: `true`
- inference_backend: `local_transformers`

## Generation diagnostics (aggregates)

- generated-token count min/median/max: 970/2048.0/2048
- configured max_new_tokens: 2048
- n reaching max_new_tokens: 10
- n ending with EOS: 4
- n beginning with `{`: 14
- n containing a balanced JSON object: 0
- n wrapped only in a Markdown JSON fence: 0
- raw completions stored: `false`

Sorted SHA-256 hashes of the 14 raw private completions (no text, no
identifiers) are in the gitignored Kaggle sanitized JSON. Completions were
not downloaded into Cursor.

## Malformed-stage counts

- `generation_truncated`: 0
- `prompt_included_in_decoded_output`: 1
- `empty_generation`: 0
- `missing_JSON_object`: 0
- `unbalanced_braces`: 0
- `Markdown_code_fence`: 0
- `JSON_decode_error`: 0
- `schema_validation_error`: 0
- `missing_target`: 0
- `unexpected_target`: 0
- `invalid_enum`: 0
- `forbidden_extra_text`: 13
- `other`: 0

All 14 decoded generations began with `{`. Ten filled `max_new_tokens=2048`.
The v1.1 parser found extra text outside one JSON object (13) or a prompt
fragment (1). That is still an output-format failure.

## Coordinator

- Do not rerun the 58-study gold pilot. This canary is not `ok`.
- Kernel version 7 remains the immutable `PILOT_EXECUTION_FAIL` gold run.
- Do not process the 4,349 report-only studies.
