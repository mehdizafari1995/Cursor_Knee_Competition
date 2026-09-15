# Second-reader synthetic canary v1.2

Synthetic reports only. No gold studies. No report-only studies.
Report text and model completions are not included.

Protocol: `second-reader-transport-v1.2`. JSON-aware stopping plus all
pinned Phi EOS / end-of-turn IDs. Kernel version 7 remains
`PILOT_EXECUTION_FAIL`. Synthetic canary kernel version 1 remains the
immutable v1.1 `CANARY_FAIL` record in
`reports/second_reader_synthetic_canary.md`.

This file is the v1.2 operator record. GPU smoke / full-canary numbers are
filled after the staged Kaggle run. Until then, software fixtures are not
scientific.

- canary_version: `second-reader-synthetic-canary-v1.2`
- transport_protocol_version: `second-reader-transport-v1.2`
- prompt_version: `second-reader-prompt-v1.1.0` (unchanged clinical prompt)
- schema_version: `second-reader-schema-v1.0.0`
- n_gold_studies_processed: 0
- n_report_only_studies_processed: 0
- scientific_evaluation: `false`
- gold_rerun_launched: `false`

## Coordinator

- Do not rerun the 58-study gold pilot until this canary is `ok`.
- Do not process the 4,349 report-only studies.
