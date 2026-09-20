# Phase 4C.7 gold-only constrained second-reader evaluation

Sanitized operator record. Report text, StudyInstanceUID values, PatientID
values, completions, full logits, credentials, and private row contents are
not included.

Phase 4C.6 remains accepted as
`CONSTRAINED_FORMAT_PASS_SEMANTIC_GATE_NOT_ESTABLISHED`. This evaluation uses
the frozen gold acceptance criteria in `configs/second_reader_pilot_v1_4.yaml`.
Do not invent a synthetic semantic threshold. Do not emit
`CONSTRAINED_FORMAT_PASS_SEMANTIC_PASS`.

## Authorization boundary

- notebook: `notebooks/09_kaggle_second_reader_gold_constrained_v14.ipynb`
- kernel slug (new): `mahdimzafari/rsna-knee-second-reader-gold-constrained-v14`
- gold studies authorized: 58 expert-labelled studies only
- report-only studies processed: 0
- `gold_rerun_authorized: false` after the run
- `HARD_WEAK_LABEL_TRAINING_ALLOWED: false`
- `SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED: false`
- notebook 07 is a historical blocked stub and was not executed
- frozen gold kernel `mahdimzafari/rsna-knee-second-reader-gold-pilot` v7 was
  not overwritten
- frozen constrained-canary
  `mahdimzafari/rsna-knee-second-reader-constrained-canary` v3 was not
  overwritten
- PR #10 stays draft and unmerged

## Runtime path

- transport: `second-reader-transport-v1.4`
- prompt: `second-reader-prompt-v1.2.0`
- schema: `second-reader-schema-v1.0.0`
- frozen v1.4 YAML `prefix_allowed_tokens_fn: true` (hash unchanged)
- effective runtime `prefix_allowed_tokens_fn: false` (source code authoritative)
- exactly one schema-mask logits processor
- Phi EOS IDs: `[32000, 32007, 32001]`
- internet off; local tokenizer/model only

## Status before the authorized Kaggle run

CPU/software verification only in this checkout:

- Gate 1 file hashes and processor-order proof: recorded in tests
- GPU gold evaluation: not executed in this environment (no CUDA / no pinned
  Phi weights)
- `n_gold_studies_processed: 0` until the private Tesla T4 kernel completes
- `n_report_only_studies_processed: 0`

## Coordinator gates

- `NOT_READY_FOR_HARD_WEAK_LABEL_TRAINING`
- `NOT_READY_TO_SCALE_SECOND_READER_TO_REPORT_ONLY`
- evaluation result is not automatic permission to scale or train
