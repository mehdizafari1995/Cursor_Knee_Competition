# Phase 4C.7 gold-only constrained second-reader evaluation

Sanitized operator and scientific record. Report text, StudyInstanceUID
values, PatientID values, completions, full logits, credentials, and
private row contents are not included. Private row files remain on the
Kaggle kernel; only SHA-256 hashes are recorded here.

Phase 4C.6 remains accepted as
`CONSTRAINED_FORMAT_PASS_SEMANTIC_GATE_NOT_ESTABLISHED`. This evaluation
uses the frozen gold acceptance criteria in
`configs/second_reader_pilot_v1_4.yaml`. Do not invent a synthetic
semantic threshold. Do not emit `CONSTRAINED_FORMAT_PASS_SEMANTIC_PASS`.
This result is not automatic permission to scale or train.

## Kernel

- kernel slug: `mahdimzafari/rsna-knee-second-reader-gold-constrained-v14`
- version: 1 (`KernelWorkerStatus.COMPLETE`)
- notebook: `notebooks/09_kaggle_second_reader_gold_constrained_v14.ipynb`
- GPU: Tesla T4, CUDA 7.5, internet off
- competition: `rsna-knee-abnormality-detection`
- source dataset: `mahdimzafari/cursor-knee-source` (Phase 4C.7 zip)
- model: `richolson/phi-3.5-mini-instruct/PyTorch/default/1`
- packaged source HEAD: `7a1072c90cb547f510fdf18b573397553879ab6f`
- Python 3.12.13, torch 2.10.0+cu128, transformers 5.0.0,
  lm-format-enforcer 0.11.3, interegular 0.3.3
- wall_seconds: 2003.583
- mean_seconds_per_study: 34.544
- peak GPU memory: 8184309248 bytes
- notebook 07 executed: false
- frozen gold kernel `mahdimzafari/rsna-knee-second-reader-gold-pilot` v7:
  remained `COMPLETE`
- frozen constrained-canary
  `mahdimzafari/rsna-knee-second-reader-constrained-canary` v3: remained
  `COMPLETE`

## Runtime path and hashes

- transport: `second-reader-transport-v1.4`
- prompt: `second-reader-prompt-v1.2.0`
- schema: `second-reader-schema-v1.0.0`
- frozen v1.4 YAML `prefix_allowed_tokens_fn: true`
  (SHA-256 `992af7a4b4483278445a6f3fb363c818611e79005618dff063460d0aed3bd7f7`)
- effective runtime `prefix_allowed_tokens_fn: false` (source code
  authoritative)
- `model.generate()` does not receive `prefix_allowed_tokens_fn`
- unconstrained n_mask=0; constrained n_mask=1
- Phi EOS IDs: `[32000, 32007, 32001]`
- Gate 1 file hashes: ok, mismatches none
- schema raw-bytes SHA-256:
  `639c41cc4f9ba1621cc1dd45404ca82ab0a6595c3e7494dca2ef2be1888f9b20`
- schema canonical JSON SHA-256:
  `c39e6236e509c466850e501969e7e8525bd40a189ab05ed3d591e99bd30d3b05`
- prompt SHA-256:
  `5f7842b00137d0a84ea75ece5503ef31a0101255eb9ab6fcd117f9efb6723289`
- weights and tokenizer hashes matched the frozen Phi-3.5 revision
  `2fe192450127e6a83f7441aef6e3ca586c338b77`

## Selection (before generate)

- n_gold_studies_selected: 58
- n_report_only_studies_present: 4349
- n_report_only_studies_selected: 0
- process_report_only_studies: false
- gold labels were not parameters of `run_second_reader`

## Execution

- status: `PILOT_EXECUTED`
- DATA_SOURCE: `real_kaggle_mount`
- scientific_evaluation: true
- scientific_performance_evaluable: true
- inference_backend: `local_transformers`
- n_gold_studies_processed: 58
- n_valid_second_reader_outputs: 58
- n_malformed_studies: 0
- malformed_output_rate: 0.0
- n_report_only_studies_processed: 0
- n_report_only_studies_refused: 4349
- 58/58 outputs began with `{` and contained balanced JSON
- n_reaching_max_new_tokens: 0
- generated-token min/median/max: 527 / 574 / 737
- code never maps uncertain / not_mentioned / unknown to negative
- `gold_rerun_authorized: false`
- `gold_rerun_executed: false`
- `HARD_WEAK_LABEL_TRAINING_ALLOWED: false`
- `SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED: false`
- forbidden outcome not used: `CONSTRAINED_FORMAT_PASS_SEMANTIC_PASS`
- `semantic_gate_status: NOT_PREDECLARED`

## Recorder (transformers logits processor)

- n_log_entries: 58
- source: `transformers_logits_processor`
- fixture_not_a_model_result: false
- n_calls: 33400
- n_generated_tokens: 33400
- n_top1_allowed: 33243
- n_top1_blocked: 157
- allowed + blocked = generated steps
- intervention_rate: 0.004701
- structural: 76 (target-key 31, property-name 37, json-structure 8,
  other 0)
- clinical-enum: 81
- raw_completions_stored: false
- logits_retained: false

## Frozen gold acceptance (primary system: strict agreement)

Coverage is reported, not a pass gate. Small-n targets cannot `PILOT_PASS`.

| Official column | status | precision | NPV | coverage | abstentions | reasons |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| ACL | `PILOT_RESTRICTED` | 1.000 | n/a | 0.172 | 48 | positive_precision_pass_npv_or_n_fail |
| MCL | `PILOT_RESTRICTED` | 1.000 | n/a | 0.017 | 57 | small_n gold_positives=9; small_n_cannot_pass |
| Medial Meniscus | `PILOT_PASS` | 1.000 | 1.000 | 0.172 | 48 | precision_and_npv_pass |
| Lateral Meniscus | `PILOT_RESTRICTED` | 0.750 | 1.000 | 0.155 | 49 | npv_pass_precision_or_n_fail |
| Medial OA | `PILOT_RESTRICTED` | 1.000 | n/a | 0.069 | 54 | positive_precision_pass_npv_or_n_fail |
| Lateral OA | `PILOT_FAIL` | 0.500 | n/a | 0.069 | 54 | small_n gold_positives=11; precision_and_npv_fail |
| PF OA | `PILOT_FAIL` | 0.600 | n/a | 0.086 | 53 | precision_and_npv_fail |
| Effusion | `PILOT_FAIL` | 0.579 | n/a | 0.328 | 39 | precision_and_npv_fail |
| Synovitis | `PILOT_FAIL` | 0.571 | n/a | 0.121 | 51 | precision_and_npv_fail |
| Baker's | `PILOT_FAIL` | 0.667 | n/a | 0.103 | 52 | small_n gold_positives=12; precision_and_npv_fail |
| Contusion | `PILOT_FAIL` | 0.833 | n/a | 0.103 | 52 | precision_and_npv_fail |
| Fracture | `PILOT_RESTRICTED` | 1.000 | n/a | 0.069 | 54 | positive_precision_pass_npv_or_n_fail |

NPV is n/a when the system produced no negatives for that target (no
denominator). That is not imputed as 0/1.

## Macro (descriptive)

| System | coverage | precision | NPV | pos capture | neg capture | selective error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| deterministic_v2 | 0.254 | 0.674 | 0.980 | 0.362 | 0.103 | 0.262 |
| second_reader | 0.190 | 0.736 | 0.583 | 0.373 | 0.014 | 0.280 |
| strict_agreement | 0.122 | 0.792 | 1.000 | 0.247 | 0.008 | 0.206 |

## Private output hashes (contents not committed)

| file | sha256 | bytes |
| --- | --- | ---: |
| `gold_deterministic_predictions.csv` | `a53150bb9077a1fdf8b270953a3d2a19dd1fd00861ce193aca0d44b3623c32cc` | 79807 |
| `gold_second_reader_predictions.csv` | `ba792cdbfaf3638a020aaecb417a54391dceadb18f8b6ed59e080f46a527f4a0` | 59969 |
| `gold_consensus_predictions.csv` | `8853f1e5e895e554f6f154c702b465deb3a90a6ec6d85a1839592648f0d830db` | 57086 |
| `gold_second_reader_status.csv` | `c21f6a0dfb2548ddfd90a302c62c62ed0bd8ebb8c080518682bd08ce5acca6de` | 11849 |

## Coordinator gates after the run

- `NOT_READY_FOR_HARD_WEAK_LABEL_TRAINING`
- `NOT_READY_TO_SCALE_SECOND_READER_TO_REPORT_ONLY`
- `gold_rerun_authorized: false`
- do not process the 4,349 report-only studies
- do not merge PR #10
