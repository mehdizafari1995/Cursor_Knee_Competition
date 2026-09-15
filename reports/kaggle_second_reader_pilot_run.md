# Kaggle gold-only second-reader pilot run

Operator record. Scientific metrics live in
`reports/second_reader_gold_pilot_comparison.md`. Report text,
StudyInstanceUID values, and PatientID values are not included.

`KAGGLE_API_TOKEN_present: True`. Weights were not downloaded in Cursor.
Latest scientific gold execution remains version 7,
`KernelWorkerStatus.COMPLETE`, recorded as `PILOT_EXECUTION_FAIL`.
That version must stay immutable. Protocol v1.1 uses a separate synthetic
canary kernel before any new gold notebook version.

| Field | Value |
| --- | --- |
| Notebook URL | https://www.kaggle.com/code/mahdimzafari/rsna-knee-second-reader-gold-pilot |
| Kernel slug | `mahdimzafari/rsna-knee-second-reader-gold-pilot` |
| Latest gold execution | version 7, `KernelWorkerStatus.COMPLETE`, `PILOT_EXECUTION_FAIL` |
| real_data_execution | `true` |
| scientific_performance_evaluable | `false` |
| reason | all second-reader outputs malformed |
| n_gold_studies_processed | 58 |
| n_valid_second_reader_outputs | 0 |
| n_report_only_studies_processed | 0 |
| Visibility | private |
| Internet | disabled |
| Accelerator | `NvidiaTeslaT4` |
| GPU observed | Tesla T4, CUDA 7.5 |
| Competition source | `rsna-knee-abnormality-detection` |
| Dataset source | `mahdimzafari/cursor-knee-source` |
| Model source | `richolson/phi-3.5-mini-instruct/pyTorch/default/1` |
| DATA_SOURCE | `real_kaggle_mount` |
| scientific_evaluation | `true` |
| inference_backend | `local_transformers` |
| n_gold_studies | 58 |
| n_report_only_studies_processed | 0 |
| n_report_only_studies_refused | 4349 |
| download_weights | `false` |
| internet_used | `false` |
| training_run / folds / leaderboard | `false` / `false` / `false` |
| wall_seconds | 2739.7 |
| mean_seconds_per_study | 47.2 |
| gpu_memory_bytes | 6368627712 |
| malformed_output_rate | 1.0 (58/58) |

Freeze hashes were written with `metrics_computed: false` before scoring.
On-Kaggle SHA-256 values for both weight shards, `config.json`,
`tokenizer_config.json`, and `tokenizer.model` matched the frozen table.

Earlier kernel versions did not produce scientific metrics:

- version 1 aborted on `assert_hard_weak_label_training_forbidden`
- version 3 failed to resolve relative prompt paths from `/kaggle/working`
- version 4 hit P100 CUDA sm_60 incompatibility (later runs requested T4)
- version 5 wrapped a tokenizer `BatchEncoding` as `input_ids`
- version 6 mixed 2024 `modeling_phi3.py` with a newer cache API, then a
  12-hour session with KV cache disabled was cancelled

Version 7 used native Phi-3 weights, KV cache, and T4. Progress logs
printed study counts only (`1/58` … `58/58`), not report text.

Private row-level CSVs were not downloaded into Cursor. Their SHA-256
hashes are in the comparison report.

Coordinator gates were not changed:

- `NOT_READY_FOR_HARD_WEAK_LABEL_TRAINING`
- `READY_FOR_SELECTIVE_LABEL_REFINEMENT`
- `NOT_READY_TO_SCALE_SECOND_READER_TO_REPORT_ONLY`

Do not process the 4,349 report-only studies. Model-only predictions are
not training labels. Kernel version 7 is `PILOT_EXECUTION_FAIL` because
every second-reader output was malformed JSON (`targets_not_object`). That
is an output-format failure, not a clinical-performance conclusion.
See `reports/second_reader_v7_execution_fail.md`.

## Cursor

```bash
source .venv/bin/activate
pytest
python scripts/run_second_reader_pilot.py   # DATA_NOT_AVAILABLE without RSNA_DATA_ROOT
```
