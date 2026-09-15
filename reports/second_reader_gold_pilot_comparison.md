# Second-reader gold-pilot comparison

Sanitized aggregates only. Report text, StudyInstanceUID values, and
PatientID values are not included. Chain-of-thought is not stored.

- pilot_version: `second-reader-pilot-v1.0.0`
- eval_protocol_version: `gold-pilot-v1.0.0`
- extractor_version: `report-labeler-v2.0.0`
- prompt_version: `second-reader-prompt-v1.0.0`
- schema_version: `second-reader-schema-v1.0.0`
- ontology_version: `pilot-1.0`
- model_id: `microsoft/Phi-3.5-mini-instruct`
- model_revision: `2fe192450127e6a83f7441aef6e3ca586c338b77`
- quantization: `none`
- data_source: `real_kaggle_mount`
- scientific_evaluation: `true`
- gold_only: `true`
- n_gold_studies: 58
- n_report_only_studies_processed: 0
- n_report_only_studies_refused: 4349
- unknown_imputed_as_negative: `false`
- internet_used: `false`
- download_weights: `false`
- training_run: `false`
- folds_created: `false`
- freeze_before_metrics: `true`

## Runtime

- n_studies_scored: 58
- wall_seconds: 2739.726603522
- mean_seconds_per_study: 47.236021067827615
- gpu_memory_bytes: 6368627712
- gpu_name: `Tesla T4`
- cuda_capability: `7.5`
- cuda_kernels_usable: `True`
- inference_backend: `local_transformers`
- malformed_output_rate: 1.0

## System `deterministic_v2`

| Official column | coverage | pos prec | NPV | pos capture | neg capture | selective error | abstentions | malformed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ACL | 0.431 | 0.875 | 1.000 | 0.583 | 0.265 | 0.080 | 33 | 0.000 |
| MCL | 0.241 | 0.500 | 1.000 | 0.333 | 0.163 | 0.214 | 44 | 0.000 |
| Medial Meniscus | 0.276 | 0.818 | 1.000 | 0.346 | 0.156 | 0.125 | 42 | 0.000 |
| Lateral Meniscus | 0.276 | 0.818 | 1.000 | 0.391 | 0.143 | 0.125 | 42 | 0.000 |
| Medial OA | 0.103 | 1.000 | 1.000 | 0.267 | 0.047 | 0.000 | 52 | 0.000 |
| Lateral OA | 0.103 | 0.400 | 1.000 | 0.182 | 0.021 | 0.500 | 52 | 0.000 |
| PF OA | 0.121 | 0.429 | n/a | 0.143 | 0.000 | 0.571 | 51 | 0.000 |
| Effusion | 0.569 | 0.607 | 1.000 | 0.486 | 0.217 | 0.333 | 25 | 0.000 |
| Synovitis | 0.207 | 0.667 | n/a | 0.296 | 0.000 | 0.333 | 46 | 0.000 |
| Baker's | 0.224 | 0.545 | 1.000 | 0.500 | 0.043 | 0.385 | 45 | 0.000 |
| Contusion | 0.259 | 0.700 | 0.800 | 0.368 | 0.103 | 0.267 | 43 | 0.000 |
| Fracture | 0.241 | 0.727 | 1.000 | 0.444 | 0.075 | 0.214 | 44 | 0.000 |

Macro coverage: 0.2543103448275862
Macro precision: 0.6738726551226552
Macro NPV: 0.9800000000000001

## System `second_reader`

| Official column | coverage | pos prec | NPV | pos capture | neg capture | selective error | abstentions | malformed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ACL | 0.000 | n/a | n/a | 0.000 | 0.000 | n/a | 58 | 1.000 |
| MCL | 0.000 | n/a | n/a | 0.000 | 0.000 | n/a | 58 | 1.000 |
| Medial Meniscus | 0.000 | n/a | n/a | 0.000 | 0.000 | n/a | 58 | 1.000 |
| Lateral Meniscus | 0.000 | n/a | n/a | 0.000 | 0.000 | n/a | 58 | 1.000 |
| Medial OA | 0.000 | n/a | n/a | 0.000 | 0.000 | n/a | 58 | 1.000 |
| Lateral OA | 0.000 | n/a | n/a | 0.000 | 0.000 | n/a | 58 | 1.000 |
| PF OA | 0.000 | n/a | n/a | 0.000 | 0.000 | n/a | 58 | 1.000 |
| Effusion | 0.000 | n/a | n/a | 0.000 | 0.000 | n/a | 58 | 1.000 |
| Synovitis | 0.000 | n/a | n/a | 0.000 | 0.000 | n/a | 58 | 1.000 |
| Baker's | 0.000 | n/a | n/a | 0.000 | 0.000 | n/a | 58 | 1.000 |
| Contusion | 0.000 | n/a | n/a | 0.000 | 0.000 | n/a | 58 | 1.000 |
| Fracture | 0.000 | n/a | n/a | 0.000 | 0.000 | n/a | 58 | 1.000 |

Macro coverage: 0.0
Macro precision: None
Macro NPV: None

## System `strict_agreement`

| Official column | coverage | pos prec | NPV | pos capture | neg capture | selective error | abstentions | malformed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ACL | 0.000 | n/a | n/a | 0.000 | 0.000 | n/a | 58 | 1.000 |
| MCL | 0.000 | n/a | n/a | 0.000 | 0.000 | n/a | 58 | 1.000 |
| Medial Meniscus | 0.000 | n/a | n/a | 0.000 | 0.000 | n/a | 58 | 1.000 |
| Lateral Meniscus | 0.000 | n/a | n/a | 0.000 | 0.000 | n/a | 58 | 1.000 |
| Medial OA | 0.000 | n/a | n/a | 0.000 | 0.000 | n/a | 58 | 1.000 |
| Lateral OA | 0.000 | n/a | n/a | 0.000 | 0.000 | n/a | 58 | 1.000 |
| PF OA | 0.000 | n/a | n/a | 0.000 | 0.000 | n/a | 58 | 1.000 |
| Effusion | 0.000 | n/a | n/a | 0.000 | 0.000 | n/a | 58 | 1.000 |
| Synovitis | 0.000 | n/a | n/a | 0.000 | 0.000 | n/a | 58 | 1.000 |
| Baker's | 0.000 | n/a | n/a | 0.000 | 0.000 | n/a | 58 | 1.000 |
| Contusion | 0.000 | n/a | n/a | 0.000 | 0.000 | n/a | 58 | 1.000 |
| Fracture | 0.000 | n/a | n/a | 0.000 | 0.000 | n/a | 58 | 1.000 |

Macro coverage: 0.0
Macro precision: None
Macro NPV: None

## Acceptance (strict agreement)

| Official column | status | precision | NPV | coverage | abstentions | reasons |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| ACL | `PILOT_FAIL` | nan | nan | 0.0 | 58 | malformed_output_rate=1.000>0.05, malformed_output_gate_failed |
| MCL | `PILOT_FAIL` | nan | nan | 0.0 | 58 | small_n gold_positives=9, malformed_output_rate=1.000>0.05, malformed_output_gate_failed |
| Medial Meniscus | `PILOT_FAIL` | nan | nan | 0.0 | 58 | malformed_output_rate=1.000>0.05, malformed_output_gate_failed |
| Lateral Meniscus | `PILOT_FAIL` | nan | nan | 0.0 | 58 | malformed_output_rate=1.000>0.05, malformed_output_gate_failed |
| Medial OA | `PILOT_FAIL` | nan | nan | 0.0 | 58 | malformed_output_rate=1.000>0.05, malformed_output_gate_failed |
| Lateral OA | `PILOT_FAIL` | nan | nan | 0.0 | 58 | small_n gold_positives=11, malformed_output_rate=1.000>0.05, malformed_output_gate_failed |
| PF OA | `PILOT_FAIL` | nan | nan | 0.0 | 58 | malformed_output_rate=1.000>0.05, malformed_output_gate_failed |
| Effusion | `PILOT_FAIL` | nan | nan | 0.0 | 58 | malformed_output_rate=1.000>0.05, malformed_output_gate_failed |
| Synovitis | `PILOT_FAIL` | nan | nan | 0.0 | 58 | malformed_output_rate=1.000>0.05, malformed_output_gate_failed |
| Baker's | `PILOT_FAIL` | nan | nan | 0.0 | 58 | small_n gold_positives=12, malformed_output_rate=1.000>0.05, malformed_output_gate_failed |
| Contusion | `PILOT_FAIL` | nan | nan | 0.0 | 58 | malformed_output_rate=1.000>0.05, malformed_output_gate_failed |
| Fracture | `PILOT_FAIL` | nan | nan | 0.0 | 58 | malformed_output_rate=1.000>0.05, malformed_output_gate_failed |

## Warnings

- Report-only studies were not processed.
- Model-only predictions are not training labels.
- conditional_sensitivity is not full-set sensitivity; use positive_capture.

## Private output hashes (contents not committed)

| file | sha256 | bytes |
| --- | --- | ---: |
| `gold_deterministic_predictions.csv` | `a53150bb9077a1fdf8b270953a3d2a19dd1fd00861ce193aca0d44b3623c32cc` | 79807 |
| `gold_second_reader_predictions.csv` | `970591582643959f6cad87695d2c09635486cd70c341a820cecf8a227f2f90af` | 58152 |
| `gold_consensus_predictions.csv` | `9d3e34639aef3a413ed16e7d2f9e121195606bd99293b0062bfd5e7838d10307` | 50562 |
| `gold_second_reader_status.csv` | `ce1aad946d0ec5f5a1e071c8468be758c9c88e988c330141a2eb4751abbb9d6a` | 7443 |


## Frozen public artefact hashes

These are committed software files, not private Kaggle outputs. They were
hashed into `freeze_manifest.json` before metrics.

| file | sha256 |
| --- | --- |
| `configs/second_reader_pilot.yaml` | `82c257f2e8cf14cb5476e1a949ed9acbe0693308e9fdf2a8470a9ab58d5a42a5` |
| `configs/second_reader_prompt.yaml` | `e5bbfd9ce21bc8c5b6c25ca19177e3ef96b5cfcd9ccde5da61f5e2f1f27645d5` |
| `configs/second_reader_schema.json` | `639c41cc4f9ba1621cc1dd45404ca82ab0a6595c3e7494dca2ef2be1888f9b20` |
| `literature/ontology.yaml` | `a9ce5da072e339c5a8688c294269b7c3b43819ecd8698fb5041c0f6f80c3342c` |
| `configs/report_labeler.yaml` | `656f3a5f3431bb554a7eb41625f2748572b06f35ba8bbd5a0d18e73d1fa7fdf8` |

On-Kaggle verified snapshot SHA-256 values matched the frozen table for
`config.json`, both weight shards, `tokenizer_config.json`, and
`tokenizer.model`.

Attached Kaggle Model: `richolson/phi-3.5-mini-instruct/pyTorch/default/1`.

## Coordinator gates

- `NOT_READY_FOR_HARD_WEAK_LABEL_TRAINING`
- `READY_FOR_SELECTIVE_LABEL_REFINEMENT`
- `NOT_READY_TO_SCALE_SECOND_READER_TO_REPORT_ONLY`

Do not accept model-only predictions as training labels. Do not process
the 4,349 report-only studies until a scientific gold pilot passes.
