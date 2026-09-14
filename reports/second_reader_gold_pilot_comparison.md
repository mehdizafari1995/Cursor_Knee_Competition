# Second-reader gold-pilot comparison

Sanitized software-verification artefact. Report text, StudyInstanceUID
values, and PatientID values are not included.

- pilot_version: `second-reader-pilot-v1.0.0`
- eval_protocol_version: `gold-pilot-v1.0.0`
- extractor_version: `report-labeler-v2.0.0`
- prompt_version: `second-reader-prompt-v1.0.0`
- schema_version: `second-reader-schema-v1.0.0`
- ontology_version: `pilot-1.0`
- model_id: `microsoft/Phi-3.5-mini-instruct`
- model_revision: `2fe192450127e6a83f7441aef6e3ca586c338b77`
- quantization: `none`
- data_source: `pending_real_kaggle_gpu_run`
- scientific_evaluation: `false`
- gold_only: `true`
- n_gold_studies_planned: 58
- n_gold_studies_processed: 0
- n_report_only_studies_processed: 0
- unknown_imputed_as_negative: `false`
- internet_used: `false`
- download_weights: `false`
- training_run: `false`
- folds_created: `false`

## SOFTWARE FIXTURE / PREFLIGHT STOP

Cursor executed parser, consensus, privacy, and coverage-aware **fixture**
tests only. No weights were downloaded. No scientific precision, NPV, or
capture numbers for the 58 expert-labelled studies are claimed here.

Preflight of the frozen shard SHA-256 values against the Hugging Face tree
for revision `2fe192450127e6a83f7441aef6e3ca586c338b77` found a mismatch on
`model-00001-of-00002.safetensors`. Protocol required a stop before
inference. Details: `reports/kaggle_second_reader_pilot_run.md`.

After the frozen shard-1 digest is corrected to the published 64-character
LFS oid for that same revision, run
`notebooks/04_kaggle_second_reader_gold_pilot.ipynb` with internet
disabled, GPU enabled, the competition dataset, the private source ZIP, and
a checksummed Phi-3.5 snapshot attached. Then replace this file with
sanitized aggregates and private-output hashes from Kaggle. Do not copy
fixture counts into a scientific table.

## Planned comparison (real run)

| System | Role |
| --- | --- |
| deterministic v2 alone | `report-labeler-v2.0.0` |
| second reader alone | frozen prompt + schema |
| strict agreement | candidate labels only on exact polarity match |

Acceptance statuses `PILOT_PASS` / `PILOT_RESTRICTED` / `PILOT_FAIL` are
issued from strict-agreement coverage-aware metrics. Coverage is reported
separately. Abstentions are not hidden in sensitivity.

## Coordinator gates

- `NOT_READY_FOR_HARD_WEAK_LABEL_TRAINING`
- `READY_FOR_SELECTIVE_LABEL_REFINEMENT`
- `NOT_READY_TO_SCALE_SECOND_READER_TO_REPORT_ONLY`

Model-only predictions are not training labels. Do not process the 4,349
report-only studies until a scientific gold pilot passes.

## Frozen public artefact hashes

These are committed software files, not private Kaggle outputs.

| file | sha256 | bytes |
| --- | --- | ---: |
| `configs/second_reader_pilot.yaml` | `82c257f2e8cf14cb5476e1a949ed9acbe0693308e9fdf2a8470a9ab58d5a42a5` | 2825 |
| `configs/second_reader_prompt.yaml` | `e5bbfd9ce21bc8c5b6c25ca19177e3ef96b5cfcd9ccde5da61f5e2f1f27645d5` | 3498 |
| `configs/second_reader_schema.json` | `639c41cc4f9ba1621cc1dd45404ca82ab0a6595c3e7494dca2ef2be1888f9b20` | 26690 |
| `literature/ontology.yaml` | `a9ce5da072e339c5a8688c294269b7c3b43819ecd8698fb5041c0f6f80c3342c` | 66808 |
| `configs/report_labeler.yaml` | `656f3a5f3431bb554a7eb41625f2748572b06f35ba8bbd5a0d18e73d1fa7fdf8` | 24509 |

## Private output hashes

Pending the real Kaggle GPU run. Fixture hashes are written under gitignored
`outputs/` during pytest and are not scientific.
