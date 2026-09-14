# Kaggle report-labeling notebook — CLI upload and run

Private internet-off notebook, pushed with the official Kaggle CLI. Report
text, StudyInstanceUID values, and PatientID values were not downloaded into
this Cursor environment. Credentials were never printed, logged, or
committed. `KAGGLE_API_TOKEN` was present as a non-empty environment secret.

| Field | Value |
| --- | --- |
| Notebook URL | https://www.kaggle.com/code/mahdimzafari/rsna-knee-report-labeling |
| Kernel slug | `mahdimzafari/rsna-knee-report-labeling` |
| Visibility | private (`is_private=true`) |
| Internet | disabled |
| GPU / TPU | disabled |
| Language / type | Python notebook |
| Competition source | `rsna-knee-abnormality-detection` |
| Dataset source | `mahdimzafari/cursor-knee-source` |
| Kaggle CLI | package version `2.2.4` |
| Latest execution | version 2, `KernelWorkerStatus.COMPLETE` |
| DATA_SOURCE | `real_kaggle_mount` |
| scientific_evaluation | `true` |
| extractor_version | `report-labeler-v1.0.0` |
| eval_protocol_version | `gold-eval-v1.0.0` |
| config_sha256 | `54dfea17063519fa33a58ff82fdfd49aa1d5bb4720b017aa65a40055f2150976` |
| n_gold_studies | 58 |
| n_report_only_studies | 4349 |
| training_run / folds / leaderboard | `false` / `false` / `false` |
| external_language_model | `false` |

## CLI sequence

1. Confirm token presence only: `KAGGLE_API_TOKEN_present: True`
2. Rebuild the allowlisted source ZIP (`python scripts/build_kaggle_bundle.py`)
   and update the private dataset `mahdimzafari/cursor-knee-source`
3. Write `kernel-metadata.json` under gitignored `outputs/`
4. Attach the competition dataset plus the private source dataset
5. `kaggle kernels push --no-run` for a private Quick Save
6. `kaggle kernels push` to execute with internet off and GPU off
7. Version 1 failed: `No module named 'knee'` because the source ZIP mounted
   at `/kaggle/input/datasets/mahdimzafari/cursor-knee-source`. Notebook
   discovery now walks nested input paths (skipping `competitions`,
   `train_series`, `test_series`).
8. Version 2 completed successfully
9. Download **sanitized** files only. Do not download or open
   `weak_labels.csv`, `gold_predictions.csv`, or `language_assignments.csv`

Temporary metadata, notebook copies, kernel logs, and downloaded outputs stay
under `outputs/` (gitignored). Kernel logs are not committed.

## Mount layout observed on Kaggle

Top-level `/kaggle/input` children were `competitions` and `datasets`:

- Competition: `/kaggle/input/competitions/rsna-knee-abnormality-detection`
- Source dataset: `/kaggle/input/datasets/mahdimzafari/cursor-knee-source`
- Working outputs: `/kaggle/working/rsna_report_label_outputs`

## Retrieved sanitized outputs (gitignored locally, then summarized)

Downloaded into `outputs/kaggle_report_label_outputs/` only:

| File | Bytes | Committed? |
| --- | ---: | --- |
| `sanitized/report_labeling_summary.md` | 7190 on Kaggle; regenerated locally with language-level and error-category sections | yes, as `reports/report_labeling_summary.md` |
| `sanitized/gold_metrics.json` | 35181 | no (aggregates copied into the markdown summary) |
| `sanitized/language_audit.json` | 1237 | no (aggregates copied into the markdown summary) |
| `sanitized/weak_label_aggregates.json` | 4347 | no (aggregates copied into the markdown summary) |
| `sanitized/error_categories.json` | 472 | no (aggregates copied into the markdown summary) |
| `sanitized/private_output_hashes.json` | 406 | hashes copied into the markdown summary |
| `sanitized/report_label_manifest.json` | 2213 | no |

Private row-level CSVs were **not** retrieved into Cursor. Their SHA-256
hashes (from the Kaggle sanitized hash file):

| file | sha256 | bytes |
| --- | --- | ---: |
| `weak_labels.csv` | `d11941721564c7df5d49a7dca978d79eff17feae4d68abdbd0f3fb1692d7834b` | 819372 |
| `gold_predictions.csv` | `0fca30d35dff4de90d0b99b930f135a590da5c52f3feed98a7bf24cd801dc5c5` | 63438 |
| `language_assignments.csv` | `57c71e90c1c819094ec67094c3ccee991f44bd8ef2a164dbdbe70a4e36cea484` | 520091 |

## Unresolved

- Heuristic language IDs for Greek, Bulgarian, Czech, and Danish are mostly
  uncertain; those studies mostly abstain except language-agnostic
  abbreviations.
- Coverage on OA, synovitis, Baker's, contusion, and fracture is low; those
  weak labels should not be treated as gold.
- No image-model folds, training, or leaderboard submission in this phase.
