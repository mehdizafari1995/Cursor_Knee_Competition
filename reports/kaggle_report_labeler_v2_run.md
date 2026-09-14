# Kaggle report-labeler v2 notebook — CLI upload and run

Private internet-off notebook, pushed with the official Kaggle CLI. Report
text, StudyInstanceUID values, and PatientID values were not downloaded into
this Cursor environment. Credentials were never printed, logged, or
committed. `KAGGLE_API_TOKEN` was present as a non-empty environment secret.

The optional second reader was **not** enabled. This run evaluates the
deterministic version-2 extractor and confidence system only. No image model
was trained and no image folds were created.

| Field | Value |
| --- | --- |
| Notebook URL | https://www.kaggle.com/code/mahdimzafari/rsna-knee-report-labeler-v2 |
| Kernel slug | `mahdimzafari/rsna-knee-report-labeler-v2` |
| Visibility | private (`is_private=true`) |
| Internet | disabled (`enable_internet=false`) |
| GPU / TPU | disabled |
| Language / type | Python notebook |
| Competition source | `rsna-knee-abnormality-detection` |
| Dataset source | `mahdimzafari/cursor-knee-source` |
| Kaggle CLI | package version `2.2.4` |
| Latest execution | version 1, `KernelWorkerStatus.COMPLETE` |
| DATA_SOURCE | `real_kaggle_mount` |
| scientific_evaluation | `true` |
| extractor_version | `report-labeler-v2.0.0` |
| eval_protocol_version | `gold-eval-v2.0.0` |
| config_sha256 | `656f3a5f3431bb554a7eb41625f2748572b06f35ba8bbd5a0d18e73d1fa7fdf8` |
| source bundle SHA-256 | `288f84471622fc58bbebd9de358e7926e05a8f660748ff6d77f708ad84507280` |
| n_train_studies / n_reports | 4407 |
| n_gold_studies | 58 |
| n_report_only_studies | 4349 |
| second_reader.enabled | `false` |
| second_reader_weights_downloaded | `false` |
| training_run / folds / leaderboard | `false` / `false` / `false` |
| external_language_model | `false` |

## CLI sequence

1. Confirm token presence only: `KAGGLE_API_TOKEN_present: True`
2. Rebuild the allowlisted source ZIP (`python scripts/build_kaggle_bundle.py`)
   and verify the ZIP SHA-256 against `outputs/kaggle_bundle/kaggle_bundle_manifest.json`
3. Update the private dataset `mahdimzafari/cursor-knee-source` with that ZIP
4. Write `kernel-metadata.json` under gitignored `outputs/kaggle_kernel_push_v2/`
5. Attach the competition dataset plus the updated private source dataset
6. `kaggle kernels push` for a private run with internet off and GPU off
7. Version 1 completed successfully (`KernelWorkerStatus.COMPLETE`)
8. Download **sanitized** files only via `--file-pattern`. Do not download or
   open `weak_labels.csv`, `gold_predictions.csv`, `language_assignments.csv`,
   or `mask_*.csv`

Temporary metadata, notebook copies, kernel logs, and downloaded outputs stay
under `outputs/` (gitignored). Kernel logs are not committed.

## Mount layout observed on Kaggle

- Competition: `/kaggle/input/competitions/rsna-knee-abnormality-detection`
- Source dataset: `/kaggle/input/datasets/mahdimzafari/cursor-knee-source`
- Working source copy: `/kaggle/working/rsna_knee_src`
- Working outputs: `/kaggle/working/rsna_report_label_outputs_v2`

## Retrieved sanitized outputs (gitignored locally, then summarized)

Downloaded into `outputs/kaggle_report_label_outputs_v2/` only:

| File | Bytes | Committed? |
| --- | ---: | --- |
| `sanitized/report_labeling_summary.md` | 16300 | no (OOF statuses superseded in comparison report) |
| `sanitized/gold_metrics.json` | 435765 | no (aggregates copied into the comparison markdown) |
| `sanitized/language_audit.json` | 1918 | no |
| `sanitized/weak_label_aggregates.json` | 4355 | no |
| `sanitized/error_categories.json` | 980 | no |
| `sanitized/private_output_hashes.json` | 1101 | hashes copied into the comparison report |
| `sanitized/report_label_manifest.json` | 3546 | no |

Private row-level CSVs were **not** retrieved into Cursor. Their SHA-256
hashes (from the Kaggle sanitized hash file) are recorded in
`reports/report_labeler_v2_kaggle_comparison.md`.

The v1 kernel `mahdimzafari/rsna-knee-report-labeling` was not overwritten.

## Unresolved

- Exact high-confidence mask true-counts remain private on Kaggle; sanitized
  aggregates record hashes, byte sizes, and extractor known/unknown counts.
- Image-model training is still not approved from these labels.
