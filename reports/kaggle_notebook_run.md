# Kaggle real-data audit notebook — CLI upload and run

Private internet-off notebook, pushed with the official Kaggle CLI. No
competition DICOMs were downloaded into this Cursor environment. Credentials
were never printed, logged, or committed. `KAGGLE_API_TOKEN` was present as a
non-empty environment secret.

| Field | Value |
| --- | --- |
| Notebook URL | https://www.kaggle.com/code/mahdimzafari/rsna-knee-real-data-audit |
| Kernel slug | `mahdimzafari/rsna-knee-real-data-audit` |
| Visibility | private (`is_private=true`) |
| Internet | disabled |
| GPU / TPU | disabled |
| Language / type | Python notebook |
| Competition source | `rsna-knee-abnormality-detection` |
| Dataset source | `mahdimzafari/cursor-knee-source` |
| Kaggle CLI | GitHub `kaggle-cli` (package version `2.2.4` with `--no-run`) |
| Latest execution | version 4, `KernelWorkerStatus.COMPLETE`, **Status: AUDITED** |
| DATA_SOURCE | `real_kaggle_mount` |
| scientific_evaluation | `true` |
| training_run / folds | `false` / `false` |

## CLI sequence

1. Confirm token presence only: `KAGGLE_API_TOKEN_present: True`
2. Install/update the official CLI (`pip install -U kaggle`, then force-install
   GitHub `kaggle-cli` so `kaggle kernels push --no-run` is available)
3. Write `kernel-metadata.json` under gitignored `outputs/kaggle_kernel_push/`
4. Attach competition + private source dataset in metadata
5. Validate `code_file` == `01_kaggle_real_data_audit.ipynb` and title slug
   `rsna-knee-real-data-audit`
6. `kaggle kernels push --no-run` → version 1 (Quick Save)
7. Confirm private kernel exists and metadata matches
8. `kaggle kernels push` to execute
9. Monitor with `kaggle kernels status` / `kaggle kernels logs` (no competition
   download)
10. `kaggle kernels output --file-pattern` for the seven audit files only

Temporary metadata, notebook copies, and downloaded outputs stay under
`outputs/` (gitignored).

## Mount layout observed on Kaggle

Top-level `/kaggle/input` children were `competitions` and `datasets`, not the
classic slug directories. Resolved paths:

- Competition: `/kaggle/input/competitions/rsna-knee-abnormality-detection`
- Source dataset: `/kaggle/input/datasets/mahdimzafari/cursor-knee-source`
- Working source copy: `/kaggle/working/rsna_knee_src`
- Audit outputs: `/kaggle/working/rsna_audit_outputs`

## Downloaded audit outputs (gitignored locally)

Downloaded into `outputs/kaggle_audit_outputs/rsna_audit_outputs/` only:

| File | Provenance | Bytes |
| --- | --- | --- |
| `real_data_audit.md` | mixed | 16468 |
| `real_data_audit.json` | mixed | 199563 |
| `target_prevalence.csv` | `full_metadata` | 978 |
| `series_summary.csv` | `full_metadata` | 514 |
| `dicom_header_summary.csv` | `one_header_per_study` | 67715 |
| `dicom_qc_sample.csv` | `sampled_dicom` | 43089 |
| `audit_manifest.json` | mixed | 1579 |

These files are competition-derived and are **not** committed. Header and sample
tables contain study-level identifiers and stay in gitignored `outputs/`.

## Unresolved

- `reports/kaggle_notebook_instructions.md` still documents the classic
  `/kaggle/input/<slug>` mount. Current Kaggle sessions also use
  `/kaggle/input/competitions/<slug>`. Runtime code now accepts both.
