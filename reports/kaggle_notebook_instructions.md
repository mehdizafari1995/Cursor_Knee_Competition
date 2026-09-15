# Kaggle real-data audit notebook

Phase 3 runs the first **real** look at the RSNA Knee Abnormality Detection
files **on Kaggle**, with internet disabled and without a Kaggle API token.
The competition dataset is attached through the Kaggle Notebook **Data** panel.

This is not a training run. It does not report AUC.

## What you attach

1. **Competition data** (official dataset):
   `rsna-knee-abnormality-detection`
   Expected mount: `/kaggle/input/rsna-knee-abnormality-detection`
   Current Kaggle sessions may instead mount it at
   `/kaggle/input/competitions/rsna-knee-abnormality-detection`.
   The notebook preflight accepts either path.
2. **Private source dataset** you create from this repository's allowlisted ZIP.
   The notebook copies or extracts that source under `/kaggle/working/rsna_knee_src`
   and never writes under `/kaggle/input`.

Do **not** put `kaggle.json`, `.env` files, DICOM files, radiology reports,
checkpoints, or predictions in the source dataset.

## Build the source ZIP (local machine, not Kaggle)

From a clone of this repository (no Kaggle token required):

```bash
source .venv/bin/activate
python scripts/build_kaggle_bundle.py
```

This writes gitignored artefacts:

- `outputs/kaggle_bundle/rsna_knee_source.zip`
- `outputs/kaggle_bundle/kaggle_bundle_manifest.json` (path + SHA-256 per file)

Create a **private** Kaggle dataset from that ZIP (Kaggle web UI). Do not make
the dataset public: the ZIP contains competition-solution source.

The ZIP allowlist is `knee/`, `scripts/`, `configs/`, `notebooks/`, plus
`pyproject.toml`, `requirements.txt`, and `README.md`. Git metadata, virtual
environments, caches, `data/`, DICOM, credentials, report dumps, checkpoints,
predictions, and `outputs/` cannot enter. Tests in `tests/test_kaggle_bundle.py`
enforce that.

## Create the Kaggle notebook

1. Upload `notebooks/01_kaggle_real_data_audit.ipynb` as a new Kaggle notebook
   (or copy its cells into a new notebook).
2. In the notebook settings:
   - **Internet: off**
   - **Accelerator: None** (CPU is enough; the first audit does not use a GPU)
   - Persistence: default is fine; outputs land in `/kaggle/working`
3. Attach the competition dataset `rsna-knee-abnormality-detection`.
4. Attach your private source dataset.
5. Run all cells.

There is no `kaggle` CLI step and no `KAGGLE_USERNAME` / `KAGGLE_KEY`.

## What the notebook does

A **preflight** cell prints detected top-level paths (not a recursive DICOM
tree) and **stops clearly** if `train.csv` or the competition mount is missing.

It then runs three labelled audit levels:

| Level | Provenance label | What it reads | What it does not do |
| --- | --- | --- | --- |
| A | `full_metadata` | All provided CSVs; study/series **directory** counts | Open DICOM files; walk ~819,640 instances |
| B | `one_header_per_study` | One representative DICOM **header** per training study (`stop_before_pixels=True`) | Load pixel arrays; create patient-grouped folds |
| C | `sampled_dicom` | All series and a few pixel arrays for a **documented, deterministic sample** | Treat sample QC as full-dataset statistics |

Sample size is configurable (`RSNA_AUDIT_SAMPLE_SIZE`, default 24). The sample
prefers one positive study per official target where available, then negatives,
then sequence-plane diversity. Sort order is `StudyInstanceUID` (stable).

Patient-grouped folds are **not** written. Level B only reports whether
`PatientID` is present and complete enough to consider later.

`DATA_SOURCE` is recorded as `real_kaggle_mount` when the notebook is reading
the Kaggle competition mount. Software-fixture executions (pytest) must keep
`scientific_evaluation=false`.

## Outputs

All generated files go to `/kaggle/working/rsna_audit_outputs`:

- `real_data_audit.md`
- `real_data_audit.json`
- `target_prevalence.csv`
- `series_summary.csv`
- `dicom_header_summary.csv`
- `dicom_qc_sample.csv`
- `audit_manifest.json`

Each table carries a `provenance` field. Radiology-report **text** is excluded;
only aggregate report-availability counts are stored.

## Local software-fixture check (not scientific)

```bash
source .venv/bin/activate
pytest
```

`tests/test_kaggle_audit.py` runs the same `run_notebook_audit()` entry point
the notebook uses, against a tiny synthetic tree. Those runs are labelled
`data_source=software_fixture` and `scientific_evaluation=false`. Do not copy
fixture counts into a scientific report.

## Hard rules

- Never modify files under `/kaggle/input`.
- Never commit the generated ZIP, DICOM, report text, credentials, weights, or predictions.
- Never enable internet to `pip install` extra packages; the allowlisted source
  plus Kaggle's default Python stack is the runtime.
- Never quote Level C sample findings as if they described the full 819,640-file dataset.

## Phase 4: report labeling notebook

Upload `notebooks/02_kaggle_report_labeling.ipynb` with the same competition
mount and the same private source ZIP (rebuild the ZIP after this phase so
`knee/report_labels.py` and `configs/report_labeler.yaml` are included).

Settings: **Internet off**, **Accelerator: None**. Outputs:

- `/kaggle/working/rsna_report_label_outputs/private` — row-level weak labels,
  gold predictions, language assignments (do not commit)
- `/kaggle/working/rsna_report_label_outputs/sanitized` — aggregate reports
  and private-file hashes only

Retrieve only the sanitized directory into git. Do not download or open the
private CSVs in Cursor. Fixture pytest runs are not scientific performance.

Current Kaggle sessions may mount the competition at
`/kaggle/input/competitions/rsna-knee-abnormality-detection` and the source
ZIP at `/kaggle/input/datasets/<user>/cursor-knee-source`. The notebook walks
those nested paths and never enables internet or a GPU.

Coordinator gates after the real gold evaluation:
`NOT_READY_FOR_HARD_WEAK_LABEL_TRAINING` and
`READY_FOR_SELECTIVE_LABEL_REFINEMENT`. Do not train an image model on the
weak-label CSV as hard 0/1. See `reports/weak_label_training_readiness.md`.

## Phase 4C: gold-only second-reader pilot

Upload `notebooks/04_kaggle_second_reader_gold_pilot.ipynb` with:

1. The competition dataset
2. The rebuilt private source ZIP
3. Checksummed instruction-model weights attached as a **Kaggle Model** or
   private dataset (never downloaded inside the notebook)

Settings: **Internet off**, **GPU on**. Process the **58 gold studies only**.
Do **not** process the 4,349 report-only studies. Do **not** train an image
model or write image folds. Retrieve sanitized aggregates and private-file
hashes only. See `reports/second_reader_pilot_protocol.md`.

Kernel version 7 is `PILOT_EXECUTION_FAIL` and must remain immutable. Before
another gold run, execute
`notebooks/05_kaggle_second_reader_synthetic_canary.ipynb` (internet off,
GPU on, synthetic reports only, no competition rows). If any canary output
is malformed, stop.

The Phase 4C.1 synthetic canary (kernel
`mahdimzafari/rsna-knee-second-reader-synthetic-canary` version 1) is
`CANARY_FAIL` (`malformed_output_rate=1.0`). Do not launch another gold
notebook version from that result. See
`reports/second_reader_synthetic_canary.md`.
