# Baseline design

Phase 2 study-level baseline for RSNA Knee Abnormality Detection. This document
describes the software that is in the repository. It is **not** a scientific
result: no competition DICOM studies were trained on in this phase.

## Problem

Each study is scored for **twelve independent binary findings**. The official
metric is the **macro-averaged ROC AUC** across those twelve targets. Training
labels may be missing or weakly supervised from multilingual reports; inference
is image-only.

Internal ids (`TARGET_KEYS` in `knee/config.py`) are unchanged so the ontology
registry cannot drift. Competition submissions use the official Kaggle header
spelling (`ACL`, `MCL`, `Medial Meniscus`, `Lateral Meniscus`, `Medial OA`,
`Lateral OA`, `PF OA`, `Effusion`, `Synovitis`, `Baker's`, `Contusion`,
`Fracture`) and `StudyInstanceUID` row order from `sample_submission.csv`.

## Data access

The dataset is not shipped with this repository.

- Data root is resolved from the `RSNA_DATA_ROOT` environment variable
  (alias `KNEE_DATA_ROOT`). No machine-specific path is hard-coded.
- Expected on-disk layout is transcribed from the official Kaggle data page,
  not invented:
  `train.csv`, `train_series.csv`, `train_series/{StudyInstanceUID}/{SeriesInstanceUID}/*.dcm`,
  and the test / sample-submission counterparts.
- If those files cannot be opened, scripts exit with `DATA_NOT_AVAILABLE` and
  write no synthetic audit statistics.
- Verified observations and assumptions are separated in `reports/data_audit.md`.

Documented CSV schemas do not include a patient column. Grouped splits therefore
**prefer `PatientID` when every study has one** (for example from a complete
DICOM tag or an added column) and otherwise group by `StudyInstanceUID`. That
fallback is recorded in fold metadata rather than treated as patient-level
splitting.

## Validation (Phase B)

`knee/folds.py` assigns deterministic, group-aware folds:

1. Collapse studies to groups (patient if available, else study).
2. Greedy multilabel iterative stratification, seeded, with group ids sorted
   before the permutation so input row order cannot leak into the assignment.
3. Repair pass: if a target has at least `n_folds` distinct positive groups,
   every validation fold must contain at least one positive study for that
   target.
4. Targets with fewer positive groups than folds are listed as
   `impossible_full_coverage_targets`. The code does not invent extra positives.
5. `assert_no_group_leakage` fails if any group appears in two folds.
6. Fold tables are written under `outputs/` (gitignored). Tests prove
   determinism and the leakage invariant on software fixtures only.

Default `n_folds` is 5 (`configs/baseline.yaml`).

## Minimal baseline (Phase C)

Configured by `configs/baseline.yaml`.

| Stage | Behaviour |
| --- | --- |
| DICOM load | Uncompressed MR instances via pydicom. Slice order is the projection of `ImagePositionPatient` onto the slice normal from `ImageOrientationPatient`, then `InstanceNumber`, then filename. |
| Intensity | Percentile normalisation (default 1st–99th) to `[0, 1]`. |
| Slice sampling | Fixed-width linspace, configurable `num_slices` (default 8). Short series are repeated via rounded linspace so the tensor width is constant. |
| Encoder | ImageNet ResNet-18 when `smoke: false`. Smoke mode uses `TinySliceEncoder` (random tiny conv, no pretrained weights, no network download). |
| Aggregation | Mean (default) or max over slice features. |
| Head | Linear layer, **12 logits**, sigmoid at inference. |
| Loss | Masked binary cross-entropy: NaN / unknown labels are excluded. Missing findings are never imputed as 0. |
| Metric | Per-target ROC AUC and macro mean; single-class or all-unknown targets are NaN and dropped from the macro. |

### Smoke vs scientific evaluation

`python scripts/train_baseline.py --smoke` and `predict_baseline.py --smoke`
run on **tiny generated tensors only**. Metrics from those runs are labelled
`scientific_evaluation: false` and must not be quoted as competition
performance.

Real training is opt-in and is **refused in this phase** even with
`--allow-training`. Full training is out of scope.

### Scripts

| Script | Role |
| --- | --- |
| `scripts/audit_competition_data.py` | Read-only audit; exit 2 on `DATA_NOT_AVAILABLE`. |
| `scripts/create_folds.py` | Deterministic grouped folds from `train.csv`. |
| `scripts/train_baseline.py` | `--smoke` software check; no full training. |
| `scripts/predict_baseline.py` | `--smoke` probabilities; real DICOM inference requires data + checkpoint. |
| `scripts/make_submission.py` | Align to `sample_submission.csv` columns and row order. |

## What is not in this phase

- Expanding the literature corpus or editing claim ledgers.
- Downloading or committing competition DICOM, reports, credentials, or weights.
- Using literature figures for training.
- A claim that the smoke AUC is a leaderboard score.
