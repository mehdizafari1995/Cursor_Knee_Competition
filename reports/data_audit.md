# Competition data audit

**Status: DATA_NOT_AVAILABLE**

This document separates **verified observations** (from local files or from the absence of local files) from **assumptions** and from **official documentation** that was not re-observed on disk.

Real competition files could not be accessed. No study counts, label prevalences, DICOM histograms, or other audit statistics were generated.

## Verified observations

- No local directory among the searched candidates contained sample_submission.csv, train.csv, train_series.csv, test.csv, or test_series.csv.
- Candidate roots searched: /workspace/data/rsna-knee-abnormality-detection, /workspace/data, /kaggle/input/rsna-knee-abnormality-detection.
- Resolver reason: `DATA_NOT_AVAILABLE: no competition data root found. RSNA_DATA_ROOT is unset (optional alias KNEE_DATA_ROOT is also unset). Searched: /workspace/data/rsna-knee-abnormality-detection (missing); /workspace/data (missing); /kaggle/input/rsna-knee-abnormality-detection (missing)`

Searched data-root candidates:

| Path | Status |
| --- | --- |
| `/workspace/data/rsna-knee-abnormality-detection` | missing |
| `/workspace/data` | missing |
| `/kaggle/input/rsna-knee-abnormality-detection` | missing |

## Assumptions

- The on-disk layout, when the dataset is mounted, matches the official Kaggle data page (see official_schema_documentation).
- A patient-level identifier may exist in DICOM PatientID even though it is not documented as a CSV column.

## Target reconciliation

TARGET_KEYS in knee/config.py is the project's internal source of truth and was not renamed. Official Kaggle sample_submission.csv headers differ in spelling; the mismatch is escalated here.

| Slot | Internal `TARGET_KEYS` | Official Kaggle column |
| --- | --- | --- |
| 1 | `acl_injury` | `ACL` |
| 2 | `mcl_injury` | `MCL` |
| 3 | `medial_meniscus_injury` | `Medial Meniscus` |
| 4 | `lateral_meniscus_injury` | `Lateral Meniscus` |
| 5 | `medial_compartment_oa` | `Medial OA` |
| 6 | `lateral_compartment_oa` | `Lateral OA` |
| 7 | `patellofemoral_oa` | `PF OA` |
| 8 | `joint_effusion` | `Effusion` |
| 9 | `synovitis` | `Synovitis` |
| 10 | `baker_cyst` | `Baker's` |
| 11 | `bone_contusion` | `Contusion` |
| 12 | `fracture` | `Fracture` |

Order matches: **True**. Spelling matches: **False**.

Competition submissions written by `scripts/make_submission.py` use the official spelling. `TARGET_KEYS` are unchanged so the ontology registry does not drift.

## Official documented schema (not a local file observation)

Evidence class: `official_documentation_not_local_observation`

- https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data
- https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview

Documented files: `train.csv`, `train_series.csv`, `test.csv`, `test_series.csv`, `sample_submission.csv`

Documented DICOM layout: `{split}_series/{StudyInstanceUID}/{SeriesInstanceUID}/*.dcm`

The data page lists twelve binary findings with the official column spelling. It does not explicitly bullet those names under the train.csv field list (which names StudyInstanceUID and Report). Whether the twelve columns are physically present in train.csv is an observation that can only be made from the local files.

The documented CSV schemas do not include PatientID. If a patient identifier exists it would have to come from DICOM tags. Grouped splits therefore fall back to StudyInstanceUID until a complete patient identifier is observed.

Organizer statements that were **not** verified locally:

- The data page states there are about 1300 studies in the (hidden) test set.
- The data page states series typically contain 20–45 slices (median 30).

## Statistics

`statistics: null` — not computed. A table of zeros would be a synthetic audit statistic and is forbidden when the files are missing.

## Checks not run

- study / patient / series / instance counts
- DICOM dimensions, spacing, orientation, slice order
- sequence descriptions
- missing and corrupt files
- label availability and prevalence
- multilingual report availability
- duplicate patients or studies
- leakage checks against local identifiers
- train/test structural comparison of local files

## Unresolved questions

- Where will the competition dataset be mounted, and will RSNA_DATA_ROOT be set to that directory?
- Does local train.csv contain the twelve official label columns, only Report text, or both?
- Is DICOM PatientID populated and stable enough to use as a grouping key?

## How to re-run

```bash
export RSNA_DATA_ROOT=/path/to/rsna-knee-abnormality-detection
python scripts/audit_competition_data.py
```
