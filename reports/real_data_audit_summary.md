# Real Kaggle data audit — sanitized aggregate summary

Coordinator analysis of the internet-off Kaggle notebook run (`status:
AUDITED`, `data_source: real_kaggle_mount`). This note uses **aggregate
counts only**. It does not quote PatientID values, StudyInstanceUID values,
SeriesInstanceUID values, report text, or DICOM pixel/row payloads.

No folds were created and no model was trained for this analysis
(`folds_created: false`, `training_run: false`).

| Field | Value |
| --- | --- |
| Audit UTC | `2026-09-14T04:10:38Z` |
| Scientific evaluation | `true` |
| GPU / internet used | `false` / `false` |
| Level C sample size | 24 studies |
| Auditor `patient_grouped_folds_allowed` | `false` (coordinator gate; never auto-flipped) |
| Source outputs | gitignored `outputs/kaggle_audit_outputs/rsna_audit_outputs/` |

## Manifest confirmation

Downloaded filenames and byte counts match `audit_manifest.json` exactly.
The local manifest file itself is 1579 bytes (not listed inside `outputs`).

| File | Manifest bytes | Local bytes | Match | Provenance |
| --- | --- | --- | --- | --- |
| `real_data_audit.md` | 16468 | 16468 | yes | mixed |
| `real_data_audit.json` | 199563 | 199563 | yes | mixed |
| `target_prevalence.csv` | 978 | 978 | yes | full_metadata |
| `series_summary.csv` | 514 | 514 | yes | full_metadata |
| `dicom_header_summary.csv` | 67715 | 67715 | yes | one_header_per_study |
| `dicom_qc_sample.csv` | 43089 | 43089 | yes | sampled_dicom |
| `audit_manifest.json` | — | 1579 | n/a | mixed |

Those files remain gitignored and are not committed.

## Leak check of `reports/kaggle_notebook_run.md`

Scanned for credentials, DICOM identifiers, report text, and local absolute
paths.

- No token values or `kaggle.json` contents.
- No StudyInstanceUID / SeriesInstanceUID / PatientID values.
- No radiology report text.
- No `/workspace`, `/home/ubuntu`, or `/opt` paths.
- Contains only the Kaggle username, public kernel URL, Kaggle mount paths
  under `/kaggle/input` and `/kaggle/working`, and repo-relative `outputs/`
  paths. Safe to keep in git.

## 1. Dataset reconciliation

Train and example-test **counts reconcile** across CSVs and folders.

| Split | CSV studies | Unique study IDs | Study dirs | CSV series | Unique series IDs | Series dirs |
| --- | --- | --- | --- | --- | --- | --- |
| Train | 4407 | 4407 | 4407 | 24371 | 24371 | 24371 |
| Example test | 3 | 3 | 3 | 15 | 15 | 15 |

- Mean series per train study: 24371 / 4407 ≈ **5.53**. Example test: 15 / 3 = **5**.
- Duplicate study IDs in `train.csv`: **0**. Duplicate series IDs: **0** (train and example test).
- `train_series.csv` reports 19964 duplicate rows on the study-ID column, and
  `test_series.csv` reports 12. That is expected (many series per study), not
  duplicated studies or series.
- Directory/table mismatches: **none**. Documented files and `train_series` /
  `test_series` directories are all present.
- Unexpected anatomical-plane values: **none**.
- Train vs example-test study-ID overlap: **0**.
- `sample_submission.csv`: 3 rows, official 12 target headers, same IDs and
  row order as `test.csv`.
- Example-test `test.csv` has only the study-ID column (no labels, no Report).
- Level A did not walk the instance tree (~819,640 DICOM files mentioned in
  the audit note). Reconciliation above is study/series only.

## 2. Label audit

Among 4407 training studies:

- Completely labelled (all 12 targets known): **58**
- Partially labelled: **0**
- Report-only / unlabelled on every target: **4349** (98.7%)
- `n_studies_with_any_label`: **58** (identical to fully labelled)
- Train Report column: present; **4407 / 4407** non-empty. **No report text
  is reproduced here.**
- Example test: no Report column.

Every target has `n_known = 58` and `n_unknown = 4349`. Prevalence below is
among known labels only.

| Official column | Internal key | n_known | n_positive | n_negative | n_unknown | Positive rate (known) | Neg:pos |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ACL | `acl_injury` | 58 | 24 | 34 | 4349 | 0.414 | 1.42 |
| MCL | `mcl_injury` | 58 | 9 | 49 | 4349 | 0.155 | 5.44 |
| Medial Meniscus | `medial_meniscus_injury` | 58 | 26 | 32 | 4349 | 0.448 | 1.23 |
| Lateral Meniscus | `lateral_meniscus_injury` | 58 | 23 | 35 | 4349 | 0.397 | 1.52 |
| Medial OA | `medial_compartment_oa` | 58 | 15 | 43 | 4349 | 0.259 | 2.87 |
| Lateral OA | `lateral_compartment_oa` | 58 | 11 | 47 | 4349 | 0.190 | 4.27 |
| PF OA | `patellofemoral_oa` | 58 | 21 | 37 | 4349 | 0.362 | 1.76 |
| Effusion | `joint_effusion` | 58 | 35 | 23 | 4349 | 0.603 | 0.66 |
| Synovitis | `synovitis` | 58 | 27 | 31 | 4349 | 0.466 | 1.15 |
| Baker's | `baker_cyst` | 58 | 12 | 46 | 4349 | 0.207 | 3.83 |
| Contusion | `bone_contusion` | 58 | 19 | 39 | 4349 | 0.328 | 2.05 |
| Fracture | `fracture` | 58 | 18 | 40 | 4349 | 0.310 | 2.22 |

Targets with fewer than 5 known positives (mathematically unable to place one
positive in each of 5 folds): **none**. Minimum is MCL = 9.

Practical tightness for 5-fold validation among the 58 labelled studies:

- MCL (9), Lateral OA (11), Baker's (12) yield only 1–3 positives per fold.
- Per-fold ROC AUC on those targets will be high-variance.
- Effusion is the only target with more known positives than negatives
  (neg:pos 0.66).

Official vs internal spelling still differs (`spelling_matches: false`,
`order_matches: true`). Submissions must use official headers;
`TARGET_KEYS` must not be renamed.

Language cues (counts only, heuristic): function-word hits in several
languages; 2613 / 4407 reports contain non-ASCII; observed scripts include
latin 4314, greek 321, cyrillic 220, thai 3, other_non_ascii 419. These are
not verified per-report language IDs.

## 3. Patient grouping

Level B read **exactly one DICOM header per training study**
(`stop_before_pixels=True`).

| Measure | Count / rate |
| --- | --- |
| Headers attempted | 4407 |
| Headers read successfully | 4407 |
| Unreadable headers | 0 |
| Missing or empty series dirs | 0 |
| PatientID present | 4407 |
| Unique PatientID values | 4407 |
| Completeness rate | **1.0** |
| `patient_id_sufficiently_complete` | `true` |
| Study UID in header matches CSV | 4407 / 4407 |
| Series UID in header matches CSV | 4407 / 4407 |
| Patients associated with multiple studies | **0** (under this sampling) |
| Intra-study PatientID conflicts | **not measured** (one header per study) |

PHI tag **names** observed: `PatientID` only (values not exported).

**Can PatientID safely define folds?** Conditionally, as a grouping key on
this train set: it is complete and 1:1 with studies. Restrictions:

1. Intra-study consistency was not checked, because only one header per study
   was read and the QC table has no PatientID column.
2. Unique PatientID count equals unique study count, so this field may be
   exam-scoped rather than a stable person identifier. That cannot be tested
   from these outputs.
3. The auditor still reports `patient_grouped_folds_allowed: false`. That flag
   is a coordinator gate and is not auto-enabled when completeness is 1.0.

**StudyInstanceUID fallback leakage:** on this training set, PatientID and
study ID are 1:1, so grouping by study ID is **equivalent** to grouping by
PatientID. Fallback would not add extra multi-study leakage **relative to
PatientID**. Residual risk is the same for both keys: if the same person
appears under different PatientID values in different exams, neither key
would group those studies together. That pattern was not observable here.

## 4. MRI structure

### Series per study

Full train set: mean **5.53** series/study. A per-study min/max histogram was
**not** exported at Level A. Level C sample (24 studies): min 4, mean 5.96,
max 10.

### Planes (full series tables)

| Plane | Train series | Example-test series |
| --- | --- | --- |
| Sagittal | 9864 | 7 |
| Coronal | 8609 | 4 |
| Axial | 5898 | 4 |
| Total | 24371 | 15 |

Level B orientation-inferred planes (one series per train study): Sagittal
1764, Coronal 1537, Axial 1106 (sums to 4407; orientation was inferable for
every sampled header).

Full-train **per-study** missing-plane counts were not tabulated. Axial series
(5898) exceed study count (4407), which is compatible with every study having
at least one axial series but does **not** prove it. In the Level C sample,
**24 / 24** studies had Axial + Coronal + Sagittal in the CSV plane column,
and CSV plane vs orientation-inferred plane mismatched on **0 / 143** series.

### Sequence descriptions (one representative series per study)

- 410 distinct `SeriesDescription` values among 4407 headers.
- `DummySeriesDesc!`: 558 (12.7%). `<empty>`: 198 (4.5%). Combined 756
  (17.2%) lack a usable description.
- 109 descriptions appear once.
- Most frequent non-placeholder strings are PD/TSE/FS sagittal, coronal, and
  axial variants plus T1 SE coronal (counts 43–71). These are protocol labels,
  not patient identifiers.

Fluid-sensitive vs fat-suppression flags are **identical** for all 24371 train
series and all 15 example-test series (14010 / 10361 train; 9 / 6 example
test). They do not provide an independent sequence axis.

### Orientation, spacing, dimensions

- Pixel spacing: 420 distinct values, **0 missing** of 4407.
- Slice thickness: 51 distinct values, **0 missing**; 3.0 mm is 2496.
- Image matrix: 102 distinct sizes. Top: 512×512 (1117), 384×384 (554),
  640×640 (526). Smallest observed: 192×192 (11 studies). Largest: 1200×1200
  (1). Non-square matrices: 350 studies.
- Modality: MR for all 4407 Level B headers.

### Manufacturer and acquisition variability (aggregate only)

Vendor strings were **not** merged in the auditor output. Family grouping
below is derived for this summary:

| Vendor family (derived) | Studies | Raw manufacturer strings (counts) |
| --- | --- | --- |
| Siemens | 1951 | Siemens Healthineers 1053, SIEMENS 804, Siemens 94 |
| Philips | 1301 | Philips Medical Systems 718, Philips 492, Philips Healthcare 91 |
| GE | 905 | GE MEDICAL SYSTEMS 868, GEHC 37 |
| Canon/Toshiba | 226 | TOSHIBA 181, CANON_MEC 45 |
| Fujifilm | 16 | FUJIFILM Healthcare Corporation 16 |
| Hitachi | 8 | Hitachi Medical Corporation 8 |

- 46 model names. Top: Ingenia 741, MAGNETOM Vida 447, Aera 400. Eight models
  appear once.
- Field strength: 1.5 T 2543, 3.0 T 1601, 1.16 T 24, 1.0 T 1, missing **238**
  (5.4%).

Acquisition variability is large enough that a model trained on one vendor /
matrix / field-strength subset can shift on others. Hidden-test vendor mix is
unknown.

## 5. DICOM sample quality

Level C is a **bounded sample**, not a full-dataset statistic
(`is_full_dataset_statistic: false`).

| Item | Count |
| --- | --- |
| Studies requested | 24 |
| Studies sampled | 24 |
| Series inspected | 143 |
| Instance headers read | 4368 |
| Pixel arrays read | 143 (one mid-slice per series) |
| Series dirs missing | 0 |
| Unreadable / corrupt pixel arrays | 0 |
| All-zero arrays | 0 |
| `fraction_finite` &lt; 1 | 0 (all 1.0) |
| `fraction_zero` &gt; 0.99 | 0 (max ≈ 0.586) |
| CSV vs inferred plane mismatches | 0 / 143 |
| Sampled studies with Ax+Cor+Sag | 24 / 24 |
| Studies with inconsistent `pixel_shape` across series | 20 / 24 |
| Studies with inconsistent `pixel_dtype` across series | 0 / 24 |

Sample construction: 12 target-positive studies (one per official column), 1
negative, 11 fill. Instances per series: min 16, median 30, max 160. Pixel
arrays were 2-D; dtypes uint16 (104) and int16 (39).

**Not measured** by this audit (no columns / flags in the QC table):

- duplicated slices
- non-monotonic slice positions
- missing ImageOrientationPatient / ImagePositionPatient as explicit QC flags
  (orientation was inferable for every Level B header and every sampled
  series, which is weaker than a dedicated missing-tag count)

**Sample-size limitation:** 24 / 4407 training studies ≈ 0.5%. Readable
pixels in this sample do not certify the remaining ~819,640 instances. Do not
treat Level C rates as dataset-wide QC.

## 6. Train / test comparison

Distinguish three populations:

1. **Training set** (4407 studies) — labels sparse; reports present; DICOM
   headers profiled.
2. **Example / public test** (3 studies in these files) — format and ID check
   only. Replaced at scoring.
3. **Hidden scoring test** (~1,300 studies per the official data page; **not
   in these files**). No local labels, reports, or headers.

Differences supported by available metadata:

- Example test has no Report and none of the 12 label columns.
- Example-test plane mix exists (Sag 7, Cor 4, Ax 4) but n = 3 is too small
  for a shift test.
- No study-ID overlap between train and the example test.
- `sample_submission.csv` matches example-test IDs and official headers.

Fields unavailable for comparison with the hidden test:

- PatientID completeness and uniqueness
- manufacturer, model, field strength, matrix, spacing
- series-description vocabulary
- label prevalence
- report language mix
- plane / sequence completeness per study

Distribution-shift risks (unmeasured on the hidden test, visible on train):

- Vendor and model heterogeneity (six vendor families, 46 models).
- Matrix sizes from 192×192 to 1200×1200.
- 1.5 T vs 3.0 T mix, plus 5.4% missing field strength.
- 17.2% of representative series lack a usable SeriesDescription.
- Labels exist for only 1.3% of train studies; hidden-test prevalence is
  unknown.
- Weak supervision from multilingual reports will not be available at test
  time.

The 3-study example test must not be used as a scientific holdout or as a
proxy for the hidden scoring test.

## 7. Fold-readiness decision

**Verdict: READY_WITH_RESTRICTIONS**

Measured support for proceeding (with restrictions), rather than
`NOT_READY`:

- Train CSVs and folders reconcile (4407 studies, 24371 series).
- No missing/duplicated study or series identifiers, no train vs example-test
  ID overlap.
- PatientID completeness is 1.0 on 4407 / 4407 headers.
- All 12 targets have at least 5 known positives (minimum MCL = 9).

Why this is not `READY_FOR_PATIENT_GROUPED_FOLDS`:

- Only 58 / 4407 studies are labelled; 4349 are unknown on every target.
- MCL, Lateral OA, and Baker's are tight for 5-fold per-fold AUC.
- Intra-study PatientID consistency and true person-level identity are
  unverified.
- The auditor flag `patient_grouped_folds_allowed` remains `false` by design
  until coordinator approval.
- Level C QC covers 0.5% of studies; slice-order and duplicate-slice checks
  were not run.
- This summary must be accepted before any fold file is written.

## 8. Proposed fold design

Design only. **Do not create fold files in this task.**

| Item | Proposal |
| --- | --- |
| Number of folds | **5** (matches `knee/folds.py` default) |
| Grouping key | **PatientID**, which is complete on this train set. Record that it is currently 1:1 with study ID. |
| Stratification | Greedy iterative multilabel stratification on the group-level label matrix (`assign_grouped_folds` in `knee/folds.py`) |
| Unknown labels | Keep as NaN. Collapse groups with `nanmax` so unknown stays unknown. Never impute 0. Stratify only on known 0/1. |
| Minimum-positive constraint | Each fold must receive ≥1 known positive for every target whenever group-level positives ≥ n_folds (true today for all 12). Report any target that cannot meet that bar; do not silently drop it. |
| Locked holdout | Do **not** use the 3-study example test as a scientific holdout. Do not treat the hidden ~1,300-study test as locally available. After this summary is accepted, lock a small labelled holdout from the 58 fully labelled studies (suggested 8–12 studies, still covering every target if possible) and keep it out of model selection. |
| Random seed | **42** (`folds_from_study_table` default) |
| Unlabelled studies | Assign the 4349 unknown-label studies to folds by group (here: one study per group) without treating unknown as negative. They may be used for semi-supervised or report-extraction experiments but must not enter supervised loss as 0. |

Gitignored files that must remain uncommitted:

- `outputs/kaggle_audit_outputs/` (this audit, including identifier-bearing CSVs)
- `outputs/` fold maps, predictions, checkpoints
- `predictions/`, `data/`, DICOM, PDFs, figures, credentials
- any table that contains study, series, or patient identifiers

Fold assignment tables contain study IDs by construction and belong under
gitignored `outputs/`, not `reports/`.

## What this analysis did not do

- Did not create folds.
- Did not train a model.
- Did not copy or commit row-level identifiers, report text, or DICOM data.
- Did not walk the full instance tree.
- Did not inspect intra-study PatientID consistency across series.
- Did not profile the hidden scoring test.
