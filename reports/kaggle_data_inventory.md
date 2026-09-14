# Kaggle data inventory — RSNA Knee Abnormality Detection

Phase 2 inventory only. No DICOM archives, zip shards, CSVs, or reports were
downloaded. The literature corpus and scientific claim ledgers were not
modified.

| Field | Value |
| --- | --- |
| Competition slug | `rsna-knee-abnormality-detection` |
| Official Data tab | https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data |
| Official Overview | https://www.kaggle.com/competitions/rsna-knee-abnormality-detection |
| Inventory UTC | `2026-09-14T01:50:13Z` |
| Kaggle CLI | `2.2.4` (installed into `.venv`; latest PyPI at inventory time) |
| Host | Cursor Cloud Agent VM (`cursor`) |

## 1. What was run

Commands (token never printed, logged, copied, or committed):

```bash
python -m pip install -U kaggle
kaggle --version
# Confirm credential *presence* only; never print the value.
python -c 'import os; print("KAGGLE_API_TOKEN_present:", bool(os.environ.get("KAGGLE_API_TOKEN")))'
kaggle competitions files rsna-knee-abnormality-detection --page-size 200 --format json -q
df -h /
df -B1 /
free -b
nproc
```

Unauthenticated HTTP probes (status codes only; no credentials sent):

| Request | HTTP |
| --- | --- |
| `GET https://www.kaggle.com/api/v1/competitions/data/list/rsna-knee-abnormality-detection` | `401 Unauthenticated` |
| `GET https://api.kaggle.com/v1/competitions/data/list/rsna-knee-abnormality-detection` | `401 Unauthenticated` |
| `GET https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data` | `200` (JS shell, ~5.7 KiB, no file table in HTML) |

## 2. Authentication

**Not confirmed.** `KAGGLE_API_TOKEN` was not set in this Cloud Agent
environment. `KAGGLE_USERNAME` / `KAGGLE_KEY` were also absent. No
`~/.kaggle/access_token` or `~/.kaggle/kaggle.json` file existed.

The official CLI (`kaggle competitions files rsna-knee-abnormality-detection`)
exited `1` with:

```
Authentication required to call the Kaggle API.
```

A `401 Unauthenticated` API response is **not** evidence that competition
rules were rejected; the client never presented an identity. Re-run the
files command after injecting the token into the environment (do not commit
it):

```bash
# Presence check only
python -c 'import os; assert os.environ.get("KAGGLE_API_TOKEN"), "missing token"'
kaggle competitions files rsna-knee-abnormality-detection --page-size 200 --format csv
```

Paginate with `--page-token` until the listing is exhausted. If the API
returns on the order of 10^5 paths, keep the raw listing out of git.

## 3. Competition rules

**Not confirmed.** Rules acceptance can only be checked after authentication.

Expected signals once a token is present:

| API / CLI outcome | Interpretation |
| --- | --- |
| File list returned | Account can see competition data (rules accepted) |
| `403` / “must accept competition rules” | Authenticated, rules not accepted |
| `401` / “Authentication required” | No usable credential (this run) |

The Data and Overview pages are publicly reachable as HTML shells. That does
not imply this environment’s Kaggle account has accepted the rules.

## 4. Size summary

| Quantity | Value | Source | Confidence |
| --- | --- | --- | --- |
| Total files in Data Explorer | **819,640** | Official Data tab text | High (official page; not CLI) |
| Data Explorer total size | **569.76 GB** | Official Data tab (“Data Explorer … 569.76 GB”) | High (official page; not CLI) |
| Total compressed download size from `kaggle competitions files` | **not obtained** | CLI auth failed | n/a |
| Expected extracted size | **≈ 569.76 GB** if the Explorer total is the unpacked DICOM + CSV tree; zip-of-DICOM often stays the same order of magnitude because pixel data is already compressed | Estimable only from Explorer total | Medium — cannot separate zip vs tree without the CLI listing |
| `sample_submission.csv` | **470 B** | Official Data tab (`sample_submission.csv(470 B)`) | High |
| Per-file sizes for other named objects | **not published on the Data tab snippets retrieved** | — | n/a |
| Available disk on this VM | **245 GiB free** (254 GiB filesystem, 8.6 GiB used, 4%) = **263,002,423,296 bytes** free | `df -h /` and `df -B1 /` at inventory UTC | High |
| RAM | 16 GiB total, ~6.7 GiB available, 0 swap | `free -b` | High |
| CPU | 4 × Intel Xeon (KVM), 1 thread/core | `nproc` / `lscpu` | High |
| GPU | **none** (`nvidia-smi` missing, no `/dev/nvidia*`) | host inspection | High |

Headroom check: 569.76 GB dataset ≰ 245 GiB free. This Cursor Cloud VM
**cannot** hold the full competition tree even before workspace, cache, or a
second copy for extraction.

## 5. Official named objects (Data tab)

Filenames below are taken from the official competition Data tab, not from
synthetic `knee/data.py` outputs and not from community mirrors. The CLI did
**not** return a complete per-file size listing, so this table is the
documented dataset *layout* plus the two sizes the Data tab actually states.

| Name | Role | Official description (condensed) |
| --- | --- | --- |
| `train.csv` | labels + reports + study metadata | One row per training study. |
| `train_series.csv` | series metadata | One row per training series (one MRI acquisition). |
| `train_series/` | training images | Training DICOMs as `train_series/<StudyInstanceUID>/<SeriesInstanceUID>/*.dcm`. One `.dcm` per slice. Series typically 20–45 slices (median 30), long tail to a few hundred. |
| `test.csv` | test study metadata (example) | Example test file with **three** study IDs from the public test set. Replaced with the actual test data at scoring. About **1,300** studies in the test set. `Report` is **not** provided at test time. |
| `test_series.csv` | test series metadata (example) | Same schema as `train_series.csv` for the example test studies. Replaced at scoring. |
| `test_series/` | test images (example) | Example test DICOMs, same layout as `train_series/`. Replaced with the real test DICOMs at scoring. |
| `sample_submission.csv` | sample submission | Valid submission with all label columns set to `0.5`. Size **470 B**. |

### 5.1 `train.csv` columns (official)

- `StudyInstanceUID` — unique study id; matches the folder name under `train_series/`.
- `Report` — free-text radiology report; may be in any of several languages.
- Twelve binary labels `(0/1)`:

| Official column | Meaning |
| --- | --- |
| `ACL` | anterior cruciate ligament injury |
| `MCL` | medial collateral ligament injury |
| `Medial Meniscus` | medial meniscus tear |
| `Lateral Meniscus` | lateral meniscus tear |
| `Medial OA` | osteoarthritis of the medial tibiofemoral compartment |
| `Lateral OA` | osteoarthritis of the lateral tibiofemoral compartment |
| `PF OA` | patellofemoral osteoarthritis |
| `Effusion` | joint effusion / excess fluid |
| `Synovitis` | inflammation of the joint lining |
| `Baker's` | Baker's cyst |
| `Contusion` | bone contusion / bone bruise |
| `Fracture` | fracture |

### 5.2 `train_series.csv` columns (official)

- `StudyInstanceUID` — parent study.
- `SeriesInstanceUID` — unique series id; matches the folder name under `train_series/<StudyInstanceUID>/`.
- `Fluid_Sensitive` — `1` if the sequence emphasizes fluid signal (T2, PD, STIR, and similar), else `0`.
- `Fat_Suppression` — `1` if fat suppression is applied, else `0`. Official note: the two flags are often correlated in training data and are **not** equivalent for every case.
- `Anatomical_Plane` — `Sagittal`, `Coronal`, or `Axial`.

### 5.3 Classification requested by this inventory

| Class | Official objects | Notes |
| --- | --- | --- |
| Metadata | `train_series.csv`, `test_series.csv`; `StudyInstanceUID` / series flags on the CSV rows | Series-level acquisition descriptors. |
| Reports | `Report` column on `train.csv` (and on the example `test.csv` if present) | Multilingual free text. **Absent at scoring.** |
| Labels | twelve binary columns on `train.csv` | Expert labels are **not** implied for every row by the Data tab text. |
| Sample submission | `sample_submission.csv` | 470 B; all targets `0.5`. |
| Training images | `train_series/**/*.dcm` | Vast majority of the 819,640 files / 569.76 GB. |
| Test images | `test_series/**/*.dcm` | Example public tree only; hidden test replaces it at scoring. |

Row counts that appear in **community** write-ups (4,407 training studies,
24,371 training series, 58 expert-labeled studies) were **not** verified from
the official file listing or from downloaded CSVs. They are **out of scope**
for this inventory.

## 6. What the CLI listing would have added

`kaggle competitions files <slug>` is the official downloadable-file catalog
(name + size, paginated, max 200 rows per page). For a Data Explorer tree of
819,640 files it may list either:

1. every Explorer path (hundreds of thousands of `.dcm` rows), or
2. a small set of downloadable archives plus the CSVs.

Until that command succeeds, **do not assume zip shard names** (`train.zip`,
`train_images/`, etc.). They were not observed.

## 7. Environment recommendation

**Use a Kaggle notebook** as the primary place to touch the real dataset and
to submit. Do not download the DICOM tree into this Cursor Cloud VM.

| Criterion | This Cursor Cloud VM | Typical local workstation | Kaggle notebook |
| --- | --- | --- | --- |
| Store ~570 GB DICOMs | No (245 GiB free) | Only if ≥ ~1.2 TB free (tree + extract + cache) | Competition data is **mounted** at `/kaggle/input/rsna-knee-abnormality-detection`; no 570 GB download into the workspace |
| Persistence | Ephemeral VM disk | Persistent | Inputs persist as the competition dataset; `/kaggle/working` is per-session |
| Memory | 16 GiB, no swap | Depends | Notebook RAM tiers (CPU/GPU sessions) |
| GPU | None | Optional / recommended for 3D MRI | GPU sessions available; **required path** for many training runs |
| Submission | Cannot submit | Cannot submit | **Required**: notebook submission, internet off, ≤ 9 h CPU or GPU, `submission.csv` |

Overview constraints that drive this choice (official competition page):

- Submissions must be notebooks.
- CPU notebook ≤ 9 hours; GPU notebook ≤ 9 hours.
- Internet disabled at scoring.
- Public external data and pre-trained models are allowed.
- Hidden test studies (~1,300) and their DICOMs are injected at scoring;
  `Report` is not available then.

**Cursor Cloud** remains appropriate for repository work (ontology, tests,
this inventory). It is the wrong place to store or train on the raw DICOMs.

**Local workstation** is a reasonable *companion* only when persistent disk
and a GPU already exist, for preprocessing experiments. It still cannot
replace the Kaggle notebook for hidden-test inference.

## 8. Target-name mismatch (do not “fix” in this phase)

`TARGET_KEYS` in `knee/config.py` is this repo’s source of truth and was
**not** edited. Official submission / label columns from the Overview and
Data tabs use different strings:

| Slot | `TARGET_KEYS` (repo) | Official column |
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

The competition-data-auditor should reconcile these against a downloaded
`sample_submission.csv` in a later phase. Do not rename `TARGET_KEYS` here.

## 9. Policy notes

- No files were written under `data/` or `literature/`.
- No PDFs, figures, DICOMs, credentials, or checkpoints were committed.
- `reports/*` remains gitignored for generated non-markdown artefacts.
  Reviewable markdown (including this inventory and the Phase 2
  `DATA_NOT_AVAILABLE` audit) is tracked via `!reports/*.md` plus an
  explicit `!reports/kaggle_data_inventory.md` exception.

## 10. Unresolved questions

1. Inject `KAGGLE_API_TOKEN` into the Cloud Agent environment and re-run
   `kaggle competitions files rsna-knee-abnormality-detection` with
   pagination to obtain **every downloadable file and its size**.
2. Confirm rules acceptance with an authenticated call (list vs `403`).
3. After a successful listing: is the downloadable catalog ~7 named
   objects, zip shards, or 819,640 individual paths?
4. Compressed vs Explorer size: are DICOMs offered as zips, and if so what
   is the zip total?
5. Exact byte sizes of `train.csv`, `train_series.csv`, and the example
   test CSVs (only `sample_submission.csv` = 470 B is published on the Data
   tab).
6. Map official label columns to `TARGET_KEYS` without renaming either
   side until the auditor has the real CSVs.

Stop. Do not download DICOM archives from this inventory.
