# Cursor_Knee_Competition

A compact, end-to-end pipeline for the **Kaggle RSNA Knee Abnormality
Detection** competition. It generates synthetic knee MRI studies, trains a
multi-label CNN, and produces a competition-style `submission.csv` plus
diagnostic plots. The project doubles as a fast smoke test that proves the
Cloud Agent development environment works.

## The task

Each knee MRI **study** is scored for **twelve independent binary findings**:

1. ACL injury
2. MCL injury
3. Medial meniscus injury
4. Lateral meniscus injury
5. Medial-compartment osteoarthritis
6. Lateral-compartment osteoarthritis
7. Patellofemoral osteoarthritis
8. Joint effusion
9. Synovitis
10. Baker cyst
11. Bone contusion
12. Fracture

- **Metric:** macro-averaged ROC AUC across the twelve targets.
- **Weak supervision:** training may use free-text (multilingual) radiology
  reports to derive labels.
- **Inference:** test-time prediction uses **MRI images only** — no report text
  is consumed at inference.

> The synthetic pipeline (`python -m knee.pipeline`) still uses generated
> studies only. The Phase 2 audit and baseline talk to real competition files
> only when `RSNA_DATA_ROOT` points at a local copy; they do not download data.

## Quick start

```bash
# One-time setup (creates .venv and installs pinned dependencies)
bash .cursor/install.sh
source .venv/bin/activate

# Run the full pipeline (data -> weak labels -> train -> image-only inference)
python -m knee.pipeline

# Train on latent labels instead of report-derived weak labels
python -m knee.pipeline --no-weak-supervision

# Run the test suite
pytest -q

# Phase 2: competition audit + baseline smoke (no full training)
export RSNA_DATA_ROOT=/path/to/rsna-knee-abnormality-detection   # if you have the data
python scripts/audit_competition_data.py                        # DATA_NOT_AVAILABLE if unset
python scripts/train_baseline.py --smoke                        # tiny generated tensors only
python scripts/predict_baseline.py --smoke
python scripts/build_kaggle_bundle.py    # Phase 3 allowlisted Kaggle source zip
```

Phase 3 adds `notebooks/01_kaggle_real_data_audit.ipynb` for an internet-off
Kaggle run against the attached competition dataset (no API token). See
`reports/kaggle_notebook_instructions.md`. The generated ZIP is gitignored.

Outputs are written to `outputs/`:

| File | Description |
| --- | --- |
| `submission.csv` | Per-study probabilities for all 12 findings (`study_id` + 12 columns) |
| `metrics.json` | Macro ROC AUC, per-target ROC AUC, and run metadata |
| `per_target_auc.png` | Per-target ROC AUC bar chart with the macro-average line |
| `samples.png` | Synthetic MRI studies annotated with their positive findings |
| `model.pt` | Trained model weights |

## Project layout

```
knee/
  config.py     # hyperparameters, TARGET_KEYS, official Kaggle column mapping
  paths.py      # RSNA_DATA_ROOT resolution (no hard-coded machine path)
  audit.py      # read-only competition-data audit
  folds.py      # grouped multilabel cross-validation
  dicom_io.py   # deterministic DICOM load + slice order
  ...
configs/baseline.yaml
scripts/        # audit, folds, train, predict, submission, kaggle bundle
notebooks/      # internet-off Kaggle real-data audit
tests/          # pipeline, agents, literature, audit, leakage, labels, metrics, kaggle
```

Official Kaggle submission headers (`ACL`, `MCL`, …, `Baker's`, …) are listed
in `knee/config.py` as `OFFICIAL_TARGET_COLUMNS`. Internal `TARGET_KEYS` are
not renamed. See `reports/baseline_design.md` and `reports/data_audit.md`.

## Development environment

The environment is configured in `.cursor/environment.json`, which runs
`.cursor/install.sh` to create a Python virtual environment and install the
pinned dependencies in `requirements.txt` (NumPy, pandas, scikit-learn,
Matplotlib, Pillow, pydicom, and CPU builds of PyTorch / torchvision).
