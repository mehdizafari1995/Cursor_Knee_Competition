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

> No real patient data is used or downloaded. Synthetic MRI studies encode each
> finding as a lesion at a fixed anatomical location so the multi-label model
> has a genuinely learnable signal, and synthetic multilingual reports provide
> the weak-supervision source.

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
```

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
  config.py     # hyperparameters, paths, the 12 target definitions
  data.py       # synthetic MRI study generation + PyTorch datasets
  reports.py    # multilingual report synthesis + weak-label extraction
  model.py      # small multi-label CNN classifier
  train.py      # training loop + macro ROC AUC evaluation
  report.py     # plots + submission writer
  pipeline.py   # end-to-end entrypoint (python -m knee.pipeline)
tests/          # fast pytest smoke tests
```

## Development environment

The environment is configured in `.cursor/environment.json`, which runs
`.cursor/install.sh` to create a Python virtual environment and install the
pinned dependencies in `requirements.txt` (NumPy, pandas, scikit-learn,
Matplotlib, Pillow, and CPU builds of PyTorch / torchvision).
