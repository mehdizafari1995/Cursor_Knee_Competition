# Cursor_Knee_Competition

A compact, end-to-end **knee osteoarthritis KL-grade classification** pipeline.
It generates synthetic knee radiographs, trains a small CNN, and produces a
competition-style `submission.csv` plus diagnostic plots. The project doubles as
a fast smoke test that proves the Cloud Agent development environment works.

> The Kellgren–Lawrence (KL) scale grades knee OA severity from 0 (none) to
> 4 (severe). Higher grades show a narrower joint space and more osteophytes;
> the synthetic data encodes exactly those signals so the model has something
> real to learn — no actual patient data is used.

## Quick start

```bash
# One-time setup (creates .venv and installs pinned dependencies)
bash .cursor/install.sh
source .venv/bin/activate

# Run the full pipeline (data -> train -> evaluate -> submission)
python -m knee.pipeline

# Run the test suite
pytest -q
```

Outputs are written to `outputs/`:

| File | Description |
| --- | --- |
| `submission.csv` | Predicted KL grade per test image (`image_id,kl_grade`) |
| `metrics.json` | Test accuracy and run metadata |
| `confusion_matrix.png` | Per-class confusion matrix on the test split |
| `samples.png` | One synthetic radiograph per KL grade |
| `model.pt` | Trained model weights |

## Project layout

```
knee/
  config.py     # hyperparameters, paths, KL grade definitions
  data.py       # synthetic radiograph generation + PyTorch datasets
  model.py      # small CNN classifier
  train.py      # training / evaluation loops
  report.py     # plots + submission writer
  pipeline.py   # end-to-end entrypoint (python -m knee.pipeline)
tests/          # fast pytest smoke tests
```

## Development environment

The environment is configured in `.cursor/environment.json`, which runs
`.cursor/install.sh` to create a Python virtual environment and install the
pinned dependencies in `requirements.txt` (NumPy, pandas, scikit-learn,
Matplotlib, Pillow, and CPU builds of PyTorch / torchvision).
