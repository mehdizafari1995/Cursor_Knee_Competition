#!/usr/bin/env python3
"""Image-only inference for the study-level baseline.

`--smoke` writes probabilities for a synthetic sample_submission contract using
tiny generated tensors. Real DICOM inference requires RSNA_DATA_ROOT and a
checkpoint, and is not run in this phase.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from knee.baseline_config import load_baseline_config  # noqa: E402
from knee.baseline_model import build_baseline_model, smoke_batch  # noqa: E402
from knee.config import OFFICIAL_TARGET_COLUMNS, STUDY_ID_COLUMN  # noqa: E402
from knee.paths import DataNotAvailable, get_data_root  # noqa: E402
from knee.submission import load_sample_submission  # noqa: E402


def _smoke_predictions(n_rows: int, seed: int) -> pd.DataFrame:
    images, _labels = smoke_batch(
        batch_size=n_rows, num_slices=4, image_size=16, seed=seed
    )
    model = build_baseline_model(smoke=True, pretrained=False)
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(images)).cpu().numpy()
    frame = pd.DataFrame(probs, columns=list(OFFICIAL_TARGET_COLUMNS))
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "baseline.yaml"))
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--sample-submission", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--output",
        default=str(ROOT / "outputs" / "baseline" / "predictions.csv"),
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    cfg = load_baseline_config(args.config, smoke=True if args.smoke else None)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.smoke or cfg.smoke:
        if args.sample_submission:
            sample = load_sample_submission(Path(args.sample_submission))
            ids = sample[STUDY_ID_COLUMN].astype(str).tolist()
        else:
            ids = [f"smoke-study-{i:03d}" for i in range(3)]
        probs = _smoke_predictions(len(ids), cfg.seed)
        probs.insert(0, STUDY_ID_COLUMN, ids)
        probs.to_csv(output, index=False)
        print(f"SMOKE MODE: wrote {output} ({len(probs)} rows). Not a scientific result.")
        (output.with_suffix(".meta.json")).write_text(
            json.dumps(
                {
                    "mode": "software_smoke",
                    "scientific_evaluation": False,
                    "n_rows": int(len(probs)),
                },
                indent=2,
            )
            + "\n"
        )
        return 0

    try:
        root = get_data_root(args.data_root)
    except DataNotAvailable as exc:
        print(exc)
        return 2

    if args.checkpoint is None:
        print(
            f"Data root resolved to {root}, but no --checkpoint was given. "
            "Refusing to invent weights. This phase does not run full inference "
            "on competition DICOM files."
        )
        return 3

    print(
        "Real DICOM inference is implemented at the library layer "
        "(knee.dataset.StudySliceDataset) but is not executed in this phase."
    )
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
