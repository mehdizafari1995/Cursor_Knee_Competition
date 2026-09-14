# Kaggle gold-only second-reader pilot run

Operator instructions for the real 58-study GPU notebook. This file is not a
scientific metrics report.

## Attachments

1. Competition dataset `rsna-knee-abnormality-detection`
2. Private source ZIP rebuilt after this phase (`python scripts/build_kaggle_bundle.py`)
3. Checksummed `microsoft/Phi-3.5-mini-instruct` revision
   `2fe192450127e6a83f7441aef6e3ca586c338b77` as a Kaggle Model or private
   dataset. Verify shard SHA-256 in `configs/second_reader_pilot.yaml`.

Do not attach a Kaggle API token. Do not enable internet. Do not `pip install`.

## Notebook settings

- File: `notebooks/04_kaggle_second_reader_gold_pilot.ipynb`
- Internet: **off**
- Accelerator: **GPU**
- Persistence: default; outputs under `/kaggle/working/rsna_second_reader_gold_pilot`

The notebook must refuse to score the 4,349 report-only studies.

## What to retrieve into git

Only `/kaggle/working/rsna_second_reader_gold_pilot/sanitized/`:

- `second_reader_gold_pilot_summary.md`
- `second_reader_gold_metrics.json`
- `freeze_manifest.json`
- `runtime_metrics.json`
- `pilot_acceptance.json`
- `private_output_hashes.json`

Do not download private CSVs into Cursor. After retrieval, replace
`reports/second_reader_gold_pilot_comparison.md` with the sanitized summary
if and only if `scientific_evaluation: true` and
`inference_backend: local_transformers`.

## Cursor

```bash
source .venv/bin/activate
pytest
python scripts/run_second_reader_pilot.py   # DATA_NOT_AVAILABLE without RSNA_DATA_ROOT
```

Fixture pytest keeps `scientific_evaluation=false`.
