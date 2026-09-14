# Kaggle gold-only second-reader pilot run

Operator / preflight record. This file is not a scientific metrics report.
Report text, StudyInstanceUID values, and PatientID values are not included.

## Preflight result: STOP — no inference

`KAGGLE_API_TOKEN_present: True`. No weights were downloaded in Cursor.
Internet was not enabled for Hugging Face weight retrieval. The real GPU
notebook was not pushed. Coordinator gates were not changed.

Required model (unchanged):

| Field | Required frozen value |
| --- | --- |
| Model | `microsoft/Phi-3.5-mini-instruct` |
| Revision | `2fe192450127e6a83f7441aef6e3ca586c338b77` |
| Licence | MIT (confirmed on that revision’s `LICENSE`) |

### Checksum comparison (metadata only; files not downloaded)

Hugging Face tree API for revision `2fe192450127e6a83f7441aef6e3ca586c338b77`
(2026-09-14). LFS oid is the SHA-256 of the blob.

| File | Frozen recorded SHA-256 | Published LFS oid / note | Match |
| --- | --- | --- | --- |
| `model-00001-of-00002.safetensors` | `c5214cdb995ed3dd716add8d9efbfe016b76bb2f1c4c1c6a95497d7a8837` (60 hex) | `c5214cdb995ed3dd716add8d9efbfe016b76bb2f1c4c1e6c1c6a95497d7a8837` (64 hex) | **NO** |
| `model-00002-of-00002.safetensors` | `41246eed2b75b66526339c5d32d6f7acdefe0bd24180f97c74303f4656877344` | `41246eed2b75b66526339c5d32d6f7acdefe0bd24180f97c74303f4656877344` | YES |
| `tokenizer.model` | not recorded in the frozen pilot yaml | `9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347` | n/a |
| `config.json` | not recorded | git blob `62d1ca8d4b3c8ab21ac9a30775e1a224be66ef58`, size 3451 | n/a |
| `tokenizer.json` | not recorded | git blob `e213d846868b09aefa5225e6026d91b41e19e7ff`, size 1844408 | n/a |
| `tokenizer_config.json` | not recorded | git blob `7280c871b0d18a2a03000fb8d638983793aa33e1`, size 3984 | n/a |

The frozen shard-1 digest is four hex characters short of SHA-256 and is not
equal to the published LFS oid. `_verify_shards` compares full strings, so
any authentic copy of this revision would be refused. Protocol: stop without
inference; do not substitute another model or revision; do not download
unverified weights.

### Kaggle snapshot search (no weight download)

`mahdimzafari` has **no** Kaggle Model for this snapshot. Existing private
dataset `mahdimzafari/cursor-knee-source` is source ZIP only.

Closest public Kaggle Model with matching **weight file sizes**:
`richolson/phi-3.5-mini-instruct/pyTorch/default/1` (created 2024-09-21,
MIT, files include both `model-0000{1,2}-of-00002.safetensors` at
4972489328 and 2669692552 bytes). README.md is 48266 bytes vs 48686 bytes
on revision `2fe19245`, so this instance is **not** proven to be that git
revision. Shard SHA-256 values were **not** computed (weights not
downloaded).

GGUF / other Phi-3.5 listings (`paultimothymooney/phi-3.5-mini-instruct`,
`Microsoft/phi-3`) are the wrong format or the wrong model family.

## What was not run

- `n_gold_studies_processed`: 0 (required 58)
- `n_report_only_studies_processed`: 0
- `scientific_evaluation`: `false`
- `internet_used`: `false`
- `download_weights`: `false`
- image training / image folds / competition submission: not run
- freeze hashes were not recorded on Kaggle before metrics (no metrics)

## Single manual action required

Correct the frozen `model-00001-of-00002.safetensors` SHA-256 in
`reports/second_reader_model_compliance.md` and
`configs/second_reader_pilot.yaml` to the published 64-character LFS oid
for the **same** model and revision:

`c5214cdb995ed3dd716add8d9efbfe016b76bb2f1c4c1e6c1c6a95497d7a8837`

Do not change the model id or revision. After that correction, re-run this
gold-pilot task so a Kaggle snapshot can be attached and every
weight/config/tokenizer checksum verified **before** inference.

## Cursor

```bash
source .venv/bin/activate
pytest
python scripts/run_second_reader_pilot.py   # DATA_NOT_AVAILABLE without RSNA_DATA_ROOT
```
