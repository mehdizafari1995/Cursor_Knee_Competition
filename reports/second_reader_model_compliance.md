# Phase 4C model-compliance report

Inspected before any weight download. No model weights were downloaded or
loaded in Cursor. Eligibility is documented here so a Kaggle operator may
**attach** a local snapshot after this check.

Retrieved 2026-09-14. Competition-rule sources are the same as
`reports/competition_pretrained_model_rules.md`.

## Competition constraints that apply

| Constraint | Source | Implication |
| --- | --- | --- |
| Public external data and pretrained models allowed | Competition overview | A public instruction model may be used if equally accessible |
| External data/models must be public, equally accessible, minimal cost | Rules §2.6 | No paid API, no gated-only weights that other participants cannot obtain |
| Scoring notebooks: internet disabled, ≤ 9 hours, CPU or GPU | Overview | Load with `local_files_only=True`; never `from_pretrained` from the hub |
| Test-time `Report` is unavailable | Data tab | Second reader is training-report only; never image-inference |
| Winner licence CC-BY-NC 4.0 | Rules | Pretrained licences need not be re-licensed, but use during the competition must have been lawful |

This repository still forbids sending reports to any external API.

## Selection criteria

A candidate must:

1. Be a public pretrained **instruction** model (encoders from Phase 4B are not sufficient for structured JSON generation)
2. Have a clear licence
3. Fit an available Kaggle GPU (typically 16 GB T4 / P100) for 58 short reports
4. Support structured JSON output
5. Be attachable as a Kaggle Model or private dataset of a pinned revision
6. Run with internet disabled
7. Have published or attach-time checksums

## Candidates (at most three)

### 1. `microsoft/Phi-3.5-mini-instruct` — SELECTED for the gold pilot

| Field | Value |
| --- | --- |
| Model | Phi-3.5-mini-instruct |
| Hugging Face id | `microsoft/Phi-3.5-mini-instruct` |
| Revision (git SHA) | `2fe192450127e6a83f7441aef6e3ca586c338b77` |
| Licence | MIT |
| Parameter count | 3,821,079,552 (BF16) |
| Quantization (pilot) | none (float16 on GPU) |
| Expected GPU memory | ~8 GB weights + KV cache for short reports |
| Structured JSON | Instruction-tuned; JSON-only prompt frozen in `configs/second_reader_prompt.yaml` |
| Source | https://huggingface.co/microsoft/Phi-3.5-mini-instruct |
| Gated | no |
| Cost | free |
| Kaggle attachment | Private dataset or Kaggle Model of this revision |
| Shard SHA-256 | `model-00001-of-00002.safetensors` = `c5214cdb995ed3dd716add8d9efbfe016b76bb2f1c4c1c6a95497d7a8837` |
| | `model-00002-of-00002.safetensors` = `41246eed2b75b66526339c5d32d6f7acdefe0bd24180f97c74303f4656877344` |
| Verdict | **PASS** — clear OSI licence, ungated, multilingual instruct, fits 16 GB |

Custom `modeling_phi3.py` is part of the snapshot (`trust_remote_code` on local files only).

### 2. `google/gemma-2-2b-it` — documented alternate

| Field | Value |
| --- | --- |
| Model | Gemma 2 2B IT |
| Hugging Face id | `google/gemma-2-2b-it` |
| Revision (git SHA) | `299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8` |
| Licence | Gemma Terms of Use |
| Parameter count | 2,614,341,888 (BF16) |
| Quantization | none (float16) |
| Expected GPU memory | ~6 GB |
| Structured JSON | Instruction-tuned |
| Source | https://huggingface.co/google/gemma-2-2b-it |
| Gated on Hugging Face | yes (accept ToU) |
| Kaggle attachment | First-party Kaggle Model `google/gemma-2` is equally accessible to participants who accept the ToU |
| Shard SHA-256 | Record from the attached snapshot at attach time. Hugging Face LFS oids were not published without ToU acceptance in this compliance pass. |
| Verdict | **PASS with restriction** — smaller GPU footprint, but HF gating means checksums must be taken from the attached Kaggle snapshot. Not the selected gold-pilot model. |

### 3. `Qwen/Qwen2.5-7B-Instruct` — documented alternate

| Field | Value |
| --- | --- |
| Model | Qwen2.5-7B-Instruct |
| Hugging Face id | `Qwen/Qwen2.5-7B-Instruct` |
| Revision (git SHA) | `a09a35458c702b33eeacc393d103063234e8bc28` |
| Licence | Apache-2.0 |
| Parameter count | 7,615,616,512 (BF16) |
| Quantization | Prefer 8-bit or 4-bit on 16 GB; BF16 (~15 GB) is tight |
| Expected GPU memory | ~8 GB at 8-bit; ~5 GB at 4-bit; ~15 GB BF16 |
| Structured JSON | Strong native JSON / tool-calling |
| Source | https://huggingface.co/Qwen/Qwen2.5-7B-Instruct |
| Gated | no |
| Shard SHA-256 | `model-00001-of-00004.safetensors` = `a1333e6293854747c481288ea83b348226af178dd565c49b6f9495ba1966aba7` |
| | `model-00002-of-00004.safetensors` = `f5d25a2772cb825164a2a2c0fb6d51a87e282abf21e4dd75bc5cfb3cd0ea6185` |
| | `model-00003-of-00004.safetensors` = `8efdec4c1bc12317ae1a38dc42b595ce777738a64deea3fcb8a0a91381bcdfd5` |
| | `model-00004-of-00004.safetensors` = `1a72d403cdf0c1ec3cb7f289f17b394a01e64394c2e9b3c0f94dbce3faf879bd` |
| Verdict | **PASS as alternate** — not selected for the 58-study pilot because BF16 is tight on 16 GB and 8-bit adds a bitsandbytes dependency. |

## Rejected for this pilot (not among the three)

| Model | Reason |
| --- | --- |
| `Qwen/Qwen2.5-3B-Instruct` | Licence `qwen-research` / `other`, not Apache-2.0 |
| `FacebookAI/xlm-roberta-base` | Encoder; cannot emit the frozen JSON schema without a separate classification head |
| `google-bert/bert-base-multilingual-cased` | Same encoder limitation |
| Hosted LLM APIs | Internet required; reports would leave Kaggle |

## What this check authorizes

- Attaching the **Phi-3.5-mini-instruct** snapshot (revision above) as a Kaggle
  Model or private dataset
- Loading it with `local_files_only=True` and `HF_HUB_OFFLINE=1` inside
  `notebooks/04_kaggle_second_reader_gold_pilot.ipynb`
- Verifying shard SHA-256 before generate

## What this check does not authorize

- Downloading weights from Hugging Face, GitHub, or any other network host
  inside Cursor or inside the Kaggle notebook
- Processing the 4,349 report-only studies
- Training an image model
- Treating model-only calls as training labels

`configs/second_reader_pilot.yaml` records `compliance.passed: true` and
`download_weights: false`. Code raises if `download_weights` is flipped.
