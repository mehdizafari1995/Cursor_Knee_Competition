# Phase 4C offline multilingual second-reader gold pilot

Evaluate a frozen instruction-model second reader on the **58 completely
expert-labelled studies only**. Image models, image folds, leaderboard
submissions, and the 4,349 report-only studies are out of scope.

Pilot version: `second-reader-pilot-v1.0.0`
Evaluation protocol: `gold-pilot-v1.0.0`
Deterministic extractor: `report-labeler-v2.0.0`
Prompt: `second-reader-prompt-v1.0.0`
Schema: `second-reader-schema-v1.0.0`
Ontology: `pilot-1.0` (`literature/ontology.yaml`)
Selected model: `microsoft/Phi-3.5-mini-instruct` (see compliance report)

The prompt, schema, ontology version, model revision and decoding
configuration are hashed into `freeze_manifest.json` **before** metrics are
computed.

## 1. Model selection

See `reports/second_reader_model_compliance.md`. Weights are never downloaded
by this repository. The gold notebook loads an attached local snapshot only
after shard SHA-256 verification.

## 2. Frozen extraction schema

Each of the twelve `TARGET_KEYS` returns only:

| Field | Values |
| --- | --- |
| `status` | `positive`, `negative`, `uncertain`, `not_mentioned` |
| `confidence` | `high`, `medium`, `low` |
| `evidence_category` | closed list in `knee/second_reader_schema.py` |
| `uncertainty_or_negation_category` | closed list in `knee/second_reader_schema.py` |

Chain-of-thought and report quotations are forbidden. The parser drops those
keys and marks the study malformed.

Required clinical distinctions (prompt + schema, not free text):

- present versus absent
- uncertain versus confirmed
- historical and postoperative
- reconstruction versus acute injury
- degeneration versus meniscal injury
- compartment-specific OA
- effusion versus synovitis
- Baker's cyst versus other popliteal findings
- marrow signal versus bone contusion
- suspected versus confirmed fracture

## 3. Privacy

- Reports stay inside the private Kaggle runtime (or a local `RSNA_DATA_ROOT`
  the operator already holds)
- Internet remains disabled
- Never print, log, download, commit, or export report text
- Do not send reports to any external API
- Row-level predictions remain under gitignored / Kaggle private outputs
- Committed reports contain aggregates and SHA-256 hashes only

## 4. Gold pilot

Use the 58 gold studies only. Unknown gold is skipped. Unknown / uncertain /
not_mentioned predictions are abstentions, never false negatives.

Per target, report:

- coverage
- positive precision
- NPV
- positive and negative capture
- selective error
- abstentions
- bootstrap confidence intervals (study-level, 1,000 resamples, seed 42)
- malformed-output rate
- runtime and GPU-memory use

Do not hide abstentions inside sensitivity. Conditional sensitivity is **not**
full-set recall; the full-set analogue is positive_capture.

## 5. Consensus

Compare three systems on the same gold studies:

| System | Rule |
| --- | --- |
| deterministic v2 alone | `report-labeler-v2.0.0` |
| second reader alone | parsed JSON statuses mapped to ternary labels |
| strict agreement | positive+positive = candidate positive; negative+negative = candidate negative; any disagreement = unknown |

Uncertain or not_mentioned is unknown unless the existing verified
`global_normal` negative rule applies. Model-only predictions (reader
positive, extractor unknown) are **not** training labels.

## 6. Acceptance

Each target receives exactly one of:

- `PILOT_PASS`
- `PILOT_RESTRICTED`
- `PILOT_FAIL`

Frozen floors (positive precision and NPV prioritized; coverage reported
separately, not a pass gate) live in `configs/second_reader_pilot.yaml`.
Small-n targets cannot `PILOT_PASS`. Malformed-output rate above 0.05 fails
the target.

Do not scale to 4,349 reports until a **scientific** gold pilot (real Kaggle
GPU, internet off, local transformers backend) contains `PILOT_PASS` or
justified `PILOT_RESTRICTED` calls. The coordinator gate
`NOT_READY_TO_SCALE_SECOND_READER_TO_REPORT_ONLY` stays in force.

## 7. Kaggle versus Cursor

Cursor runs **fixture** tests only (`generate_fn`, no weights). The real
58-study GPU run is `notebooks/04_kaggle_second_reader_gold_pilot.ipynb`
(internet off, GPU on). Retrieve sanitized aggregates and hashes only.

## 8. Out of scope

- image-model training
- image folds
- leaderboard submission
- processing report-only studies
- downloading weights
- quoting reports
