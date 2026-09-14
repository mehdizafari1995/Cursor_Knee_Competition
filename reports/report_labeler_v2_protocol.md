# Phase 4B report-labeler v2 protocol

Coverage-aware evaluation and refinement of the version-1 deterministic
extractor. Image models, patient-grouped image folds, and leaderboard
submissions remain out of scope.

Extractor version: `report-labeler-v2.0.0`
Evaluation protocol version: `gold-eval-v2.0.0`
Frozen v1 config: `configs/report_labeler_v1.yaml`
Runtime config: `configs/report_labeler.yaml`

The version-1 extractor is a baseline. Its outputs are **not** approved as
hard training labels. Version 2 does not train an image model.

## 1. Corrected evaluation

Unknown gold is never filled with 0. Unknown extractor outputs are
abstentions, never false negatives.

For every target the protocol reports:

| Quantity | Definition |
| --- | --- |
| n_gold_positive / n_gold_negative | All expert-labelled cases for that target |
| positive / negative calls and abstentions | Extractor outputs among gold-known studies |
| TP, TN, FP, FN | Among called cases only |
| abstained positives / abstained negatives | Gold polarity where the extractor returned unknown |
| positive_capture | TP / all gold positives |
| negative_capture | TN / all gold negatives |
| error_rate_all_gold | (FP + FN) / all gold-known cases (abstention is not an error) |
| selective_accuracy | (TP + TN) / called cases |
| positive precision | TP / (TP + FP) |
| NPV | TN / (TN + FN) |
| coverage | called / gold-known |
| coverage-versus-error curves | Sweep separate positive and negative thresholds |
| bootstrap CIs | Study-level percentile intervals for coverage, precision, NPV, positive_capture, negative_capture |

**conditional_sensitivity** = TP / (TP + FN) among called cases. It is **not**
full-set sensitivity. Full-set analogue is **positive_capture**. Version-1
summaries that printed `sensitivity: 0.985` were describing the conditional
quantity. That number must not be quoted as recall on all 58 gold studies.

Version-1 metrics are preserved under `v1_conditional` / `v1_alias_*` for
direct comparison.

## 2. Error diagnosis (no quotations)

Aggregate categories only, never report text or identifiers:

- negation-scope failure
- generic meniscal degeneration interpreted as injury
- postoperative or reconstructed anatomy
- historical finding
- uncertain or differential diagnosis
- compartment-unspecified OA
- generic degeneration interpreted as OA
- effusion versus synovitis
- popliteal/Baker's cyst terminology
- bone marrow signal versus contusion
- fracture exclusion statements
- language-detection failure
- unsupported terminology
- section-boundary failure

Implemented in `knee/report_error_taxonomy.py`.

## 3. Language handling

`detect_language` is an ensemble of:

1. The v1 Unicode-script + function-word heuristic
2. An independent offline character-trigram detector (`knee/report_ngram_language.py`)

No translation API, no LLM, no pretrained neural language-id weights.

Latin MRI abbreviations plus a single Greek or Cyrillic body script do not
by themselves force `uncertain`. The 321 v1 “uncertain Greek” assignments are
re-validated on Kaggle by comparing heuristic vs trigram agreement. Disagreement
stays uncertain. Uncertain language rows are kept in a separate mask and are
not promoted to high-confidence labels.

Gold performance by language is reported only when a language has at least
eight gold studies (English and Spanish in the v1 scientific run).

Only verified clinical-standard or ontology-backed terms are added. Ligament
and meniscus targets now require pathology morphology (tear/rupture/sprain);
anatomy mention alone is unknown.

## 4. Target-specific confidence policies

Each target has **separate** positive and negative thresholds.

- Positive weak labels favour **precision** (default floor 0.85).
- Negative weak labels favour **NPV** (default floor 0.85).
- Ambiguous reports stay unknown.
- Thresholds are **not** lowered globally to raise coverage.
- Risk-coverage curves are generated so a later operator can pick a point
  explicitly.
- Thresholds used to **score** a gold study are fit without that study
  (5-fold OOF). Operational thresholds fit on all gold may be applied only to
  the 4,349 report-only studies.
- v1 metrics remain available for comparison.

## 5. Optional offline second reader

See `reports/competition_pretrained_model_rules.md`. Public pretrained models
are allowed if they are freely accessible, of minimal cost, and attached
locally (internet is off at scoring). This revision **documents** that and
ships the conservative combiner. It **does not** download or load weights.
`second_reader.enabled` is false.

Combiner:

| Situation | Label |
| --- | --- |
| agreement | eligible for higher-confidence |
| disagreement | unknown |
| unsupported language | unknown |
| no finding mentioned | unknown unless a verified global-normal statement supports a negative |

## 6. Trainable masks (private)

Written only under gitignored / Kaggle working private directories:

- `mask_high_conf_positive.csv`
- `mask_high_conf_negative.csv`
- `mask_unknown.csv`
- `mask_disagreement.csv`
- `mask_unsupported_language.csv`

Unknown and disagreement are **never** converted to 0.

## 7. Acceptance statuses

Each target receives exactly one of:

- `TRAINABLE_HIGH_CONFIDENCE`
- `TRAINABLE_WITH_RESTRICTIONS`
- `NOT_TRAINABLE`

The decision uses precision, NPV, coverage, confidence intervals and
prevalence plausibility. There is no single macro threshold. Inflated
weak-label prevalence (meniscus, OA, synovitis, Baker's in v1) blocks
high-confidence **positive** use.

## 8. Kaggle vs Cursor

Cursor runs **fixture** tests only. The real 58-study evaluation and the
4,349-study weak labels stay on the private internet-off notebook
`notebooks/03_kaggle_report_labeler_v2.ipynb`. Retrieve sanitized aggregates
only. Do not download private CSVs into git.

## 9. Out of scope

- image-model training
- image folds
- leaderboard submission
- downloading second-reader weights
- quoting reports
