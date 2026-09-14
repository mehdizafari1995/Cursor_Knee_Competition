# Report-labeler v2 summary

Coverage-aware evaluation of the Phase 4 extractor. Report text,
StudyInstanceUID values, and PatientID values are not included.

- extractor_version (frozen baseline): `report-labeler-v1.0.0`
- extractor_version (this protocol): `report-labeler-v2.0.0`
- eval_protocol_version: `gold-eval-v2.0.0`
- v1 eval_protocol_version: `gold-eval-v1.0.0`
- v1 config_sha256: `54dfea17063519fa33a58ff82fdfd49aa1d5bb4720b017aa65a40055f2150976`
- data_source of the scientific counts: `real_kaggle_mount` (v1 Kaggle run)
- scientific_evaluation of the counts below: `true` (re-analysis of committed v1 aggregates; no new report access)
- v2 extractor gold evaluation in this checkout: **not computed** (competition reports are private; run `notebooks/03_kaggle_report_labeler_v2.ipynb` internet-off)
- unknown_imputed_as_negative: `false`
- external_language_model: `false`
- second_reader_enabled: `false`
- second_reader_weights_downloaded: `false`
- internet_used: `false`
- gpu_used: `false`
- training_run: `false`
- image_folds_created: `false`

The current labels are **not** approved as hard training labels.

## Evaluation correction

The v1 scientific run on 58 gold studies reported:

- macro call coverage: 0.299
- **conditional** sensitivity among called cases: 0.985
- **conditional** specificity among called cases: 0.388
- **conditional** precision among called cases: 0.621
- full-gold positive-capture (TP / all gold positives), macro: 0.422

That 0.985 figure is **conditional sensitivity** (TP / (TP+FN) among calls).
It is not full-set sensitivity. Most gold positives were abstentions, not
true positives. Version-2 artefacts never describe conditional sensitivity as
full-set sensitivity.

## Corrected v1 gold metrics (n=58 completely labelled studies)

Counts rebuilt from the committed v1 table in
`reports/report_labeling_summary.md` via `knee/report_v1_aggregates.py`.
Bootstrap intervals for capture/NPV cannot be rebuilt from cell counts alone;
they require the private row-level gold predictions on Kaggle under
gold-eval-v2.0.0.

| Official column | n_pos | n_neg | pos_calls | neg_calls | abstain | TP | TN | FP | FN | abst_pos | abst_neg | pos_capture | neg_capture | err_all | sel_acc | prec | NPV | coverage | cond_sens | cond_spec | status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ACL | 24 | 34 | 21 | 10 | 27 | 17 | 10 | 4 | 0 | 7 | 20 | 0.708 | 0.294 | 0.069 | 0.871 | 0.810 | 1.000 | 0.534 | 1.000 | 0.714 | TRAINABLE_WITH_RESTRICTIONS |
| MCL | 9 | 49 | 10 | 12 | 36 | 6 | 12 | 4 | 0 | 3 | 33 | 0.667 | 0.245 | 0.069 | 0.818 | 0.600 | 1.000 | 0.379 | 1.000 | 0.750 | TRAINABLE_WITH_RESTRICTIONS |
| Medial Meniscus | 26 | 32 | 20 | 3 | 35 | 11 | 3 | 9 | 0 | 15 | 20 | 0.423 | 0.094 | 0.155 | 0.609 | 0.550 | 1.000 | 0.397 | 1.000 | 0.250 | TRAINABLE_WITH_RESTRICTIONS |
| Lateral Meniscus | 23 | 35 | 20 | 4 | 34 | 14 | 4 | 6 | 0 | 9 | 25 | 0.609 | 0.114 | 0.103 | 0.750 | 0.700 | 1.000 | 0.414 | 1.000 | 0.400 | TRAINABLE_WITH_RESTRICTIONS |
| Medial OA | 15 | 43 | 5 | 2 | 51 | 4 | 2 | 1 | 0 | 11 | 40 | 0.267 | 0.047 | 0.017 | 0.857 | 0.800 | 1.000 | 0.121 | 1.000 | 0.667 | TRAINABLE_WITH_RESTRICTIONS |
| Lateral OA | 11 | 47 | 6 | 1 | 51 | 2 | 1 | 4 | 0 | 9 | 42 | 0.182 | 0.021 | 0.069 | 0.429 | 0.333 | 1.000 | 0.121 | 1.000 | 0.200 | TRAINABLE_WITH_RESTRICTIONS |
| PF OA | 21 | 37 | 7 | 0 | 51 | 3 | 0 | 4 | 0 | 18 | 33 | 0.143 | 0.000 | 0.069 | 0.429 | 0.429 | n/a | 0.121 | 1.000 | 0.000 | NOT_TRAINABLE |
| Effusion | 35 | 23 | 27 | 6 | 25 | 16 | 5 | 11 | 1 | 18 | 7 | 0.457 | 0.217 | 0.207 | 0.636 | 0.593 | 0.833 | 0.569 | 0.941 | 0.312 | TRAINABLE_WITH_RESTRICTIONS |
| Synovitis | 27 | 31 | 12 | 0 | 46 | 8 | 0 | 4 | 0 | 19 | 27 | 0.296 | 0.000 | 0.069 | 0.667 | 0.667 | n/a | 0.207 | 1.000 | 0.000 | NOT_TRAINABLE |
| Baker's | 12 | 46 | 11 | 2 | 45 | 6 | 2 | 5 | 0 | 6 | 39 | 0.500 | 0.043 | 0.086 | 0.615 | 0.545 | 1.000 | 0.224 | 1.000 | 0.286 | TRAINABLE_WITH_RESTRICTIONS |
| Contusion | 19 | 39 | 10 | 5 | 43 | 7 | 4 | 3 | 1 | 11 | 32 | 0.368 | 0.103 | 0.069 | 0.733 | 0.700 | 0.800 | 0.259 | 0.875 | 0.571 | TRAINABLE_WITH_RESTRICTIONS |
| Fracture | 18 | 40 | 11 | 3 | 44 | 8 | 3 | 3 | 0 | 10 | 34 | 0.444 | 0.075 | 0.052 | 0.786 | 0.727 | 1.000 | 0.241 | 1.000 | 0.500 | TRAINABLE_WITH_RESTRICTIONS |

Macro (unweighted mean over defined targets):

- coverage: 0.299
- positive_capture: 0.422
- negative_capture: 0.104
- error_rate_all_gold: 0.086
- selective_accuracy: 0.683
- precision: 0.621
- npv: 0.963
- conditional_sensitivity: 0.985 (called cases only)
- conditional_specificity: 0.388 (called cases only)

No target is `TRAINABLE_HIGH_CONFIDENCE`. n = 58 is small; capture CIs are not
available from cell counts; meniscus / OA / synovitis / Baker's weak prevalence
is inflated versus expert rates. High-confidence **negatives** may be used
where NPV is defined and the unknown mask is respected. High-confidence
**positives** are not approved for the inflated-prevalence targets.

Restrictions that apply to every `TRAINABLE_WITH_RESTRICTIONS` target:

- use only the high-confidence masks
- never impute unknown or disagreement as zero
- do not treat conditional sensitivity as full-set recall
- do not use positive calls as hard labels when weak prevalence is inflated

## Language audit (v1 scientific run, to be re-validated on Kaggle)

| language | n | n_uncertain (v1 heuristic) |
| --- | ---: | ---: |
| en | 1737 | 29 |
| es | 682 | 0 |
| tr | 546 | 0 |
| el | 321 | 321 |
| fr | 292 | 154 |
| de | 261 | 3 |
| bg | 220 | 218 |
| nl | 152 | 0 |
| cs | 96 | 96 |
| sv | 42 | 1 |
| da | 41 | 41 |
| und | 17 | 17 |

v1 marked all 321 Greek reports uncertain because Latin MRI abbreviations
triggered mixed-script uncertainty. v2 keeps an independent character-trigram
detector in-process, does not call a translation or LLM API, and will report
heuristic vs trigram agreement on Kaggle. Disagreement stays uncertain.
Gold metrics by language remain English (n=28) and Spanish (n=10) only;
Turkish (n=6) and Greek (n=3) gold subsets are underpowered.

## Error categories (v1 aggregates, no quotations)

v1 recorded 58 `fp:explicit_positive`, 2 `fn:negated_finding`, and 129
`abstain_on_positive:insufficient_evidence`. v2 maps those into the closed
taxonomy (negation-scope, generic meniscal degeneration, compartment-unspecified
OA, effusion vs synovitis, Baker terminology, bone-marrow vs contusion,
fracture exclusion, language-detection failure, section-boundary, and others)
without storing report text.

## Second reader

Competition rules allow freely and publicly available pretrained models of
minimal cost, attached locally, because scoring notebooks disable internet.
See `reports/competition_pretrained_model_rules.md`. The combiner is
implemented; weights are **not** downloaded in this revision.

## Private outputs (gitignored; produced on Kaggle)

- version-2 gold predictions
- version-2 weak labels
- positive and negative confidence masks
- unknown / disagreement / unsupported-language masks
- language assignments (heuristic + trigram + agreement)
- SHA-256 hashes and extractor version

## Cursor software verification

Fixture tests in this repository exercise abstention-aware metrics, OOF
threshold leakage controls, and privacy filters. They are not scientific
gold-label performance.

## Warnings

- Version-2 scientific gold metrics are pending the private Kaggle notebook.
- Do not train an image model or submit predictions in this phase.
- Do not convert unknown or disagreement to zero.
