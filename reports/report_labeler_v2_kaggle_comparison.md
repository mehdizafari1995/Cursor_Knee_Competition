# Report-labeler v1 versus v2 — real Kaggle comparison

Sanitized aggregate comparison of the deterministic version-2 extractor and
confidence system against the frozen version-1 scientific run. Report text,
PatientID, StudyInstanceUID, and SeriesInstanceUID are not included.

- Notebook: https://www.kaggle.com/code/mahdimzafari/rsna-knee-report-labeler-v2
- Completed kernel version: **1** (`KernelWorkerStatus.COMPLETE`)
- Visibility / internet / GPU: private / off / off
- Attachments: `rsna-knee-abnormality-detection` + `mahdimzafari/cursor-knee-source`
- DATA_SOURCE: `real_kaggle_mount`
- scientific_evaluation: `true`
- n_reports: **4407** (n_train_studies=4407 = 58 gold + 4349 report-only)
- extractor_version: `report-labeler-v2.0.0`
- eval_protocol_version: `gold-eval-v2.0.0`
- config_sha256: `656f3a5f3431bb554a7eb41625f2748572b06f35ba8bbd5a0d18e73d1fa7fdf8`
- source bundle SHA-256: `288f84471622fc58bbebd9de358e7926e05a8f660748ff6d77f708ad84507280`
- second_reader.enabled: `false`
- second_reader_weights_downloaded: `false`
- unknown_imputed_as_negative: `false`
- training_run / image_folds_created: `false` / `false`

v1 baseline is the internet-off kernel
`mahdimzafari/rsna-knee-report-labeling` version 2 (`report-labeler-v1.0.0`,
`gold-eval-v1.0.0`, config_sha256 `54dfea17063519fa33a58ff82fdfd49aa1d5bb4720b017aa65a40055f2150976`).
v1 did not use out-of-fold confidence gates.

## Recommendation: merge Phase 4B, do not train on these as hard labels

**Merge the Phase 4B PR** into the Phase 4 parent branch. The v2 extractor,
coverage-aware metrics, offline language ensemble, OOF confidence policies,
and private masks are the evaluation protocol this repository should keep.

**Do not** start image-model training from these weak labels as hard targets.
No target is `TRAINABLE_HIGH_CONFIDENCE` on genuine out-of-fold gated metrics.
v1 statuses are not retained. Several extractor-only notebook statuses were
optimistic relative to OOF (especially Lateral OA).

## Two evaluation layers

1. **Extractor comparison (v1 vs v2).** Ternary calls emitted by each
   extractor on the same 58 gold studies. This is the fair v1 versus v2
   table. v1 had no extra confidence gate.
2. **OOF confidence system (v2 only).** Each gold study is gated with
   positive/negative thresholds fit on the other folds. Full-gold operational
   thresholds may be applied only to the 4349 report-only studies.

The Kaggle notebook summary table is layer 1. Trainable statuses in this
report are layer 2. Conditional sensitivity is **not** full-set recall.

## Out-of-fold validity

Sanitized fold records omit study identifiers. Every fold has
`n_train + n_test = 58` and no train/test overlap can be present in the
sanitized payload (`train_ids` / `test_ids` are stripped).

A target is **invalid as a gold threshold-performance claim** if its
full-gold (`leaked_full_fit`) metrics are used in place of `oof_metrics`.
Those two differ for:

| Official column | OOF coverage | leaked coverage | verdict |
| --- | ---: | ---: | --- |
| ACL | 0.259 | 0.431 | **invalid** if leaked/extractor gated numbers are treated as OOF |
| Medial Meniscus | 0.172 | 0.241 | **invalid** if leaked is treated as OOF |
| Lateral Meniscus | 0.172 | 0.086 | **invalid** if leaked is treated as OOF |
| Lateral OA | 0.000 | 0.017 | **invalid** if leaked is treated as OOF |
| Baker's | 0.052 | 0.034 | **invalid** if leaked is treated as OOF |
| Contusion | 0.086 | 0.000 | **invalid** if leaked is treated as OOF |

MCL, Medial OA, PF OA, Effusion, Synovitis, and Fracture have OOF equal to
leaked (the held-out studies did not change the selected threshold). Those
OOF numbers remain valid because each study was still scored with a policy
fit without it.

ACL is the clearest leakage example: the operational positive threshold 0.7
matches the extractor and leaked tables (coverage 0.431, precision 0.875,
positive capture 0.583). Two OOF folds selected 1.01 (no positive calls).
Valid OOF ACL coverage is 0.259 and precision is 0.667.

## v1 versus v2 extractor metric table (n=58 gold)

Coverage, precision, NPV, selective error among calls, positive capture over
all gold positives, negative capture over all gold negatives, abstained
positives/negatives. v2 extractor CIs are 95% study-level bootstrap (1000
resamples). v1 capture CIs cannot be rebuilt from cell counts.

| Official | v | coverage | prec | NPV | sel_err | pos_cap | neg_cap | abst_pos | abst_neg | TP | FP | TN | FN |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ACL | v1 | 0.534 | 0.810 | 1.000 | 0.129 | 0.708 | 0.294 | 7 | 20 | 17 | 4 | 10 | 0 |
| ACL | v2 | 0.431 | 0.875 | 1.000 | 0.080 | 0.583 | 0.265 | 10 | 23 | 14 | 2 | 9 | 0 |
| MCL | v1 | 0.379 | 0.600 | 1.000 | 0.182 | 0.667 | 0.245 | 3 | 33 | 6 | 4 | 12 | 0 |
| MCL | v2 | 0.241 | 0.500 | 1.000 | 0.214 | 0.333 | 0.163 | 6 | 38 | 3 | 3 | 8 | 0 |
| Medial Meniscus | v1 | 0.397 | 0.550 | 1.000 | 0.391 | 0.423 | 0.094 | 15 | 20 | 11 | 9 | 3 | 0 |
| Medial Meniscus | v2 | 0.276 | 0.818 | 1.000 | 0.125 | 0.346 | 0.156 | 17 | 25 | 9 | 2 | 5 | 0 |
| Lateral Meniscus | v1 | 0.414 | 0.700 | 1.000 | 0.250 | 0.609 | 0.114 | 9 | 25 | 14 | 6 | 4 | 0 |
| Lateral Meniscus | v2 | 0.276 | 0.818 | 1.000 | 0.125 | 0.391 | 0.143 | 14 | 28 | 9 | 2 | 5 | 0 |
| Medial OA | v1 | 0.121 | 0.800 | 1.000 | 0.143 | 0.267 | 0.047 | 11 | 40 | 4 | 1 | 2 | 0 |
| Medial OA | v2 | 0.103 | 1.000 | 1.000 | 0.000 | 0.267 | 0.047 | 11 | 41 | 4 | 0 | 2 | 0 |
| Lateral OA | v1 | 0.121 | 0.333 | 1.000 | 0.571 | 0.182 | 0.021 | 9 | 42 | 2 | 4 | 1 | 0 |
| Lateral OA | v2 | 0.103 | 0.400 | 1.000 | 0.500 | 0.182 | 0.021 | 9 | 43 | 2 | 3 | 1 | 0 |
| PF OA | v1 | 0.121 | 0.429 | n/a | 0.571 | 0.143 | 0.000 | 18 | 33 | 3 | 4 | 0 | 0 |
| PF OA | v2 | 0.121 | 0.429 | n/a | 0.571 | 0.143 | 0.000 | 18 | 33 | 3 | 4 | 0 | 0 |
| Effusion | v1 | 0.569 | 0.593 | 0.833 | 0.364 | 0.457 | 0.217 | 18 | 7 | 16 | 11 | 5 | 1 |
| Effusion | v2 | 0.569 | 0.607 | 1.000 | 0.333 | 0.486 | 0.217 | 18 | 7 | 17 | 11 | 5 | 0 |
| Synovitis | v1 | 0.207 | 0.667 | n/a | 0.333 | 0.296 | 0.000 | 19 | 27 | 8 | 4 | 0 | 0 |
| Synovitis | v2 | 0.207 | 0.667 | n/a | 0.333 | 0.296 | 0.000 | 19 | 27 | 8 | 4 | 0 | 0 |
| Baker's | v1 | 0.224 | 0.545 | 1.000 | 0.385 | 0.500 | 0.043 | 6 | 39 | 6 | 5 | 2 | 0 |
| Baker's | v2 | 0.224 | 0.545 | 1.000 | 0.385 | 0.500 | 0.043 | 6 | 39 | 6 | 5 | 2 | 0 |
| Contusion | v1 | 0.259 | 0.700 | 0.800 | 0.267 | 0.368 | 0.103 | 11 | 32 | 7 | 3 | 4 | 1 |
| Contusion | v2 | 0.259 | 0.700 | 0.800 | 0.267 | 0.368 | 0.103 | 11 | 32 | 7 | 3 | 4 | 1 |
| Fracture | v1 | 0.241 | 0.727 | 1.000 | 0.214 | 0.444 | 0.075 | 10 | 34 | 8 | 3 | 3 | 0 |
| Fracture | v2 | 0.241 | 0.727 | 1.000 | 0.214 | 0.444 | 0.075 | 10 | 34 | 8 | 3 | 3 | 0 |

v2 extractor 95% CIs (coverage / precision / NPV / pos_capture / neg_capture):

| Official | coverage | precision | NPV | pos_capture | neg_capture |
| --- | --- | --- | --- | --- | --- |
| ACL | 0.310–0.552 | 0.700–1.000 | 1.000–1.000 | 0.389–0.769 | 0.118–0.419 |
| MCL | 0.138–0.362 | 0.000–1.000 | 1.000–1.000 | 0.000–0.667 | 0.064–0.273 |
| Medial Meniscus | 0.172–0.397 | 0.545–1.000 | 1.000–1.000 | 0.160–0.536 | 0.042–0.286 |
| Lateral Meniscus | 0.172–0.397 | 0.556–1.000 | 1.000–1.000 | 0.200–0.611 | 0.032–0.273 |
| Medial OA | 0.034–0.190 | 1.000–1.000 | 1.000–1.000 | 0.062–0.500 | 0.000–0.116 |
| Lateral OA | 0.034–0.190 | 0.000–1.000 | 1.000–1.000 | 0.000–0.438 | 0.000–0.065 |
| PF OA | 0.052–0.207 | 0.000–0.833 | n/a | 0.000–0.312 | 0.000–0.000 |
| Effusion | 0.431–0.690 | 0.417–0.778 | 1.000–1.000 | 0.314–0.656 | 0.048–0.409 |
| Synovitis | 0.103–0.310 | 0.400–0.923 | n/a | 0.130–0.467 | 0.000–0.000 |
| Baker's | 0.121–0.328 | 0.250–0.846 | 1.000–1.000 | 0.200–0.786 | 0.000–0.109 |
| Contusion | 0.155–0.379 | 0.375–1.000 | 0.333–1.000 | 0.154–0.579 | 0.023–0.200 |
| Fracture | 0.138–0.345 | 0.428–1.000 | 1.000–1.000 | 0.214–0.688 | 0.000–0.167 |

Macro extractor (unweighted mean over defined targets):

| | v1 | v2 extractor |
| --- | ---: | ---: |
| coverage | 0.299 | 0.254 |
| positive_capture | 0.422 | 0.362 |
| negative_capture | 0.104 | 0.103 |
| error_rate_all_gold | 0.086 | 0.062 |
| selective_accuracy | 0.683 | 0.738 |
| precision | 0.621 | 0.674 |
| npv | 0.963 | 0.980 |
| conditional_sensitivity (called only) | 0.985 | 0.990 |
| conditional_specificity (called only) | 0.388 | 0.491 |

v2 is more conservative (lower coverage and capture) and more precise,
especially on meniscus (fewer degeneration-as-injury false positives).
ACL, MCL, and meniscus coverage fell; OA / synovitis / Baker / contusion /
fracture extractor tables are largely unchanged.

## v2 OOF gated metrics (valid threshold evaluation)

| Official | coverage | prec | NPV | sel_err | pos_cap | neg_cap | abst_pos | abst_neg | TP | FP | TN | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ACL | 0.259 | 0.667 | 1.000 | 0.133 | 0.167 | 0.265 | 20 | 23 | 4 | 2 | 9 | 0 |
| MCL | 0.138 | n/a | 1.000 | 0.000 | 0.000 | 0.163 | 9 | 41 | 0 | 0 | 8 | 0 |
| Medial Meniscus | 0.172 | 0.800 | 1.000 | 0.100 | 0.154 | 0.156 | 22 | 26 | 4 | 1 | 5 | 0 |
| Lateral Meniscus | 0.172 | 0.600 | 1.000 | 0.200 | 0.130 | 0.143 | 20 | 28 | 3 | 2 | 5 | 0 |
| Medial OA | 0.103 | 1.000 | 1.000 | 0.000 | 0.267 | 0.047 | 11 | 41 | 4 | 0 | 2 | 0 |
| Lateral OA | 0.000 | n/a | n/a | n/a | 0.000 | 0.000 | 11 | 47 | 0 | 0 | 0 | 0 |
| PF OA | 0.000 | n/a | n/a | n/a | 0.000 | 0.000 | 21 | 37 | 0 | 0 | 0 | 0 |
| Effusion | 0.086 | n/a | 1.000 | 0.000 | 0.000 | 0.217 | 35 | 18 | 0 | 0 | 5 | 0 |
| Synovitis | 0.000 | n/a | n/a | n/a | 0.000 | 0.000 | 27 | 31 | 0 | 0 | 0 | 0 |
| Baker's | 0.052 | 0.000 | 1.000 | 0.333 | 0.000 | 0.043 | 12 | 43 | 0 | 1 | 2 | 0 |
| Contusion | 0.086 | 0.333 | 0.500 | 0.600 | 0.053 | 0.026 | 17 | 36 | 1 | 2 | 1 | 1 |
| Fracture | 0.052 | n/a | 1.000 | 0.000 | 0.000 | 0.075 | 18 | 37 | 0 | 0 | 3 | 0 |

OOF-gated labels do not have a separate bootstrap; extractor CIs above must
not be quoted as OOF intervals. Missing OOF CIs block
`TRAINABLE_HIGH_CONFIDENCE`.

## Coverage–error tradeoff

Positive-polarity confidence curves (extractor scores, 58 gold studies):

- **ACL:** from threshold 0.35 through 0.90, coverage stays 0.276 of gold
  studies as positive calls, precision 0.875, selective error 0.125,
  positive capture 0.583. At 0.95 every positive call abstains
  (coverage 0, error 0). Raising the gate does not remove the two false
  positives until coverage collapses.
- **Effusion:** the highest-coverage target. Precision stays about 0.61
  across the usable gate; OOF therefore selects positive_threshold 1.01
  (no high-conf positives) and keeps negatives at 0.85 (NPV 1.0 on 5 TN).
- **Medial OA:** the only OA target whose positive calls already meet
  precision 1.0 at the extractor gate; the curve is flat at coverage 0.103
  until higher thresholds drop the four true positives.
- **PF OA / Synovitis / Lateral OA:** low coverage and poor precision at
  every usable threshold; OOF selects 1.01/1.01 and coverage goes to 0.

Joint (pos+neg) ACL curve at the extractor operating point: coverage 0.431,
all-gold error 0.034, selective error 0.080. That point is **not** an OOF
threshold result (see ACL leaked vs OOF above).

## Per-target decision and restrictions (OOF, independent of v1)

Source: `oof_metrics` plus weak-label prevalence among the 4349 report-only
studies. v1 status is shown only for contrast and is not inherited.

| Official | v1 status (extractor counts) | v2 OOF status | restrictions |
| --- | --- | --- | --- |
| ACL | TRAINABLE_WITH_RESTRICTIONS | TRAINABLE_WITH_RESTRICTIONS | high-conf **negatives** only; OOF positives fail precision 0.667; never impute unknown/disagreement as 0 |
| MCL | TRAINABLE_WITH_RESTRICTIONS | TRAINABLE_WITH_RESTRICTIONS | small_n (9 gold positives); OOF positives abstain; high-conf **negatives** only |
| Medial Meniscus | TRAINABLE_WITH_RESTRICTIONS | TRAINABLE_WITH_RESTRICTIONS | inflated weak prevalence; positives not approved as hard labels; high-conf **negatives** allowed (NPV 1.0, OOF prec 0.800 with inflated prevalence) |
| Lateral Meniscus | TRAINABLE_WITH_RESTRICTIONS | TRAINABLE_WITH_RESTRICTIONS | inflated-prevalence list; OOF prec 0.600 so positives not approved; high-conf **negatives** only |
| Medial OA | TRAINABLE_WITH_RESTRICTIONS | TRAINABLE_WITH_RESTRICTIONS | weak/expert prevalence ratio 2.64; do not use positive calls as hard labels; high-conf **negatives** allowed |
| Lateral OA | TRAINABLE_WITH_RESTRICTIONS | **NOT_TRAINABLE** | OOF coverage 0; small_n; inflated prevalence. v1 restricted status is **not** retained |
| PF OA | NOT_TRAINABLE | NOT_TRAINABLE | OOF coverage 0; inflated prevalence |
| Effusion | TRAINABLE_WITH_RESTRICTIONS | TRAINABLE_WITH_RESTRICTIONS | OOF positives abstain (prec undefined); high-conf **negatives** only (NPV 1.0) |
| Synovitis | NOT_TRAINABLE | NOT_TRAINABLE | OOF coverage 0; inflated prevalence |
| Baker's | TRAINABLE_WITH_RESTRICTIONS | TRAINABLE_WITH_RESTRICTIONS | small_n; inflated prevalence; OOF prec 0.000 so positives not approved; high-conf **negatives** only |
| Contusion | TRAINABLE_WITH_RESTRICTIONS | **NOT_TRAINABLE** | OOF prec 0.333 and NPV 0.500 fail both polarities. v1 restricted status is **not** retained |
| Fracture | TRAINABLE_WITH_RESTRICTIONS | TRAINABLE_WITH_RESTRICTIONS | OOF positives abstain; high-conf **negatives** only |

`TRAINABLE_HIGH_CONFIDENCE`: **none**.

Shared restrictions for every restricted target: use high-confidence masks
only; never convert unknown or disagreement to 0; do not treat conditional
sensitivity as full-set recall.

## Aggregate language results

Language codes combine the Unicode-script/function-word heuristic with an
independent offline character-trigram detector. Disagreement stays uncertain.
No translation or LLM API.

| language | v1 n | v1 n_uncertain | v2 n | v2 n_uncertain |
| --- | ---: | ---: | ---: | ---: |
| en | 1737 | 29 | 1737 | 1 |
| es | 682 | 0 | 682 | 1 |
| tr | 546 | 0 | 546 | 401 |
| el | 321 | 321 | 321 | 0 |
| fr | 292 | 154 | 292 | 212 |
| de | 261 | 3 | 261 | 1 |
| bg | 220 | 218 | 220 | 0 |
| nl | 152 | 0 | 152 | 1 |
| cs | 96 | 96 | 97 | 92 |
| sv | 42 | 1 | 42 | 42 |
| da | 41 | 41 | 41 | 41 |
| it | — | — | 15 | 15 |
| pt | — | — | 1 | 1 |
| und | 17 | 17 | 0 | 0 |

v2 ensemble: 3628 detector agreements, 779 disagreements, 808 uncertain kept
separate, 0 unsupported language. Greek (321) and Bulgarian (220) are no
longer forced uncertain. Turkish, Swedish, and Italian/Portuguese become
uncertain when the ngram detector disagrees; those studies abstain except
language-agnostic abbreviations.

Gold metrics by language are reported only for n≥8 gold studies:

- English n=28 (reported). Highest coverage on ACL (0.786, prec 0.846) and
  Effusion (0.750, prec 0.625). Medial meniscus English precision 0.900.
- Spanish n=10 (reported, still small). Sparse calls; Effusion coverage 0.600
  at precision 0.500; several targets have coverage 0.
- Turkish n=6, Greek n=3, Bulgarian n=3, French n=3, German n=2, Dutch n=2,
  Swedish n=1: underpowered, not reported as performance.

## Prevalence comparison (expert n=58 vs weak among known, n=4349)

| Official | expert pos rate | v1 weak pos among known | v2 weak pos among known | v2 n_known / n_unknown |
| --- | ---: | ---: | ---: | ---: | --- |
| ACL | 0.414 | 0.308 | 0.274 | 1289 / 3060 |
| MCL | 0.155 | 0.215 | 0.210 | 816 / 3533 |
| Medial Meniscus | 0.448 | 0.704 | 0.547 | 1081 / 3268 |
| Lateral Meniscus | 0.397 | 0.461 | 0.247 | 1151 / 3198 |
| Medial OA | 0.259 | 0.703 | 0.684 | 452 / 3897 |
| Lateral OA | 0.190 | 0.673 | 0.639 | 377 / 3972 |
| PF OA | 0.362 | 0.796 | 0.795 | 405 / 3944 |
| Effusion | 0.603 | 0.619 | 0.621 | 1901 / 2448 |
| Synovitis | 0.466 | 0.809 | 0.810 | 494 / 3855 |
| Baker's | 0.207 | 0.650 | 0.654 | 1038 / 3311 |
| Contusion | 0.328 | 0.286 | 0.292 | 734 / 3615 |
| Fracture | 0.310 | 0.224 | 0.238 | 601 / 3748 |

Pathology-morphology gates reduced meniscus weak prevalence (especially
lateral). OA, synovitis, and Baker's remain inflated versus expert rates.
Those positives stay unapproved as hard labels.

## Confidence-mask counts

Private Kaggle `mask_disagreement.csv` and `mask_unsupported_language.csv`
were downloaded into a gitignored probe directory, reduced to aggregates,
and deleted. Identifiers were not printed or committed.

Both files are **correctly all-zero**, so they are byte-for-byte identical
and share SHA-256
`88c075136991356891ea0735b61692d21c8f469c9ba766ca2e0a9ece3764cdac`
(387192 bytes). This is not a copy-paste of one mask onto the other.

| check | result |
| --- | --- |
| n_rows (report-only studies) | 4349 |
| n_target columns | 12 |
| true cells (`mask_disagreement.csv`) | 0 |
| true cells (`mask_unsupported_language.csv`) | 0 |
| studies with any true cell | 0 / 0 |
| targets with any true cell | 0 / 0 |
| unique mask values | `{0}` |
| target cells identical | true |
| study-id order aligned | true |
| raw bytes identical | true |

Independent generators in `knee/report_masks.py` / `run_notebook_report_labeling`:

- **Disagreement** is filled from an optional `disagreement_frame`. The
  production call does not pass that argument. The second reader is
  disabled (`second_reader.enabled=false`, weights not downloaded), so
  every disagreement cell stays 0. That is the intended empty disagreement
  mask for this run.
- **Unsupported language** is filled from `quality_flags` containing
  `unsupported_language`, which is set when `language.supported` is false.
  v2 `n_unsupported_language` is **0**, so every unsupported-language cell
  stays 0.

v1 counted 17 unsupported reports. Those were the 17 `und` (no-cue)
classifications, which are outside the supported-language list. v2's
independent ngram detector assigned those 17 a supported language code
(`ngram_only_kept_uncertain=17`; the language table gains Italian n=15,
Portuguese n=1, and Czech 96→97) and kept them **uncertain**, not
unsupported. Uncertain language is recorded in the unknown mask, not in
`mask_unsupported_language.csv`. v2 therefore has zero
unsupported-language reports.

High-confidence positive/negative and unknown masks have distinct hashes
and are not all-zero. Exact true-counts for those three files were not
re-downloaded in this probe.

Private Kaggle outputs (row-level, not in git):

| file | sha256 | bytes |
| --- | --- | ---: |
| `weak_labels.csv` | `54ac9e781ad66602bc9530424422243e814f6855545e6face9be9f464c5f4963` | 861712 |
| `gold_predictions.csv` | `a53150bb9077a1fdf8b270953a3d2a19dd1fd00861ce193aca0d44b3623c32cc` | 79807 |
| `language_assignments.csv` | `fb9a044b2b4887f287604adc943909dae7bfdf9f621a20025863a3dcbef091f7` | 568635 |
| `mask_high_conf_positive.csv` | `23dddb8510b9fee91bdb1ea21478e1cb559b910b0ddd6fb88d8a612798dcdad1` | 387192 |
| `mask_high_conf_negative.csv` | `012306e21cf866b22c2d8769b33559fda7765856d6d4e806b3fc454a27d4ac00` | 387192 |
| `mask_unknown.csv` | `0c23a1384e9fe190fd1535ec1693b4a003dfba44514fc95941d7c245c868c962` | 387192 |
| `mask_disagreement.csv` | `88c075136991356891ea0735b61692d21c8f469c9ba766ca2e0a9ece3764cdac` | 387192 |
| `mask_unsupported_language.csv` | `88c075136991356891ea0735b61692d21c8f469c9ba766ca2e0a9ece3764cdac` | 387192 |

`mask_disagreement.csv` and `mask_unsupported_language.csv` share a hash
because both are independently written all-zero matrices (second reader
off; v2 `n_unsupported_language=0`). Unknown/disagreement are never
written as 0.

## Error taxonomy (v2 gold, counts only)

Closed vocabulary. No quotations. Dominant abstentions:
`insufficient_evidence` 399, `bone_marrow_signal_versus_contusion` 42,
`other_call_error` 42, `section_boundary_failure` 34,
`unsupported_terminology` 31. One `negation_scope_failure` FN. Two
`generic_meniscal_degeneration_as_injury` abstentions on positives.

## Merge recommendation (explicit)

Merge Phase 4B. Keep the v2 extractor and gold-eval-v2 protocol. Do not
enable the second reader in this merge. Do not train an image model from
these labels as hard targets. Use high-confidence **negative** masks only
where the OOF status is `TRAINABLE_WITH_RESTRICTIONS`, and never for
Lateral OA, PF OA, Synovitis, or Contusion.
