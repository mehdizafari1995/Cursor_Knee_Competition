# Report labeling summary

Scientific evaluation on the expert-labelled training subset from the
real Kaggle mount. Report text, StudyInstanceUID values, and PatientID
values are not included.

- extractor_version: `report-labeler-v1.0.0`
- eval_protocol_version: `gold-eval-v1.0.0`
- config_sha256: `54dfea17063519fa33a58ff82fdfd49aa1d5bb4720b017aa65a40055f2150976`
- data_source: `real_kaggle_mount`
- scientific_evaluation: `true`
- unknown_imputed_as_negative: `false`
- external_language_model: `false`
- internet_used: `false`
- gpu_used: `false`
- training_run: `false`

## Language audit (aggregate only)

- n_reports: 4407
- n_empty: 0
- n_uncertain_language: 880
- n_unsupported_language: 17
- n_mixed_scripts: 607

| language | n | n_uncertain |
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

Scripts observed: `{"cyrillic": 220, "greek": 321, "latin": 4314, "other_non_ascii": 419, "thai": 3}`
Quality flags: `{"boilerplate": 1, "empty_section": 5, "length_short": 206, "no_section_headers": 2267, "non_ascii": 2613}`

Language codes are heuristic (Unicode script + function-word and section cues). Uncertain classifications are counted separately. Report text is not exported.

## Gold-label evaluation

- n_gold_studies: 58
- bootstrap resamples: 1000
- CI level: 0.95

| Official column | TP | TN | FP | FN | coverage | sens | spec | prec | f1 | bal acc | sens CI | spec CI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| ACL | 17 | 10 | 4 | 0 | 0.534 | 1.000 | 0.714 | 0.810 | 0.895 | 0.857 | 1.000–1.000 | 0.444–0.929 |
| MCL | 6 | 12 | 4 | 0 | 0.379 | 1.000 | 0.750 | 0.600 | 0.750 | 0.875 | 1.000–1.000 | 0.529–0.938 |
| Medial Meniscus | 11 | 3 | 9 | 0 | 0.397 | 1.000 | 0.250 | 0.550 | 0.710 | 0.625 | 1.000–1.000 | 0.000–0.500 |
| Lateral Meniscus | 14 | 4 | 6 | 0 | 0.414 | 1.000 | 0.400 | 0.700 | 0.824 | 0.700 | 1.000–1.000 | 0.111–0.734 |
| Medial OA | 4 | 2 | 1 | 0 | 0.121 | 1.000 | 0.667 | 0.800 | 0.889 | 0.833 | 1.000–1.000 | 0.000–1.000 |
| Lateral OA | 2 | 1 | 4 | 0 | 0.121 | 1.000 | 0.200 | 0.333 | 0.500 | 0.600 | 1.000–1.000 | 0.000–0.667 |
| PF OA | 3 | 0 | 4 | 0 | 0.121 | 1.000 | 0.000 | 0.429 | 0.600 | 0.500 | 1.000–1.000 | 0.000–0.000 |
| Effusion | 16 | 5 | 11 | 1 | 0.569 | 0.941 | 0.312 | 0.593 | 0.727 | 0.627 | 0.800–1.000 | 0.077–0.562 |
| Synovitis | 8 | 0 | 4 | 0 | 0.207 | 1.000 | 0.000 | 0.667 | 0.800 | 0.500 | 1.000–1.000 | 0.000–0.000 |
| Baker's | 6 | 2 | 5 | 0 | 0.224 | 1.000 | 0.286 | 0.545 | 0.706 | 0.643 | 1.000–1.000 | 0.000–0.667 |
| Contusion | 7 | 4 | 3 | 1 | 0.259 | 0.875 | 0.571 | 0.700 | 0.778 | 0.723 | 0.600–1.000 | 0.167–1.000 |
| Fracture | 8 | 3 | 3 | 0 | 0.241 | 1.000 | 0.500 | 0.727 | 0.842 | 0.750 | 1.000–1.000 | 0.000–1.000 |

Macro (unweighted mean over targets with defined metrics):

- sensitivity: 0.985
- specificity: 0.388
- precision: 0.621
- recall: 0.985
- f1: 0.752
- balanced_accuracy: 0.686
- coverage: 0.299

### Small-n targets (MCL, Lateral OA, Baker's)

- **Baker's**: gold positives=12, coverage=0.224, sens=1.000, spec=0.286, F1=0.706. n is small; intervals are wide.
- **Lateral OA**: gold positives=11, coverage=0.121, sens=1.000, spec=0.200, F1=0.500. n is small; intervals are wide.
- **MCL**: gold positives=9, coverage=0.379, sens=1.000, spec=0.750, F1=0.750. n is small; intervals are wide.

Unknown extractor outputs are abstentions and are excluded from TP/TN/FP/FN. Unknown gold labels are never treated as negative.

### Language-level gold performance

Reported only when a language has at least 8 gold studies. Other
languages are listed as underpowered. Counts are study counts, not
report quotations.

| language | n_gold | reported |
| --- | ---: | --- |
| en | 28 | yes |
| es | 10 | yes |
| tr | 6 | no (n<8) |
| bg | 3 | no (n<8) |
| el | 3 | no (n<8) |
| fr | 3 | no (n<8) |
| de | 2 | no (n<8) |
| nl | 2 | no (n<8) |
| sv | 1 | no (n<8) |

#### Language `en` (n=28)

| Official column | TP | TN | FP | FN | coverage | sens | spec | prec | f1 | bal acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ACL | 11 | 9 | 4 | 0 | 0.857 | 1.000 | 0.692 | 0.733 | 0.846 | 0.846 |
| MCL | 5 | 11 | 4 | 0 | 0.714 | 1.000 | 0.733 | 0.556 | 0.714 | 0.867 |
| Medial Meniscus | 10 | 3 | 3 | 0 | 0.571 | 1.000 | 0.500 | 0.769 | 0.870 | 0.750 |
| Lateral Meniscus | 9 | 4 | 4 | 0 | 0.607 | 1.000 | 0.500 | 0.692 | 0.818 | 0.750 |
| Medial OA | 4 | 2 | 1 | 0 | 0.250 | 1.000 | 0.667 | 0.800 | 0.889 | 0.833 |
| Lateral OA | 2 | 1 | 4 | 0 | 0.250 | 1.000 | 0.200 | 0.333 | 0.500 | 0.600 |
| PF OA | 3 | 0 | 4 | 0 | 0.250 | 1.000 | 0.000 | 0.429 | 0.600 | 0.500 |
| Effusion | 10 | 5 | 6 | 0 | 0.750 | 1.000 | 0.455 | 0.625 | 0.769 | 0.727 |
| Synovitis | 4 | 0 | 3 | 0 | 0.250 | 1.000 | 0.000 | 0.571 | 0.727 | 0.500 |
| Baker's | 3 | 2 | 2 | 0 | 0.250 | 1.000 | 0.500 | 0.600 | 0.750 | 0.750 |
| Contusion | 6 | 4 | 2 | 1 | 0.464 | 0.857 | 0.667 | 0.750 | 0.800 | 0.762 |
| Fracture | 6 | 3 | 1 | 0 | 0.357 | 1.000 | 0.750 | 0.857 | 0.923 | 0.875 |

#### Language `es` (n=10)

| Official column | TP | TN | FP | FN | coverage | sens | spec | prec | f1 | bal acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ACL | 1 | 0 | 0 | 0 | 0.100 | 1.000 | n/a | 1.000 | 1.000 | n/a |
| MCL | 1 | 0 | 0 | 0 | 0.100 | 1.000 | n/a | 1.000 | 1.000 | n/a |
| Medial Meniscus | 0 | 0 | 6 | 0 | 0.600 | n/a | 0.000 | 0.000 | n/a | n/a |
| Lateral Meniscus | 4 | 0 | 2 | 0 | 0.600 | 1.000 | 0.000 | 0.667 | 0.800 | 0.500 |
| Medial OA | 0 | 0 | 0 | 0 | 0.000 | n/a | n/a | n/a | n/a | n/a |
| Lateral OA | 0 | 0 | 0 | 0 | 0.000 | n/a | n/a | n/a | n/a | n/a |
| PF OA | 0 | 0 | 0 | 0 | 0.000 | n/a | n/a | n/a | n/a | n/a |
| Effusion | 3 | 0 | 3 | 0 | 0.600 | 1.000 | 0.000 | 0.500 | 0.667 | 0.500 |
| Synovitis | 2 | 0 | 1 | 0 | 0.300 | 1.000 | 0.000 | 0.667 | 0.800 | 0.500 |
| Baker's | 0 | 0 | 1 | 0 | 0.100 | n/a | 0.000 | 0.000 | n/a | n/a |
| Contusion | 0 | 0 | 0 | 0 | 0.000 | n/a | n/a | n/a | n/a | n/a |
| Fracture | 0 | 0 | 0 | 0 | 0.000 | n/a | n/a | n/a | n/a | n/a |

### Aggregate error categories

Counts only. Categories never include report quotations.

| category | n |
| --- | ---: |
| abstain_on_negative:history_only | 1 |
| abstain_on_negative:insufficient_evidence | 349 |
| abstain_on_negative:severity_below_threshold | 1 |
| abstain_on_negative:uncertain_language | 1 |
| abstain_on_positive:anatomy_unspecified | 2 |
| abstain_on_positive:history_only | 3 |
| abstain_on_positive:insufficient_evidence | 129 |
| abstain_on_positive:severity_below_threshold | 2 |
| fn:negated_finding | 2 |
| fp:explicit_positive | 58 |
| tn | 46 |
| tp | 102 |

## Weak labels (report-only studies)

- n_report_only_studies: 4349

| Official column | n_known | n_positive | n_negative | n_unknown | unknown rate | pos rate among known |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ACL | 1664 | 513 | 1151 | 2685 | 0.617 | 0.308 |
| MCL | 1240 | 266 | 974 | 3109 | 0.715 | 0.215 |
| Medial Meniscus | 1432 | 1008 | 424 | 2917 | 0.671 | 0.704 |
| Lateral Meniscus | 1464 | 675 | 789 | 2885 | 0.663 | 0.461 |
| Medial OA | 508 | 357 | 151 | 3841 | 0.883 | 0.703 |
| Lateral OA | 440 | 296 | 144 | 3909 | 0.899 | 0.673 |
| PF OA | 431 | 343 | 88 | 3918 | 0.901 | 0.796 |
| Effusion | 1902 | 1178 | 724 | 2447 | 0.563 | 0.619 |
| Synovitis | 493 | 399 | 94 | 3856 | 0.887 | 0.809 |
| Baker's | 1034 | 672 | 362 | 3315 | 0.762 | 0.650 |
| Contusion | 734 | 210 | 524 | 3615 | 0.831 | 0.286 |
| Fracture | 599 | 134 | 465 | 3750 | 0.862 | 0.224 |

Confidence distribution (mean / p50):

- ACL: mean=0.327, p50=0.000
- MCL: mean=0.249, p50=0.000
- Medial Meniscus: mean=0.292, p50=0.000
- Lateral Meniscus: mean=0.299, p50=0.000
- Medial OA: mean=0.091, p50=0.000
- Lateral OA: mean=0.080, p50=0.000
- PF OA: mean=0.075, p50=0.000
- Effusion: mean=0.409, p50=0.000
- Synovitis: mean=0.083, p50=0.000
- Baker's: mean=0.198, p50=0.000
- Contusion: mean=0.140, p50=0.000
- Fracture: mean=0.116, p50=0.000

## Expert vs weak-label prevalence

| Official column | expert pos rate (n=gold known) | weak pos rate (among known weak) |
| --- | ---: | ---: |
| ACL | 0.414 | 0.308 |
| MCL | 0.155 | 0.215 |
| Medial Meniscus | 0.448 | 0.704 |
| Lateral Meniscus | 0.397 | 0.461 |
| Medial OA | 0.259 | 0.703 |
| Lateral OA | 0.190 | 0.673 |
| PF OA | 0.362 | 0.796 |
| Effusion | 0.603 | 0.619 |
| Synovitis | 0.466 | 0.809 |
| Baker's | 0.207 | 0.650 |
| Contusion | 0.328 | 0.286 |
| Fracture | 0.310 | 0.224 |

## Warnings

- 17 reports had unsupported/undetermined language codes; those studies abstain except language-agnostic abbreviations.
- 880 reports had uncertain language classification.
- Baker's has only 12 gold positives; intervals are wide and performance is unstable.
- Lateral OA has only 11 gold positives; intervals are wide and performance is unstable.
- MCL has only 9 gold positives; intervals are wide and performance is unstable.
- Baker's abstained on 78% of gold studies (high uncertainty).
- Contusion abstained on 74% of gold studies (high uncertainty).
- Fracture abstained on 76% of gold studies (high uncertainty).
- Lateral Meniscus abstained on 59% of gold studies (high uncertainty).
- Lateral OA abstained on 88% of gold studies (high uncertainty).
- MCL abstained on 62% of gold studies (high uncertainty).
- Medial Meniscus abstained on 60% of gold studies (high uncertainty).
- Medial OA abstained on 88% of gold studies (high uncertainty).
- PF OA abstained on 88% of gold studies (high uncertainty).
- Synovitis abstained on 79% of gold studies (high uncertainty).

## Private output hashes (contents not committed)

| file | sha256 | bytes |
| --- | --- | ---: |
| `gold_predictions.csv` | `0fca30d35dff4de90d0b99b930f135a590da5c52f3feed98a7bf24cd801dc5c5` | 63438 |
| `language_assignments.csv` | `57c71e90c1c819094ec67094c3ccee991f44bd8ef2a164dbdbe70a4e36cea484` | 520091 |
| `weak_labels.csv` | `d11941721564c7df5d49a7dca978d79eff17feae4d68abdbd0f3fb1692d7834b` | 819372 |

Row-level weak labels, gold predictions, language assignments, and report
text remain in gitignored / Kaggle working outputs only.
