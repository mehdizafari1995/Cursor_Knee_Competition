# Phase 4C.7 acceptance audit

Aggregate-only recompute from existing sanitized Phase 4C.7 gold
metrics. No model inference, no new Kaggle kernel, no private
row contents, no report-only processing, and no training. Frozen
kernel `mahdimzafari/rsna-knee-second-reader-gold-constrained-v14`
version 1, its hashes, and its scientific tables remain the original
record. This file is a reporting/implementation correction.

- status: `ACCEPTANCE_AUDIT_COMPLETE`
- primary system: `strict_agreement`
- n_gold_studies_processed: 58
- n_report_only_studies_processed: 0
- gold_eval_executed: `true`
- gold_rerun_executed: `true`
- gold_rerun_authorized: `false`
- v1.4 config SHA-256 unchanged: `992af7a4b4483278445a6f3fb363c818611e79005618dff063460d0aed3bd7f7`
- forbidden outcome not used: `CONSTRAINED_FORMAT_PASS_SEMANTIC_PASS`
- semantic_gate_status: `NOT_PREDECLARED`
- bootstrap: study-level percentile, n_resamples=1000, seed=42+target_index, ci_level=0.95
- tiny-denominator reporting max (not a YAML gate): 5
- code never maps uncertain / not_mentioned / unknown to negative

## Execution-flag correction

A run that processed 58 gold studies must record execution flags
true. `gold_rerun_authorized` remains false after the run.

- execution_flag_invariant_ok: `true`

## Medial Meniscus fail-closed CI proof

- stored_status: `PILOT_PASS`
- audited_status: `PILOT_PASS`
- precision=1.0000 CI low/high=1.0000/1.0000 (n_valid_resamples=1000)
- NPV=1.0000 CI low/high=1.0000/1.0000 (n_valid_resamples=873)
- precision_ci_finite_ge_0.70: `true`
- npv_ci_finite_ge_0.70: `true`
- still PILOT_PASS after fail-closed CI: `true`
- positive_decision_denominator=8; negative_decision_denominator=2; tiny_negative_denominator=`true` (reported, not a YAML pass gate)

## MCL tiny denominator

A single non-abstaining decision with point precision 1.0 is not
sufficient evidence. MCL cannot `PILOT_PASS` under small-n, and
the missing NPV CI is now a fail-closed non-finite CI.

- stored_status: `PILOT_RESTRICTED` audited_status: `PILOT_RESTRICTED`
- n_covered=1; positive_denom=1; negative_denom=0
- precision=1.000; NPV=n/a

## deterministic_v2

| Official | stored | audited | gold+/gold- | TP FP TN FN | abst | pos_denom | neg_denom | prec | prec_CI | NPV | NPV_CI | cov | pos_cap | neg_cap | sel_err | tiny | seed |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | --- | ---: |
| ACL | `—` | `PILOT_RESTRICTED` | 24/34 | 14 2 9 0 | 33 | 16 | 9 | 0.8750 | 0.6998/1.0000 | 1.0000 | 1.0000/1.0000 | 0.4310 | 0.5833 | 0.2647 | 0.0800 | — | 42 |
| MCL | `—` | `PILOT_RESTRICTED` | 9/49 | 3 3 8 0 | 44 | 6 | 8 | 0.5000 | 0.0000/1.0000 | 1.0000 | 1.0000/1.0000 | 0.2414 | 0.3333 | 0.1633 | 0.2143 | — | 43 |
| Medial Meniscus | `—` | `PILOT_RESTRICTED` | 26/32 | 9 2 5 0 | 42 | 11 | 5 | 0.8182 | 0.5455/1.0000 | 1.0000 | 1.0000/1.0000 | 0.2759 | 0.3462 | 0.1562 | 0.1250 | neg | 44 |
| Lateral Meniscus | `—` | `PILOT_RESTRICTED` | 23/35 | 9 2 5 0 | 42 | 11 | 5 | 0.8182 | 0.5556/1.0000 | 1.0000 | 1.0000/1.0000 | 0.2759 | 0.3913 | 0.1429 | 0.1250 | neg | 45 |
| Medial OA | `—` | `PILOT_PASS` | 15/43 | 4 0 2 0 | 52 | 4 | 2 | 1.0000 | 1.0000/1.0000 | 1.0000 | 1.0000/1.0000 | 0.1034 | 0.2667 | 0.0465 | 0.0000 | pos,neg | 46 |
| Lateral OA | `—` | `PILOT_RESTRICTED` | 11/47 | 2 3 1 0 | 52 | 5 | 1 | 0.4000 | 0.0000/1.0000 | 1.0000 | 1.0000/1.0000 | 0.1034 | 0.1818 | 0.0213 | 0.5000 | pos,neg | 47 |
| PF OA | `—` | `PILOT_FAIL` | 21/37 | 3 4 0 0 | 51 | 7 | 0 | 0.4286 | 0.0000/0.8333 | n/a | n/a/n/a | 0.1207 | 0.1429 | 0.0000 | 0.5714 | — | 48 |
| Effusion | `—` | `PILOT_RESTRICTED` | 35/23 | 17 11 5 0 | 25 | 28 | 5 | 0.6071 | 0.4167/0.7778 | 1.0000 | 1.0000/1.0000 | 0.5690 | 0.4857 | 0.2174 | 0.3333 | neg | 49 |
| Synovitis | `—` | `PILOT_FAIL` | 27/31 | 8 4 0 0 | 46 | 12 | 0 | 0.6667 | 0.4000/0.9232 | n/a | n/a/n/a | 0.2069 | 0.2963 | 0.0000 | 0.3333 | — | 50 |
| Baker's | `—` | `PILOT_RESTRICTED` | 12/46 | 6 5 2 0 | 45 | 11 | 2 | 0.5455 | 0.2500/0.8462 | 1.0000 | 1.0000/1.0000 | 0.2241 | 0.5000 | 0.0435 | 0.3846 | neg | 51 |
| Contusion | `—` | `PILOT_FAIL` | 19/39 | 7 3 4 1 | 43 | 10 | 5 | 0.7000 | 0.3750/1.0000 | 0.8000 | 0.3333/1.0000 | 0.2586 | 0.3684 | 0.1026 | 0.2667 | neg | 52 |
| Fracture | `—` | `PILOT_RESTRICTED` | 18/40 | 8 3 3 0 | 44 | 11 | 3 | 0.7273 | 0.4283/1.0000 | 1.0000 | 1.0000/1.0000 | 0.2414 | 0.4444 | 0.0750 | 0.2143 | neg | 53 |

## second_reader

| Official | stored | audited | gold+/gold- | TP FP TN FN | abst | pos_denom | neg_denom | prec | prec_CI | NPV | NPV_CI | cov | pos_cap | neg_cap | sel_err | tiny | seed |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | --- | ---: |
| ACL | `—` | `PILOT_RESTRICTED` | 24/34 | 10 1 0 1 | 46 | 11 | 1 | 0.9091 | 0.7139/1.0000 | 0.0000 | 0.0000/0.0000 | 0.2069 | 0.4167 | 0.0000 | 0.1667 | neg | 42 |
| MCL | `—` | `PILOT_FAIL` | 9/49 | 3 1 1 1 | 52 | 4 | 2 | 0.7500 | 0.2500/1.0000 | 0.5000 | 0.0000/1.0000 | 0.1034 | 0.3333 | 0.0204 | 0.3333 | pos,neg | 43 |
| Medial Meniscus | `—` | `PILOT_RESTRICTED` | 26/32 | 12 3 3 0 | 40 | 15 | 3 | 0.8000 | 0.5788/1.0000 | 1.0000 | 1.0000/1.0000 | 0.3103 | 0.4615 | 0.0938 | 0.1667 | neg | 44 |
| Lateral Meniscus | `—` | `PILOT_RESTRICTED` | 23/35 | 11 3 1 0 | 43 | 14 | 1 | 0.7857 | 0.5453/1.0000 | 1.0000 | 1.0000/1.0000 | 0.2586 | 0.4783 | 0.0286 | 0.2000 | neg | 45 |
| Medial OA | `—` | `PILOT_FAIL` | 15/43 | 7 2 0 0 | 49 | 9 | 0 | 0.7778 | 0.4000/1.0000 | n/a | n/a/n/a | 0.1552 | 0.4667 | 0.0000 | 0.2222 | — | 46 |
| Lateral OA | `—` | `PILOT_FAIL` | 11/47 | 5 4 0 0 | 49 | 9 | 0 | 0.5556 | 0.2000/0.8889 | n/a | n/a/n/a | 0.1552 | 0.4545 | 0.0000 | 0.4444 | — | 47 |
| PF OA | `—` | `PILOT_RESTRICTED` | 21/37 | 5 2 1 0 | 50 | 7 | 1 | 0.7143 | 0.3333/1.0000 | 1.0000 | 1.0000/1.0000 | 0.1379 | 0.2381 | 0.0270 | 0.2500 | neg | 48 |
| Effusion | `—` | `PILOT_FAIL` | 35/23 | 17 9 0 0 | 32 | 26 | 0 | 0.6538 | 0.4687/0.8183 | n/a | n/a/n/a | 0.4483 | 0.4857 | 0.0000 | 0.3462 | — | 49 |
| Synovitis | `—` | `PILOT_FAIL` | 27/31 | 5 3 0 0 | 50 | 8 | 0 | 0.6250 | 0.2857/1.0000 | n/a | n/a/n/a | 0.1379 | 0.1852 | 0.0000 | 0.3750 | — | 50 |
| Baker's | `—` | `PILOT_FAIL` | 12/46 | 5 2 0 0 | 51 | 7 | 0 | 0.7143 | 0.3333/1.0000 | n/a | n/a/n/a | 0.1207 | 0.4167 | 0.0000 | 0.2857 | — | 51 |
| Contusion | `—` | `PILOT_FAIL` | 19/39 | 5 2 0 0 | 51 | 7 | 0 | 0.7143 | 0.3333/1.0000 | n/a | n/a/n/a | 0.1207 | 0.2632 | 0.0000 | 0.2857 | — | 52 |
| Fracture | `—` | `PILOT_FAIL` | 18/40 | 5 1 0 1 | 51 | 6 | 1 | 0.8333 | 0.5000/1.0000 | 0.0000 | 0.0000/0.0000 | 0.1207 | 0.2778 | 0.0000 | 0.2857 | neg | 53 |

## strict_agreement

| Official | stored | audited | gold+/gold- | TP FP TN FN | abst | pos_denom | neg_denom | prec | prec_CI | NPV | NPV_CI | cov | pos_cap | neg_cap | sel_err | tiny | seed |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | --- | ---: |
| ACL | `PILOT_RESTRICTED` | `PILOT_RESTRICTED` | 24/34 | 10 0 0 0 | 48 | 10 | 0 | 1.0000 | 1.0000/1.0000 | n/a | n/a/n/a | 0.1724 | 0.4167 | 0.0000 | 0.0000 | — | 42 |
| MCL | `PILOT_RESTRICTED` | `PILOT_RESTRICTED` | 9/49 | 1 0 0 0 | 57 | 1 | 0 | 1.0000 | 1.0000/1.0000 | n/a | n/a/n/a | 0.0172 | 0.1111 | 0.0000 | 0.0000 | pos | 43 |
| Medial Meniscus | `PILOT_PASS` | `PILOT_PASS` | 26/32 | 8 0 2 0 | 48 | 8 | 2 | 1.0000 | 1.0000/1.0000 | 1.0000 | 1.0000/1.0000 | 0.1724 | 0.3077 | 0.0625 | 0.0000 | neg | 44 |
| Lateral Meniscus | `PILOT_RESTRICTED` | `PILOT_RESTRICTED` | 23/35 | 6 2 1 0 | 49 | 8 | 1 | 0.7500 | 0.4000/1.0000 | 1.0000 | 1.0000/1.0000 | 0.1552 | 0.2609 | 0.0286 | 0.2222 | neg | 45 |
| Medial OA | `PILOT_RESTRICTED` | `PILOT_RESTRICTED` | 15/43 | 4 0 0 0 | 54 | 4 | 0 | 1.0000 | 1.0000/1.0000 | n/a | n/a/n/a | 0.0690 | 0.2667 | 0.0000 | 0.0000 | pos | 46 |
| Lateral OA | `PILOT_FAIL` | `PILOT_FAIL` | 11/47 | 2 2 0 0 | 54 | 4 | 0 | 0.5000 | 0.0000/1.0000 | n/a | n/a/n/a | 0.0690 | 0.1818 | 0.0000 | 0.5000 | pos | 47 |
| PF OA | `PILOT_FAIL` | `PILOT_FAIL` | 21/37 | 3 2 0 0 | 53 | 5 | 0 | 0.6000 | 0.0000/1.0000 | n/a | n/a/n/a | 0.0862 | 0.1429 | 0.0000 | 0.4000 | pos | 48 |
| Effusion | `PILOT_FAIL` | `PILOT_FAIL` | 35/23 | 11 8 0 0 | 39 | 19 | 0 | 0.5789 | 0.3333/0.7895 | n/a | n/a/n/a | 0.3276 | 0.3143 | 0.0000 | 0.4211 | — | 49 |
| Synovitis | `PILOT_FAIL` | `PILOT_FAIL` | 27/31 | 4 3 0 0 | 51 | 7 | 0 | 0.5714 | 0.1991/1.0000 | n/a | n/a/n/a | 0.1207 | 0.1481 | 0.0000 | 0.4286 | — | 50 |
| Baker's | `PILOT_FAIL` | `PILOT_FAIL` | 12/46 | 4 2 0 0 | 52 | 6 | 0 | 0.6667 | 0.2500/1.0000 | n/a | n/a/n/a | 0.1034 | 0.3333 | 0.0000 | 0.3333 | — | 51 |
| Contusion | `PILOT_FAIL` | `PILOT_FAIL` | 19/39 | 5 1 0 0 | 52 | 6 | 0 | 0.8333 | 0.5000/1.0000 | n/a | n/a/n/a | 0.1034 | 0.2632 | 0.0000 | 0.1667 | — | 52 |
| Fracture | `PILOT_RESTRICTED` | `PILOT_RESTRICTED` | 18/40 | 4 0 0 0 | 54 | 4 | 0 | 1.0000 | 1.0000/1.0000 | n/a | n/a/n/a | 0.0690 | 0.2222 | 0.0000 | 0.0000 | pos | 53 |

## Stored vs audited primary-system labels

No primary-system status labels changed after fail-closed CI.

## Coordinator gates

- `HARD_WEAK_LABEL_TRAINING_ALLOWED: false`
- `SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED: false`
- `gold_rerun_authorized: false`
- do not process the 4,349 report-only studies
- do not merge PR #10

