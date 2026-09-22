# Phase 4C closed: second reader parked

Phase 4C format engineering succeeded. The second reader is **not**
eligible for hard weak-label training, report-only scaling, production
routing, or score-bearing submissions.

- verdict: `SECOND_READER_PARKED_NOT_SCORE_BEARING`
- `SECOND_READER_SCORE_BEARING_ALLOWED`: `false`
- `HARD_WEAK_LABEL_TRAINING_ALLOWED`: `false`
- `SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED`: `false`
- frozen kernel: `mahdimzafari/rsna-knee-second-reader-gold-constrained-v14` version 1
- frozen v1.4 config SHA-256: `992af7a4b4483278445a6f3fb363c818611e79005618dff063460d0aed3bd7f7`
- 11 of 12 targets are `PILOT_FAIL` or `PILOT_RESTRICTED` under strict agreement
- Medial Meniscus remains a formal `PILOT_PASS`, but coverage is 10/58
  and the negative decision denominator is 2. Record it as exploratory
  evidence only, not sufficient validation for deployment.
- No additional gold inference or gold-driven tuning is authorized
- Notebook 07 stays a historical blocked stub; notebook 09 is not re-run
- Do not process the 4,349 report-only studies
- Forbidden outcome not used: `CONSTRAINED_FORMAT_PASS_SEMANTIC_PASS`

Phase 4D reviews the **image-only** baseline as the score-bearing system.
