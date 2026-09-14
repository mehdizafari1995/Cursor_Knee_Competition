# Phase 4 report-labeling protocol

Versioned, deterministic, high-precision extraction of the twelve competition
targets from multilingual radiology reports. Image models, folds, and
leaderboard submissions are out of scope.

Extractor version: `report-labeler-v1.0.0`
Evaluation protocol version: `gold-eval-v1.0.0`
Runtime config: `configs/report_labeler.yaml`

## Facts from the real Kaggle audit

Source: `reports/real_data_audit_summary.md` (aggregate counts only).

- 4,407 training studies
- 58 completely expert-labelled studies
- 4,349 report-only studies (every target unknown)
- 0 partially labelled studies
- all 4,407 training reports are non-empty
- 12 official binary targets
- PatientID is complete and 1:1 with StudyInstanceUID

## Security and competition-data restrictions

1. Report text remains inside the private Kaggle environment (or a local
   `RSNA_DATA_ROOT` that the operator already holds). It is never sent to
   Cursor, OpenAI, Anthropic, Google, translation APIs, search engines, or
   any other external service.
2. Internet is disabled during report processing on Kaggle.
3. Report excerpts are never printed, committed, or written to sanitized
   logs. Tests fail if fixture report tokens appear in committed or
   sanitized artefacts.
4. Row-level labels and identifiers stay under gitignored `outputs/` or
   `/kaggle/working/rsna_report_label_outputs/private`.
5. Committed artefacts are software plus aggregate statistics.
6. No external pretrained language model is used. Adding one requires a
   competition-rules check and a protocol amendment.

## 1. Language and formatting audit

`knee/report_language.py` classifies each report locally with Unicode script
ranges, function-word cues, and report-section cues. It records:

- language code and whether the call was uncertain (low score, close
  runner-up, mixed scripts, empty text)
- aggregate language counts only
- encoding flags (replacement characters, control characters, HTML entities)
- empty named sections, boilerplate hits, length buckets

Unsupported or undetermined languages do not receive guessed target
positives except language-agnostic abbreviations such as `ACL`.

## 2. Deterministic extractor

`knee/report_labels.py` plus `knee/report_negation.py` implement a ConText-style
rule extractor. The public function is:

```text
extract_labels_from_report(report, *, language_result=None, config=None)
```

Gold / expert labels are not parameters. Tests fail if they appear in the
signature. Each target returns:

| Field | Values |
| --- | --- |
| `label` | `positive`, `negative`, or `unknown` |
| `confidence` | `[0, 1]` from the versioned config |
| `evidence_category` | code such as `explicit_positive`, `negated_finding`, `history_only`; never report text |
| `rule_id` | `{extractor_version}:{target}:{rule}` |

Behaviour that is in scope for every target:

- ontology-backed synonyms and abbreviations from `literature/ontology.yaml`,
  frozen into `configs/report_labeler.yaml`
- additional multilingual clinical-standard terms, labelled as such and not
  claimed as ontology-verified
- sentence and clause boundaries; section headers (history / findings /
  impression / technique)
- negation before and after a finding
- uncertainty and equivocal language → `unknown`
- history versus current finding
- postoperative / reconstructed structures → `unknown` unless a current
  finding is explicit
- normal-structure statements (`intact`, `no tear`) → `negative`
- anatomical compartment and meniscus laterality; unspecified laterality →
  `unknown`
- OA severity: KL 0 / no OA → negative; KL 1 or nonspecific degenerative
  change → unknown; KL 2–4 or named osteoarthritis of that compartment →
  positive. Tricompartmental OA labels all three OA targets.
- effusion vs synovitis vs Baker’s cyst are distinct; the combined
  `effusion-synovitis` construct is unknown for both effusion and synovitis
- trace / physiologic fluid is unknown (annotator threshold is not assumed)
- no mention is `unknown`, never imputed as negative
- a global “normal MRI / no significant abnormality” impression may label
  all twelve targets negative

## 3. Gold-label evaluation

Use the 58 completely labelled studies only. Unknown gold values are skipped
and never filled with 0. Unknown extractor outputs are abstentions and are
excluded from TP/TN/FP/FN (they are not false negatives).

Metrics per target: TP, TN, FP, FN, sensitivity, specificity, precision,
recall, F1, balanced accuracy, coverage. Macro averages skip undefined
targets. Study-level bootstrap percentile intervals (1,000 resamples, seed
42, 95% CI) are required because n = 58 is small.

MCL, Lateral OA, and Baker’s cyst are reported in a dedicated small-n
section (9, 11, and 12 gold positives in the real audit).

Error categories are aggregate codes plus evidence_category, without
quotations.

Each run is written to a new directory:

```text
outputs/report_labels/evaluations/{extractor_version}/{eval_protocol_version}/{utc_stamp}/
```

Existing runs are never overwritten.

Language-level gold metrics are reported only when a language has at least
eight gold studies.

## 4. Weak-label generation

After evaluation, the same extractor is applied to the 4,349 report-only
studies. The private table contains:

- `StudyInstanceUID`
- twelve weak labels (official column spelling; empty = unknown)
- twelve confidence values
- extractor version
- language code
- quality flags

That file is gitignored. The sanitized committed summary contains prevalence,
unknown rate, confidence quantiles, language-level gold metrics where n
permits, expert vs weak prevalence, and warnings.

## 5. Leakage controls

- Gold labels never enter `extract_labels_from_report`.
- Tests fail if report text enters sanitized reports, logs, or fixtures that
  are committed.
- Tests fail if StudyInstanceUID / PatientID values enter sanitized reports.
- Private files are recorded by SHA-256 without committing contents.
- Extractor version and config SHA-256 are stored with every run.

## 6. Kaggle notebook

`notebooks/02_kaggle_report_labeling.ipynb` attaches the competition dataset
and the private source ZIP, internet off, GPU off. Row-level outputs go to
`/kaggle/working/rsna_report_label_outputs/private`. Only sanitized
aggregates may be retrieved into git.

Cursor runs software-fixture tests only. A fixture run must keep
`scientific_evaluation=false` and must refuse to claim scientific
performance.

## 7. Out of scope

- image-model training
- final patient-grouped folds
- leaderboard submission
- external pretrained language models
- quoting reports
