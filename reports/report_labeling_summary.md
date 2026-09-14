# Report labeling summary

**scientific_evaluation: false (placeholder).** Real gold-label metrics are
written here only after the private internet-off Kaggle notebook has processed
the 58 expert-labelled studies. Software-fixture counts from Cursor must not
be copied into this document as scientific performance.

Extractor: `report-labeler-v1.0.0`
Evaluation protocol: `gold-eval-v1.0.0`
Config: `configs/report_labeler.yaml`
External language model: **not used**
Internet / GPU / image training / folds: **false**

## Audit facts used by this phase

From `reports/real_data_audit_summary.md`:

- 4,407 training studies, 58 fully expert-labelled, 4,349 report-only
- 4,407 / 4,407 reports non-empty
- PatientID complete and 1:1 with StudyInstanceUID

## Language audit

Not yet computed on the real reports in this committed file. Heuristic script
counts from the Phase 3 audit (not per-report language IDs): latin 4314, greek
321, cyrillic 220, thai 3, other_non_ascii 419; 2613 / 4407 reports contain
non-ASCII.

## Gold-label performance

Pending the Kaggle run. Per-target TP/TN/FP/FN, sensitivity, specificity,
precision, recall, F1, balanced accuracy, coverage, and bootstrap CIs will
replace this section. Unknown predictions remain abstentions. Unknown gold
labels are never imputed as negative.

Small-n targets called out separately: MCL (9 gold positives), Lateral OA
(11), Baker's (12).

## Weak labels

The row-level file for the 4,349 report-only studies is gitignored. Aggregate
prevalence, unknown rates, and confidence quantiles will be filled after the
Kaggle run. Private output SHA-256 hashes will be recorded without contents.

## Warnings

- This committed file is a protocol placeholder until sanitized Kaggle
  aggregates are retrieved.
- Greek, Cyrillic, Thai, and mixed-script reports are expected; unsupported
  languages abstain except language-agnostic abbreviations.
- No external pretrained language model is used.
