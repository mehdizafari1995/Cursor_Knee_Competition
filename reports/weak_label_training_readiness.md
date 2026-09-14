# Weak-label training readiness

Coordinator decision after the private internet-off gold-label evaluation of
extractor `report-labeler-v1.0.0` on the 58 completely expert-labelled
studies. Aggregate metrics only. No report text, StudyInstanceUID values, or
PatientID values.

| Gate | Verdict |
| --- | --- |
| Hard weak-label image training | `NOT_READY_FOR_HARD_WEAK_LABEL_TRAINING` |
| Selective label refinement | `READY_FOR_SELECTIVE_LABEL_REFINEMENT` |

`HARD_WEAK_LABEL_TRAINING_ALLOWED` remains `false`. That flag is a coordinator
gate in `knee/report_labels.py` and is never auto-flipped by the notebook.

No image model is trained, no fold file is written, and no leaderboard
submission is made in this decision.

## NOT_READY_FOR_HARD_WEAK_LABEL_TRAINING

Do **not** train an image model that treats the 4,349-study weak-label table
as hard 0/1 targets.

Measured reasons:

- Macro specificity is 0.388 with 58 false positives among covered gold calls.
  Hard training would treat those extractor positives as true disease.
- Macro coverage is 0.299. Most gold target-cells are abstentions. Filling
  unknown as negative is forbidden and would fabricate a large negative class.
- OA, synovitis, Baker's, contusion, and fracture abstain on 74–88% of gold
  studies. Weak-label unknown rates on the 4,349 report-only studies are
  0.563–0.901.
- Weak-label positive rates among known calls exceed expert rates for medial
  meniscus (0.704 vs 0.448), medial OA (0.703 vs 0.259), lateral OA (0.673 vs
  0.190), PF OA (0.796 vs 0.362), synovitis (0.809 vs 0.466), and Baker's
  (0.650 vs 0.207).
- MCL (9 gold positives), Lateral OA (11), and Baker's (12) have wide
  bootstrap intervals and cannot support a hard-label training claim.

## READY_FOR_SELECTIVE_LABEL_REFINEMENT

The gold evaluation is sufficient to refine the extractor without image
training.

Allowed next work:

- Edit ontology-backed rules, multilingual terms, negation, history,
  postoperative, OA-severity, and compartment checks in a **new** extractor
  version (`report-labeler-v1.x` or later), keeping `gold-eval-v1.0.0` or
  bumping the eval protocol when the comparison definition changes.
- Use the 58 expert-labelled studies only for evaluation and controlled
  development. Gold labels must still never enter `extract_labels_from_report`.
- Prioritise false-positive reduction on meniscus, OA, synovitis, and Baker's,
  and coverage recovery where abstentions are `insufficient_evidence`.
- Keep unknown as unknown in every private table. High-confidence positives
  may be reviewed as candidates; they are not hard training labels.
- Language-level English (n=28) and Spanish (n=10) gold slices may guide
  lexicon work. Other gold languages remain underpowered (`n<8`).

Not allowed under this gate:

- Imputing unknown as negative
- Training an image model on the current weak-label CSV as hard 0/1
- Creating final patient-grouped folds for that training run
- Leaderboard submission
- External pretrained language models without a competition-rules check

## Pointers

- Metrics: `reports/report_labeling_summary.md`
- Protocol: `reports/report_labeling_protocol.md`
- Kaggle run: `reports/kaggle_report_labeling_run.md`
- Code gate: `HARD_WEAK_LABEL_TRAINING_ALLOWED` in `knee/report_labels.py`
