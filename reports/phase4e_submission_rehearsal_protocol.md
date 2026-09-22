# Phase 4E gold58 submission-pipeline rehearsal

Predeclared before execution. No hyperparameter search. No v2 retuning.
Does not edit Phase 4D.1/v1 or Phase 4D.2/v2 history.
Does not create a Kaggle competition submission.

- protocol: `gold58-image-resnet18-submission-rehearsal-v1`
- kernel: `mahdimzafari/rsna-knee-gold58-submission-rehearsal-v1`
- checkpoints: five frozen v2 fold last-epoch files; dataset `mahdimzafari/gold58-oof-v2-fold-checkpoints`
- series rule: `deterministic-primary-series-v2` on test metadata/files before predictions
- preprocess: bilinear 224, percentiles [1.0, 99.0], ImageNet mean/std, 8 linspace slices, mean aggregation
- ensemble: equal-weight arithmetic mean of five fold sigmoid probabilities
- writer: `knee.submission.write_competition_submission`
- visible/example test only; hidden test is unavailable outside official scoring
- runtime guard: fail-closed if projected this-run time exceeds 32400 s
- no report text, report-only labels, second reader, target weights, sign flips, or OOF calibration
