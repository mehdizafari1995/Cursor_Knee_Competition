# Phase 4D.1 gold58 image OOF — fail-closed series preflight

Private kernel `mahdimzafari/rsna-knee-gold58-image-oof-v1` **version 1**
stopped before training. Tesla T4, internet off, competition attached.
This is not a scientific OOF score.

## Kernel

- slug: `mahdimzafari/rsna-knee-gold58-image-oof-v1`
- version: 1
- status: `KernelWorkerStatus.ERROR` (SystemExit after fail-closed preflight)
- GPU: Tesla T4
- `training_run`: false
- `competition_submission_written`: false
- `n_labelled_studies`: 58
- `n_report_only_studies_processed`: 0
- wall after GPU check: 0.312 s
- peak GPU memory: 0 bytes
- packaged source HEAD: `3a2f147b228f46792f3a6b8df0a84ca86c8c2b60`

## Frozen pins (unchanged)

- v1.4 second-reader config SHA-256:
  `992af7a4b4483278445a6f3fb363c818611e79005618dff063460d0aed3bd7f7`
- gold58 recipe SHA-256:
  `92f9d8b52d9488646a4dd19ca9740b8ae658c5f71bf6812a393b91fa3a07af9c`
- ImageNet ResNet-18 file SHA-256 (attached, not loaded because preflight stopped):
  `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`

## Series-selection preflight

Rule `sagittal-fluid-sensitive-unique-v1` requires exactly one sagittal
fluid-sensitive series per labelled study from `train_series.csv` metadata
before labels are used to choose among series.

- status: `SERIES_SELECTION_PREFLIGHT_FAIL`
- n_required: 58
- n_unique: 48
- n_zero: 2
- n_multi: 8
- skip_reason: rule did not yield exactly one series per labelled study

No PatientID values, study IDs, series UIDs, report text, or OOF rows are
published. Checkpoints were not written.

## Coordinator gates

- `HARD_WEAK_LABEL_TRAINING_ALLOWED`: false
- `SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED`: false
- `SECOND_READER_SCORE_BEARING_ALLOWED`: false
- `gold_second_reader_rerun_authorized`: false
- notebooks 07 and 09 were not executed
- no second OOF run; the series rule is not retuned from this result
