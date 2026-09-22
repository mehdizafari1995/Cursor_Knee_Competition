# Phase 4E gold58 submission rehearsal — visible test only

Private kernel `mahdimzafari/rsna-knee-gold58-submission-rehearsal-v1`
**version 1** completed on Tesla T4, internet off, competition attached,
notebook 12. Frozen v1 and v2 OOF kernels/configs/reports were not
overwritten. No Kaggle competition submission was created.

## Kernel

- slug: `mahdimzafari/rsna-knee-gold58-submission-rehearsal-v1`
- version: 1
- status: `SUBMISSION_REHEARSAL_COMPLETE`
- GPU: Tesla T4
- `training_run`: false
- `scientific_evaluation`: true
- `competition_submission_written`: false
- `consumes_submission_slot`: false
- `visible_test_only`: true
- `hidden_test_available`: false
- `n_visible_test_studies`: 3
- `n_models_loaded`: 5
- `n_report_only_studies_processed`: 0
- wall_seconds: 22.996
- peak_gpu_memory_bytes: 175136256
- device: cuda; interpolation bilinear; `imagenet_normalize`: true
- ensemble: equal-weight mean of five fold sigmoid probabilities
- packaged source HEAD: `b5f253f89139f14c22f03cee069bc699fd742b92`
- notebooks 07, 09, 10, and 11 were not executed

## Frozen pins (unchanged)

- v1.4 second-reader config SHA-256:
  `992af7a4b4483278445a6f3fb363c818611e79005618dff063460d0aed3bd7f7`
- gold58 v1 recipe SHA-256:
  `92f9d8b52d9488646a4dd19ca9740b8ae658c5f71bf6812a393b91fa3a07af9c`
- gold58 v2 recipe SHA-256:
  `4b65bc0d4d56a3dd36b6898c71d630a113c99f6e4b986e51a15ae1b66e1ea093`
- ImageNet ResNet-18 SHA-256 (architecture pin; fold weights loaded):
  `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`
- checkpoint dataset: `mahdimzafari/gold58-oof-v2-fold-checkpoints`
- fold SHA-256 match frozen Phase 4D.2 report:
  fold0 `1ea83eb228c4986655edc068c1df72b5cb38baa53d51259a7f53cc046b297dce`
  fold1 `508f887a790e56d4eae6e9aada25757af59a99f4dee36814fcd7e54f22406a8f`
  fold2 `ea9caa34a62434cb77741d8f437cfe15c96e4ad6fd39565e5fcdf92bf6b85e40`
  fold3 `c5f8fb85ecab87a5c64a44d5ed73ddcacae2b1cf6671bb0a8fe3e2de44bbb2bf`
  fold4 `4ec6e488a6801e2b3c175aec397fa6c431804f020f77eede99b34833deecad6b`

## Series-selection aggregates (`deterministic-primary-series-v2`)

Exactly one readable series per visible test study from test metadata
and files before predictions. No test UIDs published.

- n_required: 3
- n_selected: 3
- n_zero_readable: 0
- n_tier_1: 2
- n_tier_2: 1
- n_tier_3: 0
- n_tier_4: 0
- n_tie_break_none: 1
- n_tie_break_dicom_count: 0
- n_tie_break_series_description: 2
- n_tie_break_series_uid: 0
- selected_plane: sagittal 3
- selected Fluid_Sensitive: true 2 / false 1
- n_missing_series_dir: 0
- n_unreadable_instances: 0

## Rehearsal CSV (private)

Written through `knee.submission.write_competition_submission`. Kept
private. Not a competition submission. `/kaggle/working/submission.csv`
was not written.

- n_rows: 3
- n_columns: 13
- header: `StudyInstanceUID` + `OFFICIAL_TARGET_COLUMNS`
- probability_min: 0.04726875014603138
- probability_max: 0.9005700945854187
- sha256: `e42fdc386d4df39772b5305dae16f39ad2fad4be9e6ba01a37d49643f0adf454`
- deterministic repeat max_abs_diff: 0.0; within_tolerance: true; atol 1e-5

## Runtime guard

- seconds_per_study: 4.574655045333334
- safety_factor: 2.0
- projected_this_run_seconds: 27.447930272000008
- projected_hidden_seconds: 11894.103117866669 (documented hidden n=1300)
- n_studies_9h_capacity: 3541
- `this_run_fits_limit`: true
- `hidden_scoring_capacity_ok`: true
- hidden test was not available on this private kernel

## Coordinator gates

- `HARD_WEAK_LABEL_TRAINING_ALLOWED`: false
- `SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED`: false
- `SECOND_READER_SCORE_BEARING_ALLOWED`: false
- `gold_second_reader_rerun_authorized`: false
- forbidden outcome not used: `CONSTRAINED_FORMAT_PASS_SEMANTIC_PASS`
- no third OOF or training run; no v2 retuning
- no competition submission slot consumed
