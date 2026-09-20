# Phase 4D.2 gold58 image OOF v2 — first measured grouped OOF

Private kernel `mahdimzafari/rsna-knee-gold58-image-oof-v2` **version 1**
completed training on Tesla T4, internet off, competition attached,
notebook 11. Frozen v1 kernel/config/report were not overwritten.
This is a first measured gold-58 image baseline, not a submission and
not permission to retune.

## Kernel

- slug: `mahdimzafari/rsna-knee-gold58-image-oof-v2`
- version: 1
- status: `GOLD58_OOF_COMPLETE`
- GPU: Tesla T4
- `training_run`: true
- `scientific_evaluation`: true
- `competition_submission_written`: false
- `consumes_submission_slot`: false
- `n_labelled_studies`: 58
- `n_oof_rows`: 58
- `n_report_only_studies_processed`: 0
- wall_seconds: 8168.106
- peak_gpu_memory_bytes: 404743680
- device: cuda; AMP: true
- interpolation: bilinear; `imagenet_normalize`: true
- packaged source HEAD: `17a5721b39d986e7ff70f263774cb0af6a430821`
- notebooks 07, 09, and 10 were not executed

## Frozen pins (unchanged)

- v1.4 second-reader config SHA-256:
  `992af7a4b4483278445a6f3fb363c818611e79005618dff063460d0aed3bd7f7`
- gold58 v1 recipe SHA-256 (immutable fail-closed):
  `92f9d8b52d9488646a4dd19ca9740b8ae658c5f71bf6812a393b91fa3a07af9c`
- gold58 v2 recipe SHA-256:
  `4b65bc0d4d56a3dd36b6898c71d630a113c99f6e4b986e51a15ae1b66e1ea093`
- ImageNet ResNet-18 SHA-256 (loaded):
  `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`

## Series-selection aggregates (`deterministic-primary-series-v2`)

Exactly one readable series was selected for all 58 labelled studies
from metadata and files before labels were used to choose among series.

- n_required: 58
- n_selected: 58
- n_zero_readable: 0
- n_tier_1 (sagittal AND Fluid_Sensitive): 56
- n_tier_2 (sagittal, any Fluid_Sensitive): 2
- n_tier_3: 0
- n_tier_4: 0
- n_tie_break_none: 49
- n_tie_break_dicom_count: 0
- n_tie_break_series_description: 7
- n_tie_break_series_uid: 2
- selected_plane: sagittal 58
- selected Fluid_Sensitive: true 56 / false 2
- n_missing_series_dir: 0
- n_unreadable_instances: 0

## Patient grouping and folds

- group_source: PatientID from all readable series
- n_unique_patients: 58
- studies_per_patient: 1 for all 58 patients
- n_folds: 5; seed: 42
- held-out fold sizes: 11 / 11 / 11 / 12 / 13
- `train_class_support_ok`: true
- `allow_study_id_grouping_fallback`: false

## Combined 58-row OOF (primary metric)

Primary metric is combined 58-row OOF macro ROC AUC, not a mean of
fold AUCs. Patient-level percentile bootstrap, n_resamples=1000,
seed=42, ci_level=0.95.

- macro_roc_auc: 0.6237496021463534
- CI low: 0.5716772675400352
- CI high: 0.6741401637620386
- n_finite_macro_resamples: 1000
- shuffled-label diagnostic macro ROC AUC (not a gate): 0.5208846317195928

Per-target AUC (gold+/gold−; 95% CI low–high; all CIs finite, n=1000):

- ACL: 0.594363 (24/34); 0.439646–0.739624
- MCL: 0.417234 (9/49); 0.192687–0.630385
- Medial Meniscus: 0.622596 (26/32); 0.464217–0.765163
- Lateral Meniscus: 0.647205 (23/35); 0.479135–0.802664
- Medial OA: 0.716279 (15/43); 0.540384–0.863730
- Lateral OA: 0.756286 (11/47); 0.584875–0.888889
- PF OA: 0.492921 (21/37); 0.314650–0.666768
- Effusion: 0.726708 (35/23); 0.598773–0.861007
- Synovitis: 0.641577 (27/31); 0.501747–0.781346
- Baker's: 0.664855 (12/46); 0.516436–0.794907
- Contusion: 0.543860 (19/39); 0.390000–0.697717
- Fracture: 0.661111 (18/40); 0.514492–0.803957

Last-epoch train loss by fold (12 epochs each): 0.382322, 0.360170,
0.369017, 0.345092, 0.373542.

## Round-trip and private hashes

- round-trip max_abs_diff: 0.0; within_tolerance: true; atol: 1e-5
- oof_predictions_sha256:
  `79e37e8ccab00de230cfc49069e3380c68710f616a896b057f7f3f2b20b4d896`
- checkpoint SHA-256 fold0:
  `1ea83eb228c4986655edc068c1df72b5cb38baa53d51259a7f53cc046b297dce`
- fold1: `508f887a790e56d4eae6e9aada25757af59a99f4dee36814fcd7e54f22406a8f`
- fold2: `ea9caa34a62434cb77741d8f437cfe15c96e4ad6fd39565e5fcdf92bf6b85e40`
- fold3: `c5f8fb85ecab87a5c64a44d5ed73ddcacae2b1cf6671bb0a8fe3e2de44bbb2bf`
- fold4: `4ec6e488a6801e2b3c175aec397fa6c431804f020f77eede99b34833deecad6b`

Row-level OOF predictions and weights stay private. They are not
committed.

## Coordinator gates

- `HARD_WEAK_LABEL_TRAINING_ALLOWED`: false
- `SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED`: false
- `SECOND_READER_SCORE_BEARING_ALLOWED`: false
- `gold_second_reader_rerun_authorized`: false
- forbidden outcome not used: `CONSTRAINED_FORMAT_PASS_SEMANTIC_PASS`
- no hidden-test inference; no competition submission
- no third OOF run; the v2 series rule is not retuned from this result
