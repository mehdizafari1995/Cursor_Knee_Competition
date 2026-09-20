# Phase 4D competition-readiness review

Read-only. No training, inference, Kaggle kernel, or submission.
The second reader is excluded from score-bearing behaviour.

- status: `READINESS_REVIEW_COMPLETE`
- score-bearing system: image-only BaselineStudyModel (ResNet-18); not the second reader
- second reader: `SECOND_READER_PARKED_NOT_SCORE_BEARING`
- metric_reproduction_status: `NO_SAVED_COMPETITION_PREDICTIONS`
- phase4d_experiment_executed: `false`
- training_run / inference_run / submission: `false` / `false` / `false`

## Score-bearing path

- RSNA_DATA_ROOT competition DICOMs (image-only at test time)
- knee.dataset.StudySliceDataset (one series per study, linspace slices)
- knee.baseline_model.BaselineStudyModel (ImageNet ResNet-18, mean slice pool, 12 logits)
- sigmoid probabilities
- scripts/predict_baseline.py predictions.csv
- scripts/make_submission.py + knee.submission.write_competition_submission
- submission.csv with StudyInstanceUID + OFFICIAL_TARGET_COLUMNS, sample row order

## Why the second reader is parked

See `reports/phase4c_second_reader_parked.md`. Eleven of twelve
targets are FAIL/RESTRICTED. Medial Meniscus `PILOT_PASS` is
exploratory (coverage 10/58, NPV denom=2), not deployment evidence.

## Existing validation evidence

- `public_leaderboard_score`: None
- `private_leaderboard_score`: None
- `real_image_oof_macro_auc`: None
- `real_image_checkpoint`: None
- `ensemble`: None
- `postprocessing`: sigmoid only on the unimplemented real predict path
- `software_smoke_macro_auc`: present only when scripts/train_baseline.py --smoke is run; scientific_evaluation false
- `synthetic_pipeline_macro_auc`: CI fixture only; not competition data
- `phase3_audit`: AUDITED; 4407 train studies; 58 fully labelled; 4349 report-only; folds_created false; training_run false
- `phase4_weak_labels`: NOT_READY_FOR_HARD_WEAK_LABEL_TRAINING
- `phase4c_second_reader`: SECOND_READER_PARKED_NOT_SCORE_BEARING
- `submission_slots_used_this_repo`: 0

## Leakage and contract audit

- `train_vs_example_test_study_overlap`: 0
- `duplicate_train_study_ids`: 0
- `patient_id_complete_on_train_headers`: True
- `unique_patient_id_equals_unique_study_id`: True
- `patient_may_be_exam_scoped`: True
- `patient_grouped_folds_allowed_gate`: False
- `real_fold_tables_created`: False
- `example_test_n_studies`: 3
- `hidden_test_size_documented_not_local`: ~1300
- `test_reports_available`: False
- `inference_must_be_image_only`: True
- `official_vs_internal_spelling_matches`: False
- `official_vs_internal_order_matches`: True
- `competition_submission_rewrites_internal_keys`: True
- `synthetic_pipeline_submission_is_not_competition_contract`: True
- `metric_implementation`: sklearn roc_auc_score per target, skip single-class and NaN, macro mean of defined targets
- `official_kaggle_evaluator_source_not_in_repo`: True
- `second_reader_on_score_path`: False
- `unknown_imputed_as_negative_in_score_path`: False
- `nondeterminism_notes`: baseline.yaml seed 42; synthetic train_model shuffles batches; DICOM slice order is deterministic (IOP normal, InstanceNumber, name)
- `series_selection_policy`: caller-supplied; no default plane/sequence chooser
- `local_vm_cannot_hold_full_dicom_tree`: True

## Readiness blockers

- No real image checkpoint or OOF score exists
- Full training loop is not implemented; train_baseline.py returns 3 even with --allow-training
- HARD_WEAK_LABEL_TRAINING_ALLOWED=false (keep); do not train on 4349 weak labels as hard 0/1
- SECOND_READER_SCORE_BEARING_ALLOWED=false; 11/12 second-reader targets FAIL/RESTRICTED
- Medial Meniscus PILOT_PASS is exploratory: coverage 10/58, NPV denom=2
- patient_grouped_folds_allowed remains a closed coordinator gate; PatientID is 1:1 with studies
- No default series-selection policy (one series per study, caller-supplied)
- Example test is 3 rows; hidden test is not local; full DICOM tree does not fit this VM
- Synthetic write_submission uses study_id + TARGET_KEYS and must not be uploaded
- Official Kaggle evaluator source is not in the repo

## Ranked improvement backlog

### gold58-image-resnet18-grouped-oof (SELECTED)

- component: scripts/train_baseline.py + knee.baseline_model.BaselineStudyModel + knee.folds.assign_grouped_folds; gold-only 58-study image OOF
- leakage-safe validation: 5-fold grouped CV. Group by PatientID if every selected study has one, else StudyInstanceUID. assert_no_group_leakage. Train only on labelled studies. Never use report-only rows, weak 0/1 imputation, or second-reader outputs. Image-only inputs.
- expected score mechanism: OOF macro ROC AUC on the 58 labelled studies is a noisy internal estimate. It is not a public leaderboard score. A later submission (not this experiment) would fit on all 58 and predict the hidden test inside a Kaggle scoring notebook.
- compute: 58 studies, 8 slices, 224px, ResNet-18, 5 folds × a few epochs. Tesla T4: about 20–90 minutes. CPU: several hours. Dataset access is Kaggle-side (~570 GB); this VM cannot hold the tree.
- failure/rollback: Abort on group leakage, report-only rows, unknown-as-0, or second-reader inputs. If OOF AUC is undefined for most targets, stop. Delete gitignored checkpoints. Do not submit. Keep HARD_WEAK_LABEL_TRAINING_ALLOWED=false.
- consumes submission slot: `false`
- datasets: Competition train DICOMs and the 58 fully labelled train.csv target columns. No report-only labels. No hidden-test labels. No second-reader files.

### multi-series-3plane-fusion-on-gold58 (rank 2)

- component: knee.dataset series selection; fuse sagittal/coronal/axial
- leakage-safe validation: Same 5-fold grouped OOF as rank 1.
- expected score mechanism: Better anatomy coverage vs single-series baseline.
- compute: Roughly 3× rank 1.
- failure/rollback: Keep single-series checkpoint; do not submit.
- consumes submission slot: `false`
- datasets: Same 58 labelled studies plus series tables.

### stronger-encoder-on-gold58 (rank 3)

- component: ResNet-50 / EfficientNet encoder swap in BaselineStudyModel
- leakage-safe validation: Same grouped OOF.
- expected score mechanism: Possible small AUC gain; overfit risk at n=58.
- compute: Higher than rank 1; still T4-hours, not days.
- failure/rollback: Revert encoder; keep ResNet-18.
- consumes submission slot: `false`
- datasets: Same 58 labelled studies.

### slice-count-and-tta (rank 4)

- component: num_slices and test-time flips in predict_baseline.py
- leakage-safe validation: Apply TTA only at OOF/test; no extra labels.
- expected score mechanism: Variance reduction of slice sampling.
- compute: 2–8× inference cost.
- failure/rollback: Disable TTA.
- consumes submission slot: `false`
- datasets: Same images as the parent checkpoint.

### selective-weak-label-refinement-then-masked-training (rank 5)

- component: report-labeler v2 high-confidence masks + masked BCE
- leakage-safe validation: Requires a new coordinator gate; still never impute unknown as 0.
- expected score mechanism: More labelled cells if precision/NPV recover.
- compute: Full train tree on Kaggle GPU; hours.
- failure/rollback: Keep HARD_WEAK_LABEL_TRAINING_ALLOWED=false.
- consumes submission slot: `false`
- datasets: Reports + images; blocked until the hard-weak gate lifts.

### public-external-mri-pretrain (rank 6)

- component: rules-compliant public MRI checkpoint attached offline
- leakage-safe validation: Pretrain off competition patients; fine-tune with grouped OOF.
- expected score mechanism: Better features than ImageNet for MRI.
- compute: Days if training from scratch; hours if attaching a public checkpoint.
- failure/rollback: Fall back to ImageNet ResNet-18.
- consumes submission slot: `false`
- datasets: Public external MRI only; competition DICOMs for fine-tune.

## Selected next experiment (not run)

- id: `gold58-image-resnet18-grouped-oof`
- why: No image checkpoint exists. The only permitted labelled image training set is the 58 fully labelled studies. Weak labels and the second reader are gated off. This is the cheapest path to a non-random image-only scorer.
- authorized_to_run_in_phase4d: `false`
- consumes_submission_slot: `false`

## Preflight checklist and proposed command

- executed: `false`
- implemented: `false`
- command (do not run): `python scripts/train_gold58_image_oof.py --config configs/baseline.yaml --folds 5 --seed 42 --labelled-only --no-weak-labels --no-second-reader`
- current blocker: scripts/train_baseline.py refuses full training even with --allow-training because HARD_WEAK_LABEL_TRAINING_ALLOWED is false. A gold-only trainer that never touches weak labels or the second reader is not implemented. Phase 4D does not add or run it.

- [ ] HARD_WEAK_LABEL_TRAINING_ALLOWED is false
- [ ] SCALE_TO_REPORT_ONLY_SECOND_READER_ALLOWED is false
- [ ] SECOND_READER_SCORE_BEARING_ALLOWED is false
- [ ] select exactly the 58 fully labelled studies; report-only count stays 0
- [ ] group folds and assert_no_group_leakage
- [ ] unknown labels stay NaN; masked BCE; never impute 0
- [ ] image-only tensors; no Report column in the model
- [ ] official submission path unused in this experiment
- [ ] internet off if executed on Kaggle; no new kernel in Phase 4D
- [ ] STOP: do not run the command in this phase

## Coordinator gates

- `NOT_READY_FOR_HARD_WEAK_LABEL_TRAINING`
- `NOT_READY_TO_SCALE_SECOND_READER_TO_REPORT_ONLY`
- `SECOND_READER_PARKED_NOT_SCORE_BEARING`
- do not merge PR #10
