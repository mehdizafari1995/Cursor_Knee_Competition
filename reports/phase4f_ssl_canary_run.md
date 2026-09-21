# Phase 4F.1 unlabeled-image SimSiam SSL canary — kernel v1

Private kernel `mahdimzafari/rsna-knee-unlabeled-image-ssl-canary-v1`
**version 1** ran on Tesla T4, internet off, competition attached,
notebook 14. Protocol `unlabeled-image-simsiam-ssl-canary-v1`.
Outcome: `SSL_CANARY_FAIL`. Frozen Phase 4F v1, gold58 v1/v2,
rehearsal, and second-reader kernels were not overwritten. A second canary
kernel was not launched. The full 4,349-study SSL job remains unauthorized.

## Kernel

- slug: `mahdimzafari/rsna-knee-unlabeled-image-ssl-canary-v1`
- version: 1
- Kaggle worker status: `KernelWorkerStatus.ERROR`
- recorded status: `SSL_CANARY_FAIL`
- skip_reason: `uncaught_NotImplementedError` (`linalg_eigh_cuda` not implemented for AMP float16)
- GPU: Tesla T4
- internet: off
- `scientific_evaluation`: true (real_kaggle_mount)
- `training_run`: started; first optimizer step crashed before a sanitized JSON write
- `OOF_run`: false
- `inference_run`: false
- `competition_submission_written`: false
- `consumes_submission_slot`: false
- `full_ssl_run_authorized`: false
- `UNLABELED_IMAGE_SSL_EXECUTION_ALLOWED`: false
- UNLABELED_IMAGE_SSL_EXECUTION_ALLOWED=false
- `n_report_only_studies_processed`: 0
- n_real_unlabelled_studies_pixels_opened: 64
- `n_real_unlabelled_studies_pixels_opened`: 64 (inferred: cohort cache completed and the first optimizer step began; scientific path refuses any other count)
- packaged source HEAD: `22b78eb1cf19d6da39f388ce10eea6ff238c620d`
- notebooks 07, 09, 10, 11, 12, and 13 were not executed
- `/kaggle/working/submission.csv` was not written

## Timing (from kernel log timestamps only)

- source extract: 15.5 s
- preflight + GPU name: 56.8 s
- GradScaler constructed (after series inspect, 64-study pixel cache, ImageNet load): 352.6 s
- first autocast forward: 353.1 s
- crash on `effective_rank` / `eigvalsh`: 358.7 s
- notebook convert complete: 366.1 s
- optimizer steps completed: 0
- forced resume at step 20: not reached
- checkpoint round-trip: not measured
- losses / gradient norms: not recorded
- collapse probe: not recorded
- guarded 4,349-study projection: not computed (no seconds/step)

Approximate dataset/I/O wall before the first step: 352.6 − 56.8 = 295.8 s
for the 64-study header inspect plus pixel cache. That is not a complete
canary measurement and must not be used to authorize a full SSL run.

## Frozen pins (unchanged)

- v1.4 second-reader config SHA-256:
  `992af7a4b4483278445a6f3fb363c818611e79005618dff063460d0aed3bd7f7`
- gold58 v1 recipe SHA-256:
  `92f9d8b52d9488646a4dd19ca9740b8ae658c5f71bf6812a393b91fa3a07af9c`
- gold58 v2 recipe SHA-256:
  `4b65bc0d4d56a3dd36b6898c71d630a113c99f6e4b986e51a15ae1b66e1ea093`
- Phase 4F v1 SSL config SHA-256:
  `d0c714f2ec1dc818f140abee21464cc4cf8a25f9a2a882195ae6860fda48ce43`
- Phase 4F.1 canary config SHA-256:
  `3c8ad505be9e2c71c1ad3c15f894d7188f0ca1f8cf1752ad4df7d84d7ce25c4c`
- ImageNet ResNet-18 SHA-256:
  `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`

## Pass gates (all required; this run failed)

- all 64 selected studies readable: reached pixel cache / first step (no fail-closed unreadability)
- gold overlap zero: required by the scientific path before pixels open
- losses/gradients/diagnostics finite: not recorded; diagnostics crashed
- no declared collapse gate fired: not evaluated
- forced resume completed exactly: false
- guarded full-run projection with 25% safety and 3600 s reserve: not computed

Neither this `SSL_CANARY_FAIL` nor a later code fix authorizes the 4,349-study job.
Kernel v1 is frozen. Do not overwrite it as a new v1.

## Kernel version 2 — authorized repair replay (`SSL_CANARY_GPU_PREFLIGHT_FAIL`)

ChatGPT authorized exactly one additional version of the same private slug
after the diagnostic Gram/std-only fix. Local tests passed (627 pytest).
Kernel **version 2** then ran the mandatory GPU AMP preflight on Tesla T4
before `train.csv`, series metadata, or any DICOM. The preflight failed
closed. Real unlabeled pixels were not opened. A third canary is not
authorized. The full 4,349-study SSL job remains unauthorized.

- slug: `mahdimzafari/rsna-knee-unlabeled-image-ssl-canary-v1`
- version: 2
- Kaggle worker status: `KernelWorkerStatus.ERROR`
- recorded status: `SSL_CANARY_GPU_PREFLIGHT_FAIL`
- skip_reason: `nonfinite_gradient` (synthetic ResNet-18 AMP
  forward/backward software check; thresholds were not retuned)
- GPU: Tesla T4
- internet: off
- competition attached; notebook 14
- `n_real_unlabelled_studies_pixels_opened`: 0
- `n_report_text_rows_processed`: 0
- `n_second_reader_rows_processed`: 0
- report text was never read. Version 2 did not open the 64 unlabeled
  image rows (unlike kernel v1, which opened 64 pixels and then crashed).
  `n_report_only_studies_processed` remains 0 and must not be read as
  "no unlabeled image rows were in scope"; v2 stopped before that scope.
- `training_run`: false
- `OOF_run`: false
- `inference_run`: false
- `competition_submission_written`: false
- `consumes_submission_slot`: false
- `full_ssl_run_authorized`: false
- UNLABELED_IMAGE_SSL_EXECUTION_ALLOWED=false
- packaged source HEAD: `26be5d7c38f13927ed9e1fceff767b885d0f066f`
- source zip sha256: `ea749c4acef4ec03f30ea8b284ee0f96fc9b8f3eb9303a69e1ed26139c31d3e7`
- n_files in zip: 119
- notebooks 07, 09, 10, 11, 12, and 13 were not executed
- `/kaggle/working/submission.csv` was not written
- wall ~67 s (nbconvert complete); GPU preflight fail printed at 60.3 s
- optimizer steps completed: 0
- forced resume at step 20: not reached
- losses / gradient norms / probe diagnostics: not recorded
  (training segment did not start)
- eigvalsh dtype/shape: not recorded (`gpu_preflight_eigvalsh={}` because
  the AMP backward check failed before the probe eigensolve)
- guarded 4,349-study projection: not computed
- kernel v1 ERROR / `SSL_CANARY_FAIL` evidence is unchanged
- Phase 4F v1 config SHA-256 unchanged:
  `d0c714f2ec1dc818f140abee21464cc4cf8a25f9a2a882195ae6860fda48ce43`
- Phase 4F.1 canary config SHA-256 unchanged:
  `3c8ad505be9e2c71c1ad3c15f894d7188f0ca1f8cf1752ad4df7d84d7ce25c4c`

GPU-preflight values are a software check only. They were not used to
change frozen collapse thresholds or the recipe. No third canary. Do not
launch the full SSL run.
