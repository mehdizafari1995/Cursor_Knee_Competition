# Phase 4F.2 AMP-safe unlabeled-image SimSiam SSL canary — kernel v1

Private kernel `mahdimzafari/rsna-knee-unlabeled-image-ssl-canary-v2`
**version 1** ran on Tesla T4, internet off, competition attached,
notebook 15. Protocol `unlabeled-image-simsiam-ssl-canary-v2`.
Outcome: `SSL_CANARY_GPU_PREFLIGHT_FAIL`. Frozen Phase 4F.1 kernel
`mahdimzafari/rsna-knee-unlabeled-image-ssl-canary-v1` versions 1–2 were
not overwritten. A second Phase 4F.2 kernel is not authorized. The full
4,349-study SSL job remains unauthorized.

UNLABELED_IMAGE_SSL_EXECUTION_ALLOWED=false

## Frozen Phase 4F.1 failures (immutable)

- slug: `mahdimzafari/rsna-knee-unlabeled-image-ssl-canary-v1` version 1
  remains `SSL_CANARY_FAIL` (AMP float16 `eigvalsh` / `linalg_eigh_cuda`).
  `n_real_unlabelled_studies_pixels_opened: 64`.
- same slug version 2 remains `SSL_CANARY_GPU_PREFLIGHT_FAIL`
  (`nonfinite_gradient` on the synthetic AMP ResNet-18 software check).
  `n_real_unlabelled_studies_pixels_opened: 0`.
- notebooks 07, 09, 10, 11, 12, 13, and 14 were not executed for Phase 4F.2.
- Phase 4F.1 canary config SHA-256 unchanged:
  `3c8ad505be9e2c71c1ad3c15f894d7188f0ca1f8cf1752ad4df7d84d7ce25c4c`
- Phase 4F v1 SSL config SHA-256 unchanged:
  `d0c714f2ec1dc818f140abee21464cc4cf8a25f9a2a882195ae6860fda48ce43`

## Kernel

- slug: `mahdimzafari/rsna-knee-unlabeled-image-ssl-canary-v2`
- version: 1
- Kaggle worker status: `KernelWorkerStatus.ERROR`
- recorded status: `SSL_CANARY_GPU_PREFLIGHT_FAIL`
- skip_reason: `uncaught_ValueError` (GPU preflight catch-all; the
  ValueError message was not persisted because
  `run_canary_gpu_preflight` stores `uncaught_{type}` only)
- GPU: Tesla T4
- internet: off
- competition attached; notebook 15
- `scientific_evaluation`: not reached (preflight failed before
  `train.csv` / series / DICOM)
- `training_run`: false
- `OOF_run`: false
- `inference_run`: false
- `competition_submission_written`: false
- `consumes_submission_slot`: false
- `full_ssl_run_authorized`: false
- UNLABELED_IMAGE_SSL_EXECUTION_ALLOWED=false
- `n_real_unlabelled_studies_pixels_opened`: 0
- `n_report_text_rows_processed`: 0
- `n_second_reader_rows_processed`: 0
- report text was never read. Version 1 did not open the 64 unlabeled
  image rows. `train.csv` was not read after GPU name print.
- packaged source HEAD: `08ca609b6ee3938eaa6f640ea7206c24a3713d7e`
- source zip sha256: `6494cb14fec7eb6f58c31312d6befa794d033287283eb23166ac5a152e003886`
- n_files in zip: 123
- notebooks 07, 09, 10, 11, 12, 13, and 14 were not executed
- `/kaggle/working/submission.csv` was not written
- wall ~51 s (nbconvert html complete at 50.9 s); GPU preflight fail
  printed at 40.8 s
- optimizer steps completed: 0
- applied optimizer steps: 0
- skipped AMP attempts: not recorded (preflight extra fields unset)
- forced resume at applied step 20: not reached
- checkpoint round-trip: not measured
- losses / gradient norms / probe diagnostics: not recorded
- guarded 4,349-study projection: not computed

## GPU preflight aggregates (sanitized)

Printed by notebook 15 before `SystemExit`. Software-preflight fields
were not merged because the exception was a raw `ValueError`, not a
`Phase4FCanaryError`.

- gpu_name: Tesla T4
- gpu_preflight_status: `SSL_CANARY_GPU_PREFLIGHT_FAIL`
- gpu_preflight_ok: false
- gpu_preflight_eigvalsh: `{}`
- gpu_preflight_loss_dtype: None
- gpu_preflight_loss_finite: none
- gpu_preflight_grad_norm_finite: none
- gpu_preflight_applied_optimizer_steps: None
- gpu_preflight_skipped_attempts: None
- gpu_preflight_scaler_initial_scale: None
- gpu_preflight_scaler_final_scale: None
- gpu_preflight_peak_gpu_memory_bytes: None
- gpu_preflight_batch_shape: None
- n_real_unlabelled_studies_pixels_opened: 0
- n_report_text_rows_processed: 0
- n_second_reader_rows_processed: 0

Public sanitized JSON records the same fail-closed payload. This is a
numerical-software / runtime failure, not a scientific SSL result.

Hypothesis only (not confirmed; message body was swallowed): Kaggle
torch 2.10 `torch.amp.GradScaler` / `autocast` constructor mismatch, or
a `ValueError` during ImageNet ResNet-18 construction / first AMP
attempt, before extra fields were merged. A second Phase 4F.2 kernel is
not authorized to test that hypothesis.

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
- Phase 4F.2 canary config SHA-256:
  `c06e0a78848b20e5ab9c675f3e0df8c4a82aa86519408729b1149d86ab8afb26`
- ImageNet ResNet-18 SHA-256:
  `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`

## Pass gates (all required; this run failed closed)

- Tesla T4 and internet off: GPU name printed Tesla T4; internet off
- ImageNet SHA-256 verified: not recorded in the fail payload
- finite float32 loss / one applied optimizer step / finite unscaled
  gradients: not recorded
- AMP backoff limit: not recorded
- per-step eigvalsh calls=0 / probe eigvalsh float32 dim<=16: not recorded
- peak GPU memory: not recorded
- zero real pixels and zero report text during preflight: true
- 64-study scientific canary: not started
- forced resume completed exactly: false
- guarded full-run projection with 25% safety and 3600 s reserve: not computed

GPU-preflight values are a software check only. They were not used to
change frozen collapse thresholds or the recipe. No second Phase 4F.2
canary. Do not launch the full SSL run. Do not overwrite frozen Phase
4F.1 kernel versions 1–2. Kernel v1 of canary-v2 is frozen.
