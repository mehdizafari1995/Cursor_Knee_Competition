# Phase 4F.3 documented-AMP unlabeled-image SimSiam SSL canary — kernel v1

Private kernel `mahdimzafari/rsna-knee-unlabeled-image-ssl-canary-v3`
**version 1** ran on Tesla T4, internet off, competition attached,
notebook 16. Protocol `unlabeled-image-simsiam-ssl-canary-v3`.
Outcome: `SSL_CANARY_GPU_PREFLIGHT_FAIL` on both AMP and the
predeclared FP32 fallback. SimSiam is parked as
`SSL_ENGINEERING_PATH_PARKED`. Frozen Phase 4F.1 and 4F.2 kernels were
not overwritten. A second Phase 4F.3 kernel is not authorized. The full
4,349-study SSL job remains unauthorized.

UNLABELED_IMAGE_SSL_EXECUTION_ALLOWED=false

## Frozen prior SSL canary failures (immutable)

- slug: `mahdimzafari/rsna-knee-unlabeled-image-ssl-canary-v1` version 1
  remains `SSL_CANARY_FAIL` (AMP float16 `eigvalsh` / `linalg_eigh_cuda`).
  `n_real_unlabelled_studies_pixels_opened: 64`.
- same slug version 2 remains `SSL_CANARY_GPU_PREFLIGHT_FAIL`
  (`nonfinite_gradient` on the synthetic AMP ResNet-18 software check).
  `n_real_unlabelled_studies_pixels_opened: 0`.
- slug: `mahdimzafari/rsna-knee-unlabeled-image-ssl-canary-v2` version 1
  remains `SSL_CANARY_GPU_PREFLIGHT_FAIL` (`uncaught_ValueError`).
  `n_real_unlabelled_studies_pixels_opened: 0`.
- notebooks 07, 09, 10, 11, 12, 13, 14, and 15 were not executed for
  Phase 4F.3.
- Phase 4F.1 canary config SHA-256 unchanged:
  `3c8ad505be9e2c71c1ad3c15f894d7188f0ca1f8cf1752ad4df7d84d7ce25c4c`
- Phase 4F.2 canary config SHA-256 unchanged:
  `c06e0a78848b20e5ab9c675f3e0df8c4a82aa86519408729b1149d86ab8afb26`
- Phase 4F v1 SSL config SHA-256 unchanged:
  `d0c714f2ec1dc818f140abee21464cc4cf8a25f9a2a882195ae6860fda48ce43`

## Kernel

- protocol: `unlabeled-image-simsiam-ssl-canary-v3`
- slug: `mahdimzafari/rsna-knee-unlabeled-image-ssl-canary-v3`
- version: 1
- Kaggle worker status: `KernelWorkerStatus.ERROR`
- recorded status: `SSL_CANARY_GPU_PREFLIGHT_FAIL`
- skip_reason: `ValueError:stage=fp32_software_preflight:Expected more than 1 value per channel when training, got input size torch.Size([1, 2048])`
- GPU: Tesla T4
- internet: off
- competition attached; notebook 16
- documented AMP order required: scale → one unscale_ → step → update
- AMP preflight: failed (`exception_class=ValueError`, stage
  `amp_software_preflight`, last_completed_stage `model_build`)
- FP32 fallback: attempted after AMP fail; failed with the same BatchNorm
  message (`stage=fp32_software_preflight`, last_completed_stage
  `fp32_model_rebuild`)
- `precision_mode`: unset (neither mode applied an optimizer step)
- `amp_call_order`: empty
- `fp32_fallback_used`: false (fallback did not pass)
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
- packaged source HEAD: `48a7fc1f65f8841518641c0b4a57a610dffb93d6`
- source zip sha256: `17a1f929ec0f00ac28c50c440bf687a79efacdc9775223e663be554a3d79d150`
- n_files in zip: 127
- notebooks 07, 09, 10, 11, 12, 13, 14, and 15 were not executed
- `/kaggle/working/submission.csv` was not written
- wall ~53 s (nbconvert html complete at 52.9 s); GPU preflight fail
  printed at 42.4 s
- optimizer steps completed: 0
- applied optimizer steps: 0
- skipped AMP attempts: not recorded (preflight extra fields unset)
- forced resume at applied step 20: not reached
- checkpoint round-trip: not measured
- losses / gradient norms / probe diagnostics: not recorded
- guarded 4,349-study projection: not computed

## Exception diagnostics (sanitized)

AMP:

- exception_class: `ValueError`
- stage: `amp_software_preflight`
- last_completed_stage: `model_build`
- message: `Expected more than 1 value per channel when training, got input size torch.Size([1, 2048])`
- torch: `2.10.0+cu128`; CUDA: `12.8`; GPU: Tesla T4
- traceback_fingerprint: `e0f139f4808ff6891c68fc83f755040ddce81b37f308305d8f74f4846ea20273`
- grad_scaler: unset

FP32 fallback:

- exception_class: `ValueError`
- stage: `fp32_software_preflight`
- last_completed_stage: `fp32_model_rebuild`
- message: same BatchNorm batch-1 error
- traceback_fingerprint: `7e2d6490aaa910ff0434bd758bd961924fb0755a541dd9f8e8543860137688d5`

This is a third distinct SSL implementation failure after 4F.1 v1
(`eigvalsh`), 4F.1 v2 (`nonfinite_gradient`), and 4F.2
(`uncaught_ValueError`). Both numerical modes failed. SimSiam is parked.

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
- Phase 4F.3 canary config SHA-256:
  `7f8eb3b0143441d96bf9cb445ecf12a2d22bfdae6f1d9a49f3603ab41374cb4d`
- ImageNet ResNet-18 SHA-256:
  `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`

## Park and next protocol

- status: `SSL_ENGINEERING_PATH_PARKED`
- no second Phase 4F.3 kernel
- no 4,349-study SSL job
- next score-bearing protocol prepared locally, not launched:
  `gold58-image-resnet18-3plane-oof-v1`
  (58 labelled studies, grouped OOF, frozen v2 comparator, no weak
  labels, no second reader, no submission)
