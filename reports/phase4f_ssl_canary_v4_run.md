# Phase 4F.4 surgical BatchNorm-probe unlabeled-image SimSiam SSL canary — local only

Protocol `unlabeled-image-simsiam-ssl-canary-v4`. Kernel slug
`mahdimzafari/rsna-knee-unlabeled-image-ssl-canary-v4` was **not** created or
executed. Priority correction 5771289671 canceled the GPU launch before any
SSL kernel started. Local probe repair and regression tests remain. Frozen
Phase 4F.1 / 4F.2 / 4F.3 kernels were not overwritten.

UNLABELED_IMAGE_SSL_EXECUTION_ALLOWED=false
UNLABELED_IMAGE_SSL_CANARY_V4_EXECUTION_ALLOWED=false
SSL_ENGINEERING_PATH_PARKED=true

## Frozen prior SSL canary failures (immutable)

- slug: `mahdimzafari/rsna-knee-unlabeled-image-ssl-canary-v1` version 1
  remains `SSL_CANARY_FAIL` (AMP float16 `eigvalsh` / `linalg_eigh_cuda`).
  `n_real_unlabelled_studies_pixels_opened: 64`.
- same slug version 2 remains `SSL_CANARY_GPU_PREFLIGHT_FAIL`
  (`nonfinite_gradient` on the synthetic AMP ResNet-18 software check).
  `n_real_unlabelled_studies_pixels_opened: 0`.
- slug: `mahdimzafari/rsna-knee-unlabeled-image-ssl-canary-v2` version 1
  remains `SSL_CANARY_GPU_PREFLIGHT_FAIL` (`uncaught_ValueError`).
- slug: `mahdimzafari/rsna-knee-unlabeled-image-ssl-canary-v3` version 1
  remains `SSL_CANARY_GPU_PREFLIGHT_FAIL` (BatchNorm train-mode
  `torch.Size([1, 2048])`). SimSiam remains parked as
  `SSL_ENGINEERING_PATH_PARKED`.

## Kernel (not launched)

- protocol: `unlabeled-image-simsiam-ssl-canary-v4`
- slug: `mahdimzafari/rsna-knee-unlabeled-image-ssl-canary-v4`
- version: not created
- `n_real_unlabelled_studies_pixels_opened`: 0
- `n_report_text_rows_processed`: 0
- `n_second_reader_rows_processed`: 0
- full SSL: not authorized
- notebooks 07, 09, 10, 11, 12, 13, 14, 15, 16, and 17 were not executed
