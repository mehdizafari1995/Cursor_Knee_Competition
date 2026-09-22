# Phase 4F SimSiam SSL engineering path — parked

SimSiam unlabeled-image SSL is parked as `SSL_ENGINEERING_PATH_PARKED`
after four distinct implementation failures plus a canceled 4F.4 GPU launch.
Do not launch another SSL kernel. Do not start the 4,349-study job.

UNLABELED_IMAGE_SSL_EXECUTION_ALLOWED=false
UNLABELED_IMAGE_SSL_CANARY_V4_EXECUTION_ALLOWED=false
GOLD58_SSL_OOF_EXECUTION_ALLOWED=false
SSL_ENGINEERING_PATH_PARKED=true

## Frozen failures

- 4F.1 kernel v1 `mahdimzafari/rsna-knee-unlabeled-image-ssl-canary-v1`:
  `SSL_CANARY_FAIL` (AMP float16 `eigvalsh`)
- 4F.1 kernel v2 same slug: `SSL_CANARY_GPU_PREFLIGHT_FAIL`
  (`nonfinite_gradient`)
- 4F.2 kernel v1 `mahdimzafari/rsna-knee-unlabeled-image-ssl-canary-v2`:
  `SSL_CANARY_GPU_PREFLIGHT_FAIL` (`uncaught_ValueError`)
- 4F.3 kernel v1 `mahdimzafari/rsna-knee-unlabeled-image-ssl-canary-v3`:
  `SSL_CANARY_GPU_PREFLIGHT_FAIL` on AMP and FP32
  (`ValueError` BatchNorm `torch.Size([1, 2048])`)
- 4F.4 local probe repair exists (`unlabeled-image-simsiam-ssl-canary-v4`).
  GPU kernel `mahdimzafari/rsna-knee-unlabeled-image-ssl-canary-v4` was
  never started. Priority correction canceled that launch.

## Next score-bearing protocol

`gold58-image-resnet18-3plane-oof-v1` / kernel
`mahdimzafari/rsna-knee-gold58-3plane-oof-v1`: image-only deterministic
multi-series three-plane fusion on the 58 labelled studies, grouped OOF,
frozen v2 primary-series baseline as comparator. No weak labels, no
second reader, no submission.
