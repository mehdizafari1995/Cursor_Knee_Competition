# Phase 4F.3 documented-AMP unlabeled-image SimSiam SSL canary — not yet executed

Private kernel `mahdimzafari/rsna-knee-unlabeled-image-ssl-canary-v3`
is authorized for at most one Tesla T4 internet-off run after local
preflight. This placeholder records the frozen Phase 4F.1 and 4F.2
failures that must remain unchanged. Sanitized kernel evidence will
replace the pending fields after the one authorized launch. The full
4,349-study SSL job remains unauthorized.

UNLABELED_IMAGE_SSL_EXECUTION_ALLOWED=false

## Frozen prior SSL canary failures (immutable)

- protocol: `unlabeled-image-simsiam-ssl-canary-v3`
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

## Pending Phase 4F.3 kernel

- protocol: `unlabeled-image-simsiam-ssl-canary-v3`
- kernel slug: `mahdimzafari/rsna-knee-unlabeled-image-ssl-canary-v3`
- notebook: 16
- GPU: Tesla T4 required; internet off; competition attached
- documented AMP order: scale → one unscale_ → step → update
- predeclared FP32 fallback if AMP cannot apply one finite step after
  three documented scale backoffs
- UNLABELED_IMAGE_SSL_EXECUTION_ALLOWED=false
- `n_report_text_rows_processed`: 0
- `n_second_reader_rows_processed`: 0
- `full_ssl_run_authorized`: false
- pass status, if reached, remains `SSL_CANARY_PASS_FULL_RUN_NOT_AUTHORIZED`
- a second Phase 4F.3 kernel is not authorized if this one fails
- if both AMP and FP32 preflights fail: park `SSL_ENGINEERING_PATH_PARKED`

Do not launch the full SSL run. Do not overwrite frozen Phase 4F.1 or
4F.2 kernels.
