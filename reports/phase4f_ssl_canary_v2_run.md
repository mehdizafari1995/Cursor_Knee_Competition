# Phase 4F.2 AMP-safe unlabeled-image SimSiam SSL canary — not yet executed

Private kernel `mahdimzafari/rsna-knee-unlabeled-image-ssl-canary-v2` is
authorized for at most one Tesla T4 internet-off run after local preflight.
This placeholder records the frozen Phase 4F.1 failures that must remain
unchanged. Sanitized kernel evidence will replace the pending fields after
the one authorized launch. The full 4,349-study SSL job remains unauthorized.

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

## Pending Phase 4F.2 kernel

- protocol: `unlabeled-image-simsiam-ssl-canary-v2`
- kernel slug: `mahdimzafari/rsna-knee-unlabeled-image-ssl-canary-v2`
- notebook: 15
- GPU: Tesla T4 required; internet off; competition attached
- UNLABELED_IMAGE_SSL_EXECUTION_ALLOWED=false
- `n_report_text_rows_processed`: 0
- `n_second_reader_rows_processed`: 0
- `full_ssl_run_authorized`: false
- pass status, if reached, remains `SSL_CANARY_PASS_FULL_RUN_NOT_AUTHORIZED`
- a second Phase 4F.2 kernel is not authorized if this one fails

Do not launch the full SSL run. Do not overwrite frozen Phase 4F.1 kernel
versions 1–2.
