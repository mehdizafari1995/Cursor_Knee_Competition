# Phase 4F.1 unlabeled-image SimSiam SSL canary protocol

Bounded real-data canary only. The full 4,349-study SSL job is not authorized.
Phase 4F v1 config, report, notebook 13, and synthetic results stay frozen.

- protocol: `unlabeled-image-simsiam-ssl-canary-v1`
- kernel slug: `mahdimzafari/rsna-knee-unlabeled-image-ssl-canary-v1`
- SSL objective: SimSiam (Chen & He, CVPR 2021)
- preprocess: DICOM → percentile/window to [0, 1] → spatial+intensity augs on unnormalized [0, 1] → clamp [0, 1] → ImageNet last
- intensity jitter is never applied to ImageNet-normalized tensors
- encoder init: ImageNet ResNet-18 SHA-256 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`
- series rule: `deterministic-primary-series-v2` (frozen, not retuned)
- cohort: 64 non-gold studies, SHA-256 order, seed 42; do not log IDs
- gold-58 intersection must be 0 before pixels open
- train exactly 40 optimizer steps at frozen batch_size=8
- force checkpoint at step 20; restore in a fresh runtime section; continue to 40
- never continue into a full SSL run
- collapse thresholds frozen before execution: encoder/projector/predictor std min 0.001; mean_abs_cosine_max=0.99; effective_rank_min=1.5
- pass: `SSL_CANARY_PASS_FULL_RUN_NOT_AUTHORIZED`
- fail: `SSL_CANARY_FAIL`
- config SHA-256: `3c8ad505be9e2c71c1ad3c15f894d7188f0ca1f8cf1752ad4df7d84d7ce25c4c`
- UNLABELED_IMAGE_SSL_EXECUTION_ALLOWED=false
- GOLD58_SSL_OOF_EXECUTION_ALLOWED=false
- HARD_WEAK_LABEL_TRAINING_ALLOWED=false
- no `/kaggle/working/submission.csv`
- forbidden outcome not used: `CONSTRAINED_FORMAT_PASS_SEMANTIC_PASS`
