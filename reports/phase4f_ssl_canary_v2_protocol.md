# Phase 4F.2 AMP-safe unlabeled-image SimSiam SSL canary protocol

Bounded real-data canary only. The full 4,349-study SSL job is not authorized.
Phase 4F v1 config, report, notebook 13, and synthetic results stay frozen.
Phase 4F.1 config, notebooks 14, source hashes, and both failed kernel versions
stay frozen (`SSL_CANARY_FAIL` and `SSL_CANARY_GPU_PREFLIGHT_FAIL`).

- protocol: `unlabeled-image-simsiam-ssl-canary-v2`
- kernel slug: `mahdimzafari/rsna-knee-unlabeled-image-ssl-canary-v2`
- SSL objective: SimSiam (Chen & He, CVPR 2021)
- preprocess: DICOM → percentile/window to [0, 1] → spatial+intensity augs on unnormalized [0, 1] → clamp [0, 1] → ImageNet last
- intensity jitter is never applied to ImageNet-normalized tensors
- encoder init: ImageNet ResNet-18 SHA-256 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`
- series rule: `deterministic-primary-series-v2` (frozen, not retuned)
- cohort: 64 non-gold studies, SHA-256 order, seed 42; do not log IDs
- gold-58 intersection must be 0 before pixels open
- train exactly 40 applied optimizer steps at frozen batch_size=8
- force checkpoint at applied step 20; restore in a fresh runtime section; continue to 40
- never continue into a full SSL run
- collapse thresholds frozen before execution: encoder/projector/predictor std min 0.001; mean_abs_cosine_max=0.99; effective_rank_min=1.5
- per-step diagnostics: encoder/projector/predictor std only; no cosine, covariance, SVD, or eigendecomposition during optimizer steps
- probe cosine and effective rank run once after the training segment
- effective rank uses the float32 n×n centered Gram `X @ X.T / (n-1)` (probe n=16), not 2048×2048 feature covariance
- GPU AMP preflight before train.csv / series / DICOM: Tesla T4, ImageNet hash, synthetic ResNet-18 AMP step on scientific shape 8x8x1x224x224, AMP-safe float32 SimSiam loss, GradScaler init_scale=1024, std-only on AMP outputs, probe eigvalsh float32 dim<=16; fail `SSL_CANARY_GPU_PREFLIGHT_FAIL` with zero real pixels if any check fails
- AMP-safe loss: predictor and stop-grad projector in float32; cosine/mean in float32 with eps 1e-6; frozen Phase 4F v1 `simsiam_loss` is not called
- an optimizer step counts only when parameters were actually updated; scheduler steps only after an applied optimizer step
- AMP overflow: restore aug-generator, skip sampler/optimizer/scheduler, at most 3 consecutive backoffs, then fail closed
- pass: `SSL_CANARY_PASS_FULL_RUN_NOT_AUTHORIZED`
- fail: `SSL_CANARY_FAIL`
- GPU preflight fail: `SSL_CANARY_GPU_PREFLIGHT_FAIL`
- frozen Phase 4F v1 config SHA-256: `d0c714f2ec1dc818f140abee21464cc4cf8a25f9a2a882195ae6860fda48ce43`
- frozen Phase 4F.1 canary config SHA-256: `3c8ad505be9e2c71c1ad3c15f894d7188f0ca1f8cf1752ad4df7d84d7ce25c4c`
- config SHA-256: `c06e0a78848b20e5ab9c675f3e0df8c4a82aa86519408729b1149d86ab8afb26`
- UNLABELED_IMAGE_SSL_EXECUTION_ALLOWED=false
- GOLD58_SSL_OOF_EXECUTION_ALLOWED=false
- HARD_WEAK_LABEL_TRAINING_ALLOWED=false
- no `/kaggle/working/submission.csv`
- forbidden outcome not used: `CONSTRAINED_FORMAT_PASS_SEMANTIC_PASS`
