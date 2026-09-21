# Phase 4F.3 documented-AMP unlabeled-image SimSiam SSL canary protocol

Bounded real-data canary only. The full 4,349-study SSL job is not authorized.
Phase 4F v1, Phase 4F.1, and Phase 4F.2 configs, reports, notebooks 13/14/15,
and failed kernels stay frozen.


- protocol: `unlabeled-image-simsiam-ssl-canary-v3`
- kernel slug: `mahdimzafari/rsna-knee-unlabeled-image-ssl-canary-v3`
- SSL objective: SimSiam (Chen & He, CVPR 2021)
- preprocess: DICOM → percentile/window to [0, 1] → spatial+intensity
  augs on unnormalized [0, 1] → clamp [0, 1] → ImageNet last
- intensity jitter is never applied to ImageNet-normalized tensors
- encoder init: ImageNet ResNet-18 SHA-256 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`
- series rule: `deterministic-primary-series-v2` (frozen, not retuned)
- cohort: 64 non-gold studies, SHA-256 order, seed 42; do not log IDs
- gold-58 intersection must be 0 before pixels open
- train exactly 40 optimizer steps at frozen batch_size=8
- force checkpoint at step 20; restore in a fresh runtime section; continue to 40
- never continue into a full SSL run
- collapse thresholds frozen: encoder/projector/predictor std min 0.001/0.001/0.001; mean_abs_cosine_max=0.99; effective_rank_min=1.5
- per-step diagnostics: encoder/projector/predictor std only; no cosine,
  covariance, SVD, or eigendecomposition during optimizer steps
- probe cosine and effective rank run once after the training segment
- effective rank uses the float32 n×n centered Gram `X @ X.T / (n-1)`
  (probe n=16), not 2048×2048 feature covariance
- GPU preflight before train.csv / series / DICOM: Tesla T4, ImageNet
  hash, synthetic ResNet-18 on scientific shape 8x8x1x224x224,
  documented AMP order scale→unscale_→step→update, AMP-safe float32
  SimSiam loss, GradScaler init_scale=1024; if AMP cannot apply one
  finite step after 3 backoffs, reset and retry the same exact-shape
  preflight in pure FP32 (`precision_mode=fp32_fallback`); fail
  `SSL_CANARY_GPU_PREFLIGHT_FAIL` with full exception diagnostics and
  zero real pixels if both modes fail
- AMP-safe loss: predictor and stop-grad projector in float32; cosine/mean
  in float32 with eps 1e-6; frozen Phase 4F v1 `simsiam_loss` is not called
- an optimizer step counts only when a parameter-delta sentinel changes
  and AMP scale is not reduced; scheduler steps only after an applied update
- AMP overflow: restore model+BN and aug-generator, retain reduced scale,
  skip sampler/applied-count/scheduler, at most 3 consecutive backoffs
- exceptions record stage, class, sanitized message, traceback fingerprint,
  torch/CUDA/GPU, GradScaler state, and tensor shapes/dtypes
- pass: `SSL_CANARY_PASS_FULL_RUN_NOT_AUTHORIZED`
- fail: `SSL_CANARY_FAIL`
- GPU preflight fail: `SSL_CANARY_GPU_PREFLIGHT_FAIL`
- park: `SSL_ENGINEERING_PATH_PARKED` if both numerical modes fail
- config SHA-256: `7f8eb3b0143441d96bf9cb445ecf12a2d22bfdae6f1d9a49f3603ab41374cb4d`
- frozen Phase 4F v1 config SHA-256: `d0c714f2ec1dc818f140abee21464cc4cf8a25f9a2a882195ae6860fda48ce43`
- frozen Phase 4F.1 canary config SHA-256: `3c8ad505be9e2c71c1ad3c15f894d7188f0ca1f8cf1752ad4df7d84d7ce25c4c`
- frozen Phase 4F.2 canary config SHA-256: `c06e0a78848b20e5ab9c675f3e0df8c4a82aa86519408729b1149d86ab8afb26`
- UNLABELED_IMAGE_SSL_EXECUTION_ALLOWED=false
- GOLD58_SSL_OOF_EXECUTION_ALLOWED=false
- HARD_WEAK_LABEL_TRAINING_ALLOWED=false
- no `/kaggle/working/submission.csv`
- forbidden outcome not used: `CONSTRAINED_FORMAT_PASS_SEMANTIC_PASS`