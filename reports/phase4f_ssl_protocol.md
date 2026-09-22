# Phase 4F image-only SimSiam SSL protocol

Prepared, not executed on real unlabeled pixels.

- protocol: `unlabeled-image-simsiam-ssl-v1`
- kernel slug (prepared, not launched): `mahdimzafari/rsna-knee-unlabeled-image-ssl-v1`
- SSL objective: SimSiam (Chen & He, CVPR 2021)
- rationale: Tesla T4 16 GiB; gold58 v2 ResNet-18 training peaked at
  404743680 bytes with batch 2. SimSiam uses two views,
  a projector, and a predictor with stop-gradient. It needs no negative
  queue (SimCLR) and no momentum encoder (MoCo/BYOL), so peak memory
  stays compatible with T4. Frozen batch_size=8 (two views) is expected
  well under 16 GiB. SimCLR is rejected for large-batch negatives.
- encoder init: ImageNet ResNet-18 SHA-256 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`
- series rule: `deterministic-primary-series-v2` (frozen, not retuned)
- cohort: 4349 non-gold released training studies; gold 58 excluded
- never load Report, target columns, weak labels, second-reader files,
  or hidden-test paths into the dataset or trainer
- projector: 3-layer MLP 512→2048→2048→2048 with BN (tiny smoke: 8→16)
- predictor: 2-layer MLP 2048→512→2048 (tiny smoke: 16→8→16)
- augmentations: study-consistent random resized crop scale [0.6, 1.0],
  hflip p=0.5, intensity scale [0.9, 1.1], shift [-0.05, 0.05]
- crop size / slices: 224, 8 linspace slices, bilinear, ImageNet norm
- optimizer: SGD lr=0.03 momentum=0.9 weight_decay=1e-4 cosine, epochs=2
- AMP: cuda_only; seed=42; checkpoint every 200 steps plus last; resume true
- collapse: mean per-dim std of z < 0.001 fails closed
- config SHA-256: `d0c714f2ec1dc818f140abee21464cc4cf8a25f9a2a882195ae6860fda48ce43`

## Resource estimate (from frozen v2/v4E evidence, not a new measurement)

- seconds/study unlabeled estimate: 3.5207
- projected seconds / epoch for n=4349: 15311.7
- projected total (2 epochs): 30623.4
- with safety 1.2: 36748.0
- fits max_wall 39600s: True
- fits competition 9h scoring: False
  (SSL is not a scoring notebook)
- checkpoint bytes estimate: 62545274
- Throughput is scaled from gold58 v2 wall 8168.106s over ~3480 study presentations, times 1.5 for two SSL views. DICOM I/O for 4349 studies is not separately measured. Pixel cache is not used: float32 grayscale 8x224 cache would be about 1.75 GiB and is not predeclared. Checkpoint size is encoder (~41.6 MiB from 4D.2 fold mean) plus projector/predictor (~20 MiB).

## Later OOF (not authorized, not run)

- identical gold58 v2 folds, bilinear 224, ImageNet preprocess, 8 slices,
  mean aggregation, Adam 1e-4 cosine 12 epochs, official combined OOF
  macro ROC AUC
- compare SSL-v1 OOF to frozen v2 with paired patient bootstrap
- no target-specific weights, sign flips, or OOF-derived calibration
- `GOLD58_SSL_OOF_EXECUTION_ALLOWED=false`

## Gates

- UNLABELED_IMAGE_SSL_PLANNING_ALLOWED=true
- UNLABELED_IMAGE_SSL_EXECUTION_ALLOWED=false
- GOLD58_SSL_OOF_EXECUTION_ALLOWED=false
- HARD_WEAK_LABEL_TRAINING_ALLOWED=false
- no `/kaggle/working/submission.csv`
- forbidden outcome not used: `CONSTRAINED_FORMAT_PASS_SEMANTIC_PASS`
