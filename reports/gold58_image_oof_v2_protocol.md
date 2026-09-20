# Phase 4D.2 gold58 image-only grouped OOF protocol v2

Predeclared before execution. No hyperparameter search.
Does not edit v1 history. Frozen v1 kernel/config/report stay unchanged.
The second reader is not on this path.

- protocol: `gold58-image-resnet18-grouped-oof-v2`
- kernel: `mahdimzafari/rsna-knee-gold58-image-oof-v2`
- encoder: ImageNet ResNet-18 (`resnet18-f37072fd.pth`)
- ImageNet SHA-256: `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`
- slices: 8 linspace; resize 224 bilinear; percentiles [1.0, 99.0]
- ImageNet normalize after RGB repeat: mean [0.485, 0.456, 0.406] std [0.229, 0.224, 0.225]
- augmentation: `none`
- optimizer: adam lr=0.0001 weight_decay=0.0001
- schedule: cosine T_max=12
- epochs: 12; checkpoint: `last_epoch`
- loss: masked BCE; unknown stays NaN
- AMP: `cuda_only`; cudnn_deterministic: `True`
- folds: 5; seed 42; PatientID from all readable series; no StudyInstanceUID fallback
- series rule: `deterministic-primary-series-v2` (tiered ranking; exactly one readable series per labelled study)
- fold gate: every training partition has known positive and known negative for all 12 targets
- primary metric: combined 58-row OOF macro ROC AUC (not a mean of fold AUCs)
- bootstrap: patient-level percentile, n=1000, seed=42
- no report-only rows, no weak labels, no second reader, no hidden test, no submission
