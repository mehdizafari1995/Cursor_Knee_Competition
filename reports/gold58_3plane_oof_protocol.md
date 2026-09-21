# Gold-58 image-only three-plane fusion OOF protocol v1

Prepared after SimSiam SSL was parked. Not launched.
Frozen v2 primary-series OOF remains the comparator.

- protocol: `gold58-image-resnet18-3plane-oof-v1`
- kernel slug (not launched): `mahdimzafari/rsna-knee-gold58-3plane-oof-v1`
- encoder: ImageNet ResNet-18; shared weights across planes
- ImageNet SHA-256: `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`
- planes: sagittal, coronal, axial
- series rule: `deterministic-one-series-per-plane-v1` (one readable series per plane; fluid-sensitive first, then DICOM count, description, SeriesInstanceUID)
- missing plane: average available planes; min_planes_required=1
- fusion: mean of per-plane encoder embeddings, then the existing head
- slices/preprocess/optimizer/folds: copied from frozen v2, not retuned
- comparator: frozen `deterministic-primary-series-v2` OOF macro AUC 0.6237496021463534
- comparator config SHA-256: `4b65bc0d4d56a3dd36b6898c71d630a113c99f6e4b986e51a15ae1b66e1ea093`
- leakage-safe validation: same 5-fold PatientID grouped OOF as v2
- resource estimate: 3 × frozen v2 wall 8168.106 s = 24504.3 s projected; guarded ×1.25 = 30630.4 s; plus 3600 s reserve = 34230.4 s planned (~9.5 h Tesla T4); fits_private_wall=true (limit 39600 s)
- peak GPU: sequential per-plane forwards; expected near frozen v2 404743680 bytes
- no report-only rows, no weak labels, no second reader, no hidden test, no submission
- GOLD58_3PLANE_OOF_EXECUTION_ALLOWED=false
- UNLABELED_IMAGE_SSL_EXECUTION_ALLOWED=false
- HARD_WEAK_LABEL_TRAINING_ALLOWED=false
- SSL_ENGINEERING_PATH_PARKED=true
- config SHA-256: `57c8c6f3b8d6f951b9a00ee4e90028fbd1d61abcd68bc3e6c4009b1af5123ec8`
- forbidden outcome not used: `CONSTRAINED_FORMAT_PASS_SEMANTIC_PASS`
