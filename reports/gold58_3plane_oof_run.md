# Gold-58 three-plane fusion OOF — kernel v1 worker ERROR

Protocol `gold58-image-resnet18-3plane-oof-v1`. One authorized private
Tesla T4 internet-off kernel ran. Frozen gold58 v2 remains the
score-bearing comparator. Frozen SSL kernels were not overwritten. A
replacement 3plane kernel was not launched.

UNLABELED_IMAGE_SSL_EXECUTION_ALLOWED=false
GOLD58_3PLANE_OOF_EXECUTION_ALLOWED=true
SSL_ENGINEERING_PATH_PARKED=true

## Kernel

- slug: `mahdimzafari/rsna-knee-gold58-3plane-oof-v1`
- version: 1
- worker: `KernelWorkerStatus.ERROR`
- status: `GOLD58_3PLANE_TRAINING_FAIL`
- skip_reason: `worker_ERROR_empty_logs_and_outputs`
- failureMessage: unset
- GPU requested: Tesla T4; internet off; competition attached; notebook 18
- public JSON: not written
- output files: 0
- logs: empty
- `n_labelled_studies` processed: unknown (no sanitized payload)
- `n_report_only_studies_processed`: 0
- `n_report_text_rows_processed`: 0
- `n_second_reader_rows_processed`: 0
- training_run: false (no evidence of a completed fold)
- OOF_run: false
- inference_run: false
- competition_submission_written: false
- consumes_submission_slot: false
- kaggle_kernel_launched: true
- replacement kernel: not launched
- notebooks 07, 09, 10, 11, 12, 13, 14, 15, 16, and 17 were not executed
- SSL canary-v4 kernel was never created
- comparator frozen v2 macro ROC AUC: 0.6237496021463534
- packaged source HEAD: `384ea82691eb8a0b23b83ac2cc82e22205536ad1`
- source zip sha256: `28108bab3cf78f289317c6d082e4553aa94f67c7addc18b49aa7f6d6b2319c5f` (135 files)
