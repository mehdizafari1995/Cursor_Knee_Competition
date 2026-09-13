"""RSNA Knee Abnormality Detection: a compact, end-to-end multi-label knee MRI
finding-detection pipeline used to validate the dev environment.

Predicts twelve study-level binary findings and reports the competition metric,
macro-averaged ROC AUC. Training may use multilingual radiology reports for weak
supervision; test-time inference uses MRI images only."""

__version__ = "0.1.0"
