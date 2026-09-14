"""Official RSNA Knee Abnormality Detection schema, transcribed from the Kaggle data page.

Nothing here is an observation of local files. Counts, prevalence, and DICOM
properties are *not* recorded in this module. When local files are available the
auditor compares them against these documented names and reports mismatches.
"""

from __future__ import annotations

from .config import (
    OFFICIAL_TARGET_COLUMNS,
    REPORT_COLUMN,
    SERIES_ID_COLUMN,
    STUDY_ID_COLUMN,
)

# https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data
KAGGLE_DATA_PAGE = "https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data"
KAGGLE_OVERVIEW_PAGE = (
    "https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview"
)

DOCUMENTED_FILES = (
    "train.csv",
    "train_series.csv",
    "test.csv",
    "test_series.csv",
    "sample_submission.csv",
)

DOCUMENTED_DIRECTORIES = (
    "train_series",
    "test_series",
)

# Documented as: train_series/<StudyInstanceUID>/<SeriesInstanceUID>/*.dcm
DOCUMENTED_DICOM_LAYOUT = (
    "{split}_series/{StudyInstanceUID}/{SeriesInstanceUID}/*.dcm"
)

DOCUMENTED_PLANES = ("Axial", "Coronal", "Sagittal")

DOCUMENTED_SERIES_COLUMNS = (
    STUDY_ID_COLUMN,
    SERIES_ID_COLUMN,
    "Fluid_Sensitive",
    "Fat_Suppression",
    "Anatomical_Plane",
)

# The data page explicitly names these two train.csv fields. The twelve binary
# label names are documented on the same page as the dataset's findings, but
# they are *not* listed in the train.csv field bullet list. Treat their presence
# as something the auditor must verify, not as a hard-coded fact.
DOCUMENTED_TRAIN_CSV_FIELDS = (STUDY_ID_COLUMN, REPORT_COLUMN)
DOCUMENTED_TEST_CSV_FIELDS = (STUDY_ID_COLUMN,)
DOCUMENTED_LABEL_COLUMN_CANDIDATES = OFFICIAL_TARGET_COLUMNS

SAMPLE_SUBMISSION_COLUMNS = (STUDY_ID_COLUMN, *OFFICIAL_TARGET_COLUMNS)

GROUP_ID_CANDIDATES = (
    "PatientID",
    "patient_id",
    "patientId",
    STUDY_ID_COLUMN,
)
