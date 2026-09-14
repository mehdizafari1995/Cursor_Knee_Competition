"""Software-only competition-schema fixtures. Never used as scientific audit input."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from knee.competition import SAMPLE_SUBMISSION_COLUMNS
from knee.config import OFFICIAL_TARGET_COLUMNS, REPORT_COLUMN, STUDY_ID_COLUMN


def write_mr_slice(
    path: Path,
    *,
    study_uid: str,
    series_uid: str,
    instance_number: int,
    z: float,
    rows: int = 8,
    cols: int = 8,
    patient_id: str = "P001",
    series_description: str = "PD FS SAG",
    seed: int = 0,
) -> None:
    """Write an uncompressed MR instance. Pixel values are synthetic."""
    rng = np.random.default_rng(seed + instance_number)
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.Modality = "MR"
    ds.PatientID = patient_id
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.InstanceNumber = instance_number
    ds.SeriesDescription = series_description
    ds.ProtocolName = "KNEE"
    ds.ScanningSequence = "SE"
    ds.Rows = rows
    ds.Columns = cols
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelSpacing = [0.5, 0.5]
    ds.SliceThickness = 3.0
    ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    ds.ImagePositionPatient = [0.0, 0.0, float(z)]
    pixels = rng.integers(0, 1000, size=(rows, cols), dtype=np.uint16)
    ds.PixelData = pixels.tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(path, write_like_original=False)


def write_series(
    series_dir: Path,
    *,
    study_uid: str,
    series_uid: str,
    n_slices: int = 5,
    patient_id: str = "P001",
    scramble_filenames: bool = True,
) -> None:
    """Write ``n_slices`` instances. Filenames can be out of geometric order."""
    order = list(range(n_slices))
    if scramble_filenames:
        order = list(reversed(order))
    for filename_i, instance in enumerate(order, start=1):
        write_mr_slice(
            series_dir / f"{filename_i:04d}.dcm",
            study_uid=study_uid,
            series_uid=series_uid,
            instance_number=instance + 1,
            z=float(instance * 3.0),
            patient_id=patient_id,
            seed=instance,
        )


def make_mini_competition_root(root: Path) -> Path:
    """Build a tiny tree that follows the *documented* Kaggle layout.

    Two patients, four train studies, two test studies. Labels use official
    column spelling, including ``Baker's``. One train label is left unknown.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    train_studies = [
        (generate_uid(), "P001", "ACL tear. No fracture."),
        (generate_uid(), "P001", "Rotura del menisco medial."),
        (generate_uid(), "P002", "Rupture du LCA. Epanchement articulaire."),
        (generate_uid(), "P002", "Fractura. Contusion osea."),
    ]
    test_studies = [
        (generate_uid(), "P003"),
        (generate_uid(), "P004"),
    ]

    train_rows = []
    train_series_rows = []
    rng = np.random.default_rng(0)
    for study_uid, patient_id, report in train_studies:
        row = {STUDY_ID_COLUMN: study_uid, REPORT_COLUMN: report, "PatientID": patient_id}
        for i, col in enumerate(OFFICIAL_TARGET_COLUMNS):
            row[col] = int(rng.random() < 0.5)
        train_rows.append(row)
        series_uid = generate_uid()
        train_series_rows.append(
            {
                STUDY_ID_COLUMN: study_uid,
                "SeriesInstanceUID": series_uid,
                "Fluid_Sensitive": 1,
                "Fat_Suppression": 1,
                "Anatomical_Plane": "Sagittal",
            }
        )
        write_series(
            root / "train_series" / study_uid / series_uid,
            study_uid=study_uid,
            series_uid=series_uid,
            n_slices=4,
            patient_id=patient_id,
        )

    train_rows[0]["Baker's"] = np.nan  # unknown label, tests masked loss / prevalence
    pd.DataFrame(train_rows).to_csv(root / "train.csv", index=False)
    pd.DataFrame(train_series_rows).to_csv(root / "train_series.csv", index=False)

    test_rows = [{STUDY_ID_COLUMN: uid} for uid, _pid in test_studies]
    test_series_rows = []
    for study_uid, patient_id in test_studies:
        series_uid = generate_uid()
        test_series_rows.append(
            {
                STUDY_ID_COLUMN: study_uid,
                "SeriesInstanceUID": series_uid,
                "Fluid_Sensitive": 0,
                "Fat_Suppression": 0,
                "Anatomical_Plane": "Coronal",
            }
        )
        write_series(
            root / "test_series" / study_uid / series_uid,
            study_uid=study_uid,
            series_uid=series_uid,
            n_slices=3,
            patient_id=patient_id,
            scramble_filenames=False,
        )
    pd.DataFrame(test_rows).to_csv(root / "test.csv", index=False)
    pd.DataFrame(test_series_rows).to_csv(root / "test_series.csv", index=False)

    sample = pd.DataFrame({STUDY_ID_COLUMN: [uid for uid, _ in test_studies]})
    for col in OFFICIAL_TARGET_COLUMNS:
        sample[col] = 0.5
    sample = sample.loc[:, list(SAMPLE_SUBMISSION_COLUMNS)]
    sample.to_csv(root / "sample_submission.csv", index=False)
    return root
