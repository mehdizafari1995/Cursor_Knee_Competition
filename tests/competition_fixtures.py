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
    manufacturer: str = "SIEMENS",
    manufacturer_model_name: str = "Skyra",
    magnetic_field_strength: float = 3.0,
    echo_time: float = 30.0,
    repetition_time: float = 3000.0,
    body_part_examined: str = "KNEE",
    image_orientation_patient: list[float] | None = None,
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
    ds.Manufacturer = manufacturer
    ds.ManufacturerModelName = manufacturer_model_name
    ds.MagneticFieldStrength = magnetic_field_strength
    ds.EchoTime = echo_time
    ds.RepetitionTime = repetition_time
    ds.BodyPartExamined = body_part_examined
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
    ds.ImageOrientationPatient = image_orientation_patient or [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
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
    series_description: str = "PD FS SAG",
    manufacturer: str = "SIEMENS",
    image_orientation_patient: list[float] | None = None,
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
            series_description=series_description,
            manufacturer=manufacturer,
            image_orientation_patient=image_orientation_patient,
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


# Distinct report strings used only to prove Kaggle audit outputs never leak
# clinical text. These are synthetic fixture sentences, not real reports.
UNIQUE_REPORT_TOKENS = tuple(
    f"UNIQUE_FIXTURE_REPORT_TOKEN_{official.replace(' ', '_').replace(chr(39), '')}_{i:02d}"
    for i, official in enumerate(OFFICIAL_TARGET_COLUMNS)
)

AUDIT_SEQUENCE_SPECS = (
    {
        "series_description": "PD FS SAG",
        "plane": "Sagittal",
        "fluid_sensitive": 1,
        "fat_suppression": 1,
        "orientation": [0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    },
    {
        "series_description": "T1 COR",
        "plane": "Coronal",
        "fluid_sensitive": 0,
        "fat_suppression": 0,
        "orientation": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    },
    {
        "series_description": "T2 AX FS",
        "plane": "Axial",
        "fluid_sensitive": 1,
        "fat_suppression": 1,
        "orientation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    },
)


def _fixed_uid(kind: str, index: int) -> str:
    kind_code = {
        "train_study": 10,
        "train_series": 20,
        "train_series_extra": 21,
        "test_study": 30,
        "test_series": 40,
    }[kind]
    return f"1.2.840.10008.1.{kind_code}.{index}"


def make_kaggle_audit_fixture(root: Path) -> Path:
    """Build a fixture covering all 12 targets, negatives, and several sequences.

    StudyInstanceUID values are fixed so sampling is deterministic across runs.
    Each positive target has a unique report token used by leak tests.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    train_rows = []
    train_series_rows = []

    # Twelve studies, each the first positive example of one official target.
    for i, official in enumerate(OFFICIAL_TARGET_COLUMNS):
        study_uid = _fixed_uid("train_study", i)
        patient_id = f"P{i // 2:03d}"
        row = {
            STUDY_ID_COLUMN: study_uid,
            REPORT_COLUMN: UNIQUE_REPORT_TOKENS[i],
            "PatientID": patient_id,
        }
        for col in OFFICIAL_TARGET_COLUMNS:
            row[col] = int(col == official)
        train_rows.append(row)
        spec = AUDIT_SEQUENCE_SPECS[i % len(AUDIT_SEQUENCE_SPECS)]
        series_uid = _fixed_uid("train_series", i)
        train_series_rows.append(
            {
                STUDY_ID_COLUMN: study_uid,
                "SeriesInstanceUID": series_uid,
                "Fluid_Sensitive": spec["fluid_sensitive"],
                "Fat_Suppression": spec["fat_suppression"],
                "Anatomical_Plane": spec["plane"],
            }
        )
        write_series(
            root / "train_series" / study_uid / series_uid,
            study_uid=study_uid,
            series_uid=series_uid,
            n_slices=4,
            patient_id=patient_id,
            series_description=spec["series_description"],
            image_orientation_patient=spec["orientation"],
        )
        # Extra sequence on the first study so Level C can inspect multiple series.
        if i == 0:
            extra = AUDIT_SEQUENCE_SPECS[1]
            extra_uid = _fixed_uid("train_series_extra", 0)
            train_series_rows.append(
                {
                    STUDY_ID_COLUMN: study_uid,
                    "SeriesInstanceUID": extra_uid,
                    "Fluid_Sensitive": extra["fluid_sensitive"],
                    "Fat_Suppression": extra["fat_suppression"],
                    "Anatomical_Plane": extra["plane"],
                }
            )
            write_series(
                root / "train_series" / study_uid / extra_uid,
                study_uid=study_uid,
                series_uid=extra_uid,
                n_slices=3,
                patient_id=patient_id,
                series_description=extra["series_description"],
                image_orientation_patient=extra["orientation"],
                scramble_filenames=False,
            )

    # All-negative study (no known positives).
    neg_uid = _fixed_uid("train_study", 12)
    neg_patient = "P100"
    neg_row = {
        STUDY_ID_COLUMN: neg_uid,
        REPORT_COLUMN: "UNIQUE_FIXTURE_REPORT_TOKEN_NEGATIVE_12",
        "PatientID": neg_patient,
    }
    for col in OFFICIAL_TARGET_COLUMNS:
        neg_row[col] = 0
    train_rows.append(neg_row)
    neg_spec = AUDIT_SEQUENCE_SPECS[0]
    neg_series = _fixed_uid("train_series", 12)
    train_series_rows.append(
        {
            STUDY_ID_COLUMN: neg_uid,
            "SeriesInstanceUID": neg_series,
            "Fluid_Sensitive": neg_spec["fluid_sensitive"],
            "Fat_Suppression": neg_spec["fat_suppression"],
            "Anatomical_Plane": neg_spec["plane"],
        }
    )
    write_series(
        root / "train_series" / neg_uid / neg_series,
        study_uid=neg_uid,
        series_uid=neg_series,
        n_slices=3,
        patient_id=neg_patient,
        series_description=neg_spec["series_description"],
        image_orientation_patient=neg_spec["orientation"],
    )

    # One train study with an unknown label (NaN) and a missing PatientID, so
    # Level B can report incomplete grouping identifiers.
    unk_uid = _fixed_uid("train_study", 13)
    unk_row = {
        STUDY_ID_COLUMN: unk_uid,
        REPORT_COLUMN: "UNIQUE_FIXTURE_REPORT_TOKEN_UNKNOWN_13",
        "PatientID": "",
    }
    for col in OFFICIAL_TARGET_COLUMNS:
        unk_row[col] = 0
    unk_row["Baker's"] = np.nan
    train_rows.append(unk_row)
    unk_spec = AUDIT_SEQUENCE_SPECS[2]
    unk_series = _fixed_uid("train_series", 13)
    train_series_rows.append(
        {
            STUDY_ID_COLUMN: unk_uid,
            "SeriesInstanceUID": unk_series,
            "Fluid_Sensitive": unk_spec["fluid_sensitive"],
            "Fat_Suppression": unk_spec["fat_suppression"],
            "Anatomical_Plane": unk_spec["plane"],
        }
    )
    write_series(
        root / "train_series" / unk_uid / unk_series,
        study_uid=unk_uid,
        series_uid=unk_series,
        n_slices=2,
        patient_id="",
        series_description=unk_spec["series_description"],
        image_orientation_patient=unk_spec["orientation"],
    )

    pd.DataFrame(train_rows).to_csv(root / "train.csv", index=False)
    pd.DataFrame(train_series_rows).to_csv(root / "train_series.csv", index=False)

    test_studies = [_fixed_uid("test_study", i) for i in range(2)]
    test_rows = [{STUDY_ID_COLUMN: uid} for uid in test_studies]
    test_series_rows = []
    for i, study_uid in enumerate(test_studies):
        series_uid = _fixed_uid("test_series", i)
        spec = AUDIT_SEQUENCE_SPECS[i % len(AUDIT_SEQUENCE_SPECS)]
        test_series_rows.append(
            {
                STUDY_ID_COLUMN: study_uid,
                "SeriesInstanceUID": series_uid,
                "Fluid_Sensitive": spec["fluid_sensitive"],
                "Fat_Suppression": spec["fat_suppression"],
                "Anatomical_Plane": spec["plane"],
            }
        )
        write_series(
            root / "test_series" / study_uid / series_uid,
            study_uid=study_uid,
            series_uid=series_uid,
            n_slices=3,
            patient_id=f"P9{i:02d}",
            series_description=spec["series_description"],
            image_orientation_patient=spec["orientation"],
            scramble_filenames=False,
        )
    pd.DataFrame(test_rows).to_csv(root / "test.csv", index=False)
    pd.DataFrame(test_series_rows).to_csv(root / "test_series.csv", index=False)

    sample = pd.DataFrame({STUDY_ID_COLUMN: test_studies})
    for col in OFFICIAL_TARGET_COLUMNS:
        sample[col] = 0.5
    sample = sample.loc[:, list(SAMPLE_SUBMISSION_COLUMNS)]
    sample.to_csv(root / "sample_submission.csv", index=False)
    return root
