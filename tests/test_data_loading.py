"""DICOM loading, slice order, percentile normalisation, slice sampling."""

from __future__ import annotations

import numpy as np

from knee.dicom_io import infer_plane_from_orientation, load_series_volume
from knee.preprocess import percentile_normalize, sample_slice_indices, sample_slices
from tests.competition_fixtures import write_series


def test_slice_order_follows_image_position_not_filename(tmp_path):
    series_dir = tmp_path / "series"
    write_series(
        series_dir,
        study_uid="1.2.3",
        series_uid="1.2.3.4",
        n_slices=5,
        scramble_filenames=True,
    )
    volume, headers = load_series_volume(series_dir)
    assert volume.shape[0] == 5
    from pydicom import dcmread

    positions = []
    for header in headers:
        ds = dcmread(header["path"], stop_before_pixels=True)
        positions.append(float(ds.ImagePositionPatient[2]))
    assert positions == sorted(positions)
    # Filename order was reversed, so first file on disk is last geometrically.
    names = [h["path"] for h in headers]
    assert names[0].endswith("0005.dcm")
    assert names[-1].endswith("0001.dcm")


def test_percentile_normalize_range_and_constant():
    rng = np.random.default_rng(0)
    volume = rng.normal(100, 25, size=(4, 8, 8)).astype(np.float32)
    volume[0, 0, 0] = 10_000  # outlier that percentile clip should suppress
    out = percentile_normalize(volume, 1, 99)
    assert out.min() >= 0.0
    assert out.max() <= 1.0
    constant = percentile_normalize(np.ones((2, 4, 4), dtype=np.float32))
    assert np.all(constant == 0)


def test_sample_slice_indices_covers_ends():
    idx = sample_slice_indices(10, 4)
    assert idx[0] == 0
    assert idx[-1] == 9
    assert len(idx) == 4
    short = sample_slice_indices(3, 8)
    assert len(short) == 8
    assert short.min() == 0
    assert short.max() == 2


def test_sample_slices_shape():
    volume = np.arange(5 * 2 * 2, dtype=np.float32).reshape(5, 2, 2)
    sampled = sample_slices(volume, 3)
    assert sampled.shape == (3, 2, 2)


def test_infer_axial_from_orientation():
    # Standard axial: row = [1,0,0], col = [0,1,0] => normal = [0,0,1]
    assert infer_plane_from_orientation((1, 0, 0, 0, 1, 0)) == "Axial"
    assert infer_plane_from_orientation((1, 0, 0, 0, 0, 1)) == "Coronal"
    assert infer_plane_from_orientation((0, 1, 0, 0, 0, 1)) == "Sagittal"
