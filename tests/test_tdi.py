"""Tests for dmipy_tract.filtering.tdi.compute_tdi."""

import numpy as np
import pytest

from dmipy_tract.filtering.tdi import compute_tdi


@pytest.fixture()
def identity_affine():
    return np.eye(4)


def test_tdi_shape(identity_affine):
    """Output shape must match vol_shape."""
    vol_shape = (10, 10, 10)
    rng = np.random.default_rng(0)
    streamlines = [rng.uniform(1, 8, (15, 3)) for _ in range(4)]
    tdi = compute_tdi(streamlines, identity_affine, vol_shape)
    assert tdi.shape == vol_shape


def test_tdi_nonzero(identity_affine):
    """A streamline passing through known voxels produces nonzero TDI there."""
    vol_shape = (10, 10, 10)
    # Streamline along x-axis from voxel (1,5,5) to (8,5,5)
    sl = np.array([[float(x), 5.0, 5.0] for x in range(1, 9)], dtype=np.float64)
    tdi = compute_tdi([sl], identity_affine, vol_shape)
    # Voxels along the path must be nonzero
    for x in range(1, 9):
        assert tdi[x, 5, 5] > 0, f"Expected nonzero TDI at voxel ({x}, 5, 5)"


def test_tdi_length_weighted(identity_affine):
    """Longer segment should produce larger contribution than shorter one."""
    vol_shape = (20, 5, 5)
    # Short segment: 2 mm
    short_sl = np.array([[0.0, 2.0, 2.0], [2.0, 2.0, 2.0]], dtype=np.float64)
    # Long segment: 10 mm
    long_sl = np.array([[0.0, 2.0, 2.0], [10.0, 2.0, 2.0]], dtype=np.float64)

    tdi_short = compute_tdi([short_sl], identity_affine, vol_shape, length_weighted=True)
    tdi_long = compute_tdi([long_sl], identity_affine, vol_shape, length_weighted=True)

    assert tdi_long.sum() > tdi_short.sum()


def test_tdi_not_length_weighted(identity_affine):
    """When length_weighted=False, each segment contributes 1/n_samples per voxel."""
    vol_shape = (10, 5, 5)
    sl = np.array([[0.0, 2.0, 2.0], [5.0, 2.0, 2.0]], dtype=np.float64)
    tdi = compute_tdi([sl], identity_affine, vol_shape, length_weighted=False)
    # Total contributions should sum to ~1.0 (one segment)
    assert tdi.sum() == pytest.approx(1.0, abs=1e-9)


def test_tdi_empty_streamlines(identity_affine):
    """Empty streamline list returns all-zero TDI."""
    vol_shape = (5, 5, 5)
    tdi = compute_tdi([], identity_affine, vol_shape)
    assert tdi.shape == vol_shape
    assert np.all(tdi == 0.0)


def test_tdi_single_point_streamline(identity_affine):
    """Single-point streamline (no segments) produces zero TDI."""
    vol_shape = (5, 5, 5)
    sl = np.array([[2.0, 2.0, 2.0]])
    tdi = compute_tdi([sl], identity_affine, vol_shape)
    assert np.all(tdi == 0.0)
