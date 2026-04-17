"""Tests for dmipy_tract.filtering.sift.SIFTFilter (Phase 3 implementation)."""

import numpy as np
import pytest

from dmipy_tract.filtering import SIFTFilter


@pytest.fixture()
def identity_affine():
    return np.eye(4)


@pytest.fixture()
def sift():
    return SIFTFilter()


def _make_streamlines_in_box(n: int, vol_shape=(10, 10, 10), seed=0):
    """Make n streamlines that travel along the x-axis within the volume."""
    rng = np.random.default_rng(seed)
    streamlines = []
    for _ in range(n):
        y = rng.uniform(2, 7)
        z = rng.uniform(2, 7)
        pts = np.column_stack([
            np.linspace(1.0, 8.0, 10),
            np.full(10, y),
            np.full(10, z),
        ])
        streamlines.append(pts.astype(np.float64))
    return streamlines


def test_sift_filter_reduces_streamlines(sift, identity_affine):
    """Filtering with a uniformly low target_density removes some streamlines."""
    vol_shape = (10, 10, 10)
    streamlines = _make_streamlines_in_box(10, vol_shape)
    # Very low target density forces aggressive removal
    target_density = np.zeros(vol_shape, dtype=np.float64)
    filtered = sift.filter(streamlines, identity_affine, target_density)
    assert len(filtered) < len(streamlines), (
        "Expected some streamlines to be removed when target density is zero"
    )


def test_sift_filter_preserves_all_when_target_high(sift, identity_affine):
    """When target_density is very large everywhere, no streamlines are removed."""
    vol_shape = (10, 10, 10)
    streamlines = _make_streamlines_in_box(5, vol_shape)
    # Very high target — current TDI is always below target, removing any
    # streamline increases the cost
    target_density = np.full(vol_shape, 1e9, dtype=np.float64)
    filtered = sift.filter(streamlines, identity_affine, target_density)
    assert len(filtered) == len(streamlines), (
        "Expected all streamlines preserved when target density is very high"
    )


def test_sift_filter_empty_streamlines(sift, identity_affine):
    """Empty input returns empty list."""
    vol_shape = (5, 5, 5)
    target_density = np.zeros(vol_shape, dtype=np.float64)
    filtered = sift.filter([], identity_affine, target_density)
    assert filtered == []


def test_sift2_weights_shape(sift, identity_affine):
    """weights() returns array of length N_streamlines."""
    vol_shape = (10, 10, 10)
    streamlines = _make_streamlines_in_box(8, vol_shape)
    target_density = np.ones(vol_shape, dtype=np.float64) * 0.5
    w = sift.weights(streamlines, identity_affine, target_density)
    assert w.shape == (8,)


def test_sift2_weights_nonneg(sift, identity_affine):
    """All SIFT2 weights must be non-negative."""
    vol_shape = (10, 10, 10)
    streamlines = _make_streamlines_in_box(6, vol_shape)
    target_density = np.ones(vol_shape, dtype=np.float64) * 0.3
    w = sift.weights(streamlines, identity_affine, target_density)
    assert np.all(w >= 0.0), f"Found negative weights: {w[w < 0]}"


def test_sift2_weights_empty(sift, identity_affine):
    """weights() with empty streamlines returns empty array."""
    vol_shape = (5, 5, 5)
    target_density = np.zeros(vol_shape, dtype=np.float64)
    w = sift.weights([], identity_affine, target_density)
    assert w.shape == (0,)


def test_sift_instantiation_no_model():
    """SIFTFilter can be instantiated without a fitted_model."""
    f = SIFTFilter()
    assert f.fitted_model is None
    assert f.target_cf == pytest.approx(0.1)


def test_sift_instantiation_with_model():
    """SIFTFilter stores fitted_model and target_cf correctly."""
    class MockModel:
        pass

    f = SIFTFilter(fitted_model=MockModel(), target_cf=0.05)
    assert isinstance(f.fitted_model, MockModel)
    assert f.target_cf == pytest.approx(0.05)
