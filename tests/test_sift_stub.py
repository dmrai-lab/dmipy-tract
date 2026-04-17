"""Tests for dmipy_tract.filtering.sift.SIFTFilter stub."""

import numpy as np
import pytest

from dmipy_tract.filtering import SIFTFilter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _MockFittedModel:
    """Minimal stand-in for a FittedMultiCompartmentModel."""
    pass


@pytest.fixture()
def sift_filter():
    return SIFTFilter(fitted_model=_MockFittedModel(), target_cf=0.05)


@pytest.fixture()
def dummy_streamlines():
    rng = np.random.default_rng(42)
    return [rng.standard_normal((10, 3)) for _ in range(5)]


@pytest.fixture()
def identity_affine():
    return np.eye(4)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_instantiation(sift_filter):
    """SIFTFilter should instantiate and store parameters."""
    assert isinstance(sift_filter.fitted_model, _MockFittedModel)
    assert sift_filter.target_cf == pytest.approx(0.05)


def test_filter_raises_not_implemented(sift_filter, dummy_streamlines, identity_affine):
    """filter() must raise NotImplementedError with the expected message."""
    with pytest.raises(NotImplementedError, match="SIFT filtering requires per-voxel"):
        sift_filter.filter(dummy_streamlines, identity_affine)


def test_weights_raises_not_implemented(sift_filter, dummy_streamlines, identity_affine):
    """weights() must raise NotImplementedError with the expected message."""
    with pytest.raises(NotImplementedError, match="SIFT2 weighting requires per-voxel"):
        sift_filter.weights(dummy_streamlines, identity_affine)


def test_default_target_cf():
    """Default target_cf should be 0.1."""
    f = SIFTFilter(fitted_model=_MockFittedModel())
    assert f.target_cf == pytest.approx(0.1)
