"""Tests for dmipy_tract.filtering.sift.SIFTFilter basic properties.

These tests cover instantiation and API surface only.  Functional tests
are in test_sift.py.
"""

import numpy as np
import pytest

from dmipy_tract.filtering import SIFTFilter


class _MockFittedModel:
    """Minimal stand-in for a FittedMultiCompartmentModel."""
    pass


@pytest.fixture()
def sift_filter():
    return SIFTFilter(fitted_model=_MockFittedModel(), target_cf=0.05)


def test_instantiation(sift_filter):
    """SIFTFilter should instantiate and store parameters."""
    assert isinstance(sift_filter.fitted_model, _MockFittedModel)
    assert sift_filter.target_cf == pytest.approx(0.05)


def test_default_target_cf():
    """Default target_cf should be 0.1."""
    f = SIFTFilter(fitted_model=_MockFittedModel())
    assert f.target_cf == pytest.approx(0.1)


def test_filter_accepts_target_density(sift_filter):
    """filter() accepts target_density and returns a list."""
    rng = np.random.default_rng(42)
    streamlines = [rng.standard_normal((10, 3)) + 4.5 for _ in range(3)]
    affine = np.eye(4)
    vol_shape = (10, 10, 10)
    target_density = np.zeros(vol_shape, dtype=np.float64)
    result = sift_filter.filter(streamlines, affine, target_density)
    assert isinstance(result, list)


def test_weights_accepts_target_density(sift_filter):
    """weights() accepts target_density and returns a float array."""
    rng = np.random.default_rng(42)
    streamlines = [rng.standard_normal((10, 3)) + 4.5 for _ in range(3)]
    affine = np.eye(4)
    vol_shape = (10, 10, 10)
    target_density = np.zeros(vol_shape, dtype=np.float64)
    w = sift_filter.weights(streamlines, affine, target_density)
    assert isinstance(w, np.ndarray)
    assert w.shape == (3,)
