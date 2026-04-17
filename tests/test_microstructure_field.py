"""Tier 1 synthetic validation tests for MicrostructureField (Phase 0.5)."""
import numpy as np
import pytest
from dmipy_tract.core.field import MicrostructureField


class MockFittedModel:
    """Minimal mock of FittedMultiCompartmentModel for unit tests."""

    def __init__(self, peaks, params):
        """
        Parameters
        ----------
        peaks : ndarray, shape (X, Y, Z, K, 3)
        params : dict of str -> ndarray shape (X, Y, Z)
        """
        self._peaks = np.asarray(peaks, dtype=np.float32)
        self.fitted_parameters = {k: np.asarray(v, dtype=np.float32) for k, v in params.items()}

    def peaks_cartesian(self):
        return self._peaks


def _make_uniform_field(shape=(5, 5, 5), vf_ic=0.7, peak_dir=(0.0, 0.0, 1.0)):
    """Helper: create a MicrostructureField with uniform vf_ic and peak direction."""
    peak = np.array(peak_dir, dtype=np.float32)
    peak = peak / np.linalg.norm(peak)

    peaks = np.zeros(shape + (1, 3), dtype=np.float32)
    peaks[..., 0, :] = peak  # broadcast: all voxels same direction

    params = {
        'partial_volume_0': np.full(shape, vf_ic, dtype=np.float32),
    }
    model = MockFittedModel(peaks, params)
    affine = np.eye(4, dtype=np.float64)
    return MicrostructureField(model, affine)


def test_parameter_at_returns_expected_value():
    """MicrostructureField.parameter_at should return the known vf_ic=0.7 value."""
    field = _make_uniform_field(shape=(5, 5, 5), vf_ic=0.7)
    # With affine=eye(4), world position [2.5, 2.5, 2.5] maps to voxel [2.5, 2.5, 2.5]
    result = field.parameter_at('partial_volume_0', [2.5, 2.5, 2.5])
    assert abs(result - 0.7) < 0.01, f"Expected ~0.7, got {result}"


def test_peaks_at_unit_norm():
    """peaks_at should return unit-norm peaks."""
    field = _make_uniform_field(shape=(5, 5, 5), peak_dir=(0.0, 0.0, 1.0))
    peaks = field.peaks_at([2.5, 2.5, 2.5])
    # shape should be (K, 3)
    assert peaks.ndim == 2
    assert peaks.shape[1] == 3
    norm = np.linalg.norm(peaks[0])
    assert abs(norm - 1.0) < 1e-5, f"Expected unit norm, got {norm}"


def test_stopping_mask_boundary():
    """stopping_mask should be True in high-vf region and False in low-vf region."""
    shape = (5, 5, 5)
    vf = np.full(shape, 0.8, dtype=np.float32)
    # Set a sub-region to low vf_ic
    vf[:, :, 4] = 0.05

    peaks = np.zeros(shape + (1, 3), dtype=np.float32)
    peaks[..., 0, 2] = 1.0  # z-direction

    params = {'partial_volume_0': vf}
    model = MockFittedModel(peaks, params)
    field = MicrostructureField(model, np.eye(4))

    mask = field.stopping_mask(vf_ic_threshold=0.1)

    # High-vf region should be True
    assert mask[:, :, :4].all(), "Expected all True in high-vf region"
    # Low-vf region should be False
    assert not mask[:, :, 4].any(), "Expected all False in low-vf region"
