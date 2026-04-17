"""Tests for along-tract tractometry profiling (Phase 2)."""
import numpy as np
import pytest
from dmipy_tract.tractometry.profile import along_tract_profile


def test_along_tract_profile_constant_parameter():
    """For a field with constant parameter, profile should be flat."""
    class ConstantField:
        def parameter_at(self, name, pt):
            return 0.7  # constant everywhere

    # 3 straight streamlines of different lengths
    streamlines = [
        np.column_stack([np.zeros(10), np.zeros(10), np.linspace(0, 9, 10)]),
        np.column_stack([np.zeros(20), np.zeros(20), np.linspace(0, 19, 20)]),
        np.column_stack([np.zeros(15), np.zeros(15), np.linspace(0, 14, 15)]),
    ]
    field = ConstantField()
    positions, mean_profile, std_profile = along_tract_profile(
        streamlines, field, 'vf_ic', n_points=20
    )
    np.testing.assert_allclose(mean_profile, 0.7, atol=1e-6)
    np.testing.assert_allclose(std_profile, 0.0, atol=1e-6)


def test_along_tract_profile_weighted():
    """Weighted mean shifts toward the higher-weighted streamline."""
    class LinearField:
        def parameter_at(self, name, pt):
            return float(pt[2]) / 10.0  # value = z/10

    # Two streamlines: one at z=0..5 (values 0.0..0.5), one at z=5..10 (values 0.5..1.0)
    sl1 = np.column_stack([np.zeros(6), np.zeros(6), np.linspace(0, 5, 6)])
    sl2 = np.column_stack([np.zeros(6), np.zeros(6), np.linspace(5, 10, 6)])
    field = LinearField()

    # Unweighted: mean of both
    _, mean_unweighted, _ = along_tract_profile([sl1, sl2], field, 'vf', n_points=5)

    # Weighted: upweight sl2 (higher values)
    _, mean_weighted, _ = along_tract_profile(
        [sl1, sl2], field, 'vf', n_points=5, weights=np.array([1.0, 9.0])
    )
    # Mean weighted should be higher than unweighted mean
    assert mean_weighted.mean() > mean_unweighted.mean()


def test_along_tract_profile_empty_streamlines():
    """Empty list returns all-NaN profile."""
    class AnyField:
        def parameter_at(self, name, pt):
            return 0.0

    pos, mean, std = along_tract_profile([], AnyField(), 'vf', n_points=10)
    assert np.all(np.isnan(mean))
    assert np.all(np.isnan(std))


def test_along_tract_profile_single_point_streamlines_skipped():
    """Streamlines with < 2 points are skipped gracefully."""
    class AnyField:
        def parameter_at(self, name, pt):
            return 1.0

    single_pt = np.array([[0., 0., 0.]])
    valid = np.column_stack([np.zeros(5), np.zeros(5), np.linspace(0, 4, 5)])
    pos, mean, std = along_tract_profile([single_pt, valid], AnyField(), 'vf', n_points=5)
    np.testing.assert_allclose(mean, 1.0, atol=1e-6)


def test_along_tract_profile_positions_shape():
    """Returned positions array has correct shape [0, 1] with n_points elements."""
    class AnyField:
        def parameter_at(self, name, pt):
            return 0.5

    sl = np.column_stack([np.zeros(10), np.zeros(10), np.linspace(0, 9, 10)])
    positions, mean, std = along_tract_profile([sl], AnyField(), 'vf', n_points=25)
    assert positions.shape == (25,)
    assert positions[0] == pytest.approx(0.0)
    assert positions[-1] == pytest.approx(1.0)
    assert mean.shape == (25,)
    assert std.shape == (25,)


def test_along_tract_profile_std_nonzero_for_varying_streamlines():
    """Std should be non-zero when streamlines sample different parameter values."""
    class LinearField:
        def parameter_at(self, name, pt):
            return float(pt[2]) / 10.0

    sl1 = np.column_stack([np.zeros(10), np.zeros(10), np.linspace(0, 5, 10)])
    sl2 = np.column_stack([np.zeros(10), np.zeros(10), np.linspace(5, 10, 10)])
    _, _, std = along_tract_profile([sl1, sl2], LinearField(), 'vf', n_points=10)
    # Interior points should have non-zero std since sl1 and sl2 have different z ranges
    assert std.mean() > 0.0
