"""End-to-end pipeline tests: seeds → DeterministicTracker → tractometry (Phase 2)."""
import numpy as np
import pytest
from dmipy_tract.core.field import MicrostructureField
from dmipy_tract.core.seeding import seeds_from_mask
from dmipy_tract.core.stopping_criteria import CompositeStopping, BoundaryStop, VfIcThreshold
from dmipy_tract.propagation.deterministic import DeterministicTracker
from dmipy_tract.tractometry.profile import along_tract_profile


class MockFittedModel:
    """Straight-fiber phantom: peaks along z, vf_ic=0.8 everywhere."""

    def __init__(self, shape=(10, 10, 30)):
        self.shape = shape
        self.fitted_parameters = {
            'partial_volume_0': np.full(shape, 0.8, dtype=np.float32)
        }
        peaks = np.zeros(shape + (1, 3), dtype=np.float32)
        peaks[..., 0, 2] = 1.0   # z-direction
        self._peaks = peaks

    def peaks_cartesian(self):
        return self._peaks


def test_end_to_end_seed_track_profile():
    """Full pipeline: seeds → DeterministicTracker → along_tract_profile."""
    model = MockFittedModel(shape=(10, 10, 30))
    affine = np.eye(4)
    field = MicrostructureField(model, affine)

    # Seed from centre slice
    seed_mask = np.zeros((10, 10, 30), dtype=bool)
    seed_mask[5, 5, 15] = True  # single central seed
    seeds = seeds_from_mask(seed_mask, affine, density=1)
    assert len(seeds) == 1

    stopping = CompositeStopping([BoundaryStop()])
    tracker = DeterministicTracker(
        field, step_size_mm=0.5, min_length_mm=1.0, max_length_mm=15.0,
        stopping=stopping
    )
    streamlines = tracker.track(seeds)
    assert len(streamlines) >= 1, "No streamlines produced from central seed"

    sl = streamlines[0]
    # Streamline should be mostly along z
    z_extent = sl[:, 2].max() - sl[:, 2].min()
    xy_extent = max(np.ptp(sl[:, 0]), np.ptp(sl[:, 1]))
    assert z_extent > xy_extent, (
        f"Streamline not aligned with z: z_extent={z_extent:.2f}, xy={xy_extent:.2f}"
    )

    # Tractometry: vf_ic should be ~0.8 everywhere along tract
    positions, mean_profile, std_profile = along_tract_profile(
        streamlines, field, 'partial_volume_0', n_points=20
    )
    np.testing.assert_allclose(mean_profile, 0.8, atol=0.05,
                                err_msg=f"vf_ic profile not flat: {mean_profile}")


def test_end_to_end_multi_seed_tractometry_mean():
    """Multiple seeds in a uniform field → tractometry mean equals field value."""
    model = MockFittedModel(shape=(10, 10, 30))
    affine = np.eye(4)
    field = MicrostructureField(model, affine)

    seed_mask = np.zeros((10, 10, 30), dtype=bool)
    # Three seeds spread along the z-axis centre column
    seed_mask[5, 5, 5] = True
    seed_mask[5, 5, 15] = True
    seed_mask[5, 5, 25] = True
    seeds = seeds_from_mask(seed_mask, affine, density=1)
    assert len(seeds) == 3

    stopping = CompositeStopping([BoundaryStop()])
    tracker = DeterministicTracker(
        field, step_size_mm=0.5, min_length_mm=1.0, max_length_mm=20.0,
        stopping=stopping
    )
    streamlines = tracker.track(seeds)
    assert len(streamlines) >= 1

    positions, mean_profile, std_profile = along_tract_profile(
        streamlines, field, 'partial_volume_0', n_points=10
    )
    # All streamlines run through a uniform field → mean should be ~0.8
    np.testing.assert_allclose(mean_profile, 0.8, atol=0.05)


def test_end_to_end_vf_ic_stopping_truncates_streamlines():
    """Streamlines stop at low-vf boundary even in end-to-end run."""
    shape = (10, 10, 30)
    # Low vf_ic beyond z-index 15
    vf_array = np.full(shape, 0.8, dtype=np.float32)
    vf_array[:, :, 16:] = 0.0

    peaks = np.zeros(shape + (1, 3), dtype=np.float32)
    peaks[..., 0, 2] = 1.0

    class PatchedModel:
        def __init__(self):
            self.fitted_parameters = {'partial_volume_0': vf_array}
            self._peaks = peaks

        def peaks_cartesian(self):
            return self._peaks

    affine = np.eye(4)
    field = MicrostructureField(PatchedModel(), affine)

    seed_mask = np.zeros(shape, dtype=bool)
    seed_mask[5, 5, 5] = True
    seeds = seeds_from_mask(seed_mask, affine, density=1)

    stopping = CompositeStopping([BoundaryStop(), VfIcThreshold(0.1)])
    tracker = DeterministicTracker(
        field, step_size_mm=0.5, min_length_mm=1.0, max_length_mm=50.0,
        stopping=stopping
    )
    streamlines = tracker.track(seeds)
    assert len(streamlines) >= 1

    sl = streamlines[0]
    # Forward direction must stop before reaching z=17 (vf_ic drops at z-index 16)
    assert sl[:, 2].max() < 17.0, (
        f"Expected streamline to stop near vf_ic boundary, but z_max={sl[:, 2].max():.2f}"
    )
