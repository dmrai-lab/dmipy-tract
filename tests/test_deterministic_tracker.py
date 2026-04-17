"""Tier 1 straight-fiber and boundary synthetic phantom tests (Phase 0.5)."""
import numpy as np
import pytest
from dmipy_tract.core.field import MicrostructureField
from dmipy_tract.core.stopping_criteria import CompositeStopping, VfIcThreshold, BoundaryStop
from dmipy_tract.propagation.deterministic import DeterministicTracker


class MockFittedModel:
    """Minimal mock of FittedMultiCompartmentModel for unit tests."""

    def __init__(self, peaks, params):
        self._peaks = np.asarray(peaks, dtype=np.float32)
        self.fitted_parameters = {k: np.asarray(v, dtype=np.float32) for k, v in params.items()}

    def peaks_cartesian(self):
        return self._peaks


def _make_field(shape, vf_ic_array, peaks_array):
    """Helper: construct a MicrostructureField from pre-built arrays."""
    model = MockFittedModel(
        peaks=peaks_array,    # (X, Y, Z, 1, 3)
        params={'partial_volume_0': vf_ic_array},
    )
    return MicrostructureField(model, np.eye(4))


def test_straight_fiber_phantom_recovered():
    """Tracker should produce a monotonically z-increasing streamline in a straight-fiber phantom."""
    shape = (10, 10, 20)
    vf_ic = np.full(shape, 0.8, dtype=np.float32)

    # All peaks point in +z direction
    peaks = np.zeros(shape + (1, 3), dtype=np.float32)
    peaks[..., 0, 2] = 1.0  # z-component = 1

    field = _make_field(shape, vf_ic, peaks)

    stopping = CompositeStopping([
        BoundaryStop(),
        VfIcThreshold(0.1),
    ])
    tracker = DeterministicTracker(
        field,
        step_size_mm=0.5,
        max_angle_deg=30.0,
        max_length_mm=250.0,
        min_length_mm=1.0,  # small so short phantoms still return streamlines
        stopping=stopping,
    )

    seed = np.array([[5.0, 5.0, 2.0]])
    streamlines = tracker.track(seed)

    assert len(streamlines) >= 1, "Expected at least one streamline"

    sl = streamlines[0]
    assert sl.shape[1] == 3, "Streamline points should have 3 coordinates"

    # The streamline should be longer than 5 steps
    assert len(sl) > 5, f"Expected streamline length > 5 steps, got {len(sl)}"

    # The z-coordinates should be monotonically increasing or decreasing
    z_coords = sl[:, 2]
    diffs = np.diff(z_coords)
    is_monotone_increasing = np.all(diffs > -1e-3)
    is_monotone_decreasing = np.all(diffs < 1e-3)
    assert is_monotone_increasing or is_monotone_decreasing, (
        f"Expected monotone z-coordinates, got diffs range "
        f"[{diffs.min():.3f}, {diffs.max():.3f}]"
    )


def test_streamline_terminates_at_vf_ic_boundary():
    """Tracker should stop before crossing from high-vf into low-vf region."""
    shape = (10, 10, 20)
    vf_ic = np.zeros(shape, dtype=np.float32)
    vf_ic[:, :, :10] = 0.8   # first half: high WM fraction
    vf_ic[:, :, 10:] = 0.0   # second half: no WM (stop here)

    # All peaks point in +z direction
    peaks = np.zeros(shape + (1, 3), dtype=np.float32)
    peaks[..., 0, 2] = 1.0

    field = _make_field(shape, vf_ic, peaks)

    stopping = CompositeStopping([
        BoundaryStop(),
        VfIcThreshold(0.1),
    ])
    tracker = DeterministicTracker(
        field,
        step_size_mm=0.5,
        max_angle_deg=30.0,
        max_length_mm=250.0,
        min_length_mm=1.0,
        stopping=stopping,
    )

    seed = np.array([[5.0, 5.0, 2.0]])
    streamlines = tracker.track(seed)

    assert len(streamlines) >= 1, "Expected at least one streamline"

    sl = streamlines[0]
    z_max = sl[:, 2].max()

    # The forward direction should terminate near the WM boundary (z ~10), well before z=11
    assert z_max < 11.0, (
        f"Expected streamline to terminate before z=11mm, but z_max={z_max:.2f}"
    )


def test_rk4_step_respects_max_angle():
    """Tracker should stop when direction changes sharply beyond max_angle_deg."""
    shape = (10, 10, 20)
    vf_ic = np.full(shape, 0.8, dtype=np.float32)

    # Peaks: +z in voxels z=0..9, +x in voxels z=10..19 — a 90° direction change
    peaks = np.zeros(shape + (1, 3), dtype=np.float32)
    peaks[:, :, :10, 0, 2] = 1.0   # z=0..9: z-direction
    peaks[:, :, 10:, 0, 0] = 1.0   # z=10..19: x-direction

    field = _make_field(shape, vf_ic, peaks)

    stopping = CompositeStopping([
        BoundaryStop(),
        VfIcThreshold(0.1),
    ])
    tracker = DeterministicTracker(
        field,
        step_size_mm=0.5,
        max_angle_deg=30.0,  # 90° turn exceeds this limit
        max_length_mm=250.0,
        min_length_mm=1.0,
        stopping=stopping,
    )

    seed = np.array([[5.0, 5.0, 2.0]])
    streamlines = tracker.track(seed)

    assert len(streamlines) >= 1, "Expected at least one streamline (from the straight segment)"

    sl = streamlines[0]
    z_max = sl[:, 2].max()

    # The streamline should NOT cross significantly into the x-direction region (z > 11)
    # Note: RK4 intermediate sub-steps explore ahead, so slight z/x drift near the
    # boundary is expected. The key assertion is that the tracker stops near z=10.
    assert z_max <= 11.0, (
        f"Expected streamline to stop near the 90° direction boundary (z~10), "
        f"but reached z={z_max:.2f}"
    )
