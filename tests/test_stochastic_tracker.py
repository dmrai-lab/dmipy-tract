import numpy as np
import pytest
from dmipy_tract.core.field import MicrostructureField
from dmipy_tract.core.stopping_criteria import CompositeStopping, BoundaryStop
from dmipy_tract.propagation.stochastic import StochasticTracker


class MockFittedModel:
    def __init__(self, shape=(10, 10, 30)):
        self.shape = shape
        self.fitted_parameters = {
            'partial_volume_0': np.full(shape, 0.8, dtype=np.float32)
        }
        peaks = np.zeros(shape + (1, 3), dtype=np.float32)
        peaks[..., 0, 2] = 1.0
        self._peaks = peaks

    def peaks_cartesian(self):
        return self._peaks


def test_stochastic_tracker_produces_multiple_streamlines_per_seed():
    model = MockFittedModel()
    field = MicrostructureField(model, np.eye(4))
    stopping = CompositeStopping([BoundaryStop()])
    tracker = StochasticTracker(field, step_size_mm=0.5, kappa=100.0,
                                min_length_mm=1.0, n_streamlines_per_seed=5,
                                stopping=stopping, seed=42)
    seeds = np.array([[5., 5., 15.]])
    streamlines = tracker.track(seeds)
    # With kappa=100 (nearly deterministic) and 5 per seed, expect >=3 valid streamlines
    assert len(streamlines) >= 3, f"Expected >=3 streamlines, got {len(streamlines)}"


def test_stochastic_tracker_high_kappa_similar_to_deterministic():
    """At very high kappa, stochastic ≈ deterministic in z-direction."""
    from dmipy_tract.propagation.deterministic import DeterministicTracker
    model = MockFittedModel(shape=(10, 10, 30))
    field = MicrostructureField(model, np.eye(4))
    stopping = CompositeStopping([BoundaryStop()])
    seed = np.array([[5., 5., 15.]])

    det_tracker = DeterministicTracker(field, step_size_mm=0.5, min_length_mm=1.0,
                                       stopping=stopping)
    det_sls = det_tracker.track(seed)
    assert len(det_sls) >= 1

    stoch_tracker = StochasticTracker(field, step_size_mm=0.5, kappa=1000.0,
                                      min_length_mm=1.0, n_streamlines_per_seed=1,
                                      stopping=stopping, seed=0)
    stoch_sls = stoch_tracker.track(seed)
    assert len(stoch_sls) >= 1

    # Both should be mostly along z (small x,y deviation)
    for sl in stoch_sls:
        z_range = sl[:, 2].max() - sl[:, 2].min()
        xy_range = max(np.ptp(sl[:, 0]), np.ptp(sl[:, 1]))
        assert z_range > xy_range, "High-kappa stochastic not aligned with z"


def test_stochastic_tracker_low_kappa_produces_diverse_directions():
    """Low kappa should produce more variable streamlines (higher variance in endpoints)."""
    model = MockFittedModel()
    field = MicrostructureField(model, np.eye(4))
    stopping = CompositeStopping([BoundaryStop()])
    seed = np.array([[5., 5., 15.]])

    high_kappa = StochasticTracker(field, kappa=500.0, min_length_mm=0.5,
                                   n_streamlines_per_seed=20, stopping=stopping, seed=1)
    low_kappa = StochasticTracker(field, kappa=1.0, min_length_mm=0.5,
                                  n_streamlines_per_seed=20, stopping=stopping, seed=1)

    sls_high = high_kappa.track(seed)
    sls_low = low_kappa.track(seed)

    if len(sls_high) > 1 and len(sls_low) > 1:
        endpoints_high = np.array([sl[-1] for sl in sls_high])
        endpoints_low = np.array([sl[-1] for sl in sls_low])
        var_high = np.var(endpoints_high, axis=0).sum()
        var_low = np.var(endpoints_low, axis=0).sum()
        assert var_low >= var_high, (
            f"Low kappa should give more variable endpoints: "
            f"var_low={var_low:.4f}, var_high={var_high:.4f}"
        )
