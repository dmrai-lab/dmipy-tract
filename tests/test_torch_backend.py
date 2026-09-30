"""What only the torch kernel has: its counter-based draw (the numpy and torch spellings agree bit for bit), its
agreement with the JAX kernel on the deterministic rule (no draw: the same tractogram to float32 arithmetic), its
refusals, and CUDA against the CPU where a card is present. The conventions both kernels share are tested on both in
``test_tracker.py``."""
import numpy as np
import pytest
import jax

torch = pytest.importorskip("torch")

from dmipy_tract import track
from dmipy_tract._torch import uniform, uniform_torch
from conftest import uniform_field, crossing_field

T = dict(backend="torch", device="cpu")


def test_the_draw_is_the_same_in_numpy_and_torch_and_never_repeats_by_accident():
    idx = np.arange(0, 5000, dtype=np.int64)
    for key, counter in ((0, 0), (7, 1), (2 ** 31 + 3, 1001), (12345, 2 ** 20)):
        a = uniform(key, idx, counter)
        b = uniform_torch(key, torch.as_tensor(idx), counter).numpy()
        np.testing.assert_array_equal(a, b)
        assert a.dtype == np.float32 and (a >= 0).all() and (a < 1).all()
        assert len(np.unique(a)) > 4900                                             # 24-bit draws: collisions are rare
    assert not np.array_equal(uniform(0, idx, 1), uniform(0, idx, 2)) and not np.array_equal(uniform(0, idx, 1), uniform(1, idx, 1))


def test_the_deterministic_rule_agrees_with_the_jax_backend():
    """No draw: the two kernels differ only by float32 arithmetic (a near tie at a cone edge or a threshold can
    flip a streamline, which the point-wise comparison would show)."""
    f, _, crossing = crossing_field()
    seeds = np.argwhere(crossing)[::3][:50].astype(float) + np.array([0.2, 0.35, 0.1])
    a = track(f, seeds, rule='deterministic', step_mm=0.5, max_angle=30.0, max_steps=80)
    b = track(f, seeds, rule='deterministic', step_mm=0.5, max_angle=30.0, max_steps=80, **T)
    np.testing.assert_array_equal(a.n_points, b.n_points)
    np.testing.assert_allclose(a.points, b.points, atol=2e-4)
    np.testing.assert_array_equal(a.stop_reason, b.stop_reason)


def test_refusals():
    f = uniform_field((10, 10, 10))
    s = [[4.0, 4.0, 4.0]]
    with pytest.raises(ValueError, match="backend"):
        track(f, s, backend="numpy")
    with pytest.raises(ValueError, match="int key"):
        track(f, s, key=jax.random.key(0), **T)
    for name in ("phase_steps", "batch"):
        with pytest.raises(ValueError, match=f"{name} is an argument of the JAX backend"):
            track(f, s, **{name: 8}, **T)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_cuda_equals_the_cpu_bit_for_bit():
    f, _, crossing = crossing_field()
    seeds = np.argwhere(crossing)[:200].astype(float) + 0.1
    a = track(f, seeds, rule='probabilistic', key=2, backend="torch", device="cpu")
    b = track(f, seeds, rule='probabilistic', key=2, backend="torch", device="cuda")
    np.testing.assert_array_equal(a.offsets, b.offsets); np.testing.assert_array_equal(a.points, b.points)
