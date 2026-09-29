"""The torch kernel (dmipy-tract#4) on the same constructed fields as the JAX one: the stopping rules, the reasons, the
join, the backward half, the circle, the per-streamline reference with the torch draws, chunk invariance, the JAX
backend on the deterministic rule (no draw: the same tractogram to float32 arithmetic), the draw's numpy and torch
spellings agreeing bit for bit, the refusals; CUDA against the CPU when a card is present."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dmipy_tract import FODField, track
from dmipy_tract._torch import uniform, uniform_torch
from dmipy_tract.tractogram import STOP_MASK, STOP_OUTSIDE, STOP_NO_DIRECTION, STOP_MAX_STEPS
from conftest import uniform_field, crossing_field, circle_field, reference_track

X = np.array([[1.0, 0.0, 0.0]])
T = dict(backend="torch", device="cpu")


def _expected_half_points(x0, n, step, sign):
    x = x0 + sign * step * np.arange(1, 100)
    return int(np.sum((np.rint(x) >= 0) & (np.rint(x) < n)))


@pytest.mark.parametrize("x0", [2.0, 2.3, 4.5, 5.0, 6.7, 7.0])
def test_stopping_is_the_definition(x0):
    f = uniform_field((10, 10, 10))
    tg = track(f, [[x0, 4.3, 4.3]], rule='deterministic', step_mm=0.5, sphere=X, initial_directions=X, **T)
    assert tg.n_points[0] == 1 + _expected_half_points(x0, 10, 0.5, +1) + _expected_half_points(x0, 10, 0.5, -1)
    assert tg.stop_reason[0].tolist() == [STOP_OUTSIDE, STOP_OUTSIDE]
    s = tg[0]
    assert np.all(np.rint(s) >= 0) and np.all(np.rint(s) < 10)
    np.testing.assert_allclose(np.diff(s[:, 0]), 0.5, atol=1e-6)
    for n, last in ((10, 9.0), (11, 10.5)):                                  # half to even at the grid edge
        assert track(uniform_field((n, 10, 10)), [[4.5, 4.3, 4.3]], rule='deterministic', step_mm=0.5, sphere=X,
                     initial_directions=X, **T)[0][-1, 0] == pytest.approx(last)


def test_reasons_one_point_seeds_min_length_and_the_join():
    mask = np.ones((10, 10, 10), bool); mask[7:] = False
    tg = track(uniform_field((10, 10, 10), mask=mask), [[4.3, 4.3, 4.3]], rule='deterministic', step_mm=0.5, sphere=X, initial_directions=X, **T)
    assert tg.n_points[0] == 14 and tg.stop_reason[0].tolist() == [STOP_MASK, STOP_OUTSIDE] and tg[0][-1, 0] == pytest.approx(6.3)
    f = uniform_field((10, 10, 10)); sh = f.sh.copy(); sh[7:] = 0.0
    tg = track(FODField(sh, f.affine, f.mask), [[4.3, 4.3, 4.3]], rule='deterministic', step_mm=0.5, sphere=X, initial_directions=X, **T)
    assert tg.n_points[0] == 16 and tg.stop_reason[0].tolist() == [STOP_NO_DIRECTION, STOP_OUTSIDE]
    tg = track(f, [[4.3, 4.3, 4.3]], rule='deterministic', step_mm=0.5, sphere=X, initial_directions=X, max_steps=5, **T)
    assert tg.n_points[0] == 9 and tg.stop_reason[0].tolist() == [STOP_MAX_STEPS, STOP_MAX_STEPS]
    sh[:] = 0.0
    tg = track(FODField(sh, f.affine, f.mask), [[4.3, 4.3, 4.3], [2.0, 2.0, 2.0]], rule='probabilistic', **T)
    assert tg.n_points.tolist() == [1, 1] and np.all(tg.stop_reason == STOP_NO_DIRECTION)
    np.testing.assert_array_equal(tg[1], [[2.0, 2.0, 2.0]])
    mask = np.ones((10, 10, 10), bool); mask[6:] = False
    fm = uniform_field((10, 10, 10), mask=mask); seeds = np.array([[x, 4.3, 4.3] for x in (0.2, 2.0, 4.0, 5.4)])
    full = track(fm, seeds, rule='deterministic', step_mm=0.5, sphere=X, initial_directions=np.tile(X, (4, 1)), **T)
    L = (full.n_points - 1) * 0.5
    for cut in (0.0, 2.0, 4.5, 100.0):
        kept = track(fm, seeds, rule='deterministic', step_mm=0.5, sphere=X, initial_directions=np.tile(X, (4, 1)), min_length_mm=cut, **T)
        np.testing.assert_array_equal(kept.seed_index, np.flatnonzero(L >= cut))
    f = uniform_field((12, 12, 12), direction=(1, 1, 0)); seed = np.array([[5.2, 5.7, 6.1]]); d = np.array([[1, 1, 0]]) / np.sqrt(2)
    s = track(f, seed, rule='deterministic', step_mm=0.3, sphere=d, initial_directions=d, **T)[0].astype(np.float64)
    seg = np.diff(s, axis=0)
    np.testing.assert_allclose(np.linalg.norm(seg, axis=1), 0.3, atol=1e-5)
    np.testing.assert_allclose(seg / 0.3, np.tile(d, (len(seg), 1)), atol=1e-5)
    assert np.sum(np.all(np.isclose(s, seed, atol=1e-6), axis=1)) == 1 and len(np.unique(np.round(s, 5), axis=0)) == len(s)


def test_backward_half_and_the_circle():
    f, _, crossing = crossing_field()
    seeds = np.argwhere(crossing)[:40].astype(float) + 0.1
    for rule, bound_deg in (('deterministic', 1e-3), ('probabilistic', 30.0)):
        tg = track(f, seeds, rule=rule, step_mm=0.5, max_angle=30.0, key=3, **T)
        for i, s in enumerate(tg):
            k = int(np.flatnonzero(np.all(np.isclose(s, seeds[i], atol=1e-6), axis=1))[0])
            if 0 < k < len(s) - 1:
                a, b = s[k] - s[k - 1], s[k + 1] - s[k]
                cos = a @ b / np.linalg.norm(a) / np.linalg.norm(b)
                assert np.degrees(np.arccos(np.clip(cos, -1, 1))) <= bound_deg + 1e-3
    fc, centre = circle_field()
    seed = centre + np.array([9.0, 0.0, 0.0])
    tg = track(fc, seed[None], rule='deterministic', step_mm=0.3, max_angle=30.0, **T)
    r = np.linalg.norm(tg[0][:, :2] - centre[:2], axis=1)
    assert len(tg[0]) > 50 and np.abs(r - 9.0).max() < len(tg[0]) * 0.3 * 0.03   # the hemisphere's angular resolution


def test_the_draw_is_the_same_in_numpy_and_torch_and_never_repeats_by_accident():
    idx = np.arange(0, 5000, dtype=np.int64)
    for key, counter in ((0, 0), (7, 1), (2 ** 31 + 3, 1001), (12345, 2 ** 20)):
        a = uniform(key, idx, counter)
        b = uniform_torch(key, torch.as_tensor(idx), counter).numpy()
        np.testing.assert_array_equal(a, b)
        assert a.dtype == np.float32 and (a >= 0).all() and (a < 1).all()
        assert len(np.unique(a)) > 4900                                             # 24-bit draws: collisions are rare
    assert not np.array_equal(uniform(0, idx, 1), uniform(0, idx, 2)) and not np.array_equal(uniform(0, idx, 1), uniform(1, idx, 1))


@pytest.mark.parametrize("rule", ["deterministic", "probabilistic"])
def test_kernel_equals_the_per_streamline_reference(rule):
    f, _, crossing = crossing_field()
    seeds = np.argwhere(crossing)[::7][:12].astype(float) + np.array([0.2, 0.35, 0.1])
    tg = track(f, seeds, rule=rule, step_mm=0.5, max_angle=30.0, max_steps=60, key=5, **T)
    ref = reference_track(f, seeds, rule=rule, step_mm=0.5, max_angle=30.0, max_steps=60, key=5,
                          uniform=lambda i, c: float(uniform(5, np.asarray([i]), c)[0]))
    for i, (pts, reasons) in enumerate(ref):
        assert tg.n_points[i] == len(pts), (i, tg.n_points[i], len(pts))
        np.testing.assert_allclose(tg[i], pts, atol=2e-4)
        assert tg.stop_reason[i].tolist() == list(reasons)


def test_chunk_does_not_matter_and_the_same_key_repeats():
    f, _, crossing = crossing_field()
    seeds = np.argwhere(crossing)[:100].astype(float) + 0.1
    a = track(f, seeds, rule='probabilistic', key=11, chunk=7, **T)
    b = track(f, seeds, rule='probabilistic', key=11, chunk=65536, **T)
    np.testing.assert_array_equal(a.offsets, b.offsets); np.testing.assert_array_equal(a.points, b.points)
    c = track(f, seeds, rule='probabilistic', key=12, **T)
    assert not (len(c.points) == len(a.points) and np.array_equal(c.points, a.points))


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
    with pytest.raises(ValueError, match="backend"):
        track(f, [[4.0, 4.0, 4.0]], backend="numpy")
    with pytest.raises(ValueError, match="int key"):
        import jax
        track(f, [[4.0, 4.0, 4.0]], key=jax.random.key(0), **T)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_cuda_equals_the_cpu_bit_for_bit():
    f, _, crossing = crossing_field()
    seeds = np.argwhere(crossing)[:200].astype(float) + 0.1
    a = track(f, seeds, rule='probabilistic', key=2, backend="torch", device="cpu")
    b = track(f, seeds, rule='probabilistic', key=2, backend="torch", device="cuda")
    np.testing.assert_array_equal(a.offsets, b.offsets); np.testing.assert_array_equal(a.points, b.points)
