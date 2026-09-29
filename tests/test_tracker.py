"""The tracker on constructed fields: stopping, the bidirectional join, the curved field, the invariances, the
refusals, and exact agreement with the per-streamline numpy reference of the same definition."""
import numpy as np
import pytest
import jax

from dmipy_sim.replay.so3 import rotate_sh

from dmipy_tract import FODField, track, hemisphere, sh_matrix
from dmipy_tract.tractogram import STOP_MASK, STOP_OUTSIDE, STOP_NO_DIRECTION, STOP_MAX_STEPS
from conftest import uniform_field, crossing_field, circle_field, delta_sh, reference_track

X = np.array([[1.0, 0.0, 0.0]])


def expected_half_points(x0, n, step, sign):
    """Points of a half from x0 along ``sign`` x (the seed excluded) whose nearest voxel (round half to even) is in
    0..n-1: the domain of a nearest-voxel mask on an n-voxel axis."""
    x = x0 + sign * np.arange(1, 10000) * step
    return int(np.sum((np.rint(x) >= 0) & (np.rint(x) < n)))


# ------------------------------------------------------------------ 5. stopping, exhaustive over seed positions
@pytest.mark.parametrize("x0", np.round(np.arange(2.0, 7.01, 0.1), 2).tolist())
def test_uniform_field_point_count_and_outside_reason(x0):
    f = uniform_field((10, 10, 10))
    tg = track(f, [[x0, 4.3, 4.3]], rule='deterministic', step_mm=0.5, sphere=X, initial_directions=X)
    nf = expected_half_points(x0, 10, 0.5, +1)
    nb = expected_half_points(x0, 10, 0.5, -1)
    assert tg.n_points[0] == 1 + nf + nb
    assert tg.stop_reason[0].tolist() == [STOP_OUTSIDE, STOP_OUTSIDE]
    s = tg[0]
    assert np.all(np.rint(s) >= 0) and np.all(np.rint(s) < 10)             # no point outside the domain
    np.testing.assert_allclose(np.diff(s[:, 0]), 0.5, atol=1e-6)


def test_round_half_to_even_decides_the_grid_edge():
    """A point at exactly N - 0.5 is voxel N (outside) when N is even and voxel N - 1 (inside) when N is odd."""
    for n, last in ((10, 9.0), (11, 10.5)):
        f = uniform_field((n, 10, 10))
        tg = track(f, [[4.5, 4.3, 4.3]], rule='deterministic', step_mm=0.5, sphere=X, initial_directions=X)
        assert tg[0][-1, 0] == pytest.approx(last)


def test_mask_reason_and_no_direction_reason_and_max_steps():
    mask = np.ones((10, 10, 10), bool)
    mask[7:] = False
    tg = track(uniform_field((10, 10, 10), mask=mask), [[4.3, 4.3, 4.3]], rule='deterministic', step_mm=0.5,
               sphere=X, initial_directions=X)
    assert tg.n_points[0] == 14 and tg.stop_reason[0].tolist() == [STOP_MASK, STOP_OUTSIDE]
    assert tg[0][-1, 0] == pytest.approx(6.3)

    f = uniform_field((10, 10, 10))
    sh = f.sh.copy()
    sh[7:] = 0.0                                             # no FOD from x = 7 on: interpolated to zero past 7.0
    tg = track(FODField(sh, f.affine, f.mask), [[4.3, 4.3, 4.3]], rule='deterministic', step_mm=0.5, sphere=X,
               initial_directions=X)
    assert tg.n_points[0] == 16 and tg.stop_reason[0].tolist() == [STOP_NO_DIRECTION, STOP_OUTSIDE]
    assert tg[0][-1, 0] == pytest.approx(7.3)

    tg = track(f, [[4.3, 4.3, 4.3]], rule='deterministic', step_mm=0.5, sphere=X, initial_directions=X, max_steps=5)
    assert tg.n_points[0] == 9 and tg.stop_reason[0].tolist() == [STOP_MAX_STEPS, STOP_MAX_STEPS]


def test_seed_without_direction_is_a_one_point_streamline():
    f = uniform_field((10, 10, 10))
    sh = f.sh.copy()
    sh[:] = 0.0
    tg = track(FODField(sh, f.affine, f.mask), [[4.3, 4.3, 4.3], [2.0, 2.0, 2.0]], rule='probabilistic')
    assert tg.n_points.tolist() == [1, 1]
    assert np.all(tg.stop_reason == STOP_NO_DIRECTION)
    np.testing.assert_array_equal(tg[1], [[2.0, 2.0, 2.0]])


def test_min_length_drops_exactly_the_short_streamlines():
    mask = np.ones((10, 10, 10), bool)
    mask[6:] = False
    f = uniform_field((10, 10, 10), mask=mask)
    seeds = np.array([[x, 4.3, 4.3] for x in (0.2, 2.0, 4.0, 5.4)])
    full = track(f, seeds, rule='deterministic', step_mm=0.5, sphere=X, initial_directions=np.tile(X, (4, 1)))
    L = (full.n_points - 1) * 0.5
    for cut in (0.0, 2.0, 4.5, 5.0, 100.0):
        kept = track(f, seeds, rule='deterministic', step_mm=0.5, sphere=X, initial_directions=np.tile(X, (4, 1)),
                     min_length_mm=cut)
        np.testing.assert_array_equal(kept.seed_index, np.flatnonzero(L >= cut))


# ------------------------------------------------------------------ 6. the join
def test_bidirectional_join_is_one_line_with_the_seed_once():
    f = uniform_field((12, 12, 12), direction=(1, 1, 0))
    seed = np.array([[5.2, 5.7, 6.1]])
    d = np.array([[1, 1, 0]]) / np.sqrt(2)
    tg = track(f, seed, rule='deterministic', step_mm=0.3, sphere=d, initial_directions=d)
    s = tg[0].astype(np.float64)
    seg = np.diff(s, axis=0)
    np.testing.assert_allclose(np.linalg.norm(seg, axis=1), 0.3, atol=1e-5)
    np.testing.assert_allclose(seg / 0.3, np.tile(d, (len(seg), 1)), atol=1e-5)   # every segment along d
    assert np.sum(np.all(np.isclose(s, seed, atol=1e-6), axis=1)) == 1        # the seed exactly once
    assert len(np.unique(np.round(s, 5), axis=0)) == len(s)                  # no duplicated point


def test_backward_half_starts_against_the_forward_first_step():
    """The backward half's first direction is minus the forward half's actual first step (dipy): under the
    deterministic rule at a peak the backward first step is exactly collinear with the forward one; under the
    probabilistic rule it is sampled inside the cone around that reversed step, so within max_angle of it."""
    f, _, crossing = crossing_field()
    seeds = np.argwhere(crossing)[:60].astype(float) + 0.1
    for rule, bound_deg in (('deterministic', 1e-3), ('probabilistic', 30.0)):
        tg = track(f, seeds, rule=rule, step_mm=0.5, max_angle=30.0, key=3)
        for i, s in enumerate(tg):
            k = int(np.flatnonzero(np.all(np.isclose(s, seeds[i], atol=1e-6), axis=1))[0])
            if 0 < k < len(s) - 1:
                a, b = s[k] - s[k - 1], s[k + 1] - s[k]
                cos = a @ b / np.linalg.norm(a) / np.linalg.norm(b)
                assert np.degrees(np.arccos(np.clip(cos, -1, 1))) <= bound_deg + 1e-3


# ------------------------------------------------------------------ 7. the curved field
@pytest.mark.parametrize("r0", [6.0, 9.0, 12.0])
@pytest.mark.parametrize("theta0", np.deg2rad([0.0, 37.0, 200.0]).tolist())
def test_deterministic_rule_follows_the_circle(r0, theta0):
    f, centre = circle_field(40)
    V = hemisphere(4000)
    G = V @ V.T
    np.fill_diagonal(G, 0)
    delta = np.arccos(np.clip(G.max(axis=1), -1, 1)).max()      # the sphere's angular resolution, radians
    step = 0.2
    seed = centre + r0 * np.array([np.cos(theta0), np.sin(theta0), 0.0])
    tg = track(f, seed[None], rule='deterministic', step_mm=step, max_angle=30.0, max_steps=400, sphere=V)
    s = tg[0].astype(np.float64) - centre
    r = np.hypot(s[:, 0], s[:, 1])
    k = int(np.flatnonzero(np.isclose(r, r0, atol=1e-5))[0])         # the seed
    # a polygon of chords of length ``step`` on exact tangents has radius sqrt(r0^2 + n step^2) after n steps;
    # every chosen direction is within ``delta`` of the tangent, so the position is within n step delta of that
    for half in (s[k + 1:], s[:k][::-1]):
        n = np.arange(1, len(half) + 1)
        expect = np.sqrt(r0 ** 2 + n * step ** 2)
        assert np.all(np.abs(np.hypot(half[:, 0], half[:, 1]) - expect) <= n * step * delta + 1e-4)
    n_all = np.abs(np.arange(len(s)) - k)
    assert np.all(np.abs(s[:, 2]) <= n_all * step * delta + 1e-4)       # in the plane, to the same bound
    # both halves stay in the domain and reach it: a full turn is 2 pi r0 / step points per half
    assert len(s) > 2 * np.pi * r0 / step


# ------------------------------------------------------------------ 8. invariances
def test_same_key_same_tractogram_and_chunk_size_does_not_matter(cross):
    f, _, crossing = cross
    seeds = np.argwhere(crossing).astype(float) + 0.25
    a = track(f, seeds, rule='probabilistic', key=11)
    b = track(f, seeds, rule='probabilistic', key=11)
    np.testing.assert_array_equal(a.points, b.points)
    for chunk in (1, 7, 64, 2 ** 16):
        c = track(f, seeds, rule='probabilistic', key=11, chunk=chunk)
        np.testing.assert_array_equal(c.points, a.points)
        np.testing.assert_array_equal(c.offsets, a.offsets)
        np.testing.assert_array_equal(c.stop_reason, a.stop_reason)
    d = track(f, seeds, rule='probabilistic', key=12)
    assert not np.array_equal(d.points, a.points)


def test_phase_length_does_not_matter(cross):
    """Lane compaction runs in phases of ``phase_steps``; the tractogram is bitwise the same for any phasing."""
    f, _, crossing = cross
    seeds = np.argwhere(crossing).astype(float) + 0.25
    a = track(f, seeds, rule='probabilistic', key=11, max_steps=60)
    for K in (1, 5, 32, 60, 500):
        c = track(f, seeds, rule='probabilistic', key=11, max_steps=60, phase_steps=K)
        np.testing.assert_array_equal(c.points, a.points)
        np.testing.assert_array_equal(c.offsets, a.offsets)
        np.testing.assert_array_equal(c.stop_reason, a.stop_reason)
    with pytest.raises(ValueError, match="phase_steps must be at least 1"):
        track(f, seeds, phase_steps=0)


def test_rigid_affine_maps_the_identity_streamlines(cross):
    """A rotation and translation of the grid, the FOD and the sphere rotated with it: the same streamlines mapped."""
    f, _, crossing = cross
    seeds = np.argwhere(crossing).astype(float) + 0.25
    V = hemisphere(362)
    ref = track(f, seeds, rule='probabilistic', key=2, sphere=V, step_mm=0.5)
    ang = np.deg2rad(33.0)
    R = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1.0]])
    R = R @ np.array([[1, 0, 0], [0, np.cos(0.7), -np.sin(0.7)], [0, np.sin(0.7), np.cos(0.7)]])
    A = np.eye(4)
    A[:3, :3] = R
    A[:3, 3] = [5.0, -7.0, 2.5]
    sh_rot = rotate_sh(f.sh.reshape(-1, f.n_coef).astype(np.float64), R, lmax=f.order).reshape(f.sh.shape)
    g = FODField(sh_rot, A, f.mask)
    seeds_w = seeds @ R.T + A[:3, 3]
    out = track(g, seeds_w, rule='probabilistic', key=2, sphere=V @ R.T, step_mm=0.5)
    np.testing.assert_array_equal(out.offsets, ref.offsets)
    np.testing.assert_array_equal(out.stop_reason, ref.stop_reason)
    np.testing.assert_allclose(out.points, ref.points.astype(np.float64) @ R.T + A[:3, 3], atol=2e-3)


def test_isotropic_scaling_scales_the_streamlines(cross):
    f, _, crossing = cross
    seeds = np.argwhere(crossing).astype(float) + 0.25
    ref = track(f, seeds, rule='probabilistic', key=2, step_mm=0.5)
    A = np.diag([2.0, 2.0, 2.0, 1.0])
    out = track(FODField(f.sh, A, f.mask), seeds * 2.0, rule='probabilistic', key=2, step_mm=1.0)
    np.testing.assert_array_equal(out.offsets, ref.offsets)
    np.testing.assert_array_equal(out.stop_reason, ref.stop_reason)
    np.testing.assert_allclose(out.points, ref.points * 2.0, atol=2e-3)


# ------------------------------------------------------------------ 4. the samplers
def test_first_direction_is_sampled_from_the_thresholded_fod():
    """Many seeds at one crossing position: the histogram of first directions over the sphere equals the thresholded
    FOD's mass per direction to a 4 sigma binomial band, exhaustively over the directions (a delta FOD's sample is
    its direction, the degenerate case, comes free at the directions of zero mass)."""
    f, _, crossing = crossing_field()
    V = hemisphere(362)
    B = sh_matrix(f.order, V)
    pos = np.array([15.2, 14.7, 15.1])
    pmf = B @ f.interpolate(pos[None])[0]
    pmf = np.where((pmf < 0.1 * pmf.max()) | (pmf <= 0), 0.0, pmf)
    p = pmf / pmf.sum()
    n = 6000
    # a cone of 0.5 degrees holds only the first direction itself (the sphere's directions are 5 degrees apart),
    # so the forward half's first segment at max_steps=2 is that direction exactly
    tg = track(f, np.tile(pos, (n, 1)), rule='probabilistic', step_mm=0.5, max_steps=2, sphere=V, key=9,
               max_angle=0.5)
    assert np.all(tg.n_points == 3)
    seg = np.array([s[2] - s[1] for s in tg])
    seg = seg / np.linalg.norm(seg, axis=1)[:, None]
    idx = np.argmax(np.abs(seg @ V.T), axis=1)
    counts = np.bincount(idx, minlength=V.shape[0]) / n
    band = 4 * np.sqrt(p * (1 - p) / n) + 1e-9
    assert np.all(np.abs(counts - p) <= band), np.abs(counts - p).max()
    assert np.all(counts[p == 0] == 0)


def test_inverse_cdf_index_is_the_analytic_one_at_every_breakpoint_and_midpoint():
    from dmipy_tract.tracker import _sample
    import jax.numpy as jnp
    import jax
    rng = np.random.default_rng(0)
    w = rng.random(50) * (rng.random(50) > 0.3)          # zeros interleaved: plateaus in the cdf
    w[0] = 0.0
    cdf = np.cumsum(w)
    total = cdf[-1]
    # the kernel draws u = uniform * total; feed the uniform through a stub key by monkeypatching is not needed:
    # evaluate the index rule directly on the same arithmetic the kernel uses
    def kernel_index(u):
        c = jnp.cumsum(jnp.asarray(w, jnp.float32))
        return int(jnp.minimum(jnp.sum(c <= jnp.float32(u)), w.shape[0] - 1))
    for i in range(50):
        if w[i] == 0:
            continue
        lo = cdf[i - 1] if i > 0 else 0.0
        mid = 0.5 * (lo + cdf[i])
        assert kernel_index(mid) == i                        # a u inside bin i lands in i
        assert kernel_index(lo + 1e-3 * (cdf[i] - lo)) == i  # just above the breakpoint
    assert w[kernel_index(0.0)] > 0                          # u = 0 never lands on a zero-mass entry
    assert w[kernel_index(np.nextafter(np.float32(total), np.float32(0)))] > 0
    # the kernel's sampler on a one-hot pmf is that entry, for any key
    one = np.zeros(20, np.float32)
    one[13] = 1.0
    for k in range(5):
        idx, ok = _sample(jnp.asarray(one), jax.random.key(k))
        assert int(idx) == 13 and bool(ok)
    idx, ok = _sample(jnp.zeros(20, jnp.float32), jax.random.key(0))
    assert not bool(ok)


# ------------------------------------------------------------------ the kernel against the numpy definition
@pytest.mark.parametrize("rule", ['deterministic', 'probabilistic'])
def test_kernel_equals_the_per_streamline_reference(cross, rule):
    f, _, crossing = cross
    rng = np.random.default_rng(4)
    seeds = np.argwhere(crossing).astype(float)[rng.choice(crossing.sum(), 40, replace=False)] + rng.uniform(-0.4, 0.4, (40, 3))
    kw = dict(rule=rule, step_mm=0.5, max_angle=30.0, max_steps=100, key=7)
    tg = track(f, seeds, **kw)
    ref = reference_track(f, seeds, **kw)
    for i, (pts, reasons) in enumerate(ref):
        assert tg.n_points[i] == len(pts), i
        assert tg.stop_reason[i].tolist() == list(reasons), i
        np.testing.assert_allclose(tg[i], pts, atol=1e-3, err_msg=str(i))


# ------------------------------------------------------------------ refusals
def test_refusals_by_name():
    f = uniform_field((4, 4, 4))
    s = [[1.5, 1.5, 1.5]]
    with pytest.raises(ValueError, match="rule must be one of"):
        track(f, s, rule='ifod2')
    with pytest.raises(ValueError, match="step_mm must be positive"):
        track(f, s, step_mm=0.0)
    with pytest.raises(ValueError, match="max_angle must be in \\(0, 90\\]"):
        track(f, s, max_angle=95.0)
    with pytest.raises(ValueError, match="max_angle must be in \\(0, 90\\]"):
        track(f, s, max_angle=0.0)
    with pytest.raises(ValueError, match="max_steps must be at least 1"):
        track(f, s, max_steps=0)
    with pytest.raises(ValueError, match="relative_threshold must be in \\[0, 1\\]"):
        track(f, s, relative_threshold=1.5)
    with pytest.raises(ValueError, match="seeds_mm must be \\(n, 3\\)"):
        track(f, [[1.0, 2.0]])
    with pytest.raises(ValueError, match="initial_directions must be \\(1, 3\\)"):
        track(f, s, initial_directions=np.zeros((2, 3)))
    with pytest.raises(TypeError, match="FODField"):
        track(f.sh, s)
    with pytest.raises(ValueError, match="chunk must be at least 1"):
        track(f, s, chunk=0)
