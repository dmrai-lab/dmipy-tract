"""dipy's ``LocalTracking`` as the oracle: exact for the deterministic rule (no randomness to differ by), and for
the probabilistic rule a statistical floor measured from dipy against itself.

dipy is a test dependency only; nothing in the package imports it.
"""
import numpy as np
import pytest

dipy_lt = pytest.importorskip("dipy.tracking.local_tracking")
from dipy.core.sphere import Sphere
from dipy.data import default_sphere
from dipy.direction import DeterministicMaximumDirectionGetter, ProbabilisticDirectionGetter
from dipy.tracking.stopping_criterion import BinaryStoppingCriterion

from dmipy_tract import FODField, track, sh_matrix, connectivity
from dmipy_tract.field import nearest_voxel
from dmipy_tract.tractogram import STOP_MASK, STOP_OUTSIDE, STOP_NO_DIRECTION, STOP_MAX_STEPS
from conftest import uniform_field, crossing_field, circle_field

MAX_ANGLE = 30.0
STEP = 0.5
MAX_STEPS = 500


def dipy_streamlines(field, seeds, getter_cls, sphere, first_directions=None, random_seed=0):
    """dipy on the same field: the same basis (``tournier07``, non-legacy), sphere, cone, threshold and stop."""
    dg = getter_cls.from_shcoeff(field.sh.astype(np.float64), max_angle=MAX_ANGLE, sphere=sphere, pmf_threshold=0.1,
                                 basis_type='tournier07', legacy=False)
    kw = {} if first_directions is None else dict(initial_directions=np.asarray(first_directions)[:, None, :])
    lt = dipy_lt.LocalTracking(dg, BinaryStoppingCriterion(field.mask), np.asarray(seeds, float), field.affine,
                               step_size=STEP, maxlen=MAX_STEPS - 1, random_seed=random_seed, **kw)
    return [np.asarray(s) for s in lt]


def first_directions_of(field, seeds, sphere):
    """Our deterministic first direction: the whole-sphere maximum of the thresholded FOD at the seed."""
    B = sh_matrix(field.order, sphere.vertices)
    out = np.zeros((len(seeds), 3))
    for i, s in enumerate(np.asarray(seeds, float)):
        pmf = B @ field.interpolate(s[None])[0]
        j = int(np.argmax(pmf))
        if pmf[j] > 0:
            out[i] = sphere.vertices[j]
    return out


def assert_point_for_point(ours, theirs):
    assert len(ours) == len(theirs)
    for i, (a, b) in enumerate(zip(ours, theirs)):
        assert a.shape == b.shape, f"streamline {i}: {a.shape} vs dipy {b.shape}"
        np.testing.assert_allclose(a, b, atol=1e-3, err_msg=f"streamline {i}")


# ------------------------------------------------------------------ 10. exact parity, deterministic rule
@pytest.mark.parametrize("case", ["uniform", "crossing", "circle"])
def test_deterministic_rule_equals_dipy_point_for_point(case):
    sphere = default_sphere
    if case == "uniform":
        mask = np.ones((12, 12, 12), bool)
        mask[:, 9:] = False
        field = uniform_field((12, 12, 12), direction=(1.0, 0.4, 0.1), mask=mask)
        rng = np.random.default_rng(0)
        seeds = rng.uniform(1.0, 8.0, (60, 3))
    elif case == "crossing":
        field, _, crossing = crossing_field()
        rng = np.random.default_rng(1)
        seeds = np.argwhere(field.mask).astype(float)[rng.choice(field.mask.sum(), 150, replace=False)]
        seeds += rng.uniform(-0.45, 0.45, seeds.shape)
    else:
        field, centre = circle_field(30)
        rng = np.random.default_rng(2)
        seeds = np.argwhere(field.mask).astype(float)[rng.choice(field.mask.sum(), 100, replace=False)]
        seeds += rng.uniform(-0.45, 0.45, seeds.shape)
    first = first_directions_of(field, seeds, sphere)
    ours = track(field, seeds, rule='deterministic', step_mm=STEP, max_angle=MAX_ANGLE, max_steps=MAX_STEPS,
                 sphere=sphere.vertices, initial_directions=first)
    theirs = dipy_streamlines(field, seeds, DeterministicMaximumDirectionGetter, sphere, first)
    assert_point_for_point(list(ours), theirs)


def test_deterministic_rule_equals_dipy_under_a_scaled_affine():
    """Anisotropic voxels and a translation: dipy tracks in voxel space with the step divided per axis, this package
    in world space; the same streamlines."""
    A = np.array([[2.0, 0, 0, 3.0], [0, 1.0, 0, -4.0], [0, 0, 1.5, 0.5], [0, 0, 0, 1]])
    field, _, crossing = crossing_field()
    field = FODField(field.sh, A, field.mask)
    rng = np.random.default_rng(3)
    seeds_vox = np.argwhere(field.mask).astype(float)[rng.choice(field.mask.sum(), 100, replace=False)]
    seeds = field.world_from_voxel(seeds_vox + rng.uniform(-0.45, 0.45, seeds_vox.shape))
    first = first_directions_of(field, seeds, default_sphere)
    ours = track(field, seeds, rule='deterministic', step_mm=STEP, max_angle=MAX_ANGLE, max_steps=MAX_STEPS,
                 sphere=default_sphere.vertices, initial_directions=first)
    theirs = dipy_streamlines(field, seeds, DeterministicMaximumDirectionGetter, default_sphere, first)
    assert_point_for_point(list(ours), theirs)


def test_probabilistic_rule_equals_dipy_where_there_is_nothing_to_sample():
    """A field with one direction per voxel: the sampler has one choice, so the probabilistic streamlines are the
    deterministic ones and equal dipy's exactly."""
    sphere = Sphere(xyz=np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]))
    mask = np.ones((10, 10, 10), bool)
    mask[7:] = False
    field = uniform_field((10, 10, 10), mask=mask)
    seeds = np.array([[x, 4.3, 4.3] for x in np.arange(0.5, 6.5, 0.7)])
    first = np.tile([[1.0, 0, 0]], (len(seeds), 1))
    ours = track(field, seeds, rule='probabilistic', step_mm=STEP, max_angle=MAX_ANGLE, sphere=sphere.vertices,
                 initial_directions=first)
    theirs = dipy_streamlines(field, seeds, ProbabilisticDirectionGetter, sphere, first)
    assert_point_for_point(list(ours), theirs)
    assert set(ours.stop_reason[:, 0].tolist()) == {STOP_MASK} and set(ours.stop_reason[:, 1].tolist()) == {STOP_OUTSIDE}


# ------------------------------------------------------------------ 11. statistical parity, probabilistic rule
def endpoint_density(streamlines, shape):
    img = np.zeros(shape, np.int64)
    v, _ = nearest_voxel(np.concatenate([s[[0, -1]] for s in streamlines]), np.eye(4), shape)
    np.add.at(img, (v[:, 0], v[:, 1], v[:, 2]), 1)
    return img


def weighted_dice(a, b):
    return 2.0 * np.minimum(a, b).sum() / (a.sum() + b.sum())


def pair_fractions(streamlines, labels):
    """Of the streamlines whose two ends lie in regions, the fraction per unordered region pair."""
    idx, in_grid = nearest_voxel(np.array([s[[0, -1]] for s in streamlines]), np.eye(4), labels.shape)
    ends = np.sort(np.where(in_grid, labels[idx[..., 0], idx[..., 1], idx[..., 2]], 0), axis=1)
    ok = np.all(ends > 0, axis=1)
    pairs = {(1, 2): 0, (3, 4): 0, (1, 3): 0, (1, 4): 0, (2, 3): 0, (2, 4): 0}
    for a, b in ends[ok]:
        pairs[(a, b)] += 1
    n = max(ok.sum(), 1)
    return {k: v / n for k, v in pairs.items()}, int(ok.sum())


def test_probabilistic_rule_matches_dipy_within_dipy_own_scatter():
    """The step mechanics against dipy's: the same seeds and the same first directions (dipy starts one streamline
    per FOD peak; here each seed is given each peak, so both sides track 864 streamlines from the same starts), and
    dipy against itself with three random seeds sets the floor for the endpoint-density Dice and the region-pair
    fractions.

    The seeds are jittered inside their voxels: dipy seeds its per-streamline RNG from the seed's coordinate sum,
    so seeds on a regular grid share random streams within a run (translates of one streamline), which raises
    dipy's agreement with itself above what independent draws give (measured: 0.80 against 0.76).

    The floors are dipy's, measured; the margins on them (0.02 on the Dice, the wider of twice dipy's spread and a
    4 sigma band on the pair fractions) are chosen. Under the fixed keys and seeds every draw on both sides, and so
    the outcome, is the same on every run.
    """
    field, labels, crossing = crossing_field()
    base = np.argwhere(crossing).astype(float)
    rng = np.random.default_rng(0)
    seeds = np.concatenate([base + rng.uniform(-0.45, 0.45, base.shape) for _ in range(2)])
    seeds = np.concatenate([seeds, seeds])
    first = np.concatenate([np.tile([[1.0, 0, 0]], (len(seeds) // 2, 1)), np.tile([[0, 1.0, 0]], (len(seeds) // 2, 1))])
    sphere = default_sphere
    runs = [dipy_streamlines(field, seeds, ProbabilisticDirectionGetter, sphere, first, random_seed=s)
            for s in (0, 1, 2)]
    dens = [endpoint_density(r, field.shape) for r in runs]
    fracs = [pair_fractions(r, labels)[0] for r in runs]
    floor_dice = min(weighted_dice(dens[i], dens[j]) for i in range(3) for j in range(i + 1, 3))
    spread = {k: max(abs(fracs[i][k] - fracs[j][k]) for i in range(3) for j in range(i + 1, 3)) for k in fracs[0]}
    ours = track(field, seeds, rule='probabilistic', step_mm=STEP, max_angle=MAX_ANGLE, max_steps=MAX_STEPS,
                 sphere=sphere.vertices, initial_directions=first, key=0)
    our_list = list(ours)
    assert len(our_list) == len(runs[0])
    our_dens = endpoint_density(our_list, field.shape)
    our_fracs, n_conn = pair_fractions(our_list, labels)
    # endpoint density: dipy's own Dice is the floor (with a margin for the sample size of the floor itself)
    dice = min(weighted_dice(our_dens, d) for d in dens)
    assert dice >= floor_dice - 0.02, (dice, floor_dice)
    # region pairs: within dipy's spread or a 4 sigma binomial band, whichever is wider
    for k in our_fracs:
        mean = np.mean([f[k] for f in fracs])
        sigma = np.sqrt(max(mean * (1 - mean), 1e-4) / n_conn)
        assert abs(our_fracs[k] - mean) <= max(2 * spread[k], 4 * sigma), (k, our_fracs[k], mean, spread[k], sigma)
    # the crossing is followed straight through: the two true bundles carry the mass, not the turns
    assert our_fracs[(1, 2)] + our_fracs[(3, 4)] > 0.95
    # the same through the package's own connectivity
    M, _ = connectivity(ours, labels, field.affine)
    assert M[1, 2] + M[2, 1] + M[3, 4] + M[4, 3] == round((our_fracs[(1, 2)] + our_fracs[(3, 4)]) * n_conn)
