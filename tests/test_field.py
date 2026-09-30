"""FODField: validation, interpolation against closed forms and scipy, the mask lookup, and the kernel's own
interpolant against the numpy one."""
import numpy as np
import pytest
import jax.numpy as jnp
from scipy.ndimage import map_coordinates

from dmipy_tract import FODField, hemisphere, sh_matrix
from dmipy_tract.field import _trilinear_indices, nearest_voxel
from dmipy_tract.tracker import _interpolate


def random_field(shape=(6, 7, 8), n_coef=15, seed=0, affine=None):
    rng = np.random.default_rng(seed)
    sh = rng.standard_normal(shape + (n_coef,)).astype(np.float32)
    return FODField(sh, np.eye(4) if affine is None else affine, np.ones(shape, bool))


# ------------------------------------------------------------------ refusals (by name)
def test_refuses_coefficient_count_that_is_no_even_order():
    with pytest.raises(ValueError, match="16 coefficients is not an even-order"):
        FODField(np.zeros((2, 2, 2, 16)), np.eye(4), np.ones((2, 2, 2), bool))


@pytest.mark.parametrize("order", [0, 2, 4, 6, 8, 10, 12])
def test_every_even_order_is_accepted(order):
    n_coef = (order + 1) * (order + 2) // 2
    assert FODField(np.zeros((2, 2, 2, n_coef)), np.eye(4), np.ones((2, 2, 2), bool)).order == order


@pytest.mark.parametrize("n", [0, 2, 3, 4, 5, 7, 10, 14, 16, 27, 29, 44, 46])
def test_every_count_that_is_no_even_order_is_refused(n):
    with pytest.raises(ValueError, match=f"{n} coefficients is not an even-order"):
        FODField(np.zeros((2, 2, 2, n)), np.eye(4), np.ones((2, 2, 2), bool))


def test_refuses_non_4d_sh():
    with pytest.raises(ValueError, match="X, Y, Z, n_coef"):
        FODField(np.zeros((2, 2, 15)), np.eye(4), np.ones((2, 2), bool))


def test_refuses_singular_affine():
    A = np.eye(4)
    A[1, 1] = 0.0
    with pytest.raises(ValueError, match="invertible"):
        FODField(np.zeros((2, 2, 2, 15)), A, np.ones((2, 2, 2), bool))


def test_refuses_affine_that_is_not_4x4_or_not_affine():
    with pytest.raises(ValueError, match="\\(4, 4\\)"):
        FODField(np.zeros((2, 2, 2, 15)), np.eye(3), np.ones((2, 2, 2), bool))
    A = np.eye(4)
    A[3, 0] = 1.0
    with pytest.raises(ValueError, match="last row"):
        FODField(np.zeros((2, 2, 2, 15)), A, np.ones((2, 2, 2), bool))


def test_refuses_mask_of_another_shape():
    with pytest.raises(ValueError, match="mask shape \\(2, 2, 3\\) differs from the grid \\(2, 2, 2\\)"):
        FODField(np.zeros((2, 2, 2, 15)), np.eye(4), np.ones((2, 2, 3), bool))


def test_field_is_float32_and_mask_bool():
    f = FODField(np.zeros((2, 2, 2, 15), np.float64), np.eye(4), np.ones((2, 2, 2), int))
    assert f.sh.dtype == np.float32 and f.mask.dtype == bool and f.order == 4 and f.n_coef == 15


# ------------------------------------------------------------------ interpolation: closed forms
def test_interpolation_at_voxel_centres_is_the_voxel_value():
    f = random_field()
    ijk = np.stack(np.meshgrid(*[np.arange(s) for s in f.shape], indexing='ij'), -1).reshape(-1, 3)
    np.testing.assert_array_equal(f.interpolate(ijk.astype(float)), f.sh.reshape(-1, f.n_coef).astype(np.float64))


def test_interpolation_at_corners_and_edge_midpoints_is_the_mean():
    f = random_field()
    sh = f.sh.astype(np.float64)
    corner = np.array([[1.5, 2.5, 3.5]])
    expect = sh[1:3, 2:4, 3:5].reshape(-1, f.n_coef).mean(0)
    np.testing.assert_allclose(f.interpolate(corner)[0], expect, rtol=1e-12, atol=1e-12)
    edge = np.array([[1.5, 2.0, 3.0]])
    np.testing.assert_allclose(f.interpolate(edge)[0], 0.5 * (sh[1, 2, 3] + sh[2, 2, 3]), rtol=1e-12, atol=1e-12)


def test_interpolation_matches_scipy_order_1_nearest_mode_at_random_points():
    f = random_field()
    rng = np.random.default_rng(1)
    dims = np.asarray(f.shape)
    pts = rng.uniform(-0.5, dims - 0.5, size=(1000, 3))          # the whole domain, edge half-voxels included
    ours = f.interpolate(pts)
    for c in range(f.n_coef):
        ref = map_coordinates(f.sh[..., c].astype(np.float64), pts.T, order=1, mode='nearest')
        np.testing.assert_allclose(ours[:, c], ref, rtol=1e-10, atol=1e-10)


def test_outer_half_voxel_extends_the_edge_value_and_beyond_is_zero():
    f = random_field()
    inside_edge = np.array([[-0.5, 0.0, 0.0], [-0.25, 0.0, 0.0], [5.5, 0.0, 0.0], [5.49, 0.0, 0.0]])
    out = f.interpolate(inside_edge)
    np.testing.assert_array_equal(out[0], f.sh[0, 0, 0])
    np.testing.assert_array_equal(out[1], f.sh[0, 0, 0])
    np.testing.assert_array_equal(out[2], f.sh[5, 0, 0])
    np.testing.assert_array_equal(out[3], f.sh[5, 0, 0])
    beyond = np.array([[-0.51, 0.0, 0.0], [5.51, 0.0, 0.0], [0.0, 0.0, 7.5001]])
    np.testing.assert_array_equal(f.interpolate(beyond), 0.0)


def test_trilinear_indices_weights_sum_to_one_and_stay_in_grid():
    idx, w, inside = _trilinear_indices(np.random.default_rng(2).uniform(-1, 9, (500, 3)), (6, 7, 8))
    np.testing.assert_allclose(w.sum(1), 1.0, atol=1e-12)
    assert idx.min() >= 0 and np.all(idx.max(axis=(0, 1)) <= np.array([5, 6, 7]))


def test_interpolation_goes_through_the_affine():
    A = np.array([[2.0, 0, 0, 10.0], [0, 0.5, 0, -3.0], [0, 0, 1.5, 4.0], [0, 0, 0, 1]])
    f = random_field(affine=A)
    ijk = np.array([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]])
    world = f.world_from_voxel(ijk)
    np.testing.assert_allclose(f.voxel_from_world(world), ijk, atol=1e-12)
    np.testing.assert_allclose(f.interpolate(world), random_field().interpolate(ijk), rtol=1e-12)


# ------------------------------------------------------------------ the kernel's interpolant
def jax_interpolant(f, pts):
    flat = jnp.asarray(f.sh.reshape(-1, f.n_coef))
    dims = jnp.asarray(f.shape, jnp.int32)
    return np.array([_interpolate(flat, dims, jnp.asarray(p, jnp.float32)) for p in pts])


def torch_interpolant(f, pts):
    torch = pytest.importorskip("torch")
    from dmipy_tract._torch import _Field
    V = hemisphere()
    F = _Field(f, V, sh_matrix(f.order, V), 0.5, 0.5, 0.1, "cpu")
    return F.interpolate(torch.as_tensor(pts, dtype=torch.float32)).numpy()


@pytest.mark.parametrize("interpolant", [jax_interpolant, torch_interpolant], ids=["jax", "torch"])
def test_kernel_interpolant_equals_numpy_to_float32(interpolant):
    """The numpy interpolant (checked above against closed forms and scipy) is the oracle for each kernel's."""
    f = random_field()
    rng = np.random.default_rng(3)
    dims = np.asarray(f.shape)
    pts = np.concatenate([rng.uniform(-0.5, dims - 0.5, size=(300, 3)), [[-0.6, 1, 1], [1, 1, 7.6]]])
    np.testing.assert_allclose(interpolant(f, pts), f.interpolate(pts), rtol=2e-5, atol=2e-5)


# ------------------------------------------------------------------ the mask lookup
def test_mask_lookup_is_nearest_voxel_with_round_half_to_even():
    mask = np.ones((10, 10, 10), bool)
    mask[7:] = False
    f = FODField(np.zeros((10, 10, 10, 15)), np.eye(4), mask)
    pts = np.array([[6.49, 0, 0], [6.5, 0, 0], [6.51, 0, 0], [-0.5, 0, 0], [-0.51, 0, 0], [9.5, 0, 0], [0, 0, 8.5]])
    #                    in      6.5->6 in   7 out      -0.5->0 in   out    9.5->10 out    8.5->8 in
    np.testing.assert_array_equal(f.in_mask(pts), [True, True, False, True, False, False, True])
    idx, in_grid = nearest_voxel(pts, f.affine, f.shape)
    np.testing.assert_array_equal(in_grid, [True, True, True, True, False, False, True])
    np.testing.assert_array_equal(idx, [[6, 0, 0], [6, 0, 0], [7, 0, 0], [0, 0, 0], [0, 0, 0], [9, 0, 0], [0, 0, 8]])


def test_nearest_voxel_goes_through_the_affine_and_keeps_the_leading_shape():
    A = np.diag([2.0, 1.0, 1.0, 1.0])
    A[:3, 3] = [1.0, 0.0, 0.0]
    pts = np.array([[[4.0, 1, 1], [6.0, 1, 1]], [[-0.1, 1, 1], [8.1, 1, 1]]])    # voxel x 1.5, 2.5, -0.55, 3.55
    idx, in_grid = nearest_voxel(pts, A, (4, 4, 4))
    assert idx.shape == (2, 2, 3) and in_grid.shape == (2, 2)
    np.testing.assert_array_equal(idx[..., 0], [[2, 2], [0, 3]])
    np.testing.assert_array_equal(in_grid, [[True, True], [False, False]])
