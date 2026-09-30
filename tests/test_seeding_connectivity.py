"""Seeds from a mask (dipy's construction, checked against dipy) and streamline counts between regions."""
import numpy as np
import pytest

from dmipy_tract import seeds_from_mask, connectivity, endpoint_labels, Tractogram


def test_density_grid_is_centred_in_the_voxel():
    mask = np.zeros((3, 3, 3), bool)
    mask[1, 2, 0] = True
    s = seeds_from_mask(mask, np.eye(4), density=1)
    np.testing.assert_array_equal(s, [[1, 2, 0]])
    s = seeds_from_mask(mask, np.eye(4), density=2)
    assert s.shape == (8, 3)
    np.testing.assert_allclose(np.sort(np.unique(s[:, 0])), [0.75, 1.25])
    np.testing.assert_allclose(s.mean(0), [1, 2, 0])
    s = seeds_from_mask(mask, np.eye(4), density=[1, 2, 3])
    assert s.shape == (6, 3)
    np.testing.assert_allclose(np.sort(np.unique(s[:, 2])), [-1 / 3, 0.0, 1 / 3])


def test_seeds_go_through_the_affine():
    mask = np.zeros((2, 2, 2), bool)
    mask[1, 1, 1] = True
    A = np.array([[2.0, 0, 0, 10], [0, 1.0, 0, 0], [0, 0, 0.5, -1], [0, 0, 0, 1]])
    np.testing.assert_allclose(seeds_from_mask(mask, A), [[12.0, 1.0, -0.5]])


def test_seeds_refusals():
    with pytest.raises(ValueError, match="3-D"):
        seeds_from_mask(np.ones((2, 2), bool), np.eye(4))
    with pytest.raises(ValueError, match="\\(4, 4\\)"):
        seeds_from_mask(np.ones((2, 2, 2), bool), np.eye(3))
    with pytest.raises(ValueError, match="density"):
        seeds_from_mask(np.ones((2, 2, 2), bool), np.eye(4), density=0)
    for density in (2.5, 2.0, [1, 2.5, 3]):
        with pytest.raises(ValueError, match="density must be a positive int"):
            seeds_from_mask(np.ones((2, 2, 2), bool), np.eye(4), density=density)


@pytest.mark.parametrize("density", [1, 2, 4, [1, 2, 3]])
def test_seeds_equal_dipy(density):
    utils = pytest.importorskip("dipy.tracking.utils")
    rng = np.random.default_rng(0)
    mask = rng.random((5, 6, 7)) > 0.6
    A = np.eye(4)
    A[:3, :3] = np.diag([0.7, 1.3, 2.0]) @ np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    A[:3, 3] = [3.0, -2.0, 1.0]
    np.testing.assert_allclose(seeds_from_mask(mask, A, density), utils.seeds_from_mask(mask, A, density=density),
                               atol=1e-12)


# ------------------------------------------------------------------ connectivity
def two_point_tractogram(ends):
    ends = np.asarray(ends, np.float32)
    n = ends.shape[0]
    return Tractogram(ends.reshape(-1, 3), np.arange(0, 2 * n + 1, 2), np.arange(n), np.ones((n, 2), np.int8))


def test_connectivity_counts_endpoint_regions():
    labels = np.zeros((6, 6, 6), np.int32)
    labels[0] = 1
    labels[5] = 2
    labels[:, 5] = 3
    tg = two_point_tractogram([[[0.2, 2, 2], [4.6, 2, 2]],       # 1 -> 2 (4.6 rounds to 5)
                               [[4.9, 2, 2], [0.1, 2, 2]],       # 2 -> 1
                               [[0.0, 5.0, 2], [2, 2, 2]],       # 3 (x=0 and y=5: y wins, it was set last) -> 0
                               [[2, 2, 2], [2, 2, 8.0]],         # 0 -> outside = 0
                               [[0.4, 2, 2], [0.3, 2, 2]]])      # 1 -> 1
    M, ends = connectivity(tg, labels, np.eye(4))
    assert M.shape == (4, 4)
    np.testing.assert_array_equal(ends, [[1, 2], [2, 1], [3, 0], [0, 0], [1, 1]])
    expect = np.zeros((4, 4), int)
    expect[1, 2] = expect[2, 1] = expect[3, 0] = expect[0, 0] = expect[1, 1] = 1
    np.testing.assert_array_equal(M, expect)


def test_endpoint_labels_use_the_affine_and_round_half_to_even():
    labels = np.zeros((4, 4, 4), np.int32)
    labels[2] = 7
    A = np.diag([2.0, 1, 1, 1])
    tg = two_point_tractogram([[[3.0, 1, 1], [5.0, 1, 1]], [[3.0, 1, 1], [4.99, 1, 1]]])
    # world x 3.0 -> voxel 1.5 -> 2 (half to even); 5.0 -> 2.5 -> 2; 4.99 -> 2.495 -> 2
    np.testing.assert_array_equal(endpoint_labels(tg, labels, A), [[7, 7], [7, 7]])


def test_connectivity_refuses_bad_labels():
    tg = two_point_tractogram([[[0, 0, 0], [1, 1, 1]]])
    with pytest.raises(ValueError, match="non-negative integers"):
        connectivity(tg, np.zeros((2, 2, 2), float), np.eye(4))
    with pytest.raises(ValueError, match="3-D"):
        connectivity(tg, np.zeros((2, 2), np.int32), np.eye(4))
