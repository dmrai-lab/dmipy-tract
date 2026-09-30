"""The direction set and the basis: exhaustive where the set is finite."""
import numpy as np
import pytest

from dmipy_tract.sphere import hemisphere, sh_matrix


def test_hemisphere_is_unit_upper_and_antipode_free():
    V = hemisphere(362)
    assert V.shape == (362, 3)
    np.testing.assert_allclose(np.linalg.norm(V, axis=1), 1.0, atol=1e-12)
    assert np.all(V[:, 2] > 0)
    G = V @ V.T
    np.fill_diagonal(G, 0.0)
    assert G.min() > -0.999           # no two directions antipodal (within 2.6 degrees)
    nn = np.degrees(np.arccos(np.clip(G.max(axis=1), -1, 1)))    # nearest-neighbour angle per direction
    assert nn.max() < 2.5 * nn.min()  # near-uniform spacing


def test_hemisphere_refuses_empty():
    with pytest.raises(ValueError):
        hemisphere(0)


def test_delta_fod_peaks_on_its_direction_for_every_sphere_direction():
    """A one-fibre FOD (the SH of a delta) evaluated on the sphere is largest on that fibre's direction: exhaustive
    over the 362 directions, at order 8."""
    V = hemisphere(362)
    B = sh_matrix(8, V)
    amplitudes = B @ B.T          # row i: the delta at V[i], evaluated on every direction
    assert np.array_equal(np.argmax(amplitudes, axis=1), np.arange(362))


def test_sh_matrix_refuses_non_unit_and_wrong_shape():
    with pytest.raises(ValueError, match="unit"):
        sh_matrix(4, np.array([[2.0, 0, 0]]))
    with pytest.raises(ValueError, match="\\(n, 3\\)"):
        sh_matrix(4, np.zeros((3, 2)))


def test_basis_is_mrtrix_tournier07_as_dipy_writes_it():
    """The basis is dmipy-sim's; dipy's ``real_sh_tournier(legacy=False)`` is the same convention (MRtrix3), to 1e-10."""
    shm = pytest.importorskip("dipy.reconst.shm")
    V = hemisphere(100)
    theta = np.arccos(V[:, 2])
    phi = np.arctan2(V[:, 1], V[:, 0])
    B_dipy, _, _ = shm.real_sh_tournier(8, theta, phi, legacy=False)
    np.testing.assert_allclose(sh_matrix(8, V), B_dipy, atol=1e-10)
