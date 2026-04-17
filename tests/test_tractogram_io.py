import numpy as np
import pytest
from dmipy_tract.io.tractogram_io import save_trk, load_trk


def test_save_load_roundtrip(tmp_path):
    """Streamlines survive a save/load TRK roundtrip."""
    rng = np.random.default_rng(0)
    # Two streamlines of different lengths
    sl1 = rng.uniform(0, 10, (20, 3)).astype(np.float32)
    sl2 = rng.uniform(0, 10, (35, 3)).astype(np.float32)
    streamlines = [sl1, sl2]
    affine = np.eye(4, dtype=np.float32)
    path = str(tmp_path / "test.trk")

    save_trk(streamlines, affine, path)
    loaded, loaded_affine = load_trk(path)

    assert len(loaded) == 2
    assert loaded[0].shape == sl1.shape
    assert loaded[1].shape == sl2.shape
    np.testing.assert_allclose(loaded[0], sl1, atol=1e-3)
    np.testing.assert_allclose(loaded[1], sl2, atol=1e-3)


def test_save_empty_streamlines(tmp_path):
    """Empty streamline list saves and loads without error."""
    path = str(tmp_path / "empty.trk")
    save_trk([], np.eye(4), path)
    loaded, _ = load_trk(path)
    assert len(loaded) == 0
