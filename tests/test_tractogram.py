"""The ragged tractogram: validation, indexing, lengths, selection, concatenation and the ``.tck`` round trip."""
import numpy as np
import pytest

from dmipy_sim.io.strands import read_tck

from dmipy_tract import Tractogram


def make(lists, reasons=None):
    counts = np.array([len(p) for p in lists])
    n = len(lists)
    return Tractogram(np.concatenate(lists), np.concatenate([[0], np.cumsum(counts)]), np.arange(n),
                      np.ones((n, 2), np.int8) if reasons is None else reasons)


def test_validation():
    pts = np.zeros((5, 3), np.float32)
    with pytest.raises(ValueError, match="offsets"):
        Tractogram(pts, [0, 2, 4], np.arange(2), np.ones((2, 2), np.int8))          # does not end at 5
    with pytest.raises(ValueError, match="offsets"):
        Tractogram(pts, [0, 2, 2, 5], np.arange(3), np.ones((3, 2), np.int8))       # an empty streamline
    with pytest.raises(ValueError, match="seed_index must be \\(2,\\)"):
        Tractogram(pts, [0, 2, 5], np.arange(3), np.ones((2, 2), np.int8))


def test_indexing_iteration_and_counts():
    a = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], np.float32)
    b = np.array([[5, 5, 5]], np.float32)
    tg = make([a, b])
    assert len(tg) == 2
    np.testing.assert_array_equal(tg[0], a)
    np.testing.assert_array_equal(tg[-1], b)
    np.testing.assert_array_equal([len(s) for s in tg], [3, 1])
    np.testing.assert_array_equal(tg.n_points, [3, 1])
    np.testing.assert_allclose(tg.lengths_mm, [2.0, 0.0])
    np.testing.assert_array_equal(tg.endpoints, [[a[0], a[-1]], [b[0], b[0]]])
    assert [x.shape for x in tg.to_lists()] == [(3, 3), (1, 3)]


def test_lengths_are_arc_lengths_not_chords():
    a = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], np.float32)
    assert make([a]).lengths_mm[0] == pytest.approx(3.0)


def test_select_and_concatenate():
    lists = [np.full((k, 3), k, np.float32) for k in (1, 2, 3, 4)]
    tg = make(lists)
    s = tg.select(np.array([False, True, False, True]))
    assert len(s) == 2 and s.n_points.tolist() == [2, 4] and s.seed_index.tolist() == [1, 3]
    np.testing.assert_array_equal(s[1], lists[3])
    s2 = tg.select(np.array([3, 0]))
    assert s2.n_points.tolist() == [4, 1] and s2.seed_index.tolist() == [3, 0]
    c = Tractogram.concatenate([tg, s])
    assert len(c) == 6 and c.n_points.tolist() == [1, 2, 3, 4, 2, 4]
    np.testing.assert_array_equal(c[5], lists[3])
    e = Tractogram.concatenate([])
    assert len(e) == 0 and e.points.shape == (0, 3)


def test_tck_round_trip_in_millimetres(tmp_path):
    rng = np.random.default_rng(0)
    lists = [rng.uniform(-50, 50, (k, 3)).astype(np.float32) for k in (2, 7, 1, 30)]
    tg = make(lists)
    path = tmp_path / "t.tck"
    tg.to_tck(path)
    back = read_tck(path, coordinate_unit_m=1.0)              # the file's own units: millimetres
    assert len(back) == 4
    for a, b in zip(lists, back):
        np.testing.assert_allclose(b, a, atol=1e-4)
    head = path.read_bytes()[:200]
    assert head.startswith(b"mrtrix tracks") and b"count: 4" in head and b"Float32LE" in head
