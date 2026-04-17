"""Tests for dmipy_tract.vis.tractogram_viewer."""

import sys
import types
from pathlib import Path
from unittest import mock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_streamlines(n=2, length=3):
    rng = np.random.default_rng(0)
    return [rng.standard_normal((length, 3)).astype(np.float32) for _ in range(n)]


def _make_fury_mock(tmp_path: Path, png_path: Path):
    """Return a mock fury package that writes a tiny PNG on window.record()."""

    def _record(scene, out_path, size=(1200, 900)):
        # Write a minimal 1x1 white PNG (valid PNG bytes)
        import struct, zlib
        def _chunk(tag, data):
            c = zlib.crc32(tag + data) & 0xFFFFFFFF
            return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", c)
        sig = b"\x89PNG\r\n\x1a\n"
        ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        raw = b"\x00\xff\xff\xff"
        idat = _chunk(b"IDAT", zlib.compress(raw))
        iend = _chunk(b"IEND", b"")
        Path(out_path).write_bytes(sig + ihdr + idat + iend)

    mock_scene = mock.MagicMock()

    mock_window = mock.MagicMock()
    mock_window.Scene.return_value = mock_scene
    mock_window.record.side_effect = _record
    mock_window.show = mock.MagicMock()

    mock_actor = mock.MagicMock()
    mock_actor.line.return_value = mock.MagicMock()

    fury_mock = types.ModuleType("fury")
    fury_actor_mock = types.ModuleType("fury.actor")
    fury_actor_mock.line = mock_actor.line
    fury_window_mock = types.ModuleType("fury.window")
    fury_window_mock.Scene = mock_window.Scene
    fury_window_mock.record = mock_window.record
    fury_window_mock.show = mock_window.show

    return {
        "fury": fury_mock,
        "fury.actor": fury_actor_mock,
        "fury.window": fury_window_mock,
    }, mock_window, mock_actor


# ---------------------------------------------------------------------------
# Test 1: fury absent → ImportError with clear message
# ---------------------------------------------------------------------------

def test_fury_absent_raises_import_error():
    """plot_tractogram should raise ImportError with a helpful message when fury is missing."""
    import importlib
    import dmipy_tract.vis.tractogram_viewer as mod

    # Patch the lazy imports inside the function to fail
    with mock.patch.dict(sys.modules, {"fury.actor": None, "fury.window": None}):
        importlib.reload(mod)
        with pytest.raises(ImportError, match="fury is required"):
            mod.plot_tractogram(_make_streamlines())


# ---------------------------------------------------------------------------
# Test 2: output_file path — mock fury to avoid needing a real display
# ---------------------------------------------------------------------------

def test_output_file_creates_png(tmp_path):
    """plot_tractogram with output_file should produce a PNG on disk."""
    import importlib
    import dmipy_tract.vis.tractogram_viewer as mod

    out = tmp_path / "tract.png"
    fury_mocks, mock_window, mock_actor = _make_fury_mock(tmp_path, out)

    streamlines = _make_streamlines(n=2, length=5)

    with mock.patch.dict(sys.modules, fury_mocks):
        importlib.reload(mod)
        mod.plot_tractogram(
            streamlines,
            output_file=str(out),
            size=(200, 200),
        )

    assert out.exists(), "Expected PNG file was not created."
    assert out.stat().st_size > 0, "PNG file is empty."
    mock_window.record.assert_called_once()


def test_output_file_with_scalar_map(tmp_path):
    """plot_tractogram with scalar_map should produce a PNG on disk."""
    import importlib
    import dmipy_tract.vis.tractogram_viewer as mod

    out = tmp_path / "tract_colored.png"
    streamlines = _make_streamlines(n=3, length=4)
    scalar_map = np.array([0.1, 0.5, 0.9], dtype=np.float32)

    fury_mocks, mock_window, mock_actor = _make_fury_mock(tmp_path, out)

    with mock.patch.dict(sys.modules, fury_mocks):
        importlib.reload(mod)
        mod.plot_tractogram(
            streamlines,
            scalar_map=scalar_map,
            colormap="plasma",
            output_file=str(out),
            size=(200, 200),
        )

    assert out.exists(), "Expected PNG file was not created."
    assert out.stat().st_size > 0, "PNG file is empty."
    # actor.line should have been called with a colors keyword argument
    call_kwargs = mock_actor.line.call_args
    assert call_kwargs is not None, "actor.line was never called"
    colors_arg = call_kwargs.kwargs.get("colors")
    assert colors_arg is not None, "Expected per-streamline colors to be passed to actor.line"


def test_interactive_calls_show(tmp_path):
    """plot_tractogram without output_file should call window.show."""
    import importlib
    import dmipy_tract.vis.tractogram_viewer as mod

    streamlines = _make_streamlines(n=2, length=5)
    fury_mocks, mock_window, mock_actor = _make_fury_mock(tmp_path, tmp_path / "unused.png")

    with mock.patch.dict(sys.modules, fury_mocks):
        importlib.reload(mod)
        mod.plot_tractogram(streamlines)

    mock_window.show.assert_called_once()
    mock_window.record.assert_not_called()


def test_scalar_map_length_mismatch_raises(tmp_path):
    """scalar_map with wrong length should raise ValueError."""
    import importlib
    import dmipy_tract.vis.tractogram_viewer as mod

    streamlines = _make_streamlines(n=3)
    scalar_map = np.array([0.1, 0.5])  # only 2 values for 3 streamlines

    fury_mocks, _, _ = _make_fury_mock(tmp_path, tmp_path / "unused.png")

    with mock.patch.dict(sys.modules, fury_mocks):
        importlib.reload(mod)
        with pytest.raises(ValueError, match="scalar_map length"):
            mod.plot_tractogram(streamlines, scalar_map=scalar_map, output_file="/dev/null")
