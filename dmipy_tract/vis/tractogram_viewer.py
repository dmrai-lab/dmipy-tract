"""Fury-based tractogram viewer for dmipy-tract."""

from __future__ import annotations

import numpy as np


def plot_tractogram(
    streamlines: list,
    scalar_map=None,
    colormap: str = "viridis",
    linewidth: float = 1.0,
    output_file: str | None = None,
    size: tuple = (1200, 900),
) -> None:
    """Visualize streamlines using FURY.

    Parameters
    ----------
    streamlines : list of np.ndarray
        List of streamlines, each shape (L, 3) in mm space.
    scalar_map : np.ndarray, optional
        Per-streamline scalar values, shape (N_streamlines,), used for coloring.
        Values are normalized to [0, 1] and mapped through ``colormap``.
    colormap : str
        Matplotlib colormap name used when ``scalar_map`` is provided.
        Defaults to "viridis".
    linewidth : float
        Line width for rendered streamlines.
    output_file : str or None
        If given, render to a PNG file at this path instead of opening an
        interactive window.
    size : tuple of (int, int)
        Window size in pixels (width, height).

    Raises
    ------
    ImportError
        If the ``fury`` package is not installed.
    """
    try:
        import fury.actor as actor
        import fury.window as window
    except ImportError as exc:
        raise ImportError(
            "fury is required for tractogram visualization. "
            "Install it with: pip install fury"
        ) from exc

    # Build per-streamline colors ------------------------------------------------
    if scalar_map is not None:
        scalar_map = np.asarray(scalar_map, dtype=float)
        if scalar_map.shape[0] != len(streamlines):
            raise ValueError(
                f"scalar_map length ({scalar_map.shape[0]}) must match "
                f"number of streamlines ({len(streamlines)})."
            )
        # Normalize to [0, 1]
        vmin, vmax = scalar_map.min(), scalar_map.max()
        if vmax > vmin:
            norm = (scalar_map - vmin) / (vmax - vmin)
        else:
            norm = np.zeros_like(scalar_map)

        import matplotlib
        cmap = matplotlib.colormaps[colormap]
        # fury expects per-streamline colors as (N, 3) float in [0, 1]
        colors = cmap(norm)[:, :3].astype(np.float32)
    else:
        colors = None

    # Build scene ----------------------------------------------------------------
    scene = window.Scene()

    stream_actor = actor.line(streamlines, colors=colors, linewidth=linewidth)
    scene.add(stream_actor)

    if output_file is not None:
        window.record(scene, out_path=output_file, size=size)
    else:
        window.show(scene, size=size)
