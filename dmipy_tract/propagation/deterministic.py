"""
Deterministic 4th-order Runge-Kutta peak tracking.

This is the Phase 0.5 deliverable: the simplest tractography algorithm
that proves the MicrostructureField interface works end-to-end and produces
a .trk file compatible with MRview/TrackVis.
"""

import numpy as np
from ..core.stopping_criteria import CompositeStopping, default_stopping


class DeterministicTracker:
    """4th-order Runge-Kutta deterministic streamline tracker.

    Propagates along the principal peak direction from ``MicrostructureField``.
    Stops when any stopping criterion is met or the maximum length is reached.

    Parameters
    ----------
    field : MicrostructureField
    step_size_mm : float
        Step size in mm.  Default 0.5 mm.
    max_angle_deg : float
        Maximum turning angle per step in degrees.  Default 30°.
    max_length_mm : float
        Maximum streamline length in mm.  Default 250 mm.
    min_length_mm : float
        Minimum streamline length to keep.  Shorter streamlines are discarded.
    stopping : CompositeStopping or None
        Stopping criteria.  Defaults to ``default_stopping()``.
    """

    def __init__(
        self,
        field,
        step_size_mm: float = 0.5,
        max_angle_deg: float = 30.0,
        max_length_mm: float = 250.0,
        min_length_mm: float = 10.0,
        stopping: CompositeStopping | None = None,
    ):
        self.field = field
        self.step_size = float(step_size_mm)
        self.cos_max_angle = float(np.cos(np.deg2rad(max_angle_deg)))
        self.max_steps = int(max_length_mm / step_size_mm)
        self.min_steps = int(min_length_mm / step_size_mm)
        self.stopping = stopping or default_stopping()

    def track(self, seeds: np.ndarray) -> list[np.ndarray]:
        """Propagate streamlines from all seed positions.

        Parameters
        ----------
        seeds : ndarray, shape (N_seeds, 3)
            World-space seed positions.

        Returns
        -------
        streamlines : list of ndarray, each shape (L, 3)
            Completed streamlines (length >= min_length_mm only).
        """
        streamlines = []
        for seed in seeds:
            sl = self._propagate_single(seed)
            if sl is not None:
                streamlines.append(sl)
        return streamlines

    def _propagate_single(self, seed: np.ndarray) -> np.ndarray | None:
        """Propagate one streamline in both directions from seed."""
        fwd = self._propagate_direction(seed, direction=None, forward=True)
        bwd = self._propagate_direction(seed, direction=None, forward=False)
        if fwd is None and bwd is None:
            return None
        parts = []
        if bwd is not None:
            parts.append(bwd[::-1])
        parts.append(np.atleast_2d(seed))
        if fwd is not None:
            parts.append(fwd)
        sl = np.vstack(parts)
        if len(sl) < self.min_steps:
            return None
        return sl.astype(np.float32)

    def _propagate_direction(
        self,
        start: np.ndarray,
        direction: np.ndarray | None,
        forward: bool,
    ) -> np.ndarray | None:
        sign = 1.0 if forward else -1.0
        pos = np.array(start, dtype=np.float64)
        points = []

        for _ in range(self.max_steps):
            if self.stopping.should_stop(self.field, pos):
                break
            peaks = self.field.peaks_at(pos)
            if peaks is None or len(peaks) == 0 or np.all(peaks == 0):
                break

            peak = peaks[0]   # principal direction
            if direction is not None:
                # Flip peak if it points against the current direction
                if np.dot(peak, direction) < 0:
                    peak = -peak
                # Check curvature constraint
                if np.dot(peak, direction) < self.cos_max_angle:
                    break
            direction = peak

            # RK4 step — k1..k4 are unsigned unit directions (aligned with `direction`).
            # `sign` is applied only in the sub-step position offsets and final update so
            # that _get_direction's flip logic always operates on unsigned vectors.
            k1 = self._get_direction(pos, direction)
            k2 = self._get_direction(pos + sign * 0.5 * self.step_size * k1, k1)
            k3 = self._get_direction(pos + sign * 0.5 * self.step_size * k2, k2)
            k4 = self._get_direction(pos + sign * self.step_size * k3, k3)
            step_dir = (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
            norm = np.linalg.norm(step_dir)
            if norm < 1e-8:
                break
            step_dir /= norm
            direction = step_dir

            pos = pos + sign * self.step_size * step_dir
            points.append(pos.copy())

        if len(points) < self.min_steps:
            return None
        return np.array(points)

    def _get_direction(self, pos: np.ndarray, prev_dir: np.ndarray) -> np.ndarray:
        peaks = self.field.peaks_at(pos)
        if peaks is None or len(peaks) == 0 or np.all(peaks[0] == 0):
            return prev_dir
        peak = peaks[0]
        if np.dot(peak, prev_dir) < 0:
            peak = -peak
        return peak
