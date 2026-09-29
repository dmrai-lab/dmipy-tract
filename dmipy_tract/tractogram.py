"""The tracker's output: streamlines as one ragged float32 array in world millimetres, with per-streamline
provenance (which seed, why each half stopped)."""
from dataclasses import dataclass

import numpy as np

__all__ = ['Tractogram', 'STOP_REASONS', 'STOP_NONE', 'STOP_MASK', 'STOP_OUTSIDE', 'STOP_NO_DIRECTION',
           'STOP_MAX_STEPS']

STOP_NONE = 0            # the half never started (a padded lane)
STOP_MASK = 1            # the next point's nearest voxel is outside the mask
STOP_OUTSIDE = 2         # the next point's nearest voxel is outside the grid
STOP_NO_DIRECTION = 3    # no direction inside the cone above the threshold (or the FOD is not defined here)
STOP_MAX_STEPS = 4       # the half reached max_steps points
STOP_REASONS = {STOP_NONE: 'none', STOP_MASK: 'mask', STOP_OUTSIDE: 'outside', STOP_NO_DIRECTION: 'no_direction',
                STOP_MAX_STEPS: 'max_steps'}


@dataclass(frozen=True)
class Tractogram:
    """``points (total, 3)`` float32 world millimetres; streamline ``i`` is ``points[offsets[i]:offsets[i + 1]]``;
    ``seed_index (n,)`` the seed each came from; ``stop_reason (n, 2)`` why the forward and the backward half
    ended (:data:`STOP_REASONS`). The seed is a point of every streamline."""
    points: np.ndarray
    offsets: np.ndarray
    seed_index: np.ndarray
    stop_reason: np.ndarray

    def __post_init__(self):
        points = np.ascontiguousarray(np.asarray(self.points, np.float32).reshape(-1, 3))
        offsets = np.asarray(self.offsets, np.int64)
        n = offsets.shape[0] - 1
        if n < 0 or offsets[0] != 0 or offsets[-1] != points.shape[0] or np.any(np.diff(offsets) < 1):
            raise ValueError("offsets must start at 0, end at len(points) and every streamline must have a point")
        seed_index = np.asarray(self.seed_index, np.int64)
        stop_reason = np.asarray(self.stop_reason, np.int8)
        if seed_index.shape != (n,) or stop_reason.shape != (n, 2):
            raise ValueError(f"seed_index must be ({n},) and stop_reason ({n}, 2)")
        object.__setattr__(self, 'points', points)
        object.__setattr__(self, 'offsets', offsets)
        object.__setattr__(self, 'seed_index', seed_index)
        object.__setattr__(self, 'stop_reason', stop_reason)

    def __len__(self):
        return self.offsets.shape[0] - 1

    def __getitem__(self, i):
        i = int(i)
        if i < 0:
            i += len(self)
        return self.points[self.offsets[i]:self.offsets[i + 1]]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @property
    def n_points(self):
        """Points per streamline, ``(n,)``."""
        return np.diff(self.offsets)

    @property
    def lengths_mm(self):
        """Arc length per streamline, ``(n,)``, the sum of its segment lengths."""
        seg = np.linalg.norm(np.diff(self.points, axis=0).astype(np.float64), axis=1)
        keep = np.ones(seg.shape[0], bool)
        keep[self.offsets[1:-1] - 1] = False          # the join between consecutive streamlines is no segment
        seg = seg * keep
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        starts = self.offsets[:-1]
        ends = self.offsets[1:] - 1
        return cum[ends] - cum[starts]

    @property
    def endpoints(self):
        """``(n, 2, 3)``: the first and the last point of every streamline."""
        return np.stack([self.points[self.offsets[:-1]], self.points[self.offsets[1:] - 1]], axis=1)

    def select(self, keep):
        """The streamlines where ``keep (n,)`` is True (or the ones at integer indices), as a new Tractogram."""
        keep = np.asarray(keep)
        idx = np.flatnonzero(keep) if keep.dtype == bool else keep.astype(np.int64)
        n_pts = self.n_points[idx]
        starts = self.offsets[idx]
        flat = np.repeat(starts - np.concatenate([[0], np.cumsum(n_pts)[:-1]]), n_pts) + np.arange(int(n_pts.sum()))
        return Tractogram(self.points[flat], np.concatenate([[0], np.cumsum(n_pts)]), self.seed_index[idx],
                          self.stop_reason[idx])

    def to_lists(self):
        """A list of ``(n_i, 3)`` arrays (views), the shape dipy and nibabel consume."""
        return np.split(self.points, self.offsets[1:-1])

    def to_tck(self, path):
        """Write as an MRtrix ``.tck`` (Float32LE, millimetres) through dmipy-sim's writer."""
        from dmipy_sim.io.strands import write_tck
        write_tck(path, [p.astype(np.float64) * 1e-3 for p in self.to_lists()], coordinate_unit_m=1e-3)

    @classmethod
    def concatenate(cls, parts):
        parts = list(parts)
        if not parts:
            return cls(np.zeros((0, 3), np.float32), np.zeros(1, np.int64), np.zeros(0, np.int64),
                       np.zeros((0, 2), np.int8))
        counts = np.concatenate([p.n_points for p in parts])
        return cls(np.concatenate([p.points for p in parts]), np.concatenate([[0], np.cumsum(counts)]),
                   np.concatenate([p.seed_index for p in parts]), np.concatenate([p.stop_reason for p in parts]))
