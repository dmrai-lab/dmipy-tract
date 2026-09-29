"""An FOD field on an image grid: the one input every tracker in this package takes.

DiSCo, the BATMAN brain, an MRtrix ``.mif`` FOD and a dmipy-fit CSD are the same object here: even-order real
spherical-harmonics coefficients per voxel in dmipy-sim's orthonormal basis (:func:`dmipy_sim.replay.so3.real_sh`,
MRtrix's ``tournier07``), a voxel-to-world affine in millimetres, and a boolean mask that is the tracking domain.
The FOD's directions are in world coordinates, as MRtrix stores them; for an image whose affine is a scaling and
a translation this is also dipy's voxel-axis frame.

Interpolation is trilinear on the coefficients (then the FOD is evaluated), on the closed domain ``[-0.5, N - 0.5]``
per axis with the edge voxel's value extended over its outer half: ``scipy.ndimage.map_coordinates(order=1,
mode='nearest')`` and dipy's ``trilinear_interpolate4d`` compute the same numbers. A mask is looked up at the nearest
voxel (round half to even), the only meaningful lookup for a binary image.
"""
from dataclasses import dataclass, field

import numpy as np

from .sphere import order_from_ncoef

__all__ = ['FODField', 'trilinear_indices']


def trilinear_indices(voxel_coords, shape):
    """The eight corner indices ``(n, 8, 3)`` and weights ``(n, 8)`` of the clamped trilinear interpolant at
    ``voxel_coords (n, 3)``; ``inside (n,)`` is False where a coordinate leaves ``[-0.5, N - 0.5]``."""
    p = np.asarray(voxel_coords, np.float64).reshape(-1, 3)
    dims = np.asarray(shape[:3], np.int64)
    flr = np.floor(p)
    rem = p - flr
    i0 = np.clip(flr, 0, dims - 1).astype(np.int64)
    i1 = np.clip(flr + 1, 0, dims - 1).astype(np.int64)
    inside = np.all((p >= -0.5) & (p <= dims - 0.5), axis=1)
    idx = np.empty((p.shape[0], 8, 3), np.int64)
    w = np.empty((p.shape[0], 8), np.float64)
    k = 0
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                idx[:, k, 0] = i1[:, 0] if a else i0[:, 0]
                idx[:, k, 1] = i1[:, 1] if b else i0[:, 1]
                idx[:, k, 2] = i1[:, 2] if c else i0[:, 2]
                w[:, k] = ((rem[:, 0] if a else 1 - rem[:, 0]) * (rem[:, 1] if b else 1 - rem[:, 1])
                           * (rem[:, 2] if c else 1 - rem[:, 2]))
                k += 1
    return idx, w, inside


@dataclass(frozen=True)
class FODField:
    """``sh (X, Y, Z, n_coef)`` float32, ``affine (4, 4)`` voxel index to world millimetres, ``mask (X, Y, Z)`` bool.

    ``n_coef`` must be an even-order count (15, 28, 45, 66 for orders 4, 6, 8, 10); the affine must be invertible;
    the mask must have the grid's shape. Anything else is refused by name.
    """
    sh: np.ndarray
    affine: np.ndarray
    mask: np.ndarray
    order: int = field(init=False)

    def __post_init__(self):
        sh = np.ascontiguousarray(np.asarray(self.sh), dtype=np.float32)
        if sh.ndim != 4:
            raise ValueError(f"sh must be (X, Y, Z, n_coef), not shape {sh.shape}")
        order = order_from_ncoef(sh.shape[3])
        affine = np.asarray(self.affine, np.float64)
        if affine.shape != (4, 4):
            raise ValueError(f"affine must be (4, 4), not {affine.shape}")
        if not np.all(np.isfinite(affine)) or abs(np.linalg.det(affine[:3, :3])) < 1e-12 \
                or not np.allclose(affine[3], [0, 0, 0, 1]):
            raise ValueError("affine must be an invertible voxel-to-world map with last row [0, 0, 0, 1]")
        mask = np.asarray(self.mask)
        if mask.shape != sh.shape[:3]:
            raise ValueError(f"mask shape {mask.shape} differs from the grid {sh.shape[:3]}")
        mask = np.ascontiguousarray(mask.astype(bool))
        object.__setattr__(self, 'sh', sh)
        object.__setattr__(self, 'affine', affine)
        object.__setattr__(self, 'mask', mask)
        object.__setattr__(self, 'order', order)

    @property
    def shape(self):
        return self.sh.shape[:3]

    @property
    def n_coef(self):
        return self.sh.shape[3]

    @property
    def inverse_affine(self):
        return np.linalg.inv(self.affine)

    def voxel_from_world(self, points_mm):
        """World millimetres ``(n, 3)`` to continuous voxel coordinates."""
        p = np.asarray(points_mm, np.float64)
        inv = self.inverse_affine
        return p @ inv[:3, :3].T + inv[:3, 3]

    def world_from_voxel(self, voxel_coords):
        v = np.asarray(voxel_coords, np.float64)
        return v @ self.affine[:3, :3].T + self.affine[:3, 3]

    def interpolate(self, points_mm):
        """The coefficients at world points ``(n, 3)``, ``(n, n_coef)`` float64; zero outside the domain."""
        idx, w, inside = trilinear_indices(self.voxel_from_world(points_mm), self.shape)
        c = self.sh[idx[..., 0], idx[..., 1], idx[..., 2]].astype(np.float64)     # (n, 8, n_coef)
        out = np.einsum('nk,nkc->nc', w, c)
        out[~inside] = 0.0
        return out

    def in_mask(self, points_mm):
        """The mask at the nearest voxel of each world point; False outside the grid."""
        v = np.rint(self.voxel_from_world(points_mm)).astype(np.int64)
        dims = np.asarray(self.shape)
        inside = np.all((v >= 0) & (v < dims), axis=1)
        vc = np.clip(v, 0, dims - 1)
        return inside & self.mask[vc[:, 0], vc[:, 1], vc[:, 2]]

    @classmethod
    def from_mif(cls, path, mask=None):
        """An MRtrix FOD image (``dwi2fod``'s output, ``tournier07`` basis, which is this package's); the mask is
        ``mask`` or, absent, the voxels whose ``l = 0`` coefficient is positive."""
        from dmipy_sim.io.mrtrix import read_mif
        img = read_mif(path)
        sh = np.asarray(img.data, np.float32)
        if mask is None:
            mask = sh[..., 0] > 0
        return cls(sh, img.affine, mask)
