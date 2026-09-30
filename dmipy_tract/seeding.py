"""Seeds from a mask: a regular grid of ``density^3`` points per voxel, the construction of dipy's ``seeds_from_mask``
(a sub-grid centred in the voxel, then the affine), so a DiSCo seeding here is the DiSCo seeding there."""
import numpy as np

__all__ = ['seeds_from_mask']


def seeds_from_mask(mask, affine, density=1):
    """``(n, 3)`` world millimetres: ``density`` (an int, or three) points per axis in every True voxel of ``mask``,
    on the sub-grid ``(k + 0.5) / density - 0.5`` of voxel offsets, in the order ``argwhere(mask)`` lists voxels."""
    mask = np.asarray(mask).astype(bool)
    if mask.ndim != 3:
        raise ValueError(f"mask must be 3-D, not {mask.shape}")
    affine = np.asarray(affine, np.float64)
    if affine.shape != (4, 4):
        raise ValueError(f"affine must be (4, 4), not {affine.shape}")
    density = np.asarray(density)
    if not np.issubdtype(density.dtype, np.integer):
        raise ValueError(f"density must be a positive int or three of them, not {density.tolist()}")
    if density.size == 1:
        density = np.full(3, int(density))
    if density.shape != (3,) or np.any(density < 1):
        raise ValueError(f"density must be a positive int or three of them, not {density.tolist()}")
    grid = np.mgrid[0:density[0], 0:density[1], 0:density[2]].T.reshape(-1, 3)
    grid = grid / density + 0.5 / density - 0.5
    where = np.argwhere(mask)
    seeds = (where[:, None, :] + grid[None, :, :]).reshape(-1, 3)
    return seeds @ affine[:3, :3].T + affine[:3, 3]
