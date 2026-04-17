"""
Seed generation strategies for tractography.
"""

import numpy as np


def seeds_from_mask(
    mask: np.ndarray,
    affine: np.ndarray,
    density: int = 1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate seed positions from a binary voxel mask.

    Parameters
    ----------
    mask : ndarray bool, shape (X, Y, Z)
    affine : ndarray, shape (4, 4)
        Voxel-to-world affine.
    density : int
        Number of seeds per voxel.  For density=1 the seed is placed at the
        voxel centre.  For density>1, seeds are placed on a sub-voxel grid.
    rng : Generator, optional
        Random number generator for random sub-voxel jitter.

    Returns
    -------
    seeds : ndarray, shape (N_seeds, 3)
        World-space seed coordinates in mm.
    """
    voxel_coords = np.argwhere(mask)       # (N_vox, 3) integer coords
    if density == 1:
        seeds_vox = voxel_coords + 0.5     # centre of each voxel
    else:
        offsets = (np.mgrid[0:density, 0:density, 0:density].reshape(3, -1).T + 0.5) / density
        seeds_vox = (voxel_coords[:, np.newaxis, :] + offsets[np.newaxis]).reshape(-1, 3)

    # Transform to world coordinates
    seeds_h = np.hstack([seeds_vox, np.ones((len(seeds_vox), 1))])
    return (affine @ seeds_h.T).T[:, :3].astype(np.float32)


def seeds_from_vf_ic(
    field,
    vf_ic_threshold: float = 0.3,
    density: int = 1,
    parameter_name: str = "partial_volume_0",
) -> np.ndarray:
    """Generate seeds from voxels above a vf_ic threshold.

    Seeds in high-density white matter (high vf_ic) produce more biologically
    meaningful tractograms than WM-mask seeding alone.

    Parameters
    ----------
    field : MicrostructureField
    vf_ic_threshold : float
    density : int
    parameter_name : str

    Returns
    -------
    seeds : ndarray, shape (N_seeds, 3)
    """
    if parameter_name not in field.parameter_names:
        raise KeyError(f"'{parameter_name}' not in model parameters.")
    vf_map = field._params[parameter_name]
    mask = vf_map >= vf_ic_threshold
    return seeds_from_mask(mask, field.affine, density=density)
