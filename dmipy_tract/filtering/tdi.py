"""Track Density Image (TDI) computation.

Reference: Calamante et al. (2010) NeuroImage 53:1233-1243.
"""

from __future__ import annotations

import numpy as np


def compute_tdi(
    streamlines: list,
    affine: np.ndarray,
    vol_shape: tuple,
    length_weighted: bool = True,
) -> np.ndarray:
    """Compute Track Density Image.

    For each streamline segment, add its length contribution to the voxels it
    passes through using linear interpolation along the segment.

    Parameters
    ----------
    streamlines : list of (L, 3) world-space arrays
        Streamlines in mm coordinates (RASMM space).
    affine : ndarray, shape (4, 4)
        Affine mapping world mm coordinates to voxel indices.
    vol_shape : tuple of 3 ints
        Output volume shape (X, Y, Z).
    length_weighted : bool
        If True (default), weight each voxel contribution by segment length
        in mm. If False, each segment contributes 1/n_samples per sampled
        voxel regardless of length.

    Returns
    -------
    tdi : ndarray, shape (X, Y, Z), float64
        Track density image.
    """
    inv_affine = np.linalg.inv(np.asarray(affine, dtype=np.float64))
    tdi = np.zeros(vol_shape, dtype=np.float64)

    for sl in streamlines:
        sl = np.asarray(sl, dtype=np.float64)
        if sl.shape[0] < 2:
            continue

        for k in range(len(sl) - 1):
            p1 = sl[k]
            p2 = sl[k + 1]

            # Segment length in mm
            seg_len = float(np.linalg.norm(p2 - p1))
            if seg_len < 1e-9:
                continue

            # Convert endpoints to voxel coords
            p1_h = np.array([p1[0], p1[1], p1[2], 1.0])
            p2_h = np.array([p2[0], p2[1], p2[2], 1.0])
            v1 = (inv_affine @ p1_h)[:3]
            v2 = (inv_affine @ p2_h)[:3]

            # Number of samples: at least 2, proportional to voxel length
            vox_len = float(np.linalg.norm(v2 - v1))
            n_samples = max(2, int(np.ceil(vox_len * 2)))

            ts = np.linspace(0.0, 1.0, n_samples)
            # Interpolated voxel coords along segment
            voxel_coords = v1[np.newaxis, :] + ts[:, np.newaxis] * (v2 - v1)[np.newaxis, :]

            # Contribution per sample
            if length_weighted:
                contribution = seg_len / n_samples
            else:
                contribution = 1.0 / n_samples

            for vc in voxel_coords:
                xi = int(np.round(vc[0]))
                yi = int(np.round(vc[1]))
                zi = int(np.round(vc[2]))
                if (0 <= xi < vol_shape[0] and
                        0 <= yi < vol_shape[1] and
                        0 <= zi < vol_shape[2]):
                    tdi[xi, yi, zi] += contribution

    return tdi
