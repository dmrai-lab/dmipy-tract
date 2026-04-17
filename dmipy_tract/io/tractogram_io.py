"""
TRK/TCK tractogram I/O via nibabel and dipy.

The output format is DIPY ``StatefulTractogram``, which can be saved as
a .trk file and opened in MRview, TrackVis, or MI-Brain.
"""

import numpy as np


def save_trk(
    streamlines: list[np.ndarray],
    affine: np.ndarray,
    path: str,
    voxel_size: float | tuple = 1.0,
    reference_img=None,
) -> None:
    """Save streamlines to a .trk file.

    Parameters
    ----------
    streamlines : list of (L, 3) float32 arrays
        World-space streamlines in mm.
    affine : ndarray, shape (4, 4)
    path : str
        Output path (should end in .trk).
    voxel_size : float or (3,) tuple
        Voxel size in mm.  Used if ``reference_img`` is None.
    reference_img : nibabel image, optional
        If provided, metadata is taken from this image.

    Notes
    -----
    Uses ``dipy.io.stateful_tractogram.StatefulTractogram`` internally.
    """
    from dipy.io.stateful_tractogram import StatefulTractogram, Space
    from dipy.io.streamline import save_tractogram

    if reference_img is not None:
        sft = StatefulTractogram(streamlines, reference_img, Space.RASMM)
    else:
        import nibabel as nib
        shape = (1, 1, 1)
        if isinstance(voxel_size, (int, float)):
            voxel_size = (voxel_size,) * 3
        # Build a minimal reference NIfTI image; set_zooms requires ndim>0,
        # so pass the data array directly to Nifti1Image (which sets dim correctly)
        # then update the pixel dimension (zooms) via the image's header.
        ref = nib.Nifti1Image(np.zeros(shape, dtype=np.uint8), affine)
        ref.header.set_zooms(voxel_size)
        sft = StatefulTractogram(streamlines, ref, Space.RASMM)

    save_tractogram(sft, path, bbox_valid_check=False)


def load_trk(path: str) -> tuple[list[np.ndarray], np.ndarray]:
    """Load streamlines from a .trk file.

    Returns
    -------
    streamlines : list of (L, 3) float32 arrays
    affine : ndarray, shape (4, 4)
    """
    from dipy.io.streamline import load_tractogram
    from dipy.io.stateful_tractogram import Space

    sft = load_tractogram(path, "same", bbox_valid_check=False)
    sft.to_rasmm()
    return list(sft.streamlines), sft.affine
