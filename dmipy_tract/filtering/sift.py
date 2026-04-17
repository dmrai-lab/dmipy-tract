"""Biophysically-informed streamline filtering (SIFT / SIFT2).

Reference: Smith et al. (2013) NeuroImage 67:298-312.
"""

from __future__ import annotations

import numpy as np


class SIFTFilter:
    """Biophysically-informed streamline filtering (Smith et al. 2013).

    Filters streamlines so that the streamline density matches the
    model-predicted signal contribution per voxel.

    Parameters
    ----------
    fitted_model : FittedMultiCompartmentModel
        Fitted dmipy-core model providing predicted signal per voxel.
    target_cf : float
        Target coefficient of variation for the streamline-signal fit.
        Smaller values impose tighter matching. Default is 0.1.
    """

    def __init__(self, fitted_model, target_cf: float = 0.1):
        self.fitted_model = fitted_model
        self.target_cf = target_cf

    def filter(self, streamlines: list, affine: np.ndarray) -> list:
        """Filter streamlines to match model signal.

        Parameters
        ----------
        streamlines : list of np.ndarray
            Input streamlines, each shape (L, 3) in mm space.
        affine : np.ndarray
            Shape (4, 4) voxel-to-world affine for the reference image.

        Returns
        -------
        list of np.ndarray
            Subset of streamlines whose aggregate contribution best matches the
            model-predicted fibre density per voxel.

        Raises
        ------
        NotImplementedError
            Always — full algorithm is pending Phase 2 completion.
        """
        raise NotImplementedError(
            "SIFT filtering requires per-voxel streamline contribution "
            "mapping — scheduled for Phase 2 completion."
        )

    def weights(self, streamlines: list, affine: np.ndarray) -> np.ndarray:
        """Compute per-streamline weights (SIFT2 variant).

        Parameters
        ----------
        streamlines : list of np.ndarray
            Input streamlines, each shape (L, 3) in mm space.
        affine : np.ndarray
            Shape (4, 4) voxel-to-world affine for the reference image.

        Returns
        -------
        np.ndarray
            Float array of shape (N_streamlines,) with per-streamline weights.
            Larger weights indicate greater contribution to the model signal.

        Raises
        ------
        NotImplementedError
            Always — full algorithm is pending Phase 2 completion.
        """
        raise NotImplementedError(
            "SIFT2 weighting requires per-voxel streamline contribution "
            "mapping — scheduled for Phase 2 completion."
        )
