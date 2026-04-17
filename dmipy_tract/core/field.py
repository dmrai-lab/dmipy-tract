"""
MicrostructureField: the central interface between dmipy-core fitted models
and dmipy-tract tractography algorithms.

Design principle: this object knows about space (affine, shape) and exposes
interpolated biophysical quantities at any world-space coordinate.  It does
NOT contain any propagation logic.
"""

import numpy as np
from scipy.ndimage import map_coordinates


class MicrostructureField:
    """Bridges a FittedMultiCompartmentModel to tractography algorithms.

    Parameters
    ----------
    fitted_model : FittedMultiCompartmentModel
        dmipy-core fitted model with spatial parameter maps.
    affine : ndarray, shape (4, 4)
        Voxel-to-world affine (from NIfTI header).

    Examples
    --------
    >>> from dmipy_tract.core.field import MicrostructureField
    >>> field = MicrostructureField.from_fitted_model(result, nifti_affine)
    >>> peaks = field.peaks_at([50.0, 40.0, 30.0])   # world coordinates
    >>> vf = field.parameter_at('partial_volume_0', [50.0, 40.0, 30.0])
    """

    def __init__(self, fitted_model, affine: np.ndarray):
        self._model = fitted_model
        self.affine = np.asarray(affine, dtype=np.float64)
        self._inv_affine = np.linalg.inv(self.affine)
        params = fitted_model.fitted_parameters
        first_key = next(iter(params))
        self.shape = params[first_key].shape[:3]

        # Pre-cache peaks and parameter arrays as float32 for interpolation speed
        self._peaks = fitted_model.peaks_cartesian()          # (X, Y, Z, K, 3)
        self._params = {k: np.asarray(v, dtype=np.float32)
                        for k, v in params.items()}

    @classmethod
    def from_fitted_model(cls, fitted_model, affine: np.ndarray) -> "MicrostructureField":
        """Construct from a dmipy-core FittedMultiCompartmentModel."""
        return cls(fitted_model, affine)

    def _world_to_voxel(self, xyz: np.ndarray) -> np.ndarray:
        """Transform world coordinates to fractional voxel coordinates."""
        xyz_h = np.append(xyz, 1.0)
        return (self._inv_affine @ xyz_h)[:3]

    def peaks_at(self, xyz: np.ndarray) -> np.ndarray:
        """Return trilinearly interpolated fiber orientation peaks at world position.

        Parameters
        ----------
        xyz : array_like, shape (3,)
            World-space position in mm.

        Returns
        -------
        peaks : ndarray, shape (K, 3)
            Interpolated orientation peaks (unit vectors).
        """
        vc = self._world_to_voxel(np.asarray(xyz, dtype=np.float64))
        coords = vc.reshape(3, 1)
        K = self._peaks.shape[3]
        out = np.zeros((K, 3), dtype=np.float32)
        for k in range(K):
            for d in range(3):
                out[k, d] = map_coordinates(
                    self._peaks[..., k, d], coords, order=1, mode='nearest'
                )[0]
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        return out / norms

    def parameter_at(self, name: str, xyz: np.ndarray) -> float:
        """Return a trilinearly interpolated parameter value at world position.

        Parameters
        ----------
        name : str
            Parameter name as returned by ``FittedMultiCompartmentModel.fitted_parameters``.
            E.g. 'partial_volume_0', 'SD1WatsonDistributed_1_odi'.
        xyz : array_like, shape (3,)

        Returns
        -------
        value : float
        """
        if name not in self._params:
            raise KeyError(f"Parameter '{name}' not found. Available: {list(self._params)}")
        arr = self._params[name]
        if arr.ndim != 3:
            raise ValueError(f"Parameter '{name}' has shape {arr.shape}; only scalar maps supported.")
        vc = self._world_to_voxel(np.asarray(xyz, dtype=np.float64))
        return float(map_coordinates(arr, vc.reshape(3, 1), order=1, mode='nearest')[0])

    def stopping_mask(
        self,
        vf_ic_threshold: float = 0.1,
        fa_threshold: float | None = None,
    ) -> np.ndarray:
        """Compute a binary white-matter mask for tractography stopping.

        Voxels where the intra-cellular volume fraction is below
        ``vf_ic_threshold`` are marked False (stop here).

        Parameters
        ----------
        vf_ic_threshold : float
            Minimum vf_ic to continue tracking.
        fa_threshold : float or None
            Optional FA-based fallback threshold (requires 'FA' in params).

        Returns
        -------
        mask : ndarray bool, shape (X, Y, Z)
        """
        if 'partial_volume_0' in self._params:
            vf = self._params['partial_volume_0']
        else:
            # Fall back to ones if no volume fraction is available
            return np.ones(self.shape, dtype=bool)
        mask = vf >= vf_ic_threshold
        if fa_threshold is not None and 'FA' in self._params:
            mask &= self._params['FA'] >= fa_threshold
        return mask

    def seeds_from_mask(
        self,
        mask: np.ndarray,
        density: int = 1,
    ) -> np.ndarray:
        """Generate seed positions from a binary mask.

        Parameters
        ----------
        mask : ndarray bool, shape (X, Y, Z)
        density : int
            Seeds per voxel (grid seeding within each voxel).

        Returns
        -------
        seeds : ndarray, shape (N_seeds, 3)
            World-space seed coordinates.
        """
        from .seeding import seeds_from_mask
        return seeds_from_mask(mask, self.affine, density=density)

    @property
    def parameter_names(self) -> list[str]:
        return list(self._params.keys())
