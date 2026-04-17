"""Biophysically-informed streamline filtering (SIFT / SIFT2).

Reference: Smith et al. (2013) NeuroImage 67:298-312.
"""

from __future__ import annotations

import warnings

import numpy as np


class SIFTFilter:
    """Biophysically-informed streamline filtering (Smith et al. 2013).

    Filters streamlines so that the streamline density matches the
    model-predicted signal contribution per voxel.

    Parameters
    ----------
    fitted_model : FittedMultiCompartmentModel, optional
        Fitted dmipy-core model providing predicted signal per voxel.
        Not required when ``target_density`` is passed directly to
        :meth:`filter` or :meth:`weights`.
    target_cf : float
        Target coefficient of variation for the streamline-signal fit.
        Smaller values impose tighter matching. Default is 0.1.
    """

    def __init__(self, fitted_model=None, target_cf: float = 0.1):
        self.fitted_model = fitted_model
        self.target_cf = target_cf

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def filter(
        self,
        streamlines: list,
        affine: np.ndarray,
        target_density: np.ndarray,
    ) -> list:
        """Filter streamlines to match target_density map.

        Uses a greedy approach: iteratively remove the streamline whose
        removal most reduces the global cost ``C = sum_v (TDI_v - target_v)^2``.

        Parameters
        ----------
        streamlines : list of (L, 3) arrays
            World-space mm coordinates.
        affine : ndarray, shape (4, 4)
            Affine mapping mm to voxel indices.
        target_density : ndarray, shape (X, Y, Z)
            Target streamline density per voxel. Typically: intra-axonal
            volume fraction × voxel_volume / mean_streamline_length.

        Returns
        -------
        list of (L, 3) arrays
            Filtered subset of input streamlines.
        """
        from .tdi import compute_tdi

        target_density = np.asarray(target_density, dtype=np.float64)
        vol_shape = target_density.shape

        if len(streamlines) == 0:
            return []

        # Compute per-streamline contribution maps (sparse: only non-zero voxels)
        contributions = _compute_streamline_contributions(
            streamlines, affine, vol_shape
        )

        # Current TDI = sum of all contributions
        current_tdi = np.zeros(vol_shape, dtype=np.float64)
        for contrib in contributions:
            for (idx, val) in contrib:
                current_tdi[idx] += val

        active = list(range(len(streamlines)))
        removed = set()

        current_cost = _global_cost(current_tdi, target_density)

        # Greedy removal: keep removing streamlines that reduce cost
        improved = True
        while improved and len(active) > 1:
            improved = False
            best_gain = 0.0
            best_idx = -1

            for i in active:
                # Tentative TDI without streamline i
                tentative_tdi = current_tdi.copy()
                for (idx, val) in contributions[i]:
                    tentative_tdi[idx] -= val

                tentative_cost = _global_cost(tentative_tdi, target_density)
                gain = current_cost - tentative_cost

                if gain > best_gain:
                    best_gain = gain
                    best_idx = i

            if best_idx >= 0:
                # Remove this streamline
                for (idx, val) in contributions[best_idx]:
                    current_tdi[idx] -= val
                active.remove(best_idx)
                removed.add(best_idx)
                current_cost = _global_cost(current_tdi, target_density)
                improved = True

        return [streamlines[i] for i in active]

    def weights(
        self,
        streamlines: list,
        affine: np.ndarray,
        target_density: np.ndarray,
    ) -> np.ndarray:
        """Compute per-streamline SIFT2 weights.

        Instead of removing streamlines, assigns a non-negative weight w_i to
        each so that ``sum_i(w_i * contribution_i(v)) ≈ target_density(v)``
        for all voxels v. Solved via non-negative least squares (NNLS).

        Parameters
        ----------
        streamlines : list of (L, 3) arrays
            World-space mm coordinates.
        affine : ndarray, shape (4, 4)
            Affine mapping mm to voxel indices.
        target_density : ndarray, shape (X, Y, Z)
            Target streamline density per voxel.

        Returns
        -------
        weights : ndarray, shape (N_streamlines,)
            Non-negative per-streamline weights.
        """
        from scipy.optimize import nnls

        target_density = np.asarray(target_density, dtype=np.float64)
        vol_shape = target_density.shape
        n_streamlines = len(streamlines)

        if n_streamlines == 0:
            return np.zeros(0, dtype=np.float64)

        contributions = _compute_streamline_contributions(
            streamlines, affine, vol_shape
        )

        # Collect all voxels with non-zero target or any streamline contribution
        voxel_set = set()
        for contrib in contributions:
            for (idx, _) in contrib:
                voxel_set.add(idx)
        # Also include voxels with non-zero target
        nz_target = np.argwhere(target_density > 0)
        for row in nz_target:
            voxel_set.add(tuple(row))

        voxel_list = sorted(voxel_set)
        n_voxels = len(voxel_list)

        if n_voxels == 0:
            return np.ones(n_streamlines, dtype=np.float64)

        voxel_index = {v: k for k, v in enumerate(voxel_list)}

        n_entries = sum(len(c) for c in contributions)
        if n_entries > 1_000_000:
            warnings.warn(
                f"SIFT2: problem size {n_streamlines} streamlines × "
                f"{n_voxels} voxels is large ({n_entries} entries). "
                "Falling back to uniform weights.",
                RuntimeWarning,
                stacklevel=2,
            )
            return np.ones(n_streamlines, dtype=np.float64)

        # Build dense A matrix: shape (n_voxels, n_streamlines)
        A = np.zeros((n_voxels, n_streamlines), dtype=np.float64)
        for i, contrib in enumerate(contributions):
            for (idx, val) in contrib:
                if idx in voxel_index:
                    A[voxel_index[idx], i] = val

        b = np.array([target_density[v] for v in voxel_list], dtype=np.float64)

        w, _ = nnls(A, b)
        return w


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _compute_streamline_contributions(
    streamlines: list,
    affine: np.ndarray,
    vol_shape: tuple,
) -> list:
    """Return per-streamline voxel contributions as sparse lists.

    Returns
    -------
    contributions : list of list of (tuple, float)
        contributions[i] = [(voxel_index_tuple, value), ...] for streamline i.
        Multiple entries for the same voxel are possible and should be summed.
    """
    inv_affine = np.linalg.inv(np.asarray(affine, dtype=np.float64))
    contributions = []

    for sl in streamlines:
        sl = np.asarray(sl, dtype=np.float64)
        contrib_map: dict = {}

        if sl.shape[0] < 2:
            contributions.append(list(contrib_map.items()))
            continue

        for k in range(len(sl) - 1):
            p1 = sl[k]
            p2 = sl[k + 1]

            seg_len = float(np.linalg.norm(p2 - p1))
            if seg_len < 1e-9:
                continue

            p1_h = np.array([p1[0], p1[1], p1[2], 1.0])
            p2_h = np.array([p2[0], p2[1], p2[2], 1.0])
            v1 = (inv_affine @ p1_h)[:3]
            v2 = (inv_affine @ p2_h)[:3]

            vox_len = float(np.linalg.norm(v2 - v1))
            n_samples = max(2, int(np.ceil(vox_len * 2)))

            ts = np.linspace(0.0, 1.0, n_samples)
            voxel_coords = v1[np.newaxis, :] + ts[:, np.newaxis] * (v2 - v1)[np.newaxis, :]
            contribution_per_sample = seg_len / n_samples

            for vc in voxel_coords:
                xi = int(np.round(vc[0]))
                yi = int(np.round(vc[1]))
                zi = int(np.round(vc[2]))
                if (0 <= xi < vol_shape[0] and
                        0 <= yi < vol_shape[1] and
                        0 <= zi < vol_shape[2]):
                    key = (xi, yi, zi)
                    contrib_map[key] = contrib_map.get(key, 0.0) + contribution_per_sample

        contributions.append(list(contrib_map.items()))

    return contributions


def _global_cost(tdi: np.ndarray, target: np.ndarray) -> float:
    """Global cost: sum of squared differences between TDI and target."""
    diff = tdi - target
    return float(np.sum(diff * diff))
