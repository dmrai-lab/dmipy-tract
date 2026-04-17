"""
Along-tract parameter profiling (tractometry).

For each streamline, sample a dmipy-core parameter at equally-spaced arc-length
positions along the streamline.  Aggregate across streamlines to produce a
bundle-mean profile with confidence intervals.
"""

import numpy as np


def along_tract_profile(
    streamlines: list[np.ndarray],
    field,
    parameter_name: str,
    n_points: int = 100,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample a microstructure parameter along a bundle of streamlines.

    Parameters
    ----------
    streamlines : list of ndarray, each (L_i, 3)
        Streamlines in world coordinates.
    field : MicrostructureField
        Source of parameter values.
    parameter_name : str
        Parameter to profile (e.g. 'partial_volume_0', 'SD1WatsonDistributed_1_odi').
    n_points : int
        Number of equally-spaced arc-length samples per streamline.
    weights : ndarray, shape (N_streamlines,), optional
        Per-streamline weights (e.g. from SIFT2/COMMIT).

    Returns
    -------
    positions : ndarray, shape (n_points,)
        Normalised arc-length positions in [0, 1].
    mean_profile : ndarray, shape (n_points,)
        Weighted mean parameter value at each position.
    std_profile : ndarray, shape (n_points,)
        Weighted standard deviation.
    """
    positions = np.linspace(0.0, 1.0, n_points)
    all_profiles = []

    for sl in streamlines:
        if len(sl) < 2:
            continue
        arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(sl, axis=0), axis=1))])
        arc /= arc[-1]
        profile = np.interp(positions, arc, np.array([
            field.parameter_at(parameter_name, pt) for pt in sl
        ]))
        all_profiles.append(profile)

    if not all_profiles:
        nan = np.full(n_points, np.nan)
        return positions, nan, nan

    profiles = np.array(all_profiles)                    # (N, n_points)

    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        w = w / w.sum()
        mean_profile = (w[:, np.newaxis] * profiles).sum(axis=0)
        std_profile = np.sqrt((w[:, np.newaxis] * (profiles - mean_profile) ** 2).sum(axis=0))
    else:
        mean_profile = profiles.mean(axis=0)
        std_profile = profiles.std(axis=0)

    return positions, mean_profile, std_profile
