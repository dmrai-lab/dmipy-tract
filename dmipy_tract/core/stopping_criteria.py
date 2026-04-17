"""
Biophysically-informed stopping criteria for streamline tractography.

These are the primary differentiator from MRtrix3/DIPY: stopping decisions
are based on physical tissue quantities (intra-cellular volume fraction,
orientation dispersion) rather than CSD FOD amplitude or FA.
"""

import numpy as np


class StoppingCriterion:
    """Base class for stopping criteria."""

    def should_stop(self, field, xyz: np.ndarray) -> bool:
        raise NotImplementedError


class VfIcThreshold(StoppingCriterion):
    """Stop when intra-cellular volume fraction drops below threshold.

    Parameters
    ----------
    threshold : float
        Minimum vf_ic to continue.  Default 0.1.
        Use ``field.stopping_mask(vf_ic_threshold).mean()`` to calibrate
        this to your dataset's WM distribution.
    parameter_name : str
        Name of the volume fraction parameter in the fitted model.
        Default 'partial_volume_0' (dmipy-core convention for first compartment).
    """

    def __init__(self, threshold: float = 0.1, parameter_name: str = "partial_volume_0"):
        self.threshold = threshold
        self.parameter_name = parameter_name

    def should_stop(self, field, xyz: np.ndarray) -> bool:
        try:
            vf = field.parameter_at(self.parameter_name, xyz)
        except KeyError:
            return False
        return float(vf) < self.threshold


class OdiThreshold(StoppingCriterion):
    """Stop when orientation dispersion index exceeds threshold.

    High ODI indicates that the dominant orientation is poorly defined
    (isotropic tissue, crossing-fibre region with no dominant direction).
    Stopping here prevents the tracker from following noise in ambiguous regions.

    Parameters
    ----------
    threshold : float
        Maximum ODI to continue.  Default 0.8 (near-isotropic).
    parameter_name : str
        ODI parameter name.  Default 'SD1WatsonDistributed_1_odi'.
    """

    def __init__(self, threshold: float = 0.8, parameter_name: str = "SD1WatsonDistributed_1_odi"):
        self.threshold = threshold
        self.parameter_name = parameter_name

    def should_stop(self, field, xyz: np.ndarray) -> bool:
        try:
            odi = field.parameter_at(self.parameter_name, xyz)
        except KeyError:
            return False
        return float(odi) > self.threshold


class BoundaryStop(StoppingCriterion):
    """Stop when the streamline leaves the field of view."""

    def should_stop(self, field, xyz: np.ndarray) -> bool:
        vc = field._world_to_voxel(xyz)
        return bool(
            np.any(vc < 0) or
            np.any(vc >= np.array(field.shape) - 1)
        )


class CompositeStopping(StoppingCriterion):
    """Combine multiple stopping criteria with logical OR.

    Parameters
    ----------
    criteria : list of StoppingCriterion
    """

    def __init__(self, criteria: list):
        self.criteria = criteria

    def should_stop(self, field, xyz: np.ndarray) -> bool:
        return any(c.should_stop(field, xyz) for c in self.criteria)


def default_stopping(
    vf_ic_threshold: float = 0.1,
    odi_threshold: float = 0.8,
    vf_param: str = "partial_volume_0",
    odi_param: str = "SD1WatsonDistributed_1_odi",
) -> CompositeStopping:
    """Convenience constructor for the standard biophysical stopping criteria."""
    return CompositeStopping([
        BoundaryStop(),
        VfIcThreshold(vf_ic_threshold, vf_param),
        OdiThreshold(odi_threshold, odi_param),
    ])
