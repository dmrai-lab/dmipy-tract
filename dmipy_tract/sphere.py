"""The direction set a tracker chooses from, and the spherical-harmonics matrix that evaluates an FOD on it.

The basis is dmipy-sim's :func:`dmipy_sim.replay.so3.real_sh`: orthonormal real spherical harmonics, even orders,
compact layout (MRtrix's ``tournier07`` convention), and its coefficient count ``so3.n_sh_coeffs`` / order
``so3.lmax_of``; nothing here defines a basis of its own.
"""
import numpy as np

from dmipy_sim.replay import so3

__all__ = ['hemisphere', 'sh_matrix']

_GOLDEN_ANGLE = np.pi * (3.0 - np.sqrt(5.0))


def hemisphere(n=362):
    """``(n, 3)`` unit directions on the hemisphere ``z > 0``, a Fibonacci lattice: no two antipodal, near-uniform
    spacing, deterministic. The FOD is antipodally symmetric, so a hemisphere carries every direction once and a
    tracker's step is ``+v`` or ``-v`` by the sign that continues the streamline."""
    n = int(n)
    if n < 1:
        raise ValueError("a hemisphere needs at least one direction")
    i = np.arange(n, dtype=np.float64)
    z = (i + 0.5) / n
    r = np.sqrt(1.0 - z * z)
    phi = i * _GOLDEN_ANGLE
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def sh_matrix(order, dirs):
    """``(n_dirs, n_coef)``: the basis evaluated on ``dirs``, so that ``sh_matrix(order, dirs) @ coeff`` is the FOD's
    amplitude on every direction."""
    dirs = np.asarray(dirs, np.float64)
    if dirs.ndim != 2 or dirs.shape[1] != 3:
        raise ValueError(f"dirs must be (n, 3), not {dirs.shape}")
    norms = np.linalg.norm(dirs, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-6):
        raise ValueError("dirs must be unit vectors")
    return so3.real_sh(int(order), dirs)
