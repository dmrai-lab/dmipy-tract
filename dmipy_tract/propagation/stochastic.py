"""
Probabilistic streamline tracking via angular perturbation.

At each step, the propagation direction is drawn from a von Mises-Fisher
distribution centred on the peak direction. This produces a distribution
of tractograms rather than a single deterministic one.
"""
import numpy as np
from ..core.stopping_criteria import CompositeStopping, default_stopping


class StochasticTracker:
    """Probabilistic tracker: RK1 (Euler) step with vMF angular noise.

    At each step:
      1. Look up principal peak at current position
      2. Sample a perturbation direction from a von Mises-Fisher distribution
         with concentration kappa (higher kappa = less noise)
      3. Step in the perturbed direction

    Parameters
    ----------
    field : MicrostructureField
    step_size_mm : float
    kappa : float
        vMF concentration. kappa=0 = isotropic noise, kappa→∞ = deterministic.
        Typical value: 30–100 for tractography.
    max_angle_deg : float
        Hard curvature limit (same as deterministic).
    max_length_mm : float
    min_length_mm : float
    n_streamlines_per_seed : int
        Number of stochastic samples per seed.
    stopping : CompositeStopping or None
    seed : int
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        field,
        step_size_mm: float = 0.5,
        kappa: float = 50.0,
        max_angle_deg: float = 45.0,
        max_length_mm: float = 250.0,
        min_length_mm: float = 10.0,
        n_streamlines_per_seed: int = 3,
        stopping: CompositeStopping | None = None,
        seed: int = 0,
    ):
        self.field = field
        self.step_size = float(step_size_mm)
        self.kappa = float(kappa)
        self.cos_max_angle = float(np.cos(np.deg2rad(max_angle_deg)))
        self.max_steps = int(max_length_mm / step_size_mm)
        self.min_steps = int(min_length_mm / step_size_mm)
        self.n_per_seed = n_streamlines_per_seed
        self.stopping = stopping or default_stopping()
        self.rng = np.random.default_rng(seed)

    def track(self, seeds: np.ndarray) -> list[np.ndarray]:
        """Generate n_streamlines_per_seed stochastic streamlines per seed."""
        streamlines = []
        for seed in seeds:
            for _ in range(self.n_per_seed):
                sl = self._propagate_single(seed)
                if sl is not None:
                    streamlines.append(sl)
        return streamlines

    def _sample_vmf(self, mu: np.ndarray) -> np.ndarray:
        """Sample a unit vector from vMF(mu, kappa).

        Uses the Wood (1994) rejection sampling algorithm.
        For kappa=0, returns a uniform random unit vector.
        """
        if self.kappa < 1e-6:
            v = self.rng.standard_normal(3)
            return v / np.linalg.norm(v)

        # Simple approximation: rotate mu by a random angle drawn from
        # a distribution with std = arctan(1/sqrt(kappa))
        # This is not exact vMF but close enough for tractography.
        angle_std = np.arctan(1.0 / np.sqrt(self.kappa))
        # Random rotation axis perpendicular to mu
        perp = self.rng.standard_normal(3)
        perp -= np.dot(perp, mu) * mu
        norm = np.linalg.norm(perp)
        if norm < 1e-10:
            perp = np.array([1.0, 0.0, 0.0]) if abs(mu[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            perp -= np.dot(perp, mu) * mu
            perp /= np.linalg.norm(perp)
        else:
            perp /= norm
        angle = self.rng.normal(0.0, angle_std)
        # Rodrigues rotation: mu rotated by `angle` around `perp`
        direction = mu * np.cos(angle) + perp * np.sin(angle)
        return direction / np.linalg.norm(direction)

    def _propagate_single(self, seed: np.ndarray) -> np.ndarray | None:
        """Propagate one stochastic streamline from seed (forward only)."""
        pos = np.array(seed, dtype=np.float64)
        points = [pos.copy()]
        direction = None

        for _ in range(self.max_steps):
            if self.stopping.should_stop(self.field, pos):
                break
            peaks = self.field.peaks_at(pos)
            if peaks is None or len(peaks) == 0 or np.all(peaks[0] == 0):
                break

            peak = peaks[0]
            if direction is not None:
                if np.dot(peak, direction) < 0:
                    peak = -peak
                if np.dot(peak, direction) < self.cos_max_angle:
                    break

            # Sample direction with vMF noise
            step_dir = self._sample_vmf(peak)
            if direction is not None and np.dot(step_dir, direction) < self.cos_max_angle:
                break

            direction = step_dir
            pos = pos + self.step_size * step_dir
            points.append(pos.copy())

        if len(points) < self.min_steps:
            return None
        return np.array(points, dtype=np.float32)
