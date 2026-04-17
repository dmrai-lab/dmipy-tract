# dmipy-tract

Biophysically-informed tractography and tractometry powered by dmipy-core microstructure models.

dmipy-tract is not just a fiber-tracking library. The tractogram it produces is connected to
dmipy-core's fitted parameter maps at every step: seeding is driven by intra-cellular volume
fraction, stopping criteria use NDI and ODI instead of FA or CSD FOD amplitude, SIFT filtering
uses the model's fiber volume fraction as the target density, and tractometry profiles the same
parameter maps used for voxel-level analysis along the length of every bundle.

---

## Why own tractography?

The current landscape is fragmented:

- **MRtrix3** — excellent tractography, but takes CSD FODs (not biophysical model parameters),
  separate ecosystem, no Python API for tight integration.
- **DSI Studio** — proprietary, no open API.
- **dipy** — basic tractography, no biophysical model integration.

By owning both signal modelling (dmipy-core) and tractography (dmipy-tract), dmrai-lab can:

1. Feed fitted biophysical parameters (NDI, ODI, axon diameter) directly into stopping criteria
   and SIFT target density — no file conversion, no format mismatch.
2. Validate tractography against the same MC simulation engine (dmipy-sim) that validates signal
   models.
3. Use SIFT2 weights driven by the actual biophysical model's fiber volume fraction, not generic
   FOD amplitude.
4. Profile microstructure along tracts using the exact same parameter maps used for voxel-level
   analysis.

---

## File map

```
dmipy_tract/
  core/
    field.py              — MicrostructureField: wraps a fitted dmipy-core model; provides
                            peaks_at(pos) and parameter_at(name, pos) at any world-space coord
    seeding.py            — seeds_from_mask, seeds_from_vf_ic: seed generation strategies
    stopping_criteria.py  — VfIcThreshold, OdiThreshold, BoundaryStop, CompositeStopping,
                            default_stopping: biophysically-informed stopping
  propagation/
    deterministic.py      — DeterministicTracker: RK4 peak tracking, bidirectional
    stochastic.py         — StochasticTracker: Euler + vMF angular perturbation
  filtering/
    sift.py               — SIFTFilter: greedy SIFT + SIFT2 NNLS weights
    tdi.py                — compute_tdi: length-weighted Track Density Image
  tractometry/
    profile.py            — along_tract_profile: arc-length interpolation of parameter maps
  io/
    tractogram_io.py      — save_trk, load_trk (dipy-backed, bbox_valid_check=False)
    bids_export.py        — save_bids_tractogram: BIDS-derivatives .trk + JSON sidecar
  vis/
    tractogram_viewer.py  — plot_tractogram: fury-based viewer, PNG or interactive
  connectivity/
    __init__.py           — (Phase 4: connectivity matrix)

tests/
  test_deterministic_tracker.py
  test_stochastic_tracker.py
  test_microstructure_field.py
  test_seeding.py
  test_tractogram_io.py
  test_end_to_end.py
  test_viewer.py
  test_sift_stub.py
  test_sift.py
  test_tdi.py
  test_bids_export.py
  test_tractometry.py
```

---

## Key concepts

### RK4 deterministic tracking

`DeterministicTracker` propagates along the principal orientation peak using a 4th-order
Runge-Kutta integrator. At each step, four sub-step evaluations (k1..k4) sample the peak field
at intermediate positions, and the weighted average `(k1 + 2k2 + 2k3 + k4) / 6` gives the
final step direction. RK4 substantially reduces trajectory error compared to Euler integration
on the same step size, especially in high-curvature regions (e.g., the u-fibres of the
cingulum). Propagation is bidirectional: two half-streamlines are grown in opposite directions
from each seed, then concatenated.

**Sign convention**: `sign` (+1 for forward, -1 for backward) is applied only to the sub-step
position offsets and the final position update. It is never applied to the k-vectors themselves.
The direction-flip logic inside `_get_direction` operates on unsigned unit vectors, then the
caller multiplies by `sign` when computing the sub-step offset. Applying `sign` to the k-vectors
directly causes the flip logic to see a negated direction on every sub-step evaluation, which
breaks bidirectional propagation. See `CLAUDE.md` for full details.

### Stochastic tracking

`StochasticTracker` uses Euler integration (RK1) with von Mises-Fisher (vMF) angular
perturbation at each step. The peak direction at the current position is used as the vMF mean
`mu`; a sample is drawn by rotating `mu` by a random angle whose standard deviation is
`arctan(1/sqrt(kappa))`. High `kappa` (e.g. 100) produces near-deterministic behaviour; low
`kappa` (e.g. 10) produces wide angular dispersion. Multiple streamlines per seed
(`n_streamlines_per_seed`, default 3) are generated to sample the orientation uncertainty.
Stochastic tracking is preferred when crossing fibres or noisy data make a single deterministic
path unreliable, or when downstream analysis (COMMIT, Bayesian connectivity) requires a
distribution of plausible paths.

### SIFT — Spherical-deconvolution Informed Filtering of Tractograms

After tractography, the raw tractogram overrepresents long, straight fibres and underrepresents
short or curved ones. SIFT (Smith et al. 2013) corrects this by removing streamlines whose
removal reduces the global cost `C = sum_v (TDI_v - target_density_v)^2`, where `TDI` is the
Track Density Image and `target_density` is the model-predicted fiber density per voxel
(typically: `vf_ic * voxel_volume / mean_streamline_length`). The greedy implementation in
`SIFTFilter.filter()` iterates until no single removal reduces cost. SIFT2 (`SIFTFilter.weights()`)
instead assigns non-negative scalar weights to each streamline via NNLS, preserving all
streamlines while achieving the same density match.

### TDI — Track Density Image

`compute_tdi` accumulates length-weighted contributions of each streamline segment into a 3-D
volume. Each segment is sub-sampled along its length (at twice voxel resolution) and each sample
contributes `segment_length_mm / n_samples` to its nearest voxel. The result is a map whose
units are mm of fibre length per voxel, which is proportional to fiber volume fraction at the
voxel scale. TDI is the primary quantity matched by SIFT.

### Tractometry

`along_tract_profile` resamples each streamline to `n_points` equally-spaced arc-length
positions (normalised to [0, 1]), queries `MicrostructureField.parameter_at` at each position,
then returns the bundle-mean and standard-deviation profile across all streamlines. Optional
per-streamline weights (e.g. from SIFT2) allow weighted averaging. Any scalar parameter in the
fitted model can be profiled: NDI, ODI, axon diameter, diffusivity.

---

## RK4 sign convention

This is a subtle but critical correctness point documented here explicitly.

In `DeterministicTracker._propagate_direction`, `sign` is `+1.0` (forward) or `-1.0`
(backward). The sign is applied **only** to position offsets and the final position update:

```python
k2 = self._get_direction(pos + sign * 0.5 * step_size * k1, k1)
k3 = self._get_direction(pos + sign * 0.5 * step_size * k2, k2)
k4 = self._get_direction(pos + sign * step_size * k3, k3)
pos = pos + sign * step_size * step_dir
```

`_get_direction` receives the reference direction (`k_prev`) as an unsigned unit vector and
flips the retrieved peak if it points away from `k_prev`. This flip logic must always see an
unsigned direction; multiplying `k1/k2/k3` by `sign` before passing them to `_get_direction`
would cause the flip to trigger on the wrong side, producing step directions that oscillate
rather than following the fibre.

**The bug**: an earlier version applied `sign` inside the k-vector computation as
`k2 = sign * _get_direction(pos + ...)`, which silently negated the curvature-flip reference,
causing the backward half to spin and produce a single-point or zero-length streamline.

---

## Quick start

From a fitted dmipy-core model to a BIDS tractogram in 15 lines:

```python
import nibabel as nib
from dmipy_tract.core.field import MicrostructureField
from dmipy_tract.core.seeding import seeds_from_vf_ic
from dmipy_tract.propagation.deterministic import DeterministicTracker
from dmipy_tract.io.bids_export import save_bids_tractogram

# result: FittedMultiCompartmentModel from dmipy-core
ref = nib.load("sub-01_dwi.nii.gz")
field = MicrostructureField.from_fitted_model(result, ref.affine)

seeds = seeds_from_vf_ic(field, vf_ic_threshold=0.3, density=2)

tracker = DeterministicTracker(field, step_size_mm=0.5, max_angle_deg=30)
streamlines = tracker.track(seeds)

trk_path = save_bids_tractogram(
    streamlines, ref.affine,
    reference_nifti_path="sub-01_dwi.nii.gz",
    output_dir="derivatives/dmipy-tract",
    subject="sub-01",
)
print(f"Saved {len(streamlines)} streamlines to {trk_path}")
```

The resulting file lands at:
`derivatives/dmipy-tract/sub-01/dwi/sub-01_desc-wholebrain_tractography.trk`

---

## Viewer

`plot_tractogram` accepts a per-streamline scalar array for colour-coding:

```python
from dmipy_tract.vis.tractogram_viewer import plot_tractogram
from dmipy_tract.tractometry.profile import along_tract_profile

positions, ndi_mean, ndi_std = along_tract_profile(
    streamlines, field, "partial_volume_0"
)

# Per-streamline mean NDI (index 0 = start, -1 = end)
per_sl_ndi = [field.parameter_at("partial_volume_0", sl[len(sl)//2]) for sl in streamlines]

plot_tractogram(streamlines, scalar_map=per_sl_ndi, colormap="hot", output_file="tract_ndi.png")
```

For interactive viewing, omit `output_file`. In CI/headless environments, pass `output_file`
to write a PNG via off-screen rendering (fury uses OSMesa when a display is unavailable).

---

## Installation

```bash
pip install -e ".[dev]"
```

Requires Python >= 3.10. Core dependencies: numpy, scipy, nibabel, dipy.
Optional: fury (visualisation), dmipy-core (full biophysical model integration).

---

## Running tests

```bash
/home/rutger/dmipy-core/.venv/bin/python -m pytest tests/
```

## Documentation
- [`docs/faq.md`](docs/faq.md) — 20 frequently asked questions
- [`docs/architecture.md`](docs/architecture.md) — pipeline diagram and module responsibilities
- [`docs/rk4_and_sign_convention.md`](docs/rk4_and_sign_convention.md) — RK4 implementation and the sign convention
