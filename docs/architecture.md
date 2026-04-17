# dmipy-tract Architecture

## Pipeline Diagram

```
                         ┌─────────────────────────────────┐
                         │   FittedMultiCompartmentModel    │
                         │         (dmipy-core)             │
                         └───────────────┬─────────────────┘
                                         │ .peaks_cartesian()
                                         │ .fitted_parameters
                                         ▼
                         ┌─────────────────────────────────┐
                         │        MicrostructureField       │
                         │  core/field.py                   │
                         │  • peaks_at(xyz)                 │
                         │  • parameter_at(name, xyz)       │
                         │  • stopping_mask(...)            │
                         └────┬─────────────┬──────────────┘
                              │             │
               ┌──────────────┘             └─────────────────┐
               ▼                                               ▼
  ┌────────────────────────┐               ┌────────────────────────────┐
  │    seeds_from_mask()   │               │    seeds_from_vf_ic()      │
  │    core/seeding.py     │               │    core/seeding.py         │
  │    (arbitrary mask)    │               │    (threshold on vf_ic)    │
  └────────────┬───────────┘               └───────────────┬────────────┘
               └───────────────┬───────────────────────────┘
                               │  seeds: ndarray (N_seeds, 3)
                               ▼
          ┌────────────────────────────────────────────────┐
          │              stopping criteria                  │
          │              core/stopping_criteria.py          │
          │  BoundaryStop | VfIcThreshold | OdiThreshold    │
          └────────────────────┬───────────────────────────┘
                               │
               ┌───────────────┴───────────────────┐
               ▼                                   ▼
  ┌────────────────────────┐         ┌─────────────────────────────┐
  │  DeterministicTracker  │         │     StochasticTracker       │
  │  propagation/          │         │     propagation/            │
  │  deterministic.py      │         │     stochastic.py           │
  │  RK4 + bidirectional   │         │     Euler + vMF noise       │
  └────────────┬───────────┘         └──────────────┬──────────────┘
               └───────────────┬───────────────────┘
                               │  streamlines: list of (L, 3) arrays
                               ▼
                  ┌────────────────────────────┐
                  │        SIFTFilter          │
                  │  filtering/sift.py         │
                  │  .filter()  → subset       │
                  │  .weights() → w per stream │
                  └────────────┬───────────────┘
                               │  filtered streamlines / weights
                    ┌──────────┴──────────────────────────────┐
                    │                  │                       │
                    ▼                  ▼                       ▼
  ┌─────────────────────┐  ┌────────────────────┐  ┌──────────────────────┐
  │ save_bids_tractogram│  │  plot_tractogram   │  │ along_tract_profile  │
  │ io/bids_export.py   │  │  vis/              │  │ tractometry/         │
  │ .trk + .json sidecar│  │  tractogram_viewer │  │ profile.py           │
  │ BIDS derivatives    │  │  fury-based        │  │ arc-length profiles  │
  └─────────────────────┘  └────────────────────┘  └──────────────────────┘
```

---

## Module Responsibilities

| Module | File(s) | Responsibility |
|--------|---------|----------------|
| `core/field.py` | `MicrostructureField` | Bridges a `FittedMultiCompartmentModel` to the tractography layer. Caches peaks and parameter maps as float32 arrays; provides trilinearly interpolated `peaks_at(xyz)` and `parameter_at(name, xyz)` at any world-space coordinate. Owns the voxel-to-world affine and its inverse. |
| `core/seeding.py` | `seeds_from_mask`, `seeds_from_vf_ic` | Generates world-space seed coordinates from a binary mask or from an intra-cellular volume fraction threshold. Supports sub-voxel grid seeding via `density > 1`. |
| `core/stopping_criteria.py` | `VfIcThreshold`, `OdiThreshold`, `BoundaryStop`, `CompositeStopping`, `default_stopping` | Biophysically-informed stopping logic. Each criterion implements `should_stop(field, xyz) -> bool`. `CompositeStopping` combines criteria with logical OR. `default_stopping()` returns the standard three-criterion combination. |
| `propagation/deterministic.py` | `DeterministicTracker` | 4th-order Runge-Kutta peak tracking. Propagates bidirectionally from each seed. Implements the sign convention: `sign` applied only to position offsets, never to k-vectors. |
| `propagation/stochastic.py` | `StochasticTracker` | Euler integration with vMF angular perturbation at each step. Generates `n_streamlines_per_seed` samples per seed. Uses Wood (1994) approximate vMF sampling. |
| `filtering/sift.py` | `SIFTFilter` | SIFT greedy streamline removal (`filter()`) and SIFT2 NNLS per-streamline weighting (`weights()`). Cost function: `C = sum_v (TDI_v - target_v)^2`. |
| `filtering/tdi.py` | `compute_tdi` | Length-weighted Track Density Image. Sub-samples each segment at twice voxel resolution; accumulates `segment_length_mm / n_samples` per voxel. Units: mm of fibre length per voxel. |
| `tractometry/profile.py` | `along_tract_profile` | Arc-length resampling of a microstructure parameter along a bundle of streamlines. Returns normalised positions, weighted mean, and weighted standard deviation profiles. |
| `io/tractogram_io.py` | `save_trk`, `load_trk` | dipy-backed .trk I/O. Always uses `bbox_valid_check=False`. Builds a minimal synthetic NIfTI reference image when no real reference is supplied, using `Nifti1Image(data, affine)` to avoid the `Nifti1Header dim` bug. |
| `io/bids_export.py` | `save_bids_tractogram` | Writes a .trk file and JSON sidecar to a BIDS Derivatives directory tree: `<output_dir>/<subject>/[<session>/]dwi/`. Returns the absolute path to the .trk file. |
| `vis/tractogram_viewer.py` | `plot_tractogram` | fury-based 3-D visualisation. Imports fury lazily inside the function to keep the package importable in headless environments. Supports per-streamline scalar colour maps. |
| `connectivity/` | (Phase 4) | Connectivity matrix computation — not yet implemented. |

---

## Stopping Criteria

| Criterion | Class | Parameter used | Default threshold | Biophysical meaning |
|-----------|-------|---------------|-------------------|---------------------|
| Field of view boundary | `BoundaryStop` | — | voxel coordinates outside `[0, shape-1]` | Stop at image edge; prevents out-of-bounds access |
| Intra-cellular volume fraction | `VfIcThreshold` | `partial_volume_0` | 0.1 | Below this NDI the voxel is not white matter; fibre orientation cannot be reliably estimated |
| Orientation dispersion index | `OdiThreshold` | `SD1WatsonDistributed_1_odi` | 0.8 | Above this ODI the orientation distribution is near-isotropic; the dominant peak direction is undefined |

All three are combined by `default_stopping()` using `CompositeStopping` (logical OR: stop if any triggers). Parameter names can be overridden to support non-NODDI model types.
