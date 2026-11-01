# Changelog

All notable changes are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
the project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### 0.1.0

dmipy-tract tracks streamlines on fibre-orientation-distribution fields, batched on the device.

- **One data model, one tracker, one output.** `FODField(sh, affine, mask)` holds spherical-harmonic FODs in
  dmipy-sim's basis (`FODField.from_mif` reads an MRtrix FOD; a dmipy-fit CSD fit is the same basis).
  `track(field, seeds, rule="probabilistic" | "deterministic", backend="jax" | "torch")` returns a ragged `Tractogram`
  (`.to_tck` writes MRtrix format). `seeds_from_mask` seeds a mask and `connectivity` counts streamlines between
  regions.
- **dipy's conventions.** The stopping, seeding and direction rules are those of dipy's `LocalTracking`; the
  deterministic rule reproduces dipy point for point and the probabilistic rule agrees within dipy's own scatter.
- **Reproducible.** Randomness is counter-based: a streamline is a function of the key, its seed index and the field,
  not of the chunk size, the batch or the other seeds.
- **Pairs with** dmipy-sim (`>=2.2,<4` in the package metadata, `>=3,<4` with the first dmipy-sim 3 on PyPI), JAX >= 0.4.30,
  NumPy >= 1.24; the `torch` extra adds the PyTorch kernel and the `cuda` extra JAX on CUDA 12. Python >= 3.10.
- **Licence** AGPL-3.0-only or commercial.
