# dmipy-tract — rules for agents

Read `README.md` first: it says what the tracker is. This file is what an agent must not get wrong.

## What lives where

| file | holds |
|---|---|
| `dmipy_tract/field.py` | `FODField(sh, affine, mask)`: validation, the clamped trilinear interpolant (numpy), the nearest-voxel mask lookup |
| `dmipy_tract/sphere.py` | `hemisphere(n)` (Fibonacci, z > 0), `sh_matrix(order, dirs)` = `dmipy_sim.replay.so3.real_sh`, order/count conversions |
| `dmipy_tract/tracker.py` | `track()`: the lockstep kernel (`_compiled_half`, `_compiled_first`), the host join, chunking |
| `dmipy_tract/tractogram.py` | `Tractogram` (ragged points/offsets, seed_index, stop_reason), `.to_tck` via `dmipy_sim.io.strands.write_tck` |
| `dmipy_tract/seeding.py` | `seeds_from_mask`: dipy's sub-grid construction |
| `dmipy_tract/connectivity.py` | endpoint labels (nearest voxel) and the count matrix |
| `tests/conftest.py` | the constructed fields (uniform, crossing, circle) and `reference_track`, the per-streamline numpy definition |
| `benchmarks/disco.py` | the DiSCo acceptance; `benchmarks/scaling.py` the brain-scale timing |

## Rules

- **dipy is an oracle, never a runtime import.** The package imports numpy, jax and dmipy-sim only. Shared
  mathematics (the SH basis, `.tck`/`.mif`) lives in dmipy-sim, once.
- **The basis is `so3.real_sh`** (orthonormal, even orders, MRtrix `tournier07` non-legacy). A dipy `shm_coeff` in
  `descoteaux07` must be converted (`dmipy_sim.replay.fod.FOD.from_sh`) before it becomes an `FODField`; nothing here
  guesses a basis.
- **FOD directions are world coordinates** (MRtrix's convention). Tracking is in world millimetres; interpolation
  and the mask lookup go through the inverse affine. For a scaling-plus-translation affine this is dipy's voxel-axis
  frame too; under a rotation, dipy's field would need rotating (`so3.rotate_sh`).
- **The conventions are dipy's, measured, and tested point for point.** Changing any of them breaks
  `tests/test_dipy_parity.py` by design: a streamline never holds a point outside the domain (the failing step is
  not taken); the mask is read at `rint` (half to even) of the voxel coordinate; the interpolant is clamped on
  `[-0.5, N - 0.5]`; the threshold is 10 % of the sphere-wide maximum, applied before the cone; the backward half
  starts against the forward half's actual first step; the sign of a chosen direction follows `dot > 0`.
- **Every matmul in the kernel is `Precision.HIGHEST`.** A float32 `jnp.dot` on CUDA is TF32 (10-bit mantissa);
  amplitudes move at 1e-3 and near-tie choices flip. This bit the ecosystem three times before this package existed.
- **Randomness is counter-based.** Streamline `i` draws from `fold_in(fold_in(key, i), 1 + 2 t + half)` at step `t`
  and `fold_in(fold_in(key, i), 0)` for its first direction. Do not thread a key through the loop: the chunk-size
  invariance test exists to catch that.
- **Deterministic tests, statistics only for random quantities, floors measured not chosen.** A new mechanism gets a
  test on a constructed input with a known answer. When the quantity is random, the floor is dipy against itself
  on jittered seeds (dipy seeds its RNG from the seed's coordinate sum; a regular grid correlates its streamlines).
- **Declarative docstrings.** What a thing is; no history of what it used to be.
- **One function per application.** No compatibility spellings; a rename converts every caller.

## Traps

- `jnp.asarray(device_array)` is read-only on the host; `np.array(...)` before writing into it.
- `jax.random.uniform` in the kernel is float32; `u * total` with `total` the float32 CDF end. The numpy reference
  in `tests/conftest.py` mirrors that arithmetic; a float64 reference disagrees at CDF breakpoints.
- The Fibonacci hemisphere's nearest direction to an in-plane vector has a small positive `z`, so a streamline in a
  planar field drifts out of plane by up to `n_steps × step × angular_resolution` (the circle test's bound), as
  it does with dipy's hemisphere.
- `chunk` defaults to the smallest power of two holding the seeds, capped by `DMIPY_TRACT_CHUNK` (65,536): one
  compile per distinct chunk, `max_steps`, `n_coef`, `n_dirs` and rule.
- The JAX CUDA plugin needs the nvidia libraries on `LD_LIBRARY_PATH` in a uv venv (see the L40S notes in the
  memory), else it falls back to CPU silently; `JAX_PLATFORMS=cpu` in `tests/conftest.py` is a default, a GPU run
  sets `JAX_PLATFORMS=cuda`.
