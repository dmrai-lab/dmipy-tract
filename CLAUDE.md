# dmipy-tract — rules for agents

Read `README.md` first: it says what the tracker is. This file is what an agent must not get wrong.

## What lives where

| file | holds |
|---|---|
| `dmipy_tract/field.py` | `FODField(sh, affine, mask)`: validation, the clamped trilinear interpolant (numpy), the nearest-voxel mask lookup |
| `dmipy_tract/sphere.py` | `hemisphere(n)` (Fibonacci, z > 0), `sh_matrix(order, dirs)` = `dmipy_sim.replay.so3.real_sh` (the order/count conversions are `so3.lmax_of` / `so3.n_sh_coeffs`) |
| `dmipy_tract/tracker.py` | `track()`: the first-direction kernel (`_compiled_first`), the phase kernel (`_compiled_phase`, K steps for one lane count), `_Batch` (device-resident state of one batch of seeds; phases with compaction of the active lanes; the device-side join of the phase slabs into the ragged output) and its fixed-shape helpers |
| `dmipy_tract/_torch.py` | the torch kernel behind `track(backend="torch")` (#4): `_Field` (field, sphere, settings on the device; `amplitudes`, `choose`, `mask_at`), `_half` (one step for every active lane, in chunks; a lockstep half, so point `t` of a lane is taken at loop step `t` and the join is a scatter per step), `track_torch`; `uniform` / `uniform_torch`, the counter-based draw in numpy and torch (identical bits) |
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
- **Two backends are two kernels of one tracker.** `track()` validates every argument before dispatch, and refuses
  by name an argument the chosen backend does not use (`phase_steps`, `batch` on torch; `device` on JAX); the conventions, the
  RNG contract (counter-based, chunk-invariant, device-invariant), `Tractogram`, seeding and connectivity are
  shared. A convention changed in one kernel is changed in the other and in `tests/conftest.py`'s reference. The
  torch draw is not JAX's stream: probabilistic tractograms differ between backends by design, the deterministic
  rule must agree (`tests/test_torch_backend.py`).

## Traps

- `jnp.asarray(device_array)` is read-only on the host; `np.array(...)` before writing into it.
- `jax.random.uniform` in the kernel is float32; `u * total` with `total` the float32 CDF end. The numpy reference
  in `tests/conftest.py` mirrors that arithmetic; a float64 reference disagrees at CDF breakpoints.
- The Fibonacci hemisphere's nearest direction to an in-plane vector has a small positive `z`, so a streamline in a
  planar field drifts out of plane by up to `n_steps × step × angular_resolution` (the circle test's bound), as
  it does with dipy's hemisphere.
- Lanes run in phases of `phase_steps` (`DMIPY_TRACT_PHASE`, 32); after each phase only the active lanes go on,
  pooled over the batch and padded to the smallest of 1, 16, 256, 4,096, `chunk` (65,536) lanes that holds them.
  All active lanes stand at the same point index at a phase boundary, so `t0` is one scalar; `max_steps` and
  `half_id` are traced.
- Everything is fixed-shape so that it compiles a bounded number of times: the state rows are 4,096, 65,536 or
  2**20 (`_state_rows`), the lane count comes from the ladder, the ragged output on the device is a power of four
  from 2**16 rows (`_points_rows`) plus a dump row that every unused scatter entry targets. A new helper must take
  its shapes from these ladders, never from `n`; the test suite (many small `n`) is the check, it was 5 minutes of
  compiles at one point and is 45 s.
- `real` (a seed, not a padding row) is not `ok` (has a first direction): a seed without a direction is a real
  one-point streamline and gets its seed row; padding rows get nothing.
- A phase slab can have zero columns (no lane took a step): guard `slab.shape[1] == 0` before indexing column 0.
- The JAX CUDA plugin needs the nvidia libraries on `LD_LIBRARY_PATH` in a uv venv (see the L40S notes in the
  memory), else it falls back to CPU silently; `JAX_PLATFORMS=cpu` in `tests/conftest.py` is a default, a GPU run
  sets `JAX_PLATFORMS=cuda,cpu` (both, for the CUDA-against-CPU test).
