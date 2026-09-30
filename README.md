# dmipy-tract

Streamline tractography on FOD fields, batched on the JAX device. One data model, one tracker, one output:

```python
from dmipy_tract import FODField, seeds_from_mask, track, connectivity

field = FODField(sh, affine, mask)                     # (X, Y, Z, n_coef) SH in dmipy-sim's basis, voxel->mm, domain
seeds = seeds_from_mask(rois > 0, affine, density=4)   # (n, 3) mm, dipy's construction
tg = track(field, seeds, rule='probabilistic', step_mm=0.5, max_angle=30.0, key=0)
matrix, ends = connectivity(tg, rois, affine)          # streamline counts between regions
tg.to_tck("tracks.tck")                                # MRtrix format, through dmipy-sim
```

`FODField.from_mif("wmfod.mif")` reads an MRtrix FOD; a dmipy-fit `csd_tournier07_jax` fit's `sh_coeff` is the
same basis, so no basis conversion, but its directions are those of the gradient table it was fitted with and an
`FODField`'s are world coordinates: fit with the b-vectors in the world frame (for a scaling-plus-translation
affine the voxel frame is the world frame; under a rotation, rotate the b-vectors or the coefficients with
`so3.rotate_sh`). DiSCo, the BATMAN brain and an HCP subject are the same input; the seeding density,
not the grid, sets the time.

## What the tracker is

The two direction rules every reference has, on one lockstep kernel (`jax.lax.while_loop` over steps, `vmap` over
seeds, chunks of 65,536 lanes):

- **probabilistic**: the FOD interpolated trilinearly on its coefficients and evaluated on a 362-direction
  hemisphere, amplitudes below 10 % of the sphere-wide maximum set to zero, restricted to the cone of `max_angle`
  around the previous direction, sampled by inverse CDF (dipy's `ProbabilisticDirectionGetter`);
- **deterministic**: the maximum inside the cone (dipy's `DeterministicMaximumDirectionGetter`).

The first direction at a seed is drawn from the FOD on the whole sphere; the backward half starts against the
forward half's actual first step; the streamline is the backward half reversed followed by the forward half, with
the seed once. A step whose landing point has its nearest voxel outside the mask or the grid is not taken and ends
the half with that reason (`stop_reason`); no direction above the threshold inside the cone ends it too; `max_steps`
points per half is the cap. These are dipy's `LocalTracking` conventions, measured on constructed fields, so the
deterministic rule reproduces dipy point for point (a test).

Randomness is counter-based: streamline `i` draws from `fold_in(fold_in(key, i), 1 + 2 t + h)` at step `t` of half
`h` (0 forward, 1 backward) and from `fold_in(fold_in(key, i), 0)` for its first direction, so it is a function of
the key, seed `i` and the field, not of the chunk size, the phase length, the batch or the other seeds. Positions are float32 millimetres; every matrix product
is at `Precision.HIGHEST` (a float32 matmul on CUDA is TF32 otherwise).

**Two backends, one tracker** (`track(..., backend="jax" | "torch")`). The torch kernel
(`dmipy_tract/_torch.py`, `pip install dmipy-tract[torch]`) exists for hosts that run PyTorch only (Hugging Face's
shared GPU pool); it is eager, one step for every active lane at once in chunks, the same conventions, TF32 off for
the call. Its draws come from its own counter-based stream (the splitmix64 finaliser of `(key, streamline, counter)`,
the same bits on the CPU, on CUDA and in numpy), so the two backends give different, equally valid probabilistic
tractograms; on the deterministic rule they agree to float32 arithmetic. One test set runs on both kernels
(stopping, reasons, join, backward half, circle, invariances, the per-streamline reference under each kernel's own
draws); the CUDA-against-CPU tests of both run where a card is present.

## Measured

The DiSCo acceptance (`benchmarks/disco.py`, `benchmarks/out/`): the reference replay volume, CSD with dmipy-fit's
`csd_tournier07_jax` (order 8, the single-fibre response, 9 s), 659,840 seeds (density 4 in the 16 regions, 64 per
voxel), step 0.5, 30°, 500 steps per half. All 120 region pairs come out connected in every pipeline (25 true, 95
false, 0 missed), as in the published dipy runs.

| tracker | streamlines | Pearson vs strand count | vs area | tracking |
|---|---|---|---|---|
| this package, L40S | 659,840 | **0.927** | 0.929 | **1.6 s** steady, 11 s first call (compile) |
| this package, CPU (16 of 72 cores, shared box, v0 before compaction) | 659,840 | 0.927 | 0.929 | 659 s |
| dipy `LocalTracking` on the same FOD and seeds, L40S host | 2,822,911 (one per FOD peak per seed) | 0.917 | 0.917 | 1016 s |
| the replay paper's dipy pipeline (dipy CSD), dmipy-sim#505 | 1.6 M | 0.904 – 0.912 | | 250 – 500 s |

The two connectivity matrices (this tracker and dipy on the same field) correlate at 0.997. On the L40S on
2026-09-29 the CPU and GPU tractograms of this run were bit-identical (lengths, stop reasons and positions): a
measurement on that host, not a guarantee of the design (another device or XLA version may order a float32 sum
differently); `tests/test_tracker.py::test_jax_cuda_equals_the_cpu_bit_for_bit` and its torch twin check it where
a card is present. Tractograms at any chunk size, phase length or batch size are bit-identical by construction
(tested).

Where the 1.6 s go (`benchmarks/profile_track.py`, L40S): the phase kernels 0.9 s (40 calls at 65,536 or 4,096
lanes, phases of 32 steps between compactions of the active lanes), the device-side join and its one transfer 0.2 s,
lane selection and state updates on the device 0.05 s, the rest Python dispatch and scalar syncs. The v0 lockstep
kernel took 6.5 s on its own (each chunk ran to its longest lane, 266 steps for a mean streamline of 35 points),
and the host-side ragged join 0.9 s. The kernel does 6.6e7 lane-steps/s flat out (1 ms per iteration of 65,536 lanes
at order 8 on 362 directions), insensitive to the rule, the sphere size and the order (`benchmarks/ablate2.py`):
three quarters of an iteration is the eight-corner gather of the coefficients, the design's intended bound.

Brain scale (`benchmarks/scaling.py`, a 145 × 174 × 145 order-8 field, streamlines of 227 points, L40S):

| seeds | tracking | host RSS |
|---|---|---|
| 10⁵ | 1.4 s | 3.4 GB |
| 10⁶ | 6.9 s (3.3e7 lane-steps/s) | 6.2 GB |
| 10⁷ | the output is 2.3 × 10⁹ points (27 GB float32); the batches (2²⁰ seeds each) bound the device memory, an in-memory `Tractogram` does not fit | |

**Next, measured not guessed:** (1) a streaming `.tck` writer over the batches, the 10⁷-seed case; (2) the first
call pays about 10 s of compile for the fixed-shape kernels, which `JAX_COMPILATION_CACHE_DIR` (JAX's persistent
cache) removes from the second process on; (3) the coefficient gather, if a brain at 10⁶ seeds must go below 7 s.

## Tests

`pytest tests -q` (CPU, 75 s on the shared 72-core host; dipy is a test dependency, never imported by the package). Every mechanism has
a deterministic test on constructed inputs: interpolation against closed forms and `scipy.ndimage.map_coordinates`;
the basis against a delta on every sphere direction and against dipy's `tournier07`; the inverse-CDF sampler at every
breakpoint; stopping exhaustive over seed positions and over the four reasons; the bidirectional join; the deterministic
rule on concentric circles to the sphere's angular-resolution bound; key, chunk-size, rigid-affine and isotropic-scale
invariances; refusals by name. The batched kernel is checked against a per-streamline numpy tracker of the same
definition, then against dipy: point for point for the deterministic rule (uniform, crossing and circle fields, a
scaled affine), and for the probabilistic rule within dipy's own scatter (endpoint-density Dice and region-pair
fractions, with the floor measured from dipy against itself on jittered seeds: dipy seeds its per-streamline RNG
from the seed's coordinate sum, so a regular seed grid shares random streams within a run).

## Not in v0

PTT, anatomically constrained stopping (5TT), SIFT/SIFT2, TDI, tractometry, learned trackers: all reachable on the
same kernel and the same `Tractogram`, none of them here.

## Install

```
pip install "dmipy-sim @ git+https://github.com/dmrai-lab/dmipy-sim.git"
pip install -e .            # CPU; pip install -e ".[cuda]" for a CUDA device
pip install -e ".[test]"    # dipy, scipy, nibabel, dmipy-fit, huggingface_hub for the tests and benchmarks
```
