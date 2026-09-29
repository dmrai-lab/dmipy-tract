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
same basis, no conversion. DiSCo, the BATMAN brain and an HCP subject are the same input; the seeding density,
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

Randomness is counter-based (`fold_in(fold_in(key, seed_index), step)`): streamline `i` is a function of the key,
seed `i` and the field, not of the chunk size or the device. Positions are float32 millimetres; every matrix product
is at `Precision.HIGHEST` (a float32 matmul on CUDA is TF32 otherwise).

## Measured

The DiSCo acceptance (`benchmarks/disco.py`, `benchmarks/out/`): the reference replay volume, CSD with dmipy-fit's
`csd_tournier07_jax` (order 8, the single-fibre response, 9 s), 659,840 seeds (density 4 in the 16 regions, 64 per
voxel), step 0.5, 30°, 500 steps per half. All 120 region pairs come out connected in every pipeline (25 true, 95
false, 0 missed), as in the published dipy runs.

| tracker | streamlines | Pearson vs strand count | vs area | tracking |
|---|---|---|---|---|
| this package, L40S | 659,840 | **0.927** | 0.929 | **2.5 s** steady, 10.8 s first call (compile) |
| this package, CPU (16 of 72 cores, shared box, before compaction) | 659,840 | 0.927 | 0.929 | 659 s |
| dipy `LocalTracking` on the same FOD and seeds, L40S host | 2,822,911 (one per FOD peak per seed) | 0.917 | 0.917 | 1016 s |
| the replay paper's dipy pipeline (dipy CSD), dmipy-sim#505 | 1.6 M | 0.904 – 0.912 | | 250 – 500 s |

The two connectivity matrices (this tracker and dipy on the same field) correlate at 0.997. CPU and GPU tractograms
are bit-identical (lengths, stop reasons and positions), and so are tractograms at any chunk size or phase length.

Where the 2.5 s go (`benchmarks/profile_track.py`, L40S): the phase kernels 0.9 s (40 calls at 65,536 or 4,096 lanes,
phases of 32 steps between compactions of the active lanes), the ragged scatter on the host 0.9 s, host state and
transfers 0.7 s. Before compaction the kernels alone took 6.5 s, because each chunk ran to its longest lane (266 steps
for a mean streamline of 35 points). The kernel does 6.6e7 lane-steps/s flat out (1 ms per iteration of 65,536 lanes at
order 8 on 362 directions), insensitive to the rule, the sphere size and the order (`benchmarks/ablate2.py`): it is
gather-bound on the field, as designed. Phase lengths 16 and 32 measure the same; 64 costs 1 s more.

Brain scale (`benchmarks/scaling.py`, a 145 × 174 × 145 order-8 field, streamlines of 227 points, L40S):

| seeds | tracking | host RSS |
|---|---|---|
| 10⁵ | 7.2 s | 3.6 GB |
| 10⁶ | 17 s | 9.1 GB |
| 10⁷ | the output itself is 2.3 × 10⁹ points (27 GB float32): not a case for an in-memory tractogram | |

At brain scale the host side (the ragged scatter, three passes over the points) is most of the time; the three
scatter formulations tried run within 5 % of each other on the host, so the next gain is a device-side join with a
streaming `.tck` writer, which also covers 10⁷ seeds. The first call pays about 8 s of compile for the three kernel
shapes; a persistent compile cache is the other pending item.

## Tests

`pytest tests -q` (CPU, about a minute; dipy is a test dependency, never imported by the package). Every mechanism has
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
