# dmipy-tract — Agent Rules and Implementation Notes

This file documents correctness-critical implementation details, known pitfalls, and conventions
for agents working on dmipy-tract. Read this before touching any propagation, I/O, or filtering
code.

---

## RK4 sign convention (critical — read before touching deterministic.py)

### The rule

In `DeterministicTracker._propagate_direction`, the variable `sign` is `+1.0` (forward half)
or `-1.0` (backward half). **`sign` is applied only to sub-step position offsets and the final
position update. It is never applied to the k-vectors (k1, k2, k3, k4).**

Correct:
```python
k1 = self._get_direction(pos, direction)
k2 = self._get_direction(pos + sign * 0.5 * self.step_size * k1, k1)
k3 = self._get_direction(pos + sign * 0.5 * self.step_size * k2, k2)
k4 = self._get_direction(pos + sign * self.step_size * k3, k3)
step_dir = (k1 + 2*k2 + 2*k3 + k4) / 6.0
pos = pos + sign * self.step_size * step_dir
```

Wrong (do not do this):
```python
k2 = sign * self._get_direction(pos + 0.5 * self.step_size * k1, k1)  # BAD
```

### Why it matters

`_get_direction` receives a reference direction `prev_dir` and flips the retrieved peak if
`dot(peak, prev_dir) < 0`. This polarity-alignment logic must always see unsigned (non-negated)
unit vectors. If `sign` is folded into `k_prev`, the flip triggers on the wrong hemisphere and
the backward half produces step directions that reverse on every sub-step, causing the backward
streamline to oscillate at the seed point and produce a zero-length or single-point half, which
gets discarded. The symptom is that all streamlines appear to propagate in only one direction.

### The bug that was fixed

An early prototype had:
```python
k2 = self._get_direction(pos + 0.5 * step_size * (sign * k1), sign * k1)
```
This silently negated the reference direction on every sub-step evaluation. The curvature
constraint `dot(peak, prev_dir) < cos_max_angle` would then reject valid peaks, terminating the
backward half immediately. The fix: leave k-vectors unsigned; multiply by `sign` only when
computing where to sample the field next.

---

## TRK I/O: bbox_valid_check=False required on both save and load

dipy's `save_tractogram` and `load_tractogram` validate that all streamline points lie within
the bounding box defined by the reference NIfTI header. dmipy-tract streamlines use world-space
(RASMM) coordinates and may extend slightly outside the bounding box of a minimal reference
image (especially when a synthetic 1x1x1 dummy header is used). Always pass
`bbox_valid_check=False`:

```python
save_tractogram(sft, path, bbox_valid_check=False)   # in save_trk
sft = load_tractogram(path, "same", bbox_valid_check=False)   # in load_trk
```

Omitting this flag raises `ValueError: The file ... is not valid` even when the streamlines are
geometrically correct.

### Nifti1Header dim bug

When constructing a minimal reference image without a real NIfTI source (e.g. to save a
tractogram without a reference scan), do not attempt to set zooms on a header constructed from
scratch without first providing data:

```python
# Correct: construct Nifti1Image with data, THEN set zooms
ref = nib.Nifti1Image(np.zeros((1, 1, 1), dtype=np.uint8), affine)
ref.header.set_zooms(voxel_size)
```

nibabel's `Nifti1Header.set_zooms` requires `dim[0]` (ndim) to be set, which only happens when
data is attached via `Nifti1Image`. Creating a bare `Nifti1Header()` and calling `set_zooms`
raises an `AttributeError` or silently sets wrong pixel dimensions.

---

## SIFT cost function and algorithm

### SIFT (greedy filtering)

Cost function: `C = sum_v (TDI_v - target_density_v)^2`

The greedy loop in `SIFTFilter.filter()`:
1. Compute per-streamline voxel contribution maps (sparse: only non-zero voxels).
2. Initialise `current_tdi = sum of all contributions`.
3. On each iteration, find the streamline whose removal maximally reduces C.
4. Remove it, update `current_tdi`, repeat until no removal reduces C.

Time complexity: O(N_streamlines^2 * N_voxels_per_streamline) in the worst case. For large
tractograms (>100k streamlines), switch to SIFT2 weights.

### SIFT2 (weighted, NNLS)

`SIFTFilter.weights()` builds a dense matrix A of shape `(n_voxels, n_streamlines)` where
`A[v, i]` is the length contribution of streamline i to voxel v. Solves `min ||Aw - b||^2`
subject to `w >= 0` using `scipy.optimize.nnls`. Falls back to uniform weights with a
`RuntimeWarning` if the problem has more than 1,000,000 entries.

---

## Fury import: lazy, headless CI uses mocking

Fury is an optional dependency for visualisation. It must be imported **inside** the function
body, never at module level:

```python
def plot_tractogram(...):
    try:
        import fury.actor as actor
        import fury.window as window
    except ImportError as exc:
        raise ImportError("fury is required ...") from exc
```

This ensures that importing `dmipy_tract` does not fail on headless servers without fury.

In CI tests, mock fury at the module level:
```python
import sys
from unittest.mock import MagicMock
sys.modules["fury"] = MagicMock()
sys.modules["fury.actor"] = MagicMock()
sys.modules["fury.window"] = MagicMock()
```

Do not use `pytest.importorskip("fury")` as the primary guard — this skips the test rather than
verifying that the mock-based code path works.

---

## BIDS path structure convention

`save_bids_tractogram` writes to:
```
<output_dir>/<subject>/[<session>/]dwi/<subject>[_<session>]_desc-<desc>_<suffix>.trk
<output_dir>/<subject>/[<session>/]dwi/<subject>[_<session>]_desc-<desc>_<suffix>.json
```

Example with session:
```
derivatives/dmipy-tract/sub-01/ses-01/dwi/sub-01_ses-01_desc-wholebrain_tractography.trk
```

Example without session:
```
derivatives/dmipy-tract/sub-01/dwi/sub-01_desc-wholebrain_tractography.trk
```

The JSON sidecar contains: `n_streamlines`, `affine`, `TrackerSoftware`, `GeneratedBy`.
Do not add fields not present in the implementation without updating both the writer and the
tests.

---

## MicrostructureField: peaks_at return shape and principal direction

`MicrostructureField.peaks_at(xyz)` returns `ndarray, shape (K, 3)` where K is the number of
peaks in the fitted model. **Index 0 is always the principal (dominant) peak direction.**
`DeterministicTracker` and `StochasticTracker` both use `peaks[0]` as the propagation
direction.

Peaks are returned as unit vectors (L2-normalised). A zero-length peak is possible at voxels
where the fitted model did not converge; both trackers guard against this with:
```python
if np.all(peaks[0] == 0):
    break
```

`parameter_at(name, xyz)` supports only scalar (3-D) parameter maps. Calling it with a
parameter that has shape `(X, Y, Z, K)` (e.g. a raw peaks array) raises `ValueError`.

---

## Stopping criteria parameter names

Default dmipy-core parameter names used by the stopping criteria:

| Criterion | Default parameter_name |
|---|---|
| VfIcThreshold | `partial_volume_0` |
| OdiThreshold | `SD1WatsonDistributed_1_odi` |

These are the dmipy-core conventions for a single-fibre NODDI model. For multi-fibre models or
non-NODDI compartments, pass `parameter_name` explicitly.

---

## Write access for agents

This repo contains runnable tractography code. The implementation lives in `dmipy_tract/`.
Validation engineers write to `tests/` and `benchmarks/`. See the top-level dmipy
`CLAUDE.md` for the full write-access policy.
