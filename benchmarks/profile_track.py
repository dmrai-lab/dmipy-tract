"""Where the DiSCo tracking time goes: the two half kernels per chunk, the backward first direction, the host join."""
import os, sys, time, numpy as np
sys.path.insert(0, os.path.expanduser("~/dmipy-tract/benchmarks"))
import disco as bench, nibabel as nib, jax, jax.numpy as jnp
from dmipy_tract import FODField, seeds_from_mask, track
from dmipy_tract import tracker as T
REF = os.environ.get("DISCO_REF", os.path.expanduser("~/disco-ref"))
data, bvals, bvecs, _ = bench.load_volume(None, None, None)
mask = nib.load(f"{REF}/DiSCo_mask.nii.gz").get_fdata() > 0
rois = nib.load(f"{REF}/DiSCo_ROIs.nii.gz").get_fdata().astype(np.int32)
sh, _ = bench.csd(data, bench.scheme_for(bvals, bvecs), data[..., 0] > 0, "csd_tournier07_jax")
field = FODField(sh, np.eye(4), mask | (rois > 0))
seeds = seeds_from_mask(rois > 0, np.eye(4), density=4)
print("seeds", len(seeds), "device", jax.devices()[0], flush=True)
# instrument by monkeypatching the compiled functions with timed wrappers
times = {"first": 0.0, "half": 0.0, "back": 0.0, "join": 0.0, "n_half": 0}
orig_half, orig_first, orig_join, orig_back = T._compiled_half, T._compiled_first, T._join, T._backward_first_direction
def timed_half(*a):
    fn = orig_half(*a)
    def w(*args):
        t0 = time.perf_counter(); out = fn(*args); jax.block_until_ready(out); times["half"] += time.perf_counter() - t0; times["n_half"] += 1; return out
    return w
def timed_first(*a):
    fn = orig_first(*a)
    def w(*args):
        t0 = time.perf_counter(); out = fn(*args); jax.block_until_ready(out); times["first"] += time.perf_counter() - t0; return out
    return w
def timed_join(*a):
    t0 = time.perf_counter(); out = orig_join(*a); times["join"] += time.perf_counter() - t0; return out
def timed_back(*a):
    t0 = time.perf_counter(); out = orig_back(*a); jax.block_until_ready(out); times["back"] += time.perf_counter() - t0; return out
T._compiled_half, T._compiled_first, T._join, T._backward_first_direction = timed_half, timed_first, timed_join, timed_back
track(field, seeds[:4096], key=0)   # warm
for k in times: times[k] = 0
t0 = time.perf_counter(); tg = track(field, seeds, key=0); total = time.perf_counter() - t0
print(f"total {total:.2f} s: first-direction {times['first']:.2f}, halves {times['half']:.2f} ({times['n_half']} calls), backward-dir {times['back']:.2f}, host join {times['join']:.2f}, rest (transfers, padding) {total - sum(v for k, v in times.items() if k != 'n_half'):.2f}", flush=True)
print("n_points mean", tg.n_points.mean().round(1), "max", tg.n_points.max(), "stop reasons", np.bincount(tg.stop_reason.ravel(), minlength=5).tolist(), flush=True)
# the buffer-layout hypothesis: one half call with max_steps 500 vs 100 on the same chunk (cost per step)
T._compiled_half = orig_half
import functools
for ms in (100, 500):
    tt = []
    for rep in range(3):
        t0 = time.perf_counter(); tg2 = track(field, seeds[:65536], key=0, max_steps=ms); tt.append(time.perf_counter() - t0)
    print(f"one chunk of 65536, max_steps {ms}: {min(tt):.2f} s (n_points mean {tg2.n_points.mean():.1f})", flush=True)
