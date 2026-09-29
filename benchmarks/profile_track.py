"""Where the DiSCo tracking time goes: the first-direction call, the phase kernels, the device-side join with its
transfer, and the rest (lane selection and state updates on the device).

    DISCO_REF=~/disco-ref python benchmarks/profile_track.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import disco as bench                                   # noqa: E402
import nibabel as nib                                   # noqa: E402
import jax                                              # noqa: E402
from dmipy_tract import FODField, seeds_from_mask, track   # noqa: E402
from dmipy_tract import tracker as T                    # noqa: E402

REF = os.environ.get("DISCO_REF", os.path.expanduser("~/disco-ref"))
data, bvals, bvecs, _ = bench.load_volume(None, None, None)
mask = nib.load(f"{REF}/DiSCo_mask.nii.gz").get_fdata() > 0
rois = nib.load(f"{REF}/DiSCo_ROIs.nii.gz").get_fdata().astype(np.int32)
sh, _ = bench.csd(data, bench.scheme_for(bvals, bvecs), data[..., 0] > 0, "csd_tournier07_jax")
field = FODField(sh, np.eye(4), mask | (rois > 0))
seeds = seeds_from_mask(rois > 0, np.eye(4), density=4)
print("seeds", len(seeds), "device", jax.devices()[0], flush=True)

times = {"first": 0.0, "phase_kernel": 0.0, "join": 0.0, "select": 0.0, "gather": 0.0, "update": 0.0,
         "first_point": 0.0}
calls = {"phase": 0}
lanes_seen = []
orig_phase, orig_first, orig_join = T._compiled_phase, T._compiled_first, T._Batch.join


def timed_helper(name, orig):
    def maker(*a):
        fn = orig(*a)

        def w(*args):
            t0 = time.perf_counter()
            out = fn(*args)
            jax.block_until_ready(out)
            times[name] += time.perf_counter() - t0
            return out
        return w
    return maker


for _name, _attr in (("select", "_compiled_select"), ("gather", "_compiled_gather"), ("update", "_compiled_update"),
                     ("first_point", "_compiled_first_point")):
    setattr(T, _attr, timed_helper(_name, getattr(T, _attr)))


def timed_phase(*a):
    fn = orig_phase(*a)

    def w(*args):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        times["phase_kernel"] += time.perf_counter() - t0
        calls["phase"] += 1
        lanes_seen.append(int(args[0].shape[0]))
        return out
    return w


def timed_first(*a):
    fn = orig_first(*a)

    def w(*args):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        times["first"] += time.perf_counter() - t0
        return out
    return w


def timed_join(self, *a, **k):
    t0 = time.perf_counter()
    out = orig_join(self, *a, **k)
    times["join"] += time.perf_counter() - t0
    return out


T._compiled_phase, T._compiled_first, T._Batch.join = timed_phase, timed_first, timed_join
track(field, seeds, key=0)                              # compile every shape this run uses
for k in times:
    times[k] = 0.0
calls["phase"] = 0
lanes_seen.clear()
t0 = time.perf_counter()
tg = track(field, seeds, key=0)
total = time.perf_counter() - t0
rest = total - sum(times.values())
print(f"total {total:.2f} s: first-direction {times['first']:.2f}, phase kernels {times['phase_kernel']:.2f} "
      f"({calls['phase']} calls, lane counts {sorted(set(lanes_seen))}), device join + transfer {times['join']:.2f}, "
      f"select {times['select']:.2f}, gather {times['gather']:.2f}, update {times['update']:.2f}, "
      f"first_point {times['first_point']:.2f}, rest {rest:.2f}", flush=True)
print("n_points mean", tg.n_points.mean().round(1), "max", tg.n_points.max(), "stop reasons",
      np.bincount(tg.stop_reason.ravel(), minlength=5).tolist(), flush=True)
