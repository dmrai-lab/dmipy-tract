"""Brain-scale timing on a synthetic field: a 145 x 174 x 145 grid at order 8 (an HCP-like volume, 660 MB in
float32) of two crossing bundles through a spherical mask, seeded at 1e5, 1e6 and 1e7 seeds; first call and steady
state per seed count, the resident-memory high-water mark, and the same on CPU with ``--cpu``.

    python benchmarks/scaling.py [--seeds 100000 1000000 10000000] [--cpu] [--out out/scaling.json]
"""
import argparse
import json
import os
import resource
import time

import numpy as np


def synthetic_field(shape=(145, 174, 145), order=8):
    from dmipy_tract import FODField, sh_matrix
    d1 = sh_matrix(order, np.array([[1.0, 0.0, 0.0]]))[0]
    d2 = sh_matrix(order, np.array([[0.0, 0.6, 0.8]]))[0]
    ijk = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing='ij'), -1).astype(np.float32)
    c = (np.asarray(shape, np.float32) - 1) / 2
    r = np.linalg.norm((ijk - c) / c, axis=-1)
    mask = r < 1.0
    w2 = np.clip(1.0 - r, 0.0, 1.0)[..., None].astype(np.float32)
    sh = (mask[..., None] * (d1[None, None, None] * (1.0 - 0.5 * w2) + d2[None, None, None] * 0.7 * w2)).astype(np.float32)
    return FODField(sh, np.diag([1.25, 1.25, 1.25, 1.0]), mask)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[100_000, 1_000_000, 10_000_000])
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "scaling.json"))
    a = ap.parse_args()
    if a.cpu:
        os.environ["JAX_PLATFORMS"] = "cpu"
    import jax
    from dmipy_tract import track, seeds_from_mask
    field = synthetic_field()
    all_seeds = seeds_from_mask(field.mask, field.affine, density=1)
    rng = np.random.default_rng(0)
    rows = []
    for n in a.seeds:
        seeds = all_seeds[rng.choice(len(all_seeds), n, replace=n > len(all_seeds))] + rng.uniform(-0.6, 0.6, (n, 3))
        t0 = time.perf_counter()
        tg = track(field, seeds, step_mm=0.6, max_steps=a.max_steps, key=1)
        t_first = time.perf_counter() - t0                  # includes the compiles this seed count needs
        del tg
        t0 = time.perf_counter()
        tg = track(field, seeds, step_mm=0.6, max_steps=a.max_steps, key=1)
        t = time.perf_counter() - t0
        lane_steps = int((tg.n_points - 1).sum()) + 2 * len(tg)
        rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        row = dict(seeds=n, streamlines=len(tg), points=int(tg.points.shape[0]), n_points_mean=float(tg.n_points.mean()),
                   first_call_seconds=t_first, seconds=t, lane_steps=lane_steps,
                   lane_steps_per_second=lane_steps / t, host_rss_gb=rss_gb)
        rows.append(row)
        print(f"{str(jax.devices()[0]).split(':')[0]}: {n:>9d} seeds -> {len(tg)} streamlines, {row['n_points_mean']:.0f} "
              f"points each, {t:.1f} s ({row['lane_steps_per_second']:.2e} lane-steps/s), first call {t_first:.1f} s, "
              f"host RSS {rss_gb:.1f} GB", flush=True)
        del tg
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(dict(device=str(jax.devices()[0]).split(':')[0], shape=list(field.shape), rows=rows), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
