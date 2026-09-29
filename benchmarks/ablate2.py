"""Per-iteration cost of the half kernel's pieces, by removing them one at a time (a fixed 100-iteration loop of
65536 lanes on the synthetic field)."""
import os, sys, time, numpy as np
sys.path.insert(0, os.path.expanduser("~/dmipy-tract/benchmarks"))
import jax, jax.numpy as jnp
from scaling import synthetic_field
from dmipy_tract import hemisphere, sh_matrix, seeds_from_mask
from dmipy_tract.tracker import _interpolate, _amplitudes, _sample, _mask_at, _HI
f = synthetic_field(); V = hemisphere(); B = sh_matrix(8, V)
chunk, steps = 65536, 100
rng = np.random.default_rng(0)
allseeds = seeds_from_mask(f.mask, f.affine, density=1)
seeds = jnp.asarray(allseeds[rng.choice(len(allseeds), chunk, replace=False)] + rng.uniform(-0.6, 0.6, (chunk, 3)), jnp.float32)
dims = jnp.asarray(f.shape, jnp.int32); field_flat = jnp.asarray(f.sh.reshape(-1, 45)); mask_flat = jnp.asarray(f.mask.reshape(-1))
inv = f.inverse_affine; inv_lin = jnp.asarray(inv[:3, :3], jnp.float32); inv_off = jnp.asarray(inv[:3, 3], jnp.float32)
Bd = jnp.asarray(B, jnp.float32); Vd = jnp.asarray(V, jnp.float32)
keys = jax.vmap(jax.random.fold_in, (None, 0))(jax.random.key(0), jnp.arange(chunk, dtype=jnp.uint32))
d0 = jnp.tile(jnp.asarray([[1.0, 0, 0]], jnp.float32), (chunk, 1))
def build(write=True, gather=True, sample=True, rngon=True, maskon=True, layout="lane_major"):
    def lane(pos, d, key):
        if gather:
            pmf = _amplitudes(field_flat, dims, inv_lin, inv_off, Bd, jnp.float32(0.1), pos)
        else:
            c = jnp.ones(45, jnp.float32) * 0.01; pmf = jnp.dot(Bd, c, precision=_HI); pmf = jnp.where(pmf <= 0, 0.0, pmf)
        cone = jnp.abs(jnp.dot(Vd, d, precision=_HI)) >= jnp.float32(0.866)
        if sample:
            idx, ok = _sample(jnp.where(cone, pmf, 0.0), key)
        else:
            w = jnp.where(cone, pmf, 0.0); idx = jnp.argmax(w); ok = w[idx] > 0
        nd = Vd[idx]; nd = jnp.where(jnp.dot(nd, d, precision=_HI) > 0, nd, -nd)
        new = pos + 0.6 * nd
        if maskon:
            in_grid, in_mask = _mask_at(mask_flat, dims, inv_lin, inv_off, new)
        else:
            in_grid = in_mask = jnp.bool_(True)
        return new, nd, ok & in_mask
    lane_v = jax.vmap(lane)
    fold_v = jax.vmap(jax.random.fold_in, (0, None))
    def run(seed_pos, first_dir, lane_keys):
        buf = jnp.zeros((chunk, steps, 3) if layout == "lane_major" else (steps, chunk, 3), jnp.float32)
        def body(c):
            buf, pos, d, active, t = c
            k = fold_v(lane_keys, 1 + 2 * t) if rngon else lane_keys
            new, nd, ok = lane_v(pos, d, k)
            taken = active & ok
            if write:
                if layout == "lane_major":
                    buf = buf.at[:, t].set(jnp.where(taken[:, None], new, buf[:, t]))
                else:
                    buf = buf.at[t].set(jnp.where(taken[:, None], new, 0.0))
            pos = jnp.where(taken[:, None], new, pos); d = jnp.where(taken[:, None], nd, d)
            return buf, pos, d, taken, t + 1
        return jax.lax.while_loop(lambda c: c[4] < steps, body, (buf, seed_pos, first_dir, jnp.ones(chunk, bool), jnp.int32(1)))
    return jax.jit(run)
for name, kw in [("full", {}), ("no buffer write", dict(write=False)), ("no field gather", dict(gather=False)),
                 ("argmax instead of sample", dict(sample=False)), ("no per-step fold_in", dict(rngon=False)),
                 ("no mask lookup", dict(maskon=False)), ("step-major buffer", dict(layout="step_major")),
                 ("no write, no gather, no rng", dict(write=False, gather=False, rngon=False))]:
    fn = build(**kw); out = fn(seeds, d0, keys); jax.block_until_ready(out)
    best = 1e9
    for _ in range(3):
        t0 = time.perf_counter(); out = fn(seeds, d0, keys); jax.block_until_ready(out); best = min(best, time.perf_counter() - t0)
    print(f"{name:32s}: {best*1e3/steps:6.2f} ms per iteration of {chunk} lanes  ({chunk*steps/best:.2e} lane-steps/s)", flush=True)
