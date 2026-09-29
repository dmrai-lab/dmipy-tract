"""Streamline tracking on an FOD field, every seed of a chunk advanced in lockstep on the JAX device.

The two direction rules every reference has, on one kernel:

* ``probabilistic``: the FOD interpolated at the position (coefficients, then evaluated on the sphere), amplitudes
  below ``relative_threshold`` times the sphere-wide maximum set to zero, restricted to the directions within
  ``max_angle`` of the previous one, sampled by inverse CDF (dipy's ``ProbabilisticDirectionGetter``, MRtrix's
  iFOD1 up to its rejection sampler and its lack of a threshold);
* ``deterministic``: the same amplitudes, their maximum inside the cone (dipy's ``DeterministicMaximumDirectionGetter``,
  MRtrix's SD_STREAM up to interpolation).

At the seed the first direction is drawn from the FOD on the whole sphere (sampled, or its maximum); the forward
half then steps from the seed with the cone around that direction, the backward half starts against the forward
half's actual first step, and the streamline is the backward half reversed followed by the forward half, with the
seed once. A step whose landing point has its nearest voxel outside the mask or outside the grid is not taken and
ends the half with that reason; a position where no direction inside the cone is above the threshold ends it with
``no_direction``; ``max_steps`` points per half is the cap. These are dipy's ``LocalTracking`` conventions,
measured on constructed fields (a streamline never holds a point outside the domain).

Randomness is counter-based: the draw at (seed ``i``, half ``h``, step ``t``) comes from
``fold_in(fold_in(key, i), 1 + 2 t + h)`` (``fold_in(fold_in(key, i), 0)`` for the first direction), so streamline ``i``
is a function of the key, seed ``i`` and the field alone: not of the chunk size, the other seeds or the device.
Positions are float32 millimetres on the device; every matrix product is at ``Precision.HIGHEST`` (a float32 matmul
on CUDA is TF32 otherwise, which moves amplitudes at 1e-3 and flips near-tie choices).
"""
import functools
import os

import numpy as np
import jax
import jax.numpy as jnp

from .field import FODField
from .sphere import hemisphere, sh_matrix
from .tractogram import Tractogram, STOP_MASK, STOP_OUTSIDE, STOP_NO_DIRECTION, STOP_MAX_STEPS

__all__ = ['track', 'RULES']

RULES = ('probabilistic', 'deterministic')
_HI = jax.lax.Precision.HIGHEST


# ----------------------------------------------------------------------------------------------------------------
# per-lane pieces (traced once per compiled shape)
def _interpolate(field_flat, dims, v):
    """Clamped trilinear coefficients at voxel coordinate ``v (3,)``; zero outside ``[-0.5, N - 0.5]``."""
    flr = jnp.floor(v)
    rem = v - flr
    i0 = jnp.clip(flr, 0, dims - 1).astype(jnp.int32)
    i1 = jnp.clip(flr + 1, 0, dims - 1).astype(jnp.int32)
    ix = jnp.stack([i0[0], i1[0]])
    iy = jnp.stack([i0[1], i1[1]])
    iz = jnp.stack([i0[2], i1[2]])
    wx = jnp.stack([1.0 - rem[0], rem[0]])
    wy = jnp.stack([1.0 - rem[1], rem[1]])
    wz = jnp.stack([1.0 - rem[2], rem[2]])
    idx = (ix[:, None, None] * (dims[1] * dims[2]) + iy[None, :, None] * dims[2] + iz[None, None, :]).reshape(8)
    w = (wx[:, None, None] * wy[None, :, None] * wz[None, None, :]).reshape(8)
    c = jnp.dot(w, field_flat[idx], precision=_HI)
    inside = jnp.all(v >= -0.5) & jnp.all(v <= dims - 0.5)
    return jnp.where(inside, c, jnp.zeros_like(c))


def _amplitudes(field_flat, dims, inv_lin, inv_off, B, rel_thr, pos):
    """The thresholded FOD on the sphere at world position ``pos``: amplitudes below ``rel_thr`` of the maximum,
    and non-positive ones, are zero."""
    v = jnp.dot(inv_lin, pos, precision=_HI) + inv_off
    c = _interpolate(field_flat, dims, v)
    pmf = jnp.dot(B, c, precision=_HI)
    mx = jnp.max(pmf)
    return jnp.where((pmf < rel_thr * mx) | (pmf <= 0), jnp.zeros_like(pmf), pmf)


def _sample(pmf, key):
    """Inverse-CDF index into ``pmf`` and whether any mass exists."""
    cdf = jnp.cumsum(pmf)
    total = cdf[-1]
    u = jax.random.uniform(key, dtype=cdf.dtype) * total
    idx = jnp.minimum(jnp.sum(cdf <= u), pmf.shape[0] - 1)
    return idx, total > 0


def _argmax(pmf):
    idx = jnp.argmax(pmf)
    return idx, pmf[idx] > 0


def _mask_at(mask_flat, dims, inv_lin, inv_off, pos):
    """``(in_grid, in_mask)`` of the nearest voxel of world position ``pos``."""
    v = jnp.dot(inv_lin, pos, precision=_HI) + inv_off
    iv = jnp.rint(v).astype(jnp.int32)
    in_grid = jnp.all(iv >= 0) & jnp.all(iv < dims)
    ivc = jnp.clip(iv, 0, dims - 1)
    in_mask = in_grid & mask_flat[ivc[0] * (dims[1] * dims[2]) + ivc[1] * dims[2] + ivc[2]]
    return in_grid, in_mask


@functools.lru_cache(maxsize=None)
def _compiled_first(rule, n_coef, n_dirs, chunk):
    """The first direction at each seed from the whole-sphere FOD: ``(dirs (chunk, 3), ok (chunk,))``."""
    prob = rule == 'probabilistic'

    def one(pos, lane_key, field_flat, dims, inv_lin, inv_off, B, V, rel_thr):
        pmf = _amplitudes(field_flat, dims, inv_lin, inv_off, B, rel_thr, pos)
        idx, ok = _sample(pmf, jax.random.fold_in(lane_key, 0)) if prob else _argmax(pmf)
        return jnp.where(ok, V[idx], jnp.zeros(3, V.dtype)), ok

    return jax.jit(jax.vmap(one, in_axes=(0, 0, None, None, None, None, None, None, None)))


@functools.lru_cache(maxsize=None)
def _compiled_phase(rule, n_coef, n_dirs, K, lanes):
    """``K`` steps of one half for ``lanes`` lanes that all stand at point index ``t0``: ``(slab (lanes, K, 3), count
    (lanes,), pos, d, active, reason)``; the slab holds the points each lane took this phase, ``count`` how many."""
    prob = rule == 'probabilistic'

    def lane_step(field_flat, dims, inv_lin, inv_off, mask_flat, B, V, step, cos_max, rel_thr, pos, d, key):
        pmf = _amplitudes(field_flat, dims, inv_lin, inv_off, B, rel_thr, pos)
        cone = jnp.abs(jnp.dot(V, d, precision=_HI)) >= cos_max
        if prob:
            idx, ok = _sample(jnp.where(cone, pmf, 0.0), key)
        else:
            idx, ok = _argmax(jnp.where(cone, pmf, 0.0))
        nd = V[idx]
        nd = jnp.where(jnp.dot(nd, d, precision=_HI) > 0, nd, -nd)
        new_pos = pos + step * nd
        in_grid, in_mask = _mask_at(mask_flat, dims, inv_lin, inv_off, new_pos)
        return new_pos, nd, ok, in_grid, in_mask

    lane_step_v = jax.vmap(lane_step, in_axes=(None,) * 10 + (0, 0, 0))
    fold_v = jax.vmap(jax.random.fold_in, in_axes=(0, None))

    def phase(pos, d, active0, lane_keys, half_id, t0, max_steps, field_flat, dims, inv_lin, inv_off, mask_flat, B, V,
              step, cos_max, rel_thr):
        slab = jnp.zeros((lanes, K, 3), jnp.float32)

        def cond(c):
            k = c[6]
            return jnp.any(c[3]) & (k < K) & (t0 + k < max_steps)

        def body(c):
            slab, pos, d, active, cnt, reason, k = c
            keys = fold_v(lane_keys, 1 + 2 * (t0 + k) + half_id)
            new_pos, nd, ok, in_grid, in_mask = lane_step_v(field_flat, dims, inv_lin, inv_off, mask_flat, B, V,
                                                              step, cos_max, rel_thr, pos, d, keys)
            taken = active & ok & in_mask
            slab = slab.at[:, k].set(jnp.where(taken[:, None], new_pos, 0.0))
            pos = jnp.where(taken[:, None], new_pos, pos)
            d = jnp.where(taken[:, None], nd, d)
            r = jnp.where(~ok, STOP_NO_DIRECTION, jnp.where(~in_grid, STOP_OUTSIDE, STOP_MASK)).astype(jnp.int8)
            reason = jnp.where(active & ~taken, r, reason)
            return slab, pos, d, taken, cnt + taken, reason, k + 1

        init = (slab, pos, d, active0, jnp.zeros(lanes, jnp.int32), jnp.zeros(lanes, jnp.int8), jnp.int32(0))
        slab, pos, d, active, cnt, reason, _ = jax.lax.while_loop(cond, body, init)
        return slab, cnt, pos, d, active, reason

    return jax.jit(phase)


_PHASE_LANES_SMALL = 4096


class _Half:
    """One half of every streamline, run in phases of ``K`` steps: after each phase only the lanes still active go on,
    pooled across the whole seed set and padded to 65,536 lanes (or 4,096 when fewer remain), so a phase's cost
    follows the lanes that are alive rather than the longest one. The points each lane took in a phase come back as
    a slab; :meth:`scatter` writes them into the ragged output."""

    def __init__(self, n, K, chunk, phase_fn, fold_v, key, static, half_id, max_steps):
        self.n, self.K, self.chunk, self.phase_fn, self.fold_v, self.key = n, K, chunk, phase_fn, fold_v, key
        self.static, self.half_id, self.max_steps = static, half_id, max_steps
        self.count = np.zeros(n, np.int32)
        self.reason = np.zeros(n, np.int8)
        self.slabs = []                               # (lane indices, slab (m, kmax, 3), count before, count here)

    def run(self, pos, d, active):
        """``pos``, ``d`` (n, 3) float32 and ``active`` (n,) bool are updated in place; returns the final ``active``."""
        f32 = jnp.float32
        t0 = 1
        while t0 < self.max_steps:
            idx = np.flatnonzero(active)
            if idx.size == 0:
                break
            lanes = self.chunk if idx.size > _PHASE_LANES_SMALL else min(_PHASE_LANES_SMALL, self.chunk)
            for s in range(0, idx.size, lanes):
                sel = idx[s:s + lanes]
                m = sel.size
                pad = lanes - m
                pos_in = jnp.asarray(np.concatenate([pos[sel], np.zeros((pad, 3), np.float32)]))
                d_in = jnp.asarray(np.concatenate([d[sel], np.zeros((pad, 3), np.float32)]))
                act_in = jnp.asarray(np.arange(lanes) < m)
                keys = self.fold_v(self.key, jnp.asarray(np.concatenate([sel, np.zeros(pad, np.int64)]), jnp.uint32))
                slab, cnt, pos_o, d_o, act_o, reason_o = self.phase_fn(
                    pos_in, d_in, act_in, keys, jnp.int32(self.half_id), jnp.int32(t0), jnp.int32(self.max_steps),
                    *self.static)
                cnt = np.array(cnt)[:m]
                kmax = int(cnt.max()) if m else 0
                slab = np.asarray(slab[:m, :kmax])
                pos[sel] = np.asarray(pos_o)[:m]
                d[sel] = np.asarray(d_o)[:m]
                active[sel] = np.asarray(act_o)[:m]
                reason_o = np.array(reason_o)[:m]
                self.reason[sel[reason_o != 0]] = reason_o[reason_o != 0]
                self.slabs.append((sel, slab, self.count[sel].copy(), cnt))
                self.count[sel] += cnt
            t0 += self.K
        self.reason[active] = STOP_MAX_STEPS
        return active

    def first_step(self, first_dir):
        """The first point after the seed per lane, or the seed's direction where no step was taken (for the
        backward half's start)."""
        out = first_dir.copy()
        for sel, slab, before, cnt in self.slabs:
            if slab.shape[1] == 0:
                continue
            first = (before == 0) & (cnt > 0)
            out[sel[first]] = slab[first, 0]
        return out

    def scatter(self, points, dest0, forward):
        """Write every slab's points into ``points``: lane ``i``'s ``j``-th point (0-based, after the seed) goes to
        ``dest0[i] + 1 + j`` for the forward half and ``dest0[i] - 1 - j`` for the backward one, ``dest0`` the seed's row."""
        for sel, slab, before, cnt in self.slabs:
            total = int(cnt.sum())
            if total == 0:
                continue
            rep = np.repeat(np.arange(sel.size), cnt)
            k = np.arange(total) - np.repeat(np.cumsum(cnt) - cnt, cnt)
            j = np.repeat(before, cnt) + k
            dest = dest0[sel[rep]] + (1 + j if forward else -1 - j)
            points[dest] = slab[rep, k]


def _default_chunk(n):
    cap = int(os.environ.get("DMIPY_TRACT_CHUNK", "65536"))
    return max(1, min(cap, 1 << max(0, int(n) - 1).bit_length()))


def track(field, seeds_mm, *, rule='probabilistic', step_mm=0.5, max_angle=30.0, max_steps=500,
          relative_threshold=0.1, min_length_mm=0.0, sphere=None, initial_directions=None, key=0, chunk=None,
          phase_steps=None):
    """Streamlines from every seed of ``seeds_mm (n, 3)`` (world millimetres) on ``field``.

    Parameters
    ----------
    field : FODField
    seeds_mm : array (n, 3)
    rule : 'probabilistic' | 'deterministic'
    step_mm : float
        The step, in millimetres, along the chosen direction.
    max_angle : float in (0, 90]
        The cone around the previous direction, degrees.
    max_steps : int
        Points per half at most, the seed included.
    relative_threshold : float in [0, 1]
        Amplitudes below this fraction of the sphere-wide maximum are not followed (dipy's ``pmf_threshold``).
    min_length_mm : float
        Streamlines shorter than this (``(n_points - 1) * step_mm``) are dropped.
    sphere : array (n_dirs, 3), optional
        The directions chosen from; default :func:`~dmipy_tract.sphere.hemisphere` with 362 directions.
    initial_directions : array (n, 3), optional
        The first direction per seed instead of drawing it from the FOD; a zero row means "no direction".
    key : int or jax key
        The randomness; an int is a seed.
    chunk : int, optional
        Lanes per device call (default the smallest power of two holding the seeds, at most
        ``DMIPY_TRACT_CHUNK`` = 65536); the result does not depend on it.
    phase_steps : int, optional
        Steps per phase between compactions of the active lanes (default ``DMIPY_TRACT_PHASE`` = 32); the result
        does not depend on it.
    """
    if not isinstance(field, FODField):
        raise TypeError("field must be an FODField")
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}, not {rule!r}")
    step_mm = float(step_mm)
    if not step_mm > 0:
        raise ValueError(f"step_mm must be positive, not {step_mm}")
    max_angle = float(max_angle)
    if not 0.0 < max_angle <= 90.0:
        raise ValueError(f"max_angle must be in (0, 90] degrees, not {max_angle}")
    max_steps = int(max_steps)
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    relative_threshold = float(relative_threshold)
    if not 0.0 <= relative_threshold <= 1.0:
        raise ValueError(f"relative_threshold must be in [0, 1], not {relative_threshold}")
    seeds = np.asarray(seeds_mm, np.float64)
    if seeds.ndim != 2 or seeds.shape[1] != 3:
        raise ValueError(f"seeds_mm must be (n, 3), not {seeds.shape}")
    n = seeds.shape[0]
    V = hemisphere() if sphere is None else np.asarray(sphere, np.float64)
    B = sh_matrix(field.order, V)
    n_dirs = V.shape[0]
    if initial_directions is not None:
        initial_directions = np.asarray(initial_directions, np.float64)
        if initial_directions.shape != (n, 3):
            raise ValueError(f"initial_directions must be ({n}, 3), not {initial_directions.shape}")
    key = jax.random.key(int(key)) if isinstance(key, (int, np.integer)) else key
    chunk = _default_chunk(n) if chunk is None else int(chunk)
    if chunk < 1:
        raise ValueError("chunk must be at least 1")
    K = int(os.environ.get("DMIPY_TRACT_PHASE", "32")) if phase_steps is None else int(phase_steps)
    if K < 1:
        raise ValueError("phase_steps must be at least 1")
    K = min(K, max_steps)

    f32 = jnp.float32
    dims = jnp.asarray(field.shape, jnp.int32)
    field_flat = jnp.asarray(field.sh.reshape(-1, field.n_coef), f32)
    mask_flat = jnp.asarray(field.mask.reshape(-1))
    inv = field.inverse_affine
    inv_lin = jnp.asarray(inv[:3, :3], f32)
    inv_off = jnp.asarray(inv[:3, 3], f32)
    Bd = jnp.asarray(B, f32)
    Vd = jnp.asarray(V, f32)
    static = (field_flat, dims, inv_lin, inv_off, mask_flat, Bd, Vd, jnp.asarray(step_mm, f32),
              jnp.asarray(np.cos(np.deg2rad(max_angle)), f32), jnp.asarray(relative_threshold, f32))
    fold_v = jax.jit(jax.vmap(jax.random.fold_in, in_axes=(None, 0)))

    # the first direction of every seed, in chunks
    seeds32 = seeds.astype(np.float32)
    first_dir = np.zeros((n, 3), np.float32)
    ok = np.zeros(n, bool)
    if initial_directions is None:
        first_fn = _compiled_first(rule, field.n_coef, n_dirs, chunk)
        for s in range(0, n, chunk):
            m = min(chunk, n - s)
            seed_pos = np.zeros((chunk, 3), np.float32)
            seed_pos[:m] = seeds32[s:s + m]
            keys = fold_v(key, jnp.asarray(np.arange(s, s + chunk), jnp.uint32))
            fd, okd = first_fn(jnp.asarray(seed_pos), keys, field_flat, dims, inv_lin, inv_off, Bd, Vd, static[-1])
            first_dir[s:s + m] = np.asarray(fd)[:m]
            ok[s:s + m] = np.asarray(okd)[:m]
    else:
        nrm = np.linalg.norm(initial_directions, axis=1)
        ok = nrm > 0
        first_dir[ok] = (initial_directions[ok] / nrm[ok, None]).astype(np.float32)

    phase_fn = {}
    def phase_for(lanes):
        if lanes not in phase_fn:
            phase_fn[lanes] = _compiled_phase(rule, field.n_coef, n_dirs, K, lanes)
        return phase_fn[lanes]

    class _Dispatch:
        def __call__(self, pos, d, act, keys, half_id, t0, ms, *st):
            return phase_for(pos.shape[0])(pos, d, act, keys, half_id, t0, ms, *st)

    halves = []
    for half_id in (0, 1):
        h = _Half(n, K, chunk, _Dispatch(), fold_v, key, static, half_id, max_steps)
        if half_id == 0:
            d0 = first_dir.copy()
        else:
            step1 = halves[0].first_step(seeds32)                  # the first forward point, or the seed itself
            d0 = step1 - seeds32
            nrm = np.linalg.norm(d0, axis=1)
            took = nrm > 0
            d0[took] = -d0[took] / nrm[took, None]
            d0[~took] = -first_dir[~took]
        h.run(seeds32.copy(), d0, ok.copy())
        halves.append(h)
    fwd, bwd = halves
    reason = np.stack([fwd.reason, bwd.reason], axis=1)
    reason[~ok] = STOP_NO_DIRECTION

    # the ragged output: backward points reversed, the seed, forward points
    n_pts = 1 + fwd.count + bwd.count
    offsets = np.concatenate([[0], np.cumsum(n_pts)]).astype(np.int64)
    points = np.empty((int(offsets[-1]), 3), np.float32)
    seed_row = offsets[:-1] + bwd.count
    points[seed_row] = seeds32
    fwd.scatter(points, seed_row, forward=True)
    bwd.scatter(points, seed_row, forward=False)
    tg = Tractogram(points, offsets, np.arange(n), reason)
    if min_length_mm > 0:
        tg = tg.select((tg.n_points - 1) * step_mm >= min_length_mm)
    return tg
