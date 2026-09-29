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
def _compiled_half(rule, n_coef, n_dirs, max_steps, chunk):
    """One half of every lane of a chunk: ``(positions (chunk, max_steps, 3), n_points (chunk,), reason (chunk,))``."""
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

    def half(seed_pos, first_dir, active0, lane_keys, half_id, field_flat, dims, inv_lin, inv_off, mask_flat, B, V,
             step, cos_max, rel_thr):
        buf = jnp.zeros((chunk, max_steps, 3), jnp.float32).at[:, 0].set(seed_pos)
        n0 = active0.astype(jnp.int32)
        reason0 = jnp.zeros(chunk, jnp.int8)

        def cond(c):
            return jnp.logical_and(jnp.any(c[3]), c[6] < max_steps)

        def body(c):
            buf, pos, d, active, n, reason, t = c
            keys = fold_v(lane_keys, 1 + 2 * t + half_id)
            new_pos, nd, ok, in_grid, in_mask = lane_step_v(field_flat, dims, inv_lin, inv_off, mask_flat, B, V,
                                                              step, cos_max, rel_thr, pos, d, keys)
            taken = active & ok & in_mask
            buf = buf.at[:, t].set(jnp.where(taken[:, None], new_pos, buf[:, t]))
            pos = jnp.where(taken[:, None], new_pos, pos)
            d = jnp.where(taken[:, None], nd, d)
            r = jnp.where(~ok, STOP_NO_DIRECTION, jnp.where(~in_grid, STOP_OUTSIDE, STOP_MASK)).astype(jnp.int8)
            reason = jnp.where(active & ~taken, r, reason)
            return buf, pos, d, taken, n + taken, reason, t + 1

        init = (buf, seed_pos, first_dir, active0, n0, reason0, jnp.int32(1))
        buf, _, _, active, n, reason, _ = jax.lax.while_loop(cond, body, init)
        reason = jnp.where(active, jnp.int8(STOP_MAX_STEPS), reason)
        return buf, n, reason

    return jax.jit(half, static_argnums=(4,))


@jax.jit
def _backward_first_direction(buf_f, n_f, first_dir):
    """Against the forward half's actual first step where it took one, else against its first direction (dipy)."""
    d = buf_f[:, 1] - buf_f[:, 0]
    nrm = jnp.linalg.norm(d, axis=1)
    use = (n_f >= 2) & (nrm > 0)
    return jnp.where(use[:, None], -d / jnp.where(use, nrm, 1.0)[:, None], -first_dir)


# ----------------------------------------------------------------------------------------------------------------
def _join(buf_f, n_f, buf_b, n_b):
    """The backward half reversed (without its seed) followed by the forward half, as ragged points and offsets."""
    nb = n_b - 1
    L = nb + n_f
    offsets = np.concatenate([[0], np.cumsum(L)])
    total = int(offsets[-1])
    i = np.repeat(np.arange(L.shape[0]), L)
    k = np.arange(total) - offsets[i]
    from_b = k < nb[i]
    jb = np.clip(nb[i] - k, 0, buf_b.shape[1] - 1)
    jf = np.clip(k - nb[i], 0, buf_f.shape[1] - 1)
    pts = np.where(from_b[:, None], buf_b[i, jb], buf_f[i, jf])
    return pts, offsets


def _default_chunk(n):
    cap = int(os.environ.get("DMIPY_TRACT_CHUNK", "65536"))
    return max(1, min(cap, 1 << max(0, int(n) - 1).bit_length()))


def track(field, seeds_mm, *, rule='probabilistic', step_mm=0.5, max_angle=30.0, max_steps=500,
          relative_threshold=0.1, min_length_mm=0.0, sphere=None, initial_directions=None, key=0, chunk=None):
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

    f32 = jnp.float32
    dims = jnp.asarray(field.shape, jnp.int32)
    field_flat = jnp.asarray(field.sh.reshape(-1, field.n_coef), f32)
    mask_flat = jnp.asarray(field.mask.reshape(-1))
    inv = field.inverse_affine
    inv_lin = jnp.asarray(inv[:3, :3], f32)
    inv_off = jnp.asarray(inv[:3, 3], f32)
    Bd = jnp.asarray(B, f32)
    Vd = jnp.asarray(V, f32)
    step = jnp.asarray(step_mm, f32)
    cos_max = jnp.asarray(np.cos(np.deg2rad(max_angle)), f32)
    rel_thr = jnp.asarray(relative_threshold, f32)
    first_fn = _compiled_first(rule, field.n_coef, n_dirs, chunk)
    half_fn = _compiled_half(rule, field.n_coef, n_dirs, max_steps, chunk)
    fold_v = jax.jit(jax.vmap(jax.random.fold_in, in_axes=(None, 0)))

    parts = []
    for s in range(0, n, chunk):
        m = min(chunk, n - s)
        lane_index = np.arange(s, s + chunk)
        valid = np.arange(chunk) < m
        seed_pos = np.zeros((chunk, 3), np.float32)
        seed_pos[:m] = seeds[s:s + m]
        seed_pos = jnp.asarray(seed_pos)
        lane_keys = fold_v(key, jnp.asarray(lane_index, jnp.uint32))
        if initial_directions is None:
            first_dir, ok = first_fn(seed_pos, lane_keys, field_flat, dims, inv_lin, inv_off, Bd, Vd, rel_thr)
        else:
            d0 = np.zeros((chunk, 3), np.float64)
            d0[:m] = initial_directions[s:s + m]
            nrm = np.linalg.norm(d0, axis=1)
            okn = nrm > 0
            d0[okn] /= nrm[okn, None]
            first_dir, ok = jnp.asarray(d0, f32), jnp.asarray(okn)
        active0 = jnp.asarray(valid) & ok
        buf_f, n_f, r_f = half_fn(seed_pos, first_dir, active0, lane_keys, 0, field_flat, dims, inv_lin, inv_off,
                                  mask_flat, Bd, Vd, step, cos_max, rel_thr)
        first_b = _backward_first_direction(buf_f, n_f, first_dir)
        buf_b, n_b, r_b = half_fn(seed_pos, first_b, active0, lane_keys, 1, field_flat, dims, inv_lin, inv_off,
                                  mask_flat, Bd, Vd, step, cos_max, rel_thr)
        no_first = np.asarray(~ok)[:m]
        n_f = np.array(n_f)[:m]
        n_b = np.array(n_b)[:m]
        n_f[no_first] = 1
        n_b[no_first] = 1
        reason = np.stack([np.array(r_f)[:m], np.array(r_b)[:m]], axis=1)
        reason[no_first] = STOP_NO_DIRECTION
        # only the prefix of each buffer that any lane filled leaves the device
        buf_f = np.asarray(buf_f[:m, :int(n_f.max())])
        buf_b = np.asarray(buf_b[:m, :int(n_b.max())])
        pts, offsets = _join(buf_f, n_f, buf_b, n_b)
        part = Tractogram(pts, offsets, lane_index[:m], reason)
        if min_length_mm > 0:
            part = part.select((part.n_points - 1) * step_mm >= min_length_mm)
        parts.append(part)
    return Tractogram.concatenate(parts)
