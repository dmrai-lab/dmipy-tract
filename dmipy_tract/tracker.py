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
from .tractogram import Tractogram, STOP_NONE, STOP_MASK, STOP_OUTSIDE, STOP_NO_DIRECTION, STOP_MAX_STEPS

__all__ = ['track', 'RULES', 'BACKENDS']

RULES = ('probabilistic', 'deterministic')
_HI = jax.lax.Precision.HIGHEST


def _counter(t, half):
    """The RNG counter of step ``t`` of half ``half`` (0 forward, 1 backward); counter 0 is the first direction's."""
    return 1 + 2 * t + half


_fold_v = jax.jit(jax.vmap(jax.random.fold_in, in_axes=(None, 0)))     # one key, many streamline indices


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


def _index(pmf, u):
    """The inverse-CDF index into ``pmf`` of the uniform ``u`` in ``[0, 1)``, and whether any mass exists: the number
    of CDF entries at or below ``u`` times the total, so a ``u`` on a breakpoint lands in the next entry of positive
    mass (never on a zero-mass entry)."""
    cdf = jnp.cumsum(pmf)
    total = cdf[-1]
    idx = jnp.minimum(jnp.sum(cdf <= u * total), pmf.shape[0] - 1)
    return idx, total > 0


def _sample(pmf, key):
    """:func:`_index` of the draw of ``key``."""
    return _index(pmf, jax.random.uniform(key, dtype=pmf.dtype))


def _argmax(pmf):
    """The index of the maximum of ``pmf`` and whether it is positive."""
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


@functools.partial(jax.jit, static_argnames=('prob',))
def _first(pos, lane_keys, field_flat, dims, inv_lin, inv_off, B, V, rel_thr, *, prob):
    """The first direction at each seed ``pos (lanes, 3)`` from the whole-sphere FOD: ``(dirs (lanes, 3), ok
    (lanes,))``, sampled (``prob``) or the maximum."""
    def one(p, lane_key):
        pmf = _amplitudes(field_flat, dims, inv_lin, inv_off, B, rel_thr, p)
        idx, ok = _sample(pmf, jax.random.fold_in(lane_key, 0)) if prob else _argmax(pmf)
        return jnp.where(ok, V[idx], jnp.zeros(3, V.dtype)), ok

    return jax.vmap(one)(pos, lane_keys)


def _lane_step(field_flat, dims, inv_lin, inv_off, mask_flat, B, V, step, cos_max, rel_thr, pos, d, key, prob):
    """One step of one lane: ``(new_pos, new_d, ok, in_grid, in_mask)`` of the candidate step from ``pos`` along the
    direction chosen inside the cone around ``d``."""
    pmf = _amplitudes(field_flat, dims, inv_lin, inv_off, B, rel_thr, pos)
    cone = jnp.abs(jnp.dot(V, d, precision=_HI)) >= cos_max
    idx, ok = _sample(jnp.where(cone, pmf, 0.0), key) if prob else _argmax(jnp.where(cone, pmf, 0.0))
    nd = V[idx]
    nd = jnp.where(jnp.dot(nd, d, precision=_HI) > 0, nd, -nd)
    new_pos = pos + step * nd
    in_grid, in_mask = _mask_at(mask_flat, dims, inv_lin, inv_off, new_pos)
    return new_pos, nd, ok, in_grid, in_mask


@functools.partial(jax.jit, static_argnames=('K', 'prob'))
def _phase(pos, d, active0, lane_keys, half_id, t0, max_steps, field_flat, dims, inv_lin, inv_off, mask_flat, B, V,
           step, cos_max, rel_thr, *, K, prob):
    """``K`` steps of one half for the lanes of ``pos (lanes, 3)``, which all stand at point index ``t0``: ``(slab
    (lanes, K, 3), count (lanes,), pos, d, active, reason)``; the slab holds the points each lane took this phase,
    ``count`` how many."""
    lanes = pos.shape[0]
    step_v = jax.vmap(functools.partial(_lane_step, prob=prob), in_axes=(None,) * 10 + (0, 0, 0))
    fold_v = jax.vmap(jax.random.fold_in, in_axes=(0, None))

    def cond(c):
        k = c[6]
        return jnp.any(c[3]) & (k < K) & (t0 + k < max_steps)

    def body(c):
        slab, pos, d, active, cnt, reason, k = c
        keys = fold_v(lane_keys, _counter(t0 + k, half_id))
        new_pos, nd, ok, in_grid, in_mask = step_v(field_flat, dims, inv_lin, inv_off, mask_flat, B, V, step,
                                                   cos_max, rel_thr, pos, d, keys)
        taken = active & ok & in_mask
        slab = slab.at[:, k].set(jnp.where(taken[:, None], new_pos, 0.0))
        pos = jnp.where(taken[:, None], new_pos, pos)
        d = jnp.where(taken[:, None], nd, d)
        r = jnp.where(~ok, STOP_NO_DIRECTION, jnp.where(~in_grid, STOP_OUTSIDE, STOP_MASK)).astype(jnp.int8)
        reason = jnp.where(active & ~taken, r, reason)
        return slab, pos, d, taken, cnt + taken, reason, k + 1

    init = (jnp.zeros((lanes, K, 3), jnp.float32), pos, d, active0, jnp.zeros(lanes, jnp.int32),
            jnp.zeros(lanes, jnp.int8), jnp.int32(0))
    slab, pos, d, active, cnt, reason, _ = jax.lax.while_loop(cond, body, init)
    return slab, cnt, pos, d, active, reason


_LANE_LADDER = (1, 16, 256, 4096, 65536)


# ----------------------------------------------------------------------------------------------------------------
# device-resident state of one batch of seeds (``n_rows`` lanes plus the padding row ``n_rows``), and the
# fixed-shape helpers that move lanes in and out of a phase
@functools.partial(jax.jit, static_argnames=('L',))
def _select(remaining, *, L):
    """The first ``L`` lanes still to run (the padding row where fewer remain), and ``remaining`` without them."""
    idx = jnp.nonzero(remaining, size=L, fill_value=remaining.shape[0] - 1)[0]
    return idx, remaining.at[idx].set(False)


@jax.jit
def _gather(pos, d, count, lane_keys, idx):
    """The state of the lanes ``idx``, and which of them are lanes rather than the padding row."""
    return pos[idx], d[idx], count[idx], lane_keys[idx], idx < pos.shape[0] - 1


@functools.partial(jax.jit, donate_argnums=(0, 1, 2, 3, 4))
def _update(pos, d, active, count, reason, idx, pos_o, d_o, act_o, cnt, reason_o):
    """The state with the phase's output for the lanes ``idx`` written back (in place: the state is donated)."""
    return (pos.at[idx].set(pos_o), d.at[idx].set(d_o), active.at[idx].set(act_o), count.at[idx].add(cnt),
            reason.at[idx].set(jnp.where(reason_o != 0, reason_o, reason[idx])))


@functools.partial(jax.jit, donate_argnums=0)
def _first_point(first_pt, idx, before, cnt, slab):
    """Record the first point a lane took (a slab whose ``before`` is 0), for the backward half's start."""
    took = ((before == 0) & (cnt > 0))[:, None]
    return first_pt.at[idx].set(jnp.where(took, slab[:, 0], first_pt[idx]))


@functools.partial(jax.jit, static_argnames=('forward',), donate_argnums=0)
def _scatter(points, dest0, idx, before, cnt, slab, *, forward):
    """Write one slab ``(L, K, 3)`` into the ragged ``points (P, 3)``: lane ``i``'s ``j``-th point after the seed
    goes to ``dest0[i] + 1 + j`` (forward) or ``dest0[i] - 1 - j`` (backward); unused entries go to the dump row
    ``P - 1``."""
    k = jnp.arange(slab.shape[1], dtype=jnp.int32)
    j = before[:, None] + k[None, :]
    dest = dest0[idx][:, None] + (1 + j if forward else -(1 + j))
    dest = jnp.where(k[None, :] < cnt[:, None], dest, points.shape[0] - 1)
    return points.at[dest].set(slab)


@jax.jit
def _layout(real, cf, cb):
    """Rows of the ragged output: ``n_pts`` per lane (0 for the padding row), exclusive offsets, and the seed's
    row."""
    n_pts = jnp.where(real, 1 + cf + cb, 0)
    offsets = jnp.cumsum(n_pts) - n_pts
    return n_pts, offsets, offsets + cb


@functools.partial(jax.jit, donate_argnums=0)
def _seed_rows(points, dest0, real, seeds):
    """Write every seed into its row of the ragged ``points``; the padding row's goes to the dump row."""
    return points.at[jnp.where(real, dest0, points.shape[0] - 1)].set(seeds)


class _Batch:
    """One batch of ``n_rows`` seed lanes (plus a padding row) on the device, tracked half by half in phases of ``K``
    steps: after each phase only the lanes still active go on, pooled and padded to the smallest of 1, 16, 256,
    4,096 or 65,536 lanes that holds them, at least ``lane_floor`` (4,096, or the batch's seed count rounded up to a
    power of two when smaller) and at most ``chunk`` (:func:`_lanes_for`), so a phase costs what the live lanes cost.
    Every phase's slab stays on the device; :meth:`join` scatters them into the ragged output there."""

    def __init__(self, n_rows, n_seeds, seeds, first_dir, ok, real, key, offset, K, chunk, max_steps, prob, static):
        self.n_rows, self.K, self.chunk, self.max_steps = n_rows, K, chunk, max_steps
        self.lane_floor = min(4096, 1 << max(0, n_seeds - 1).bit_length())      # a small run stays small
        self.prob, self.static = prob, static
        self.seeds = seeds                                   # (n_rows + 1, 3) float32, device
        self.first_dir = first_dir
        self.ok = ok                                         # has a first direction
        self.real = real                                     # a seed, not padding
        self.lane_keys = _fold_v(key, jnp.arange(offset, offset + n_rows + 1, dtype=jnp.uint32))

    def half(self, half_id, d0):
        """``(slabs, count, reason)`` of the half ``half_id`` from the seeds along ``d0``."""
        n1 = self.n_rows + 1
        pos, d, active = jnp.copy(self.seeds), jnp.copy(d0), jnp.copy(self.ok)       # _update donates them
        count = jnp.zeros(n1, jnp.int32)
        reason = jnp.zeros(n1, jnp.int8)
        slabs = []
        t0 = 1
        while t0 < self.max_steps:
            s = int(jnp.sum(active))
            if s == 0:
                break
            L = _lanes_for(s, self.chunk, self.lane_floor)
            remaining = active
            for _ in range((s + L - 1) // L):
                idx, remaining = _select(remaining, L=L)
                pos_c, d_c, before, keys, act_c = _gather(pos, d, count, self.lane_keys, idx)
                slab, cnt, pos_o, d_o, act_o, reason_o = _phase(pos_c, d_c, act_c, keys, jnp.int32(half_id),
                                                                 jnp.int32(t0), jnp.int32(self.max_steps),
                                                                 *self.static, K=self.K, prob=self.prob)
                pos, d, active, count, reason = _update(pos, d, active, count, reason, idx, pos_o, d_o, act_o, cnt,
                                                        reason_o)
                slabs.append((t0, idx, before, cnt, slab))
            t0 += self.K
        reason = jnp.where(active, jnp.int8(STOP_MAX_STEPS), reason)
        return slabs, count, reason

    def first_point(self, slabs):
        """Each lane's first forward point (its seed where it took none); only the first phase's slabs hold one."""
        first_pt = jnp.copy(self.seeds)                      # _first_point donates it
        for t0, idx, before, cnt, slab in slabs:
            if t0 != 1:
                break
            first_pt = _first_point(first_pt, idx, before, cnt, slab)
        return first_pt

    def join(self, fwd, bwd):
        """``(points (total, 3), n_pts (n_rows,))`` on the host: backward points reversed, the seed, forward points."""
        (slabs_f, cf, _), (slabs_b, cb, _) = fwd, bwd
        n_pts, offsets, dest0 = _layout(self.real, cf, cb)
        total = int(jnp.sum(n_pts))
        points = _seed_rows(jnp.zeros((_points_rows(total), 3), jnp.float32), dest0, self.real, self.seeds)
        for forward, slabs in ((True, slabs_f), (False, slabs_b)):
            for _, idx, before, cnt, slab in slabs:
                points = _scatter(points, dest0, idx, before, cnt, slab, forward=forward)
        return np.asarray(points[:total]), np.asarray(n_pts)[:self.n_rows]


_STATE_ROWS = (4096, 65536, 1 << 20)


def _lanes_for(s, chunk, floor):
    """The phase's lane count for ``s`` active lanes: the smallest of 1, 16, 256, 4096, 65536 that holds them, at
    least ``floor`` and at most ``chunk`` (more lanes than that run in several calls). Few sizes, so few compiles of
    the phase kernel; the floor keeps a large run to two sizes."""
    for L in _LANE_LADDER:
        if s <= L and L >= floor:
            return min(L, chunk)
    return min(_LANE_LADDER[-1], chunk)


def _state_rows(n):
    """The smallest of the three state sizes holding ``n`` lanes: the fixed-shape helpers compile for at most three
    row counts in a session (each with the lane counts of the ladder)."""
    for r in _STATE_ROWS:
        if n <= r:
            return r
    return _STATE_ROWS[-1]


def _points_rows(total):
    """The ragged output's device size: a power of four from 2**16, plus the dump row."""
    P = 1 << 16
    while P < total + 1:
        P <<= 2
    return P


BACKENDS = ('jax', 'torch')


def track(field, seeds_mm, *, rule='probabilistic', step_mm=0.5, max_angle=30.0, max_steps=500,
          relative_threshold=0.1, min_length_mm=0.0, sphere=None, initial_directions=None, key=0, chunk=None,
          phase_steps=None, batch=None, backend='jax', device=None):
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
        Lanes per phase kernel call (default ``DMIPY_TRACT_CHUNK`` = 65536); the result does not depend on it.
    phase_steps : int, optional
        JAX backend only. Steps per phase between compactions of the active lanes (default ``DMIPY_TRACT_PHASE``
        = 32); the result does not depend on it.
    batch : int, optional
        JAX backend only. Seeds per device-resident batch (default ``DMIPY_TRACT_BATCH`` = 2**20); bounds the device
        memory (the phase slabs of a batch stay on the device until its streamlines are joined); the result does not
        depend on it. A batch is at most the state size that holds all ``n`` seeds (4,096, 65,536 or 2**20
        rows), so a larger value runs as that size.
    backend : 'jax' | 'torch'
        The kernel (dmipy-tract#4): the JAX one (the reference), or :mod:`dmipy_tract._torch` for hosts that run
        PyTorch only. The conventions are the same; the probabilistic draws come from each backend's own
        counter-based stream, so the two give different, equally valid tractograms, each independent of ``chunk``.
    device : optional
        Torch backend only. The torch device (the current CUDA device when one exists, else the CPU).

    Every argument is validated before the backend runs; an argument of the other backend is refused by name.

    Returns
    -------
    Tractogram
        One streamline per seed in seed order (``seed_index`` is ``arange(n)``), less the ones ``min_length_mm``
        drops; empty for zero seeds.
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
    if initial_directions is not None:
        initial_directions = np.asarray(initial_directions, np.float64)
        if initial_directions.shape != (n, 3):
            raise ValueError(f"initial_directions must be ({n}, 3), not {initial_directions.shape}")
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, not {backend!r}")
    chunk = int(os.environ.get("DMIPY_TRACT_CHUNK", "65536")) if chunk is None else int(chunk)
    if chunk < 1:
        raise ValueError("chunk must be at least 1")
    if backend == 'torch':
        for name, value in (('phase_steps', phase_steps), ('batch', batch)):
            if value is not None:
                raise ValueError(f"{name} is an argument of the JAX backend; the torch backend does not take it")
        if not isinstance(key, (int, np.integer)):
            raise ValueError("the torch backend takes an int key")
    else:
        if device is not None:
            raise ValueError("device is an argument of the torch backend; the JAX backend runs on JAX's default "
                             "device")
        K = int(os.environ.get("DMIPY_TRACT_PHASE", "32")) if phase_steps is None else int(phase_steps)
        if K < 1:
            raise ValueError("phase_steps must be at least 1")
        batch = int(os.environ.get("DMIPY_TRACT_BATCH", str(1 << 20))) if batch is None else int(batch)
        if batch < 1:
            raise ValueError("batch must be at least 1")
    if n == 0:
        return Tractogram(np.zeros((0, 3), np.float32), np.zeros(1, np.int64), np.zeros(0, np.int64),
                          np.zeros((0, 2), np.int8))

    seeds32 = seeds.astype(np.float32)
    first = None
    if initial_directions is not None:
        nrm = np.linalg.norm(initial_directions, axis=1)
        ok = nrm > 0
        first_dir = np.zeros((n, 3), np.float32)
        first_dir[ok] = (initial_directions[ok] / nrm[ok, None]).astype(np.float32)
        first = (first_dir, ok)
    settings = dict(rule=rule, step_mm=step_mm, cos_max=float(np.cos(np.deg2rad(max_angle))),
                    relative_threshold=relative_threshold, max_steps=max_steps, chunk=chunk)
    if backend == 'torch':
        from ._torch import track_torch
        points, n_pts, reasons = track_torch(field, seeds32, first, V, B, key=int(key), device=device, **settings)
    else:
        key = jax.random.key(int(key)) if isinstance(key, (int, np.integer)) else key
        points, n_pts, reasons = _track_jax(field, seeds32, first, V, B, key=key, K=min(K, max_steps), batch=batch,
                                            **settings)
    reasons = np.where(reasons == STOP_NONE, STOP_NO_DIRECTION, reasons).astype(np.int8)   # seeds without a direction
    tg = Tractogram(points, np.concatenate([[0], np.cumsum(n_pts)]).astype(np.int64), np.arange(n), reasons)
    if min_length_mm > 0:
        tg = tg.select((tg.n_points - 1) * step_mm >= min_length_mm)
    return tg


def _track_jax(field, seeds32, first, V, B, *, rule, step_mm, cos_max, relative_threshold, max_steps, chunk, key, K,
               batch):
    """The JAX kernel on validated arguments: ``(points (total, 3) float32, n_pts (n,), reasons (n, 2))``, the
    reasons :data:`~dmipy_tract.tractogram.STOP_NONE` for a seed without a first direction. ``first`` is
    ``(first_dir (n, 3), ok (n,))`` or None to draw the first directions from the FOD."""
    n = seeds32.shape[0]
    prob = rule == 'probabilistic'
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
              jnp.asarray(cos_max, f32), jnp.asarray(relative_threshold, f32))

    if first is None:                                   # the first direction of every seed, in chunks
        first_dir = np.zeros((n, 3), np.float32)
        ok = np.zeros(n, bool)
        first_chunk = min(chunk, 1 << max(0, n - 1).bit_length())
        for s in range(0, n, first_chunk):
            m = min(first_chunk, n - s)
            seed_pos = np.zeros((first_chunk, 3), np.float32)
            seed_pos[:m] = seeds32[s:s + m]
            keys = _fold_v(key, jnp.asarray(np.arange(s, s + first_chunk), jnp.uint32))
            fd, okd = _first(jnp.asarray(seed_pos), keys, field_flat, dims, inv_lin, inv_off, Bd, Vd, static[-1],
                             prob=prob)
            first_dir[s:s + m] = np.asarray(fd)[:m]
            ok[s:s + m] = np.asarray(okd)[:m]
    else:
        first_dir, ok = first

    parts_points, parts_npts, parts_reasons = [], [], []
    batch = min(batch, _state_rows(n))                                  # the state's row count: 4096, 65536 or 2**20
    for b0 in range(0, n, batch):
        m = min(batch, n - b0)
        seeds_b = np.zeros((batch + 1, 3), np.float32)
        seeds_b[:m] = seeds32[b0:b0 + m]
        fd_b = np.zeros((batch + 1, 3), np.float32)
        fd_b[:m] = first_dir[b0:b0 + m]
        ok_b = np.zeros(batch + 1, bool)
        ok_b[:m] = ok[b0:b0 + m]
        real_b = np.arange(batch + 1) < m
        bt = _Batch(batch, m, jnp.asarray(seeds_b), jnp.asarray(fd_b), jnp.asarray(ok_b), jnp.asarray(real_b), key,
                    b0, K, chunk, max_steps, prob, static)
        fwd = bt.half(0, bt.first_dir)
        step1 = bt.first_point(fwd[0]) - bt.seeds                       # the first forward step, or zero
        nrm = jnp.linalg.norm(step1, axis=1)
        took = nrm > 0
        d0 = jnp.where(took[:, None], -step1 / jnp.where(took, nrm, 1.0)[:, None], -bt.first_dir)
        bwd = bt.half(1, d0)
        pts, n_pts = bt.join(fwd, bwd)
        parts_points.append(pts)
        parts_npts.append(n_pts[:m])
        parts_reasons.append(np.stack([np.asarray(fwd[2])[:m], np.asarray(bwd[2])[:m]], axis=1))
    points = parts_points[0] if len(parts_points) == 1 else np.concatenate(parts_points)
    return points, np.concatenate(parts_npts), np.concatenate(parts_reasons)
