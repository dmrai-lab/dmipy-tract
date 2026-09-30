"""Constructed FOD fields with known answers, and a per-streamline numpy tracker that spells out the kernel's
semantics one step at a time (the oracle for "the batched kernel does what the definition says")."""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")            # tests are about semantics; a GPU run sets this itself

import numpy as np
import pytest
import jax

from dmipy_tract import FODField, sh_matrix, hemisphere
from dmipy_tract.field import nearest_voxel
from dmipy_tract.tracker import _counter
from dmipy_tract.tractogram import STOP_MASK, STOP_OUTSIDE, STOP_NO_DIRECTION, STOP_MAX_STEPS

ORDER = 8


def delta_sh(direction, order=ORDER):
    """The SH of a unit-mass delta at ``direction`` (the FOD of one straight fibre)."""
    d = np.asarray(direction, np.float64)
    d = d / np.linalg.norm(d)
    return sh_matrix(order, d[None])[0]


def uniform_field(shape=(10, 10, 10), direction=(1.0, 0.0, 0.0), affine=None, mask=None, order=ORDER):
    sh = np.tile(delta_sh(direction, order), tuple(shape) + (1,))
    return FODField(sh, np.eye(4) if affine is None else affine, np.ones(shape, bool) if mask is None else mask)


def crossing_field(n=30, half_width=3, order=ORDER):
    """Two orthogonal bundles (along x and along y) of square section ``2 half_width`` through the centre of an
    ``n^3`` grid; the FOD is the sum of the two deltas where they cross. Labels 1..4 mark the four end caps
    (x low, x high, y low, y high, three voxels deep)."""
    sh = np.zeros((n, n, n, sh_matrix(order, np.eye(3)).shape[1]), np.float64)
    c = n // 2
    band = slice(c - half_width, c + half_width)
    dx, dy = delta_sh((1, 0, 0), order), delta_sh((0, 1, 0), order)
    sh[:, band, band] += dx
    sh[band, :, band] += dy
    mask = np.abs(sh[..., 0]) > 0
    labels = np.zeros((n, n, n), np.int32)
    labels[:3][mask[:3]] = 1
    labels[n - 3:][mask[n - 3:]] = 2
    labels[:, :3][mask[:, :3]] = 3
    labels[:, n - 3:][mask[:, n - 3:]] = 4
    crossing = np.zeros((n, n, n), bool)
    crossing[band, band, band] = True
    return FODField(sh, np.eye(4), mask), labels, crossing


def circle_field(n=40, order=ORDER):
    """FODs tangent to the circles around the z axis through the grid centre; the centre voxel column is masked out."""
    ijk = np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing='ij'), axis=-1).astype(np.float64)
    centre = np.array([(n - 1) / 2.0] * 3)
    rel = ijk - centre
    r = np.hypot(rel[..., 0], rel[..., 1])
    t = np.stack([-rel[..., 1], rel[..., 0], np.zeros_like(r)], axis=-1) / np.maximum(r, 1e-12)[..., None]
    t = t.reshape(-1, 3)
    t[r.reshape(-1) < 1e-9] = [1.0, 0.0, 0.0]
    sh = sh_matrix(order, t).reshape(n, n, n, -1)
    mask = r > 0.5
    return FODField(sh, np.eye(4), mask), centre


# ----------------------------------------------------------------------------------------------------------------
def jax_uniform(root, i, counter):
    """The kernel's draw on the JAX backend: ``uniform(fold_in(fold_in(root, i), counter))`` as float32."""
    return float(jax.random.uniform(jax.random.fold_in(jax.random.fold_in(root, i), counter), dtype=np.float32))


def reference_half(field, V, B, pos, direction, *, rule, step_mm, max_angle, max_steps, relative_threshold, lane_key,
                   half_id, uniform=None):
    """One half, one streamline, the definition step by step in float64. The RNG calls are the kernel's:
    ``uniform(counter) -> float32`` draws for this streamline (the JAX stream by default)."""
    if uniform is None:
        uniform = lambda counter: float(jax.random.uniform(jax.random.fold_in(lane_key, counter), dtype=np.float32))
    cos_max = np.cos(np.deg2rad(max_angle))
    pts = [np.asarray(pos, np.float64)]
    d = np.asarray(direction, np.float64)
    for t in range(1, max_steps):
        pmf = B @ field.interpolate(pts[-1][None])[0]
        mx = pmf.max()
        pmf = np.where((pmf < relative_threshold * mx) | (pmf <= 0), 0.0, pmf)
        cone = np.abs(V @ d) >= cos_max
        w = np.where(cone, pmf, 0.0)
        if rule == 'probabilistic':
            cdf = np.cumsum(w)
            if cdf[-1] <= 0:
                return np.array(pts), STOP_NO_DIRECTION
            u = uniform(_counter(t, half_id))
            idx = min(int(np.sum(cdf.astype(np.float32) <= np.float32(u * cdf[-1]))), V.shape[0] - 1)
        else:
            idx = int(np.argmax(w))
            if w[idx] <= 0:
                return np.array(pts), STOP_NO_DIRECTION
        nd = V[idx] if V[idx] @ d > 0 else -V[idx]
        new = pts[-1] + step_mm * nd
        if not nearest_voxel(new, field.affine, field.shape)[1]:
            return np.array(pts), STOP_OUTSIDE
        if not field.in_mask(new):
            return np.array(pts), STOP_MASK
        pts.append(new)
        d = nd
    return np.array(pts), STOP_MAX_STEPS


def reference_track(field, seeds, *, rule='deterministic', step_mm=0.5, max_angle=30.0, max_steps=500,
                    relative_threshold=0.1, sphere=None, initial_directions=None, key=0, uniform=None):
    """The whole definition, per seed, in numpy: a list of ``(points, (reason_f, reason_b))``. ``uniform(i, counter)``
    is the backend's draw for streamline ``i`` (the JAX stream by default; the torch backend's is
    ``dmipy_tract._torch.uniform``)."""
    V = hemisphere() if sphere is None else np.asarray(sphere, np.float64)
    B = sh_matrix(field.order, V)
    root = jax.random.key(key)
    if uniform is None:
        uniform = lambda i, counter: jax_uniform(root, i, counter)
    out = []
    for i, s in enumerate(np.asarray(seeds, np.float64)):
        lane_key = jax.random.fold_in(root, i)
        draw = lambda counter, i=i: uniform(i, counter)
        if initial_directions is None:
            pmf = B @ field.interpolate(s[None])[0]
            mx = pmf.max()
            pmf = np.where((pmf < relative_threshold * mx) | (pmf <= 0), 0.0, pmf)
            if rule == 'probabilistic':
                cdf = np.cumsum(pmf)
                ok = cdf[-1] > 0
                if ok:
                    u = draw(0)
                    idx = min(int(np.sum(cdf.astype(np.float32) <= np.float32(u * cdf[-1]))), V.shape[0] - 1)
            else:
                idx = int(np.argmax(pmf))
                ok = pmf[idx] > 0
            first = V[idx] if ok else np.zeros(3)
        else:
            first = np.asarray(initial_directions[i], np.float64)
            ok = np.linalg.norm(first) > 0
            if ok:
                first = first / np.linalg.norm(first)
        if not ok:
            out.append((s[None].copy(), (STOP_NO_DIRECTION, STOP_NO_DIRECTION)))
            continue
        kw = dict(rule=rule, step_mm=step_mm, max_angle=max_angle, max_steps=max_steps,
                  relative_threshold=relative_threshold, lane_key=lane_key, uniform=draw)
        F, rf = reference_half(field, V, B, s, first, half_id=0, **kw)
        back = -(F[1] - F[0]) / np.linalg.norm(F[1] - F[0]) if len(F) >= 2 else -first
        Bk, rb = reference_half(field, V, B, s, back, half_id=1, **kw)
        out.append((np.concatenate([Bk[:0:-1], F]), (rf, rb)))
    return out


@pytest.fixture
def cross():
    return crossing_field()
