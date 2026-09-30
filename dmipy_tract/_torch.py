"""The tracker's kernel in PyTorch (dmipy-tract#4): the same conventions as :mod:`dmipy_tract.tracker`, eager, for
hosts that run PyTorch only (Hugging Face's shared GPU pool, where the device exists only inside a call).

One step of one half for every active lane at once, the lanes in chunks: the FOD interpolated at the position
(clamped trilinear, zero outside ``[-0.5, N - 0.5]``), evaluated on the sphere, amplitudes below
``relative_threshold`` of the sphere-wide maximum and non-positive ones set to zero, the cone
``|cos| >= cos(max_angle)`` around the previous direction, the inverse-CDF draw (or the maximum), the sign by
``dot > 0``, and the landing point's nearest voxel tested against the grid and the mask -- a failing step is not
taken and ends the half with its reason. Lanes stop in lockstep, so the point a lane takes at loop step ``t`` is its
``t``-th after the seed, and the ragged output is a scatter per step.

Randomness is counter-based and shared bit for bit between the CPU and CUDA and with numpy
(:func:`uniform`): the draw of streamline ``i`` at counter ``c`` (``0`` for the first direction, ``1 + 2 t + h``
at step ``t`` of half ``h``) is the splitmix64 finaliser of ``(key, i, c)`` turned into a float32 in ``[0, 1)`` from
its top 24 bits; a lane's draw depends on its streamline index alone, never on the chunk it sits in. The torch
stream is not JAX's ``fold_in`` stream, so the two backends give different, equally valid probabilistic tractograms;
the deterministic rule has no draw and the two backends agree to float32 arithmetic.

TF32 is off for the call (``torch.backends.cuda.matmul.allow_tf32``): a float32 matmul on CUDA is TF32 by default,
which moves amplitudes at 1e-3 and flips near-tie choices.
"""
import contextlib

import numpy as np

from .tracker import _counter
from .tractogram import Tractogram, STOP_MASK, STOP_OUTSIDE, STOP_NO_DIRECTION, STOP_MAX_STEPS

_PHI = 0x9E3779B97F4A7C15
_M1 = 0xBF58476D1CE4E5B9
_M2 = 0x94D049BB133111EB
_MASK = (1 << 64) - 1


def _signed(c):
    """A uint64 constant as the int64 torch stores (two's complement)."""
    return c - (1 << 64) if c >= (1 << 63) else c


def uniform(key, index, counter):
    """The float32 uniform in ``[0, 1)`` of streamline ``index`` at ``counter`` under ``key``, in numpy (uint64
    arithmetic): ``mix(mix(key * phi + index) + counter) >> 40`` over ``2^24``, ``mix`` the splitmix64 finaliser.
    ``index`` may be an array."""
    with np.errstate(over="ignore"):
        i = np.asarray(index, np.uint64)
        z = np.uint64(int(key) & _MASK) * np.uint64(_PHI) + i
        z = _mix_np(z) + np.uint64(int(counter) & _MASK)
        z = _mix_np(z)
    return ((z >> np.uint64(40)).astype(np.float64) / float(1 << 24)).astype(np.float32)


def _mix_np(z):
    z = (z ^ (z >> np.uint64(30))) * np.uint64(_M1)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(_M2)
    return z ^ (z >> np.uint64(31))


def _lsr(z, k):
    """Logical right shift of int64 by ``k`` (torch's ``>>`` is arithmetic)."""
    return (z >> k) & ((1 << (64 - k)) - 1)


def _mix_t(z):
    z = (z ^ _lsr(z, 30)) * _signed(_M1)
    z = (z ^ _lsr(z, 27)) * _signed(_M2)
    return z ^ _lsr(z, 31)


def uniform_torch(key, index, counter):
    """:func:`uniform` on torch int64 tensors (wrapping arithmetic), the same bits on every device."""
    import torch
    z = torch.as_tensor(index, dtype=torch.int64, device=index.device) + _signed((int(key) & _MASK) * _PHI & _MASK)
    z = _mix_t(z) + _signed(int(counter) & _MASK)
    z = _mix_t(z)
    return _lsr(z, 40).to(torch.float32) / float(1 << 24)


@contextlib.contextmanager
def _full_precision():
    import torch
    m, c = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = m; torch.backends.cudnn.allow_tf32 = c


class _Field:
    """The field, the sphere and the settings on the device."""

    def __init__(self, field, V, B, step_mm, max_angle, relative_threshold, device):
        import torch
        f32 = torch.float32
        self.dims = torch.as_tensor(np.asarray(field.shape, np.int64), device=device)
        self.dims_f = self.dims.to(f32)
        self.field_flat = torch.as_tensor(np.ascontiguousarray(field.sh.reshape(-1, field.n_coef), np.float32), device=device)
        self.mask_flat = torch.as_tensor(np.ascontiguousarray(field.mask.reshape(-1)), device=device)
        inv = field.inverse_affine
        self.inv_lin_T = torch.as_tensor(np.ascontiguousarray(inv[:3, :3].T, np.float32), device=device)
        self.inv_off = torch.as_tensor(np.asarray(inv[:3, 3], np.float32), device=device)
        self.B_T = torch.as_tensor(np.ascontiguousarray(B.T, np.float32), device=device)      # (n_coef, n_dirs)
        self.V = torch.as_tensor(np.asarray(V, np.float32), device=device)
        self.V_T = self.V.T.contiguous()
        self.step = float(step_mm); self.cos_max = float(np.cos(np.deg2rad(max_angle))); self.rel_thr = float(relative_threshold)
        self.n_dirs = int(V.shape[0]); self.device = device
        d = self.dims
        self.strides = torch.stack([d[1] * d[2], d[2], torch.ones_like(d[2])])
        c = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]], device=device)
        self.corners = c                                                                         # (8, 3)

    def voxel(self, pos):
        return pos @ self.inv_lin_T + self.inv_off

    def amplitudes(self, pos):
        """The thresholded FOD on the sphere at world positions ``pos (a, 3)``: ``(a, n_dirs)``."""
        import torch
        v = self.voxel(pos)
        flr = torch.floor(v); rem = v - flr
        i = torch.stack([torch.clamp(flr, torch.zeros_like(flr), self.dims_f - 1), torch.clamp(flr + 1, torch.zeros_like(flr), self.dims_f - 1)], 1).long()   # (a, 2, 3)
        wgt = torch.stack([1.0 - rem, rem], 1)                                                 # (a, 2, 3)
        corner_i = torch.stack([i[:, self.corners[:, k], k] for k in range(3)], -1)             # (a, 8, 3)
        corner_w = torch.stack([wgt[:, self.corners[:, k], k] for k in range(3)], -1).prod(-1)   # (a, 8)
        flat = (corner_i * self.strides).sum(-1)                                                 # (a, 8)
        c = (self.field_flat[flat] * corner_w[:, :, None]).sum(1)                                # (a, n_coef)
        inside = ((v >= -0.5) & (v <= self.dims_f - 0.5)).all(1)
        c = torch.where(inside[:, None], c, torch.zeros_like(c))
        pmf = c @ self.B_T
        mx = pmf.max(1, keepdim=True).values
        return torch.where((pmf < self.rel_thr * mx) | (pmf <= 0), torch.zeros_like(pmf), pmf)

    def choose(self, w, u, prob):
        """The direction index per lane from the weights ``w (a, n_dirs)``: inverse CDF with ``u`` or the maximum;
        and whether any mass exists."""
        import torch
        if prob:
            cdf = torch.cumsum(w, 1); total = cdf[:, -1]
            uu = (u * total)[:, None]
            idx = torch.clamp((cdf <= uu).sum(1), max=self.n_dirs - 1)
            return idx, total > 0
        idx = torch.argmax(w, 1)
        return idx, w.gather(1, idx[:, None])[:, 0] > 0

    def mask_at(self, pos):
        import torch
        iv = torch.round(self.voxel(pos)).long()
        in_grid = ((iv >= 0) & (iv < self.dims)).all(1)
        ivc = torch.minimum(torch.clamp(iv, min=0), self.dims - 1)
        return in_grid, in_grid & self.mask_flat[(ivc * self.strides).sum(-1)]


def _half(F, pos0, d0, ok, gindex, key, half_id, prob, max_steps, chunk):
    """One half for every lane: ``(slabs [(t, idx, points)], count, reason)``; slab ``t`` holds the ``t``-th point
    after the seed of the lanes ``idx`` that took step ``t``."""
    import torch
    n = pos0.shape[0]
    pos, d = pos0.clone(), d0.clone()
    active = ok.clone(); count = torch.zeros(n, dtype=torch.int32, device=F.device); reason = torch.zeros(n, dtype=torch.int8, device=F.device)
    slabs = []
    for t in range(1, max_steps):
        idx_all = torch.nonzero(active, as_tuple=True)[0]
        if idx_all.numel() == 0:
            break
        for s in range(0, idx_all.numel(), chunk):
            c = idx_all[s:s + chunk]
            pmf = F.amplitudes(pos[c])
            dc = d[c]
            cone = (dc @ F.V_T).abs() >= F.cos_max
            w = torch.where(cone, pmf, torch.zeros_like(pmf))
            u = uniform_torch(key, gindex[c], _counter(t, half_id)) if prob else None
            idx, okc = F.choose(w, u, prob)
            nd = F.V[idx]
            nd = torch.where(((nd * dc).sum(1) > 0)[:, None], nd, -nd)
            new_pos = pos[c] + F.step * nd
            in_grid, in_mask = F.mask_at(new_pos)
            taken = okc & in_mask
            r = torch.where(~okc, STOP_NO_DIRECTION, torch.where(~in_grid, STOP_OUTSIDE, STOP_MASK)).to(torch.int8)
            pos[c[taken]] = new_pos[taken]; d[c[taken]] = nd[taken]
            reason[c[~taken]] = r[~taken]; active[c[~taken]] = False; count[c[taken]] += 1
            if taken.any():
                slabs.append((t, c[taken], new_pos[taken]))
    reason[active] = STOP_MAX_STEPS
    return slabs, count, reason


def track_torch(field, seeds, *, rule, step_mm, max_angle, max_steps, relative_threshold, min_length_mm, V, B,
                initial_directions, key, chunk, device=None):
    """The tractogram of ``seeds`` on ``field`` with the torch kernel; the arguments as :func:`dmipy_tract.track`
    validates them (``V``, ``B`` the sphere and its SH matrix, ``key`` an int)."""
    import torch
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    prob = rule == 'probabilistic'
    n = seeds.shape[0]
    with _full_precision(), torch.no_grad():
        F = _Field(field, V, B, step_mm, max_angle, relative_threshold, device)
        pos = torch.as_tensor(np.asarray(seeds, np.float32), device=device)
        gindex = torch.arange(n, dtype=torch.int64, device=device)
        # the first direction of every seed from the whole-sphere FOD
        if initial_directions is None:
            first = torch.zeros((n, 3), dtype=torch.float32, device=device); ok = torch.zeros(n, dtype=torch.bool, device=device)
            for s in range(0, n, chunk):
                c = slice(s, s + chunk)
                pmf = F.amplitudes(pos[c])
                u = uniform_torch(key, gindex[c], 0) if prob else None
                idx, okc = F.choose(pmf, u, prob)
                first[c] = torch.where(okc[:, None], F.V[idx], torch.zeros_like(F.V[idx])); ok[c] = okc
        else:
            nrm = np.linalg.norm(initial_directions, axis=1)
            ok_np = nrm > 0
            fd = np.zeros((n, 3), np.float32); fd[ok_np] = (initial_directions[ok_np] / nrm[ok_np, None]).astype(np.float32)
            first = torch.as_tensor(fd, device=device); ok = torch.as_tensor(ok_np, device=device)
        slabs_f, cf, rf = _half(F, pos, first, ok, gindex, key, 0, prob, max_steps, chunk)
        first_pt = pos.clone()
        for t, idx, pts in slabs_f:
            if t == 1:
                first_pt[idx] = pts
        step1 = first_pt - pos; nrm = torch.linalg.norm(step1, dim=1); took = nrm > 0
        d0 = torch.where(took[:, None], -step1 / torch.where(took, nrm, torch.ones_like(nrm))[:, None], -first)
        slabs_b, cb, rb = _half(F, pos, d0, ok, gindex, key, 1, prob, max_steps, chunk)
        # the ragged output: the backward points reversed, the seed, the forward points
        n_pts = 1 + cf + cb
        offsets = torch.cumsum(n_pts, 0) - n_pts
        dest0 = offsets + cb
        total = int(n_pts.sum())
        points = torch.zeros((total, 3), dtype=torch.float32, device=device)
        points[dest0] = pos
        for t, idx, pts in slabs_f:
            points[dest0[idx] + t] = pts
        for t, idx, pts in slabs_b:
            points[dest0[idx] - t] = pts
        reason = np.full((n, 2), STOP_NO_DIRECTION, np.int8)
        ok_np = ok.cpu().numpy()
        reason[ok_np] = np.stack([rf.cpu().numpy()[ok_np], rb.cpu().numpy()[ok_np]], 1)
        n_pts_np = n_pts.cpu().numpy().astype(np.int64)
    off = np.concatenate([[0], np.cumsum(n_pts_np)]).astype(np.int64)
    tg = Tractogram(points.cpu().numpy(), off, np.arange(n), reason)
    if min_length_mm > 0:
        tg = tg.select((tg.n_points - 1) * step_mm >= min_length_mm)
    return tg
