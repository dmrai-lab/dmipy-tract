# RK4 Implementation and the Sign Convention

## The RK4 Algorithm for Streamline Propagation

4th-order Runge-Kutta integration approximates the integral of a direction field by evaluating it at four intermediate positions per step and combining the results with a weighted average. For streamline tractography, the "direction field" is `peaks_at(xyz)` — the principal orientation peak at each world-space position.

Given the current position `pos` and a step size `h`, one RK4 step computes:

```
k1 = direction at pos
k2 = direction at pos + h/2 * k1     (half-step using k1)
k3 = direction at pos + h/2 * k2     (half-step using k2)
k4 = direction at pos + h * k3       (full step using k3)

step_dir = (k1 + 2*k2 + 2*k3 + k4) / 6
pos_new  = pos + h * step_dir
```

Each k-vector is a unit orientation vector ("which way does the fibre run here?"), returned by `_get_direction(pos, prev_dir)`. The `prev_dir` argument is the reference direction from the previous sub-step; `_get_direction` flips the retrieved peak if it points against `prev_dir`, ensuring all k-vectors are consistently signed relative to the propagation direction.

The local truncation error is O(h^5), compared to O(h^2) for Euler (RK1). In practice, for a 0.5 mm step size through curved white matter pathways (e.g. the cingulum or corona radiata), RK4 substantially reduces drift relative to Euler integration, producing smoother streamlines that follow the true fibre geometry more accurately.

---

## Why `sign` Is Needed: Bidirectional Propagation

Tractography from a seed point must grow the streamline in both directions along the fibre axis, because the fibre passes through the seed going in both directions. `DeterministicTracker._propagate_single` calls `_propagate_direction` twice: once with `forward=True` (sign = +1.0) and once with `forward=False` (sign = -1.0), then concatenates the two half-streamlines around the seed.

The fundamental challenge is that the orientation field `peaks_at(xyz)` returns a direction vector with an arbitrary sign: a fibre running left-right could return `[1, 0, 0]` or `[-1, 0, 0]` depending on the local fitting result. The polarity is disambiguated by the flip logic in `_get_direction`: if `dot(peak, prev_dir) < 0`, the peak is negated to align with the previous direction. This keeps all k-vectors consistently aligned along the propagation direction.

To propagate backward, the tracker must move in the opposite physical direction while still using the same polarity-alignment logic. The `sign` variable encodes this: it multiplies the position offsets to move toward the correct sampling location, while the k-vectors themselves remain unsigned.

---

## The Original Bug

An early prototype applied `sign` directly to the k-vectors:

```python
# WRONG — do not do this
k2 = self._get_direction(
    pos + 0.5 * self.step_size * (sign * k1),
    sign * k1                                    # <-- sign applied to reference direction
)
k3 = self._get_direction(
    pos + 0.5 * self.step_size * (sign * k2),
    sign * k2
)
k4 = self._get_direction(
    pos + self.step_size * (sign * k3),
    sign * k3
)
```

Or equivalently:

```python
# Also wrong — same problem, different syntax
k2 = sign * self._get_direction(pos + 0.5 * self.step_size * k1, k1)
```

**What went wrong:** When `sign = -1.0` (backward half), passing `sign * k1` as the reference direction to `_get_direction` presented a negated vector as the polarity anchor. The flip check `if dot(peak, prev_dir) < 0: peak = -peak` then fired on the wrong hemisphere: valid peaks that were correctly aligned with the fibre got flipped, and peaks that were actually anti-aligned were left unchanged. The resulting k-vectors were oriented inconsistently with each other, and their weighted sum `(k1 + 2*k2 + 2*k3 + k4) / 6` had a norm close to zero because the four contributions partially cancelled. The curvature constraint `dot(step_dir, direction) < cos_max_angle` then immediately terminated the backward half, or the norm test `if norm < 1e-8: break` did. Either way, the backward half produced at most one or two points and was discarded by the minimum-length filter.

**The visible symptom:** All streamlines appeared to propagate in only one direction from each seed. The tractogram had roughly half the expected streamline length and, for many seeds, no streamline at all because the forward-only result was too short to pass the minimum-length threshold.

---

## The Fix: Sign Applied Only to Position Offsets

The correct implementation separates the two concerns:

- **k-vectors** are always unsigned unit directions, computed by `_get_direction` with an unsigned reference. They encode fibre orientation, not propagation direction.
- **sign** is applied only when computing where to sample the field next (position offsets in sub-steps) and when advancing the position at the end of the step.

```python
# CORRECT
sign = 1.0 if forward else -1.0

k1 = self._get_direction(pos, direction)
k2 = self._get_direction(pos + sign * 0.5 * self.step_size * k1, k1)
#                              ^^^^                                ^^^
#                    sign scales the offset              k1 passed unsigned
k3 = self._get_direction(pos + sign * 0.5 * self.step_size * k2, k2)
k4 = self._get_direction(pos + sign * self.step_size * k3, k3)

step_dir = (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
norm = np.linalg.norm(step_dir)
step_dir /= norm

pos = pos + sign * self.step_size * step_dir
#           ^^^^
#     sign controls which physical direction we advance
```

With this convention:
- `_get_direction` always receives unsigned reference directions; its flip logic operates correctly.
- All four k-vectors are consistently oriented along the fibre axis (aligned with `direction`).
- Their weighted average `step_dir` has norm close to 1.0 and points in the correct fibre direction.
- Multiplying by `sign` in the final position update moves the tracker forward or backward as required.

---

## Summary: Correct vs. Incorrect Patterns

```python
# -----------------------------------------------------------------------
# INCORRECT — sign folded into k-vector reference argument
# -----------------------------------------------------------------------
k2 = self._get_direction(pos + sign * 0.5 * h * k1, sign * k1)  # BAD
# _get_direction sees a negated reference; flip logic fires on wrong side

# -----------------------------------------------------------------------
# INCORRECT — sign applied as a multiplier on the returned k-vector
# -----------------------------------------------------------------------
k2 = sign * self._get_direction(pos + 0.5 * h * k1, k1)          # BAD
# k2 is now negated; passed as reference to k3, breaking k3 alignment

# -----------------------------------------------------------------------
# CORRECT — sign in position offset only; k-vectors always unsigned
# -----------------------------------------------------------------------
k2 = self._get_direction(pos + sign * 0.5 * h * k1, k1)          # GOOD
# position offset moves toward backward sub-step location
# k1 (unsigned) anchors polarity for k2 correctly
```

The invariant to remember: **k-vectors are orientation, `sign` is direction.** Never multiply them together.
