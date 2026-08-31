"""
Course terrain + carrot goal-planning — PURE logic (no Isaac imports)
=====================================================================
Shaped training arenas: straight / L / T corridors where the terrain changes
along the path (flat / rough / stairs up / stairs down) and the robot must
follow the lane — including a direction change at the bend — to a goal at the
far end.

Design (agreed):
  * One terrain CELL = one whole course. Column (variant seed) decides the
    SHAPE + segment order; the difficulty ROW scales HARSHNESS ONLY (stair
    height, rough noise). The centerline geometry is therefore identical
    across rows of a column — the env only needs the variant id to know the
    path.
  * Goal planning = a moving CARROT waypoint on the course centerline,
    ~lookahead m ahead of the robot's (monotonic) projection onto the path.
    The carrot clamps at the course end, so goal_tol / goal_bonus can only
    fire at the true finish. Path shape lives HERE, not in the policy — the
    policy only ever sees a nearby goal, so curves/L/T generalize for free.
  * Spawn at a RANDOM END of the course with random yaw (no "always forward"
    memorization); the goal is the opposite end.

Everything in this module is numpy/torch-only so it can be unit-tested
without Isaac. The Isaac side (env_cfg) wraps `paint_course` in a
@height_field_to_mesh function; nav_env uses `CourseRuntime` for the carrot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# ── Layout ──────────────────────────────────────────────────────────────

SEG_KINDS = ("flat", "rough", "stairs_up", "stairs_down")


def effective_cell_size(cell_size: float, horizontal_scale: float,
                        border_width: float = 0.0) -> float:
    """The cell size Isaac Lab actually hands the painter — NOT `cell_size`.

    `height_field_to_mesh` allocates `int(size/hs) + 1` pixels per axis,
    reserves `int(border_width/hs) + 1` of them as border on EACH side (note
    the +1: there is always a one-pixel border, even at border_width=0), and
    then calls the painter with `cfg.size` shrunk to the remaining span. With
    an 18 m cell at 0.1 m/px that is 17.9 m, not 18 m.

    Both the painter and the env must build their layout from this value or
    they disagree about where the course is: the terrain gets painted for one
    centreline while the carrot follows another, and the robot spawns off the
    end pad. env_cfg passes the shrunk `cfg.size` straight through, so the env
    side calls this to arrive at the same number.

    `border_width` is the SUB-TERRAIN cfg's (HfTerrainBaseCfg, default 0.0) —
    TerrainGeneratorCfg.border_width surrounds the whole grid and is not
    copied onto sub-terrains.
    """
    width_px = int(cell_size / horizontal_scale) + 1
    border_px = int(border_width / horizontal_scale) + 1
    return (width_px - 2 * border_px) * horizontal_scale


@dataclass
class CourseLayout:
    """Deterministic geometry of one course variant (cell-local coords,
    origin at the CELL CENTER, x/y in metres)."""
    variant: int
    shape: str                                  # "straight" | "L" | "T"
    route: np.ndarray                           # (P, 2) polyline vertices
    stub: np.ndarray | None                     # (2, 2) T dead-end arm or None
    seg_s: np.ndarray                           # (S+1,) arc-length breakpoints
    seg_kind: list[str]                         # (S,) kind per segment
    dense: np.ndarray = field(default=None)     # (K, 2) densified route
    dense_s: np.ndarray = field(default=None)   # (K,) arc-length per point

    @property
    def length(self) -> float:
        return float(self.dense_s[-1])


def _densify(poly: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray]:
    """Resample a polyline at ~`step` m spacing. Returns (points, arclen)."""
    pts = [poly[0]]
    s = [0.0]
    for a, b in zip(poly[:-1], poly[1:]):
        d = float(np.linalg.norm(b - a))
        n = max(1, int(math.ceil(d / step)))
        for i in range(1, n + 1):
            pts.append(a + (b - a) * (i / n))
            s.append(s[-1] + d / n)
    return np.asarray(pts, dtype=np.float32), np.asarray(s, dtype=np.float32)


def build_layout(
    variant: int,
    *,
    cell_size: float,
    margin: float = 1.5,
    seg_len: float = 4.5,
    end_pad: float = 2.0,
    bend_pad: float = 1.2,
    shapes: tuple[str, ...] = ("straight", "L", "T"),
    dense_step: float = 0.25,
    base_seed: int = 1234,
) -> CourseLayout:
    """Deterministic course layout for one variant (column). Same function is
    used by the terrain painter AND the env, so they always agree.

    Shape and segment ORDER depend only on `variant` (harshness-only rows:
    difficulty never changes geometry).
    """
    rng = np.random.default_rng(base_seed + 7919 * variant)
    h = cell_size / 2.0
    shape = shapes[variant % len(shapes)]  # even mix; order shuffled by seed below
    if len(shapes) > 1 and rng.random() < 0.5:
        shape = shapes[int(rng.integers(0, len(shapes)))]

    ya = float(rng.uniform(-cell_size / 6, cell_size / 6))
    stub = None
    if shape == "straight":
        route = np.array([[-h + margin, ya], [h - margin, ya]], dtype=np.float32)
    else:
        # entry along +x to a bend at bx, then turn +/-90deg to a y edge.
        sgn = 1.0 if rng.random() < 0.5 else -1.0
        lo, hi = -cell_size / 8, h - margin - 3.0
        bx = float(rng.uniform(lo, max(lo + 0.5, hi)))
        A = np.array([-h + margin, ya], dtype=np.float32)
        B = np.array([bx, ya], dtype=np.float32)
        C = np.array([bx, sgn * (h - margin)], dtype=np.float32)
        route = np.stack([A, B, C])
        if shape == "T":
            # walkable dead-end arm opposite the turn — a distractor lane that
            # tests carrot-following (the robot must NOT take the stub).
            D = np.array([bx, -sgn * (h - margin) * 0.6], dtype=np.float32)
            stub = np.stack([B, D])

    dense, dense_s = _densify(route, dense_step)
    total = float(dense_s[-1])

    # Segment the arc-length: flat pads at both ends and around the bend
    # (turning on stairs is brutal — ease the corner), random kinds between.
    protected: list[tuple[float, float]] = [(0.0, end_pad), (total - end_pad, total)]
    if shape in ("L", "T"):
        s_bend = float(np.linalg.norm(route[1] - route[0]))
        protected.append((s_bend - bend_pad, s_bend + bend_pad))
    protected.sort()

    breaks: list[float] = [0.0]
    kinds: list[str] = []

    def _emit(s0: float, s1: float, kind: str):
        if s1 - s0 < 0.25:
            return
        breaks.append(s1)
        kinds.append(kind)

    cursor = 0.0
    for p0, p1 in protected + [(total, total)]:
        # Random chunks between protected flats. Stairs are emitted as PAIRED
        # half-chunks of EQUAL length — up-then-down ("hill") or down-then-up
        # ("valley") — so the same step count cancels exactly and the lane
        # always returns to level 0. Both end pads are therefore at z=0
        # regardless of variant/difficulty, which pins the spawn height for
        # either traversal direction.
        while p0 - cursor > 0.25:
            ln = min(seg_len * float(rng.uniform(0.8, 1.2)), p0 - cursor)
            kind = ("flat", "rough", "hill", "valley")[int(rng.integers(0, 4))]
            if kind in ("hill", "valley") and ln >= 1.5:
                half = ln / 2.0
                first, second = (
                    ("stairs_up", "stairs_down") if kind == "hill"
                    else ("stairs_down", "stairs_up")
                )
                _emit(cursor, cursor + half, first)
                _emit(cursor + half, cursor + ln, second)
            else:
                _emit(cursor, cursor + ln, "flat" if kind in ("hill", "valley") else kind)
            cursor += ln
        if p1 > cursor:
            _emit(cursor, min(p1, total), "flat")
            cursor = min(p1, total)
        if cursor >= total:
            break

    return CourseLayout(
        variant=variant, shape=shape, route=route, stub=stub,
        seg_s=np.asarray(breaks, dtype=np.float32), seg_kind=kinds,
        dense=dense, dense_s=dense_s,
    )


# ── Painting (numpy heightfield; Isaac wrapper adds @height_field_to_mesh) ──

def paint_course(
    layout: CourseLayout,
    difficulty: float,
    *,
    cell_size: float,
    horizontal_scale: float,
    vertical_scale: float,
    lane_width: float = 3.5,
    wall_height: float = 1.0,
    stair_height_range: tuple[float, float] = (0.06, 0.28),
    rough_noise_range: tuple[float, float] = (0.02, 0.16),
    step_width: float = 0.32,
) -> np.ndarray:
    """Rasterize one course cell into an int16 heightfield (units of
    vertical_scale, shape (x_px, y_px)) — the @height_field_to_mesh contract.

    Lane pixels take the path's elevation profile (stairs accumulate along
    arc-length; rough adds noise); everything OFF-lane is a plateau at
    (max lane height + wall_height) so the robot cannot shortcut the bend
    and the walls are visible to scandots/depth.
    """
    hs, vs = float(horizontal_scale), float(vertical_scale)
    n = max(2, int(round(cell_size / hs)))
    d = float(np.clip(difficulty, 0.0, 1.0))
    stair_h = stair_height_range[0] + d * (stair_height_range[1] - stair_height_range[0])
    rough_lo = rough_noise_range[0]
    rough_hi = rough_lo + d * (rough_noise_range[1] - rough_noise_range[0])
    rng = np.random.default_rng(4242 + layout.variant)  # noise pattern per variant

    # Elevation profile h(s) along the route (metres), from the segment plan.
    ds = 0.05
    prof_s = np.arange(0.0, layout.length + ds, ds, dtype=np.float32)
    prof_h = np.zeros_like(prof_s)
    level = 0.0
    for k, kind in enumerate(layout.seg_kind):
        s0, s1 = float(layout.seg_s[k]), float(layout.seg_s[k + 1])
        m = (prof_s >= s0) & (prof_s < s1)
        if kind in ("stairs_up", "stairs_down"):
            sgn = 1.0 if kind == "stairs_up" else -1.0
            nsteps = max(1, int((s1 - s0) / step_width))
            idx = np.minimum(((prof_s[m] - s0) / step_width).astype(int), nsteps - 1)
            prof_h[m] = level + sgn * stair_h * (idx + 1)
            level += sgn * stair_h * nsteps
        else:
            prof_h[m] = level
    prof_h[prof_s >= float(layout.seg_s[-1])] = level

    # Pixel grid (cell-local metres, origin at center).
    xs = (np.arange(n) + 0.5) * hs - cell_size / 2.0
    ys = (np.arange(n) + 0.5) * hs - cell_size / 2.0
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    px = np.stack([gx.ravel(), gy.ravel()], axis=1)          # (N, 2)

    # Distance to route (via dense points) -> lane mask + arc-length s.
    dd = px[:, None, :] - layout.dense[None, :, :]           # (N, K, 2)
    dist2 = np.einsum("nkc,nkc->nk", dd, dd)
    near = np.argmin(dist2, axis=1)
    dist = np.sqrt(dist2[np.arange(len(px)), near])
    s_px = layout.dense_s[near]
    on_lane = dist <= lane_width / 2.0

    h_px = np.interp(s_px, prof_s, prof_h)                   # lane elevation

    # Rough segments: add noise on lane pixels within those arc ranges.
    rough_mask = np.zeros(len(px), dtype=bool)
    for k, kind in enumerate(layout.seg_kind):
        if kind == "rough":
            s0, s1 = float(layout.seg_s[k]), float(layout.seg_s[k + 1])
            rough_mask |= (s_px >= s0) & (s_px < s1)
    noise = rng.uniform(rough_lo, rough_hi, size=len(px))
    h_px = np.where(on_lane & rough_mask, h_px + noise, h_px)

    # T stub: walkable flat dead-end at the junction's elevation.
    if layout.stub is not None:
        sd, _ = _densify(layout.stub, 0.25)
        dd2 = px[:, None, :] - sd[None, :, :]
        sdist = np.sqrt(np.min(np.einsum("nkc,nkc->nk", dd2, dd2), axis=1))
        stub_lane = (sdist <= lane_width / 2.0) & ~on_lane
        s_junction = float(np.linalg.norm(layout.route[1] - layout.route[0]))
        h_junction = float(np.interp(s_junction, prof_s, prof_h))
        h_px = np.where(stub_lane, h_junction, h_px)
        on_lane = on_lane | stub_lane

    wall = float(np.max(prof_h)) + wall_height
    h_px = np.where(on_lane, h_px, wall)

    # NOTE: nothing is carved at the cell centre. An earlier version flattened
    # a 0.35 m disc there to try to force Isaac's env-origin z to 0, which
    # cannot work: height_field_to_mesh takes origin_z = np.max() over a 2 m
    # box at the centre, and lowering pixels never moves a maximum while one
    # taller pixel remains in the window. Widening the disc to cover the whole
    # window would instead punch a pit into any course whose lane crosses the
    # middle. nav_env therefore ignores env_origins[:, 2] on courses and spawns
    # from the end pads, which the segment planner pins to exactly z=0.
    return np.round(h_px.reshape(n, n) / vs).astype(np.int16)


# ── Carrot runtime (torch; used by nav_env every step) ──────────────────

class CourseRuntime:
    """Vectorized carrot goal-planning over per-variant centerlines.

    All envs share the variant table (V, K, 2) (cell-local, padded to Kmax);
    per-env state is (variant, direction, monotonic path index). Direction
    reversal = index mirroring, so both traversal directions come free.
    """

    def __init__(self, layouts: list[CourseLayout], device, dense_step: float = 0.25):
        import torch
        self.torch = torch
        self.step = dense_step
        kmax = max(len(l.dense) for l in layouts)
        table = np.zeros((len(layouts), kmax, 2), dtype=np.float32)
        klen = np.zeros(len(layouts), dtype=np.int64)
        for i, l in enumerate(layouts):
            table[i, : len(l.dense)] = l.dense
            table[i, len(l.dense):] = l.dense[-1]            # pad with endpoint
            klen[i] = len(l.dense)
        self.table = torch.as_tensor(table, device=device)    # (V, K, 2)
        self.klen = torch.as_tensor(klen, device=device)      # (V,)

    def _world_point(self, variant, direction, idx, origin_xy):
        """Path point at traversal index `idx` (0 = spawn end) in world xy."""
        t = self.torch
        last = self.klen[variant] - 1
        i = t.where(direction == 0, idx.clamp(min=0), last - idx.clamp(min=0))
        i = i.clamp(min=0)
        i = t.minimum(i, last)
        pts = self.table[variant, i]                          # (B, 2)
        return origin_xy + pts

    def reset(self, variant, direction, origin_xy):
        """Returns (spawn_xy, cur_idx=0). Spawn = index 0 of the traversal."""
        t = self.torch
        zero = t.zeros_like(variant)
        return self._world_point(variant, direction, zero, origin_xy), zero

    def advance(self, variant, direction, cur_idx, origin_xy, robot_xy,
                lookahead: float, window: int = 24):
        """Monotonic projection + carrot. Searches only a FORWARD window from
        cur_idx (prevents backtracking and jumping across the T stub), then
        places the carrot `lookahead` m further along, clamped to the end.
        Returns (new_idx, carrot_xy, frac_complete)."""
        t = self.torch
        B = variant.shape[0]
        last = self.klen[variant] - 1
        offs = t.arange(window, device=variant.device)         # (W,)
        cand = (cur_idx.unsqueeze(1) + offs.unsqueeze(0))      # (B, W)
        cand = t.minimum(cand, last.unsqueeze(1))
        i_tab = t.where(direction.unsqueeze(1) == 0, cand, last.unsqueeze(1) - cand)
        pts = origin_xy.unsqueeze(1) + self.table[variant.unsqueeze(1).expand_as(i_tab), i_tab]
        d2 = ((pts - robot_xy.unsqueeze(1)) ** 2).sum(-1)      # (B, W)
        best = d2.argmin(dim=1)
        new_idx = t.maximum(cur_idx, t.gather(cand, 1, best.unsqueeze(1)).squeeze(1))
        carrot_idx = t.minimum(new_idx + int(round(lookahead / self.step)), last)
        carrot = self._world_point(variant, direction, carrot_idx, origin_xy)
        frac = new_idx.float() / last.clamp(min=1).float()
        return new_idx, carrot, frac
