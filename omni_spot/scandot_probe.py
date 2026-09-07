"""
Scandot Geometry Probe — pure numpy, no Isaac Lab
==================================================
Answers two questions that decide `ScandotsCfg.forward_offset` WITHOUT
needing a GPU, Isaac Lab, or a training run:

  1. Where do the 187 scandot grid points actually sit relative to the base?
  2. Which of them can the student's depth camera physically see?

(2) is the one that matters. The teacher is distilled into a student whose
only exteroception is one forward depth camera. Any scandot the camera can
never see is teacher information the student cannot recover — wasted encoder
capacity at best, a distillation floor at worst. Turning that into a scalar
(coverage %) converts a design argument into a measurement.

Frames
------
* **yaw frame** — world-horizontal, yaw-aligned to the base, origin at the
  base body origin. The RayCaster uses `ray_alignment="yaw"`
  (`attach_yaw_only=True`), so the scandot grid lives HERE: it never pitches
  or rolls with the body.
* **body frame** — rigidly attached to the base, so it DOES pitch/roll. The
  camera is mounted in this frame, which is why body pitch changes coverage
  while leaving the grid untouched.

Everything is float64 numpy; there is no torch dependency so this imports in
any environment that can import `omni_spot.configs`.
"""

from __future__ import annotations

import math

import numpy as np

from .configs.base import CameraMountCfg, CameraRigCfg, ExperimentCfg, ScandotsCfg


# ════════════════════════════════════════════════════════════════════════════
# Grid construction
# ════════════════════════════════════════════════════════════════════════════

def grid_offsets(
    sc: ScandotsCfg, order: str = "ij"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scandot grid XY offsets in the yaw frame, in RayCaster flatten order.

    Mirrors Isaac Lab's ``GridPatternCfg``: a regular lattice spanning
    ``size`` at ``resolution`` spacing, shifted forward by
    ``forward_offset``.

    Args:
        sc: the ScandotsCfg being probed.
        order: flatten convention of the underlying meshgrid. ``"ij"`` (the
            Isaac Lab default) gives index = ix * grid_y + iy; ``"xy"`` gives
            index = iy * grid_x + ix. Use `infer_grid_order` against a live
            sensor rather than trusting this default.

    Returns:
        (xy, ix, iy) — xy is (N, 2) in metres, ix/iy are the integer grid
        coordinates of each flattened point (ix along +X/forward,
        iy along +Y/left).
    """
    size_x, size_y = sc.size
    x = np.linspace(-size_x / 2.0, size_x / 2.0, sc.grid_x) + sc.forward_offset
    y = np.linspace(-size_y / 2.0, size_y / 2.0, sc.grid_y)
    if order == "ij":
        gx, gy = np.meshgrid(x, y, indexing="ij")
        ixg, iyg = np.meshgrid(
            np.arange(sc.grid_x), np.arange(sc.grid_y), indexing="ij"
        )
    elif order == "xy":
        gx, gy = np.meshgrid(x, y, indexing="xy")
        ixg, iyg = np.meshgrid(
            np.arange(sc.grid_x), np.arange(sc.grid_y), indexing="xy"
        )
    else:
        raise ValueError(f"order must be 'ij' or 'xy', got {order!r}")
    xy = np.stack([gx.reshape(-1), gy.reshape(-1)], axis=-1)
    return xy, ixg.reshape(-1), iyg.reshape(-1)


def infer_grid_order(local_xy: np.ndarray, sc: ScandotsCfg) -> str:
    """Determine the real flatten order from LIVE sensor hit positions.

    ``local_xy`` is (N, 2): the sensor's ray hit points expressed in the yaw
    frame (world hit XY minus base XY, de-rotated by base yaw). Rays are cast
    straight down, so their XY *is* the grid XY.

    Returns "ij" or "xy". Raises if neither matches — that means the pattern
    is not the lattice this repo assumes, which would silently scramble every
    heatmap and every spatial conclusion drawn from one.
    """
    best, best_err = None, float("inf")
    for order in ("ij", "xy"):
        ref, _, _ = grid_offsets(sc, order=order)
        if ref.shape != local_xy.shape:
            continue
        err = float(np.abs(ref - local_xy).max())
        if err < best_err:
            best, best_err = order, err
    if best is None or best_err > 0.5 * sc.spacing:
        raise RuntimeError(
            f"Ray hit XY does not match either 'ij' or 'xy' flatten order of a "
            f"{sc.grid_x}x{sc.grid_y} @ {sc.spacing} m grid "
            f"(best mismatch {best_err:.4f} m). The heatmap reshape would be "
            f"wrong — inspect the RayCaster pattern before trusting any "
            f"spatial diagnosis."
        )
    return best


def to_heatmap(
    heights: np.ndarray, sc: ScandotsCfg, order: str = "ij"
) -> np.ndarray:
    """Flat (N,) scandot vector -> (grid_x, grid_y) array.

    Row index is ix (0 = rearmost, grid_x-1 = frontmost); column index is iy
    (0 = rightmost, grid_y-1 = leftmost), matching the yaw frame's +X forward
    / +Y left convention.
    """
    n = sc.n_points
    if heights.shape[-1] != n:
        raise ValueError(
            f"expected {n} scandots ({sc.grid_x}x{sc.grid_y}), "
            f"got {heights.shape[-1]}"
        )
    out = np.empty((sc.grid_x, sc.grid_y), dtype=heights.dtype)
    _, ix, iy = grid_offsets(sc, order=order)
    out[ix, iy] = heights
    return out


# ════════════════════════════════════════════════════════════════════════════
# Camera geometry
# ════════════════════════════════════════════════════════════════════════════

def quat_to_matrix(q: tuple[float, float, float, float]) -> np.ndarray:
    """(w, x, y, z) unit quaternion -> 3x3 rotation matrix."""
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        raise ValueError(f"degenerate quaternion {q}")
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def half_fov(cam: CameraRigCfg) -> tuple[float, float]:
    """(horizontal, vertical) half field-of-view in radians.

    `_build_cameras` sets ``horizontal_aperture = 2*tan(h_fov/2)`` at
    ``focal_length = 1.0``; the vertical aperture follows from square pixels,
    i.e. scaled by the image aspect ratio.
    """
    h_half = math.radians(cam.h_fov_deg) / 2.0
    v_half = math.atan(math.tan(h_half) * cam.height / cam.width)
    return h_half, v_half


def body_to_cam(mount: CameraMountCfg) -> np.ndarray:
    """Rotation mapping CAMERA OPTICAL frame vectors into the BODY frame.

    With the "ros" convention the optical frame is +Z forward, +X right,
    +Y down; `mount.rot` is exactly that body<-camera rotation.
    """
    if mount.convention != "ros":
        raise NotImplementedError(
            f"only the 'ros' camera convention is modelled here, got "
            f"{mount.convention!r}. Add the conversion before trusting the "
            f"coverage numbers."
        )
    return quat_to_matrix(mount.rot)


def _rpy_body_to_yaw(pitch_rad: float, roll_rad: float) -> np.ndarray:
    """Rotation mapping BODY frame vectors into the YAW frame.

    Positive ``pitch_rad`` is nose DOWN (it tilts the camera toward the
    ground, increasing coverage); positive ``roll_rad`` is right-side down.
    """
    cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
    cr, sr = math.cos(roll_rad), math.sin(roll_rad)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return ry @ rx


def frustum_mask(
    points_yaw: np.ndarray,
    cam: CameraRigCfg,
    mount: CameraMountCfg,
    *,
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
) -> np.ndarray:
    """Which yaw-frame points fall inside the depth camera frustum.

    Args:
        points_yaw: (N, 3) points in the yaw frame (base origin, world-level).
        cam / mount: camera rig and the mount being tested.
        pitch_deg: body pitch, positive = nose down.
        roll_deg: body roll, positive = right side down.

    Returns:
        (N,) bool mask. A point counts as visible when it is within the near/
        far clipping range and inside both the horizontal and vertical FOV.
        Occlusion is NOT modelled — this is an upper bound on visibility, so a
        point marked invisible here is definitively unseeable.
    """
    r_by = _rpy_body_to_yaw(math.radians(pitch_deg), math.radians(roll_deg))
    cam_pos_yaw = r_by @ np.asarray(mount.pos, dtype=float)
    r_yaw_cam = r_by @ body_to_cam(mount)

    rel = np.asarray(points_yaw, dtype=float) - cam_pos_yaw
    local = rel @ r_yaw_cam                     # == (R^T @ rel^T)^T
    right, down, depth = local[:, 0], local[:, 1], local[:, 2]

    h_half, v_half = half_fov(cam)
    return (
        (depth >= cam.min_depth)
        & (depth <= cam.max_depth)
        & (np.abs(right) <= np.tan(h_half) * depth)
        & (np.abs(down) <= np.tan(v_half) * depth)
    )


# ════════════════════════════════════════════════════════════════════════════
# Coverage
# ════════════════════════════════════════════════════════════════════════════

def scandot_coverage(
    x: ExperimentCfg,
    *,
    forward_offset: float | None = None,
    base_height: float | None = None,
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
    mount_idx: int = 0,
    order: str = "ij",
) -> dict:
    """Fraction of scandots the depth camera can see, on nominal flat ground.

    The grid is placed at ``base_height`` below the base origin — the height
    the reward drives the robot to hold, and the height at which a scandot
    reads exactly 0.0.

    Returns a dict with the coverage fraction, the visible mask, and the
    geometric limits that explain it.
    """
    sc = x.scandots
    if forward_offset is not None:
        sc = ScandotsCfg(**{**vars(sc), "forward_offset": forward_offset})
    if base_height is None:
        base_height = x.reward.target_height
    mount = x.camera.mounts[mount_idx]

    xy, ix, iy = grid_offsets(sc, order=order)
    pts = np.concatenate(
        [xy, np.full((xy.shape[0], 1), -float(base_height))], axis=-1
    )
    mask = frustum_mask(
        pts, x.camera, mount, pitch_deg=pitch_deg, roll_deg=roll_deg
    )

    h_half, v_half = half_fov(x.camera)
    # Where the BOTTOM edge of the frustum meets flat ground. Solved by
    # intersecting the actual lower-edge ray with the ground plane rather
    # than with a horizontal-camera closed form, so the number stays correct
    # under pitch (which both tilts the camera AND lowers it).
    r_by = _rpy_body_to_yaw(math.radians(pitch_deg), math.radians(roll_deg))
    cam_pos_yaw = r_by @ np.asarray(mount.pos, dtype=float)
    r_yaw_cam = r_by @ body_to_cam(mount)
    drop = float(base_height) + float(cam_pos_yaw[2])
    lower_edge = r_yaw_cam @ np.array([0.0, math.tan(v_half), 1.0])
    if lower_edge[2] < -1e-9:
        t = (-float(base_height) - cam_pos_yaw[2]) / lower_edge[2]
        ground_hit_x = float(cam_pos_yaw[0] + t * lower_edge[0])
        min_ground_dist = ground_hit_x - float(cam_pos_yaw[0])
    else:
        # Frustum bottom edge points at or above the horizon: never hits ground.
        ground_hit_x = float("inf")
        min_ground_dist = float("inf")

    return {
        "forward_offset": float(sc.forward_offset),
        "base_height": float(base_height),
        "pitch_deg": float(pitch_deg),
        "roll_deg": float(roll_deg),
        "n_points": int(sc.n_points),
        "n_visible": int(mask.sum()),
        "coverage": float(mask.mean()),
        "mask": mask,
        "ix": ix,
        "iy": iy,
        "grid_x_range": (float(xy[:, 0].min()), float(xy[:, 0].max())),
        "grid_y_range": (float(xy[:, 1].min()), float(xy[:, 1].max())),
        "h_fov_deg": math.degrees(2 * h_half),
        "v_fov_deg": math.degrees(2 * v_half),
        "camera_drop_m": drop,
        # Distance (ahead of the CAMERA) at which flat ground enters the FOV.
        "min_ground_dist_from_cam": min_ground_dist,
        # Same thing as a grid X coordinate in the yaw frame (base at x=0).
        # Exact for roll=0; with roll the near edge is not a single number.
        "min_visible_grid_x": ground_hit_x,
        "rows_visible": sorted({int(i) for i in ix[mask]}),
    }


def sweep_forward_offset(
    x: ExperimentCfg,
    offsets: np.ndarray,
    *,
    pitch_deg: float = 0.0,
    base_height: float | None = None,
    order: str = "ij",
) -> tuple[np.ndarray, np.ndarray]:
    """Coverage fraction as a function of forward_offset. Returns (offsets, cov)."""
    cov = np.array([
        scandot_coverage(
            x, forward_offset=float(o), pitch_deg=pitch_deg,
            base_height=base_height, order=order,
        )["coverage"]
        for o in offsets
    ])
    return np.asarray(offsets, dtype=float), cov


# ════════════════════════════════════════════════════════════════════════════
# Terminal rendering (works headless; no matplotlib needed)
# ════════════════════════════════════════════════════════════════════════════

_RAMP = " .:-=+*#%@"


def ascii_heatmap(
    grid: np.ndarray, vmin: float = -1.0, vmax: float = 1.0,
    mask: np.ndarray | None = None, ramp: str | None = None,
) -> str:
    """Render a (grid_x, grid_y) array as text, FRONT OF ROBOT AT THE TOP.

    Cells outside an optional bool `mask` (same shape) are drawn as '·' —
    used to overlay depth-camera visibility onto the height field.
    """
    r = ramp if ramp is not None else _RAMP
    lines = []
    span = max(vmax - vmin, 1e-9)
    for ix in range(grid.shape[0] - 1, -1, -1):
        row = []
        for iy in range(grid.shape[1] - 1, -1, -1):   # +Y is left -> print left first
            if mask is not None and not mask[ix, iy]:
                row.append("·")
                continue
            t = (float(grid[ix, iy]) - vmin) / span
            k = int(round(min(max(t, 0.0), 1.0) * (len(r) - 1)))
            row.append(r[k])
        lines.append("".join(row))
    return "\n".join(lines)


def saturation_report(heights: np.ndarray, clip: float) -> dict:
    """How much of the scandot signal is being destroyed by `height_clip`.

    A band of solid saturation means the encoder receives a constant where
    terrain structure used to be — invisible in any 3D viewport.
    """
    h = np.asarray(heights, dtype=float)
    tol = 1e-6
    hi = h >= clip - tol
    lo = h <= -clip + tol
    return {
        "frac_at_pos_clip": float(hi.mean()),
        "frac_at_neg_clip": float(lo.mean()),
        "frac_saturated": float((hi | lo).mean()),
        "min": float(h.min()),
        "max": float(h.max()),
        "mean": float(h.mean()),
        "std": float(h.std()),
    }
