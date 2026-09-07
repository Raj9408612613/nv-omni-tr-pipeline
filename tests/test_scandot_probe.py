"""
CPU tests for the scandot geometry probe (no Isaac, no torch).

These pin the geometry that the forward_offset decision rests on: grid
construction, flatten-order round-tripping, camera frustum math, and the
coverage measurement. If any of these drift, the coverage number silently
becomes fiction — which is worse than not measuring at all.

    python tests/test_scandot_probe.py     # or: pytest tests/test_scandot_probe.py
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from omni_spot.configs import get_experiment_cfg
from omni_spot.configs.base import CameraMountCfg, ScandotsCfg
from omni_spot.scandot_probe import (
    ascii_heatmap,
    body_to_cam,
    frustum_mask,
    grid_offsets,
    half_fov,
    infer_grid_order,
    quat_to_matrix,
    saturation_report,
    scandot_coverage,
    sweep_forward_offset,
    to_heatmap,
)


def test_grid_shape_and_extent():
    # Pin the raw lattice, independent of the shipped forward_offset.
    sc = ScandotsCfg(forward_offset=0.0)
    xy, ix, iy = grid_offsets(sc)
    assert xy.shape == (sc.n_points, 2) == (187, 2)
    assert np.isclose(xy[:, 0].min(), -0.8) and np.isclose(xy[:, 0].max(), 0.8)
    assert np.isclose(xy[:, 1].min(), -0.5) and np.isclose(xy[:, 1].max(), 0.5)
    # spacing is honoured on both axes
    assert np.isclose(np.diff(np.unique(xy[:, 0])).max(), sc.spacing)
    assert np.isclose(np.diff(np.unique(xy[:, 1])).max(), sc.spacing)
    assert ix.max() == sc.grid_x - 1 and iy.max() == sc.grid_y - 1
    print("ok: grid shape/extent")


def test_forward_offset_shifts_only_x():
    a, _, _ = grid_offsets(ScandotsCfg(forward_offset=0.0))
    b, _, _ = grid_offsets(ScandotsCfg(forward_offset=0.7))
    assert np.allclose(b[:, 0] - a[:, 0], 0.7)
    assert np.allclose(b[:, 1], a[:, 1])
    print("ok: forward_offset shifts +X only")


def test_order_inference_and_heatmap_roundtrip():
    sc = ScandotsCfg()
    for order in ("ij", "xy"):
        xy, ix, iy = grid_offsets(sc, order=order)
        assert infer_grid_order(xy, sc) == order
        # A value that encodes its own grid position must land in the right cell
        vals = ix * 100.0 + iy
        grid = to_heatmap(vals, sc, order=order)
        assert grid.shape == (sc.grid_x, sc.grid_y)
        for r in range(sc.grid_x):
            for c in range(sc.grid_y):
                assert grid[r, c] == r * 100.0 + c
    print("ok: order inference + heatmap round-trip (ij and xy)")


def test_order_inference_rejects_garbage():
    sc = ScandotsCfg()
    rng = np.random.default_rng(0)
    try:
        infer_grid_order(rng.normal(size=(sc.n_points, 2)) * 5.0, sc)
    except RuntimeError:
        print("ok: order inference rejects a non-lattice pattern")
        return
    raise AssertionError("expected RuntimeError on a scrambled pattern")


def test_quat_matrix_is_rotation_and_axis_is_forward():
    mount = CameraMountCfg()
    r = body_to_cam(mount)
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(r), 1.0)
    # ros convention: optical +Z is forward -> body +X; optical +Y is down.
    assert np.allclose(r @ np.array([0.0, 0.0, 1.0]), [1.0, 0.0, 0.0], atol=1e-9)
    assert np.allclose(r @ np.array([0.0, 1.0, 0.0]), [0.0, 0.0, -1.0], atol=1e-9)
    print("ok: camera quaternion -> optical axis along body +X, +Y down")


def test_half_fov_matches_aspect():
    x = get_experiment_cfg("spot")
    h, v = half_fov(x.camera)
    assert np.isclose(math.degrees(2 * h), 87.0)
    # square pixels: tan(v) / tan(h) == height / width
    assert np.isclose(math.tan(v) / math.tan(h),
                      x.camera.height / x.camera.width)
    print(f"ok: FOV {math.degrees(2*h):.1f} x {math.degrees(2*v):.1f} deg")


def test_frustum_basic_cases():
    """Axis-relative so the assertions hold for any mount tilt."""
    x = get_experiment_cfg("spot")
    cam, mount = x.camera, x.camera.mounts[0]
    origin = np.asarray(mount.pos, dtype=float)
    axis = body_to_cam(mount) @ np.array([0.0, 0.0, 1.0])   # optical axis, body frame
    side = body_to_cam(mount) @ np.array([1.0, 0.0, 0.0])   # image right

    def seen(p):
        return bool(frustum_mask(np.array([p]), cam, mount)[0])

    assert seen(origin + 2.0 * axis), "on-axis at 2 m must be visible"
    assert not seen(origin - 2.0 * axis), "behind the camera must be rejected"
    assert not seen(origin + (cam.max_depth + 1.0) * axis), "far clip"
    assert not seen(origin + 0.05 * axis), "near clip"
    assert not seen(origin + 0.5 * axis + 5.0 * side), "outside horizontal FOV"
    print("ok: frustum accepts on-axis / rejects behind, near, far, off-axis")


def test_nose_down_pitch_increases_coverage_with_a_level_mount():
    """With no mount tilt, more nose-down body pitch always sees more grid."""
    x = _level_camera_cfg()
    cov = [scandot_coverage(x, pitch_deg=p)["coverage"]
           for p in (10.0, 15.0, 20.0)]
    assert cov[0] < cov[1] < cov[2], cov
    print(f"ok: nose-down pitch raises coverage {[round(c, 3) for c in cov]}")


def test_mount_tilt_trades_far_range_for_near_coverage():
    """Tilting past the half-vertical-FOV puts a hard ceiling on sight distance.

    The frustum's TOP edge sits at (mount_tilt + body_pitch) - v_half below the
    horizon. Once that is positive the camera can no longer see the horizon,
    and flat ground beyond drop/tan(top_edge) falls above the image. The
    shipped 30 deg mount leaves only ~2 deg of margin, so ordinary nose-down
    walking pitch starts truncating the far field.
    """
    x = get_experiment_cfg("spot")
    _, v_half = half_fov(x.camera)
    v_half_deg = math.degrees(v_half)
    tilt = x.camera.mounts[0].pitch_deg
    margin = v_half_deg - tilt
    assert margin > 0.0, (
        f"mount tilt {tilt} deg exceeds the half vertical FOV {v_half_deg:.1f} "
        f"deg — the camera cannot see the horizon even standing level"
    )

    drop = x.reward.target_height + x.camera.mounts[0].pos[2]

    def far_limit(body_pitch):
        top = tilt + body_pitch - v_half_deg
        return float("inf") if top <= 0 else drop / math.tan(math.radians(top))

    assert far_limit(0.0) == float("inf")
    # Monotonically shrinking as the robot pitches nose-down.
    limits = [far_limit(p) for p in (10.0, 15.0, 20.0)]
    assert limits[0] > limits[1] > limits[2], limits
    print(f"ok: mount tilt {tilt:.0f} deg leaves {margin:.1f} deg of horizon "
          f"margin; far ground limit at 10/15/20 deg body pitch = "
          f"{limits[0]:.1f}/{limits[1]:.1f}/{limits[2]:.1f} m")


def test_coverage_is_monotonic_in_forward_offset():
    x = get_experiment_cfg("spot")
    offs, cov = sweep_forward_offset(x, np.arange(0.0, 2.6, 0.05))
    assert np.all(np.diff(cov) >= -1e-9), "coverage must not decrease"
    assert cov[-1] == 1.0
    print(f"ok: coverage monotonic, reaches 1.0 over offsets "
          f"{offs[0]:.2f}..{offs[-1]:.2f}")


def _level_camera_cfg():
    """Shipped config with the camera tilt removed (the pre-fix geometry)."""
    x = get_experiment_cfg("spot")
    m = x.camera.mounts[0]
    x.camera.mounts = (
        CameraMountCfg(name=m.name, pos=m.pos, pitch_deg=0.0,
                       convention=m.convention),
    )
    return x


def test_mount_pitch_tilts_optical_axis_down():
    """`CameraMountCfg.pitch_deg` must tilt the axis DOWN and stay unit-norm."""
    level = body_to_cam(CameraMountCfg(pitch_deg=0.0))
    tilted = body_to_cam(CameraMountCfg(pitch_deg=30.0))
    for r in (level, tilted):
        assert np.allclose(r @ r.T, np.eye(3), atol=1e-9)
    axis_level = level @ np.array([0.0, 0.0, 1.0])
    axis_tilt = tilted @ np.array([0.0, 0.0, 1.0])
    assert np.isclose(axis_level[2], 0.0, atol=1e-9), axis_level
    # 30 deg nose-down -> z component of the optical axis is -sin(30) = -0.5
    assert np.isclose(axis_tilt[2], -0.5, atol=1e-6), axis_tilt
    assert axis_tilt[0] > 0.0, "camera must still look forward"
    print(f"ok: pitch_deg=30 tilts optical axis to {np.round(axis_tilt, 4)}")


def test_level_camera_with_no_offset_sees_nothing():
    """The original failure mode, pinned so it cannot silently return.

    A level camera at Spot's ride height cannot see ground closer than
    ~1.3 m, while a centred grid ends at 0.8 m — zero overlap.
    """
    x = _level_camera_cfg()
    r = scandot_coverage(x, forward_offset=0.0)
    assert r["n_visible"] == 0, r["n_visible"]
    assert r["grid_x_range"][1] < r["min_visible_grid_x"]
    print(f"ok: level camera + centred grid -> {r['n_visible']}/{r['n_points']} "
          f"(ground enters FOV at x={r['min_visible_grid_x']:.2f} m, "
          f"grid ends at x={r['grid_x_range'][1]:.2f} m)")


def test_shipped_spot_config_has_real_overlap():
    """Guards the fix: the shipped config must keep teacher/student overlap.

    Both halves matter — the camera must reach scandots, and the grid must
    still cover ground near the feet that the camera cannot see.
    """
    x = get_experiment_cfg("spot")
    r = scandot_coverage(x)
    assert r["coverage"] > 0.15, (
        f"shipped config exposes only {r['n_visible']}/{r['n_points']} scandots "
        f"to the depth camera; the student cannot recover the rest"
    )
    assert r["grid_x_range"][0] < 0.0, (
        "grid no longer covers ground behind the base origin — the near-field "
        "foot-placement signal is gone"
    )
    print(f"ok: shipped spot config {r['n_visible']}/{r['n_points']} "
          f"({100 * r['coverage']:.1f}%) visible, grid spans "
          f"{r['grid_x_range'][0]:+.2f}..{r['grid_x_range'][1]:+.2f} m")



def test_saturation_report():
    clip = 1.0
    h = np.array([-1.0, -1.0, 0.0, 0.5, 1.0])
    s = saturation_report(h, clip)
    assert np.isclose(s["frac_at_neg_clip"], 0.4)
    assert np.isclose(s["frac_at_pos_clip"], 0.2)
    assert np.isclose(s["frac_saturated"], 0.6)
    print("ok: saturation report")


def test_ascii_heatmap_orientation():
    """Front row must print first; robot's left must print leftmost."""
    sc = ScandotsCfg(grid_x=3, grid_y=3, spacing=0.1)
    grid = np.zeros((3, 3))
    grid[2, :] = 1.0        # frontmost row (ix = grid_x-1)
    lines = ascii_heatmap(grid, 0.0, 1.0).splitlines()
    assert lines[0] == "@@@" and lines[-1] == "   ", lines
    grid = np.zeros((3, 3))
    grid[:, 2] = 1.0        # leftmost column (iy = grid_y-1, +Y is left)
    lines = ascii_heatmap(grid, 0.0, 1.0).splitlines()
    assert all(ln[0] == "@" for ln in lines), lines
    print("ok: ascii heatmap orientation (front up, left left)")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"\nAll {len(fns)} scandot-probe tests passed.")
