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
    sc = ScandotsCfg()
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
    x = get_experiment_cfg("spot")
    cam, mount = x.camera, x.camera.mounts[0]
    # Straight ahead at camera height, 2 m out -> visible.
    p = np.array([[mount.pos[0] + 2.0, 0.0, mount.pos[2]]])
    assert frustum_mask(p, cam, mount)[0]
    # Directly behind -> never.
    p = np.array([[-2.0, 0.0, mount.pos[2]]])
    assert not frustum_mask(p, cam, mount)[0]
    # Beyond the far clip -> never.
    p = np.array([[mount.pos[0] + cam.max_depth + 1.0, 0.0, mount.pos[2]]])
    assert not frustum_mask(p, cam, mount)[0]
    # Far off to the side at short range -> outside horizontal FOV.
    p = np.array([[mount.pos[0] + 0.5, 5.0, mount.pos[2]]])
    assert not frustum_mask(p, cam, mount)[0]
    print("ok: frustum accepts ahead / rejects behind, too-far, off-axis")


def test_nose_down_pitch_increases_coverage():
    x = get_experiment_cfg("spot")
    cov = [
        scandot_coverage(x, forward_offset=1.0, pitch_deg=p)["coverage"]
        for p in (0.0, 10.0, 20.0)
    ]
    assert cov[0] < cov[1] < cov[2], cov
    print(f"ok: nose-down pitch raises coverage {[round(c, 3) for c in cov]}")


def test_coverage_is_monotonic_in_forward_offset():
    x = get_experiment_cfg("spot")
    offs, cov = sweep_forward_offset(x, np.arange(0.0, 2.6, 0.05))
    assert np.all(np.diff(cov) >= -1e-9), "coverage must not decrease"
    assert cov[0] == 0.0
    assert cov[-1] == 1.0
    print(f"ok: coverage monotonic 0.0 -> 1.0 over offsets "
          f"{offs[0]:.2f}..{offs[-1]:.2f}")


def test_spot_default_has_zero_overlap():
    """The finding this tool exists to surface, pinned as a regression."""
    x = get_experiment_cfg("spot")
    r = scandot_coverage(x)
    assert r["forward_offset"] == 0.0
    assert r["n_visible"] == 0, (
        f"expected zero teacher/student overlap at the shipped config, got "
        f"{r['n_visible']}/{r['n_points']}"
    )
    # The grid ends well short of where the ground enters the frustum.
    assert r["grid_x_range"][1] < r["min_visible_grid_x"]
    print(f"ok: shipped spot config has {r['n_visible']}/{r['n_points']} "
          f"overlap; ground enters FOV at x={r['min_visible_grid_x']:.2f} m "
          f"but grid ends at x={r['grid_x_range'][1]:.2f} m")


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
