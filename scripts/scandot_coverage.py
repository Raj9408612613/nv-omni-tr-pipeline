#!/usr/bin/env python3
"""
Scandot / depth-camera coverage report — runs ANYWHERE (no Isaac, no GPU)
=========================================================================
Settles `ScandotsCfg.forward_offset` by measuring, not by argument: how many
of the teacher's scandots can the student's depth camera physically see?

Scandots the camera can never see are teacher exteroception the student has
no way to recover. Phase 2 distils the teacher's ACTIONS, so any behaviour
that depends on unseeable scandots becomes an irreducible DAgger floor.

Usage
-----
    python scripts/scandot_coverage.py --robot spot
    python scripts/scandot_coverage.py --robot spot --pitch 10
    python scripts/scandot_coverage.py --robot spot --plot cov.png

Reads only `omni_spot.configs` (plain dataclasses), so it runs on a laptop.
Occlusion is not modelled, so every number here is an UPPER BOUND on what the
camera sees: a point reported invisible is definitively invisible.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from omni_spot.configs import get_experiment_cfg  # noqa: E402
from omni_spot.scandot_probe import (  # noqa: E402
    ascii_heatmap,
    grid_offsets,
    scandot_coverage,
    sweep_forward_offset,
    to_heatmap,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robot", default="spot", help="config module in omni_spot/configs/")
    p.add_argument("--pitch", type=float, default=0.0,
                   help="body pitch in degrees, positive = nose down")
    p.add_argument("--roll", type=float, default=0.0)
    p.add_argument("--base_height", type=float, default=None,
                   help="base height over terrain (default: reward.target_height)")
    p.add_argument("--order", default="ij", choices=["ij", "xy"],
                   help="grid flatten order (verify against a live sensor "
                        "with scripts/scandot_inspect.py)")
    p.add_argument("--sweep_max", type=float, default=2.5)
    p.add_argument("--sweep_step", type=float, default=0.05)
    p.add_argument("--plot", default=None, metavar="PNG",
                   help="also write a coverage-vs-offset plot (needs matplotlib)")
    a = p.parse_args()

    x = get_experiment_cfg(a.robot)
    sc, cam = x.scandots, x.camera
    mount = cam.mounts[0]

    r = scandot_coverage(
        x, pitch_deg=a.pitch, roll_deg=a.roll,
        base_height=a.base_height, order=a.order,
    )

    print("=" * 74)
    print(f"SCANDOT / DEPTH COVERAGE — robot={a.robot}")
    print("=" * 74)
    print(f"  scandot grid      {sc.grid_x} x {sc.grid_y} = {sc.n_points} pts "
          f"@ {sc.spacing} m  (extent {sc.size[0]} x {sc.size[1]} m)")
    print(f"  forward_offset    {sc.forward_offset:+.3f} m")
    print(f"  grid X span       {r['grid_x_range'][0]:+.2f} .. "
          f"{r['grid_x_range'][1]:+.2f} m  (base at x=0)")
    print(f"  grid Y span       {r['grid_y_range'][0]:+.2f} .. "
          f"{r['grid_y_range'][1]:+.2f} m")
    print(f"  height_clip       +/-{sc.height_clip} m")
    print()
    print(f"  camera            {cam.width}x{cam.height} @ "
          f"{1.0 / cam.update_period_s:.0f} Hz, mount pos {tuple(mount.pos)}")
    print(f"  FOV               {r['h_fov_deg']:.1f} deg horizontal, "
          f"{r['v_fov_deg']:.1f} deg vertical (from aspect ratio)")
    print(f"  clip range        {cam.min_depth} .. {cam.max_depth} m")
    print(f"  body pitch/roll   {r['pitch_deg']:+.1f} / {r['roll_deg']:+.1f} deg")
    print()
    print(f"  base height       {r['base_height']:.2f} m over terrain")
    print(f"  camera height     {r['camera_drop_m']:.2f} m over terrain")
    print(f"  ground enters FOV {r['min_ground_dist_from_cam']:.2f} m ahead of "
          f"the camera")
    print(f"  => nearest visible ground is at grid x = "
          f"{r['min_visible_grid_x']:+.2f} m")
    print()
    print(f"  VISIBLE SCANDOTS  {r['n_visible']} / {r['n_points']}  "
          f"({100.0 * r['coverage']:.1f}%)")
    if r["rows_visible"]:
        print(f"  visible rows (ix) {r['rows_visible']}  "
              f"of 0..{sc.grid_x - 1} (0 = rearmost)")
    else:
        print("  visible rows (ix) NONE")
    print()

    # ── Visibility map over the grid ────────────────────────────────────
    xy, ix, iy = grid_offsets(sc, order=a.order)
    vis_grid = to_heatmap(r["mask"].astype(float), sc, order=a.order)
    print("  Camera visibility over the scandot grid")
    print("  (front of robot at TOP, robot's left at LEFT; "
          "'@' = camera sees it, ' ' = never)")
    for line in ascii_heatmap(vis_grid, 0.0, 1.0).splitlines():
        print(f"      {line}")
    print(f"      {'^' * sc.grid_y}  <- base is at row "
          f"{int(round((0.0 - xy[:, 0].min()) / sc.spacing))} from the bottom")
    print()

    # ── Sweep ───────────────────────────────────────────────────────────
    offs = np.arange(0.0, a.sweep_max + 1e-9, a.sweep_step)
    offs, cov = sweep_forward_offset(
        x, offs, pitch_deg=a.pitch, base_height=a.base_height, order=a.order
    )
    print(f"  Coverage vs forward_offset (pitch {a.pitch:+.1f} deg)")
    print("  offset[m]  coverage  bar")
    for o, c in zip(offs, cov):
        if abs(o / a.sweep_step - round(o / a.sweep_step)) > 1e-6:
            continue
        bar = "#" * int(round(c * 50))
        star = "  <- current" if abs(o - sc.forward_offset) < 1e-9 else ""
        print(f"   {o:5.2f}     {100 * c:5.1f}%  {bar}{star}")
    print()

    first = np.argmax(cov > 0) if np.any(cov > 0) else None
    full = np.argmax(cov >= 0.999) if np.any(cov >= 0.999) else None
    if first is not None and cov[first] > 0:
        print(f"  first non-zero coverage at forward_offset = {offs[first]:.2f} m")
    else:
        print("  NO offset in the swept range gives non-zero coverage")
    if full is not None and cov[full] >= 0.999:
        print(f"  100% coverage at forward_offset = {offs[full]:.2f} m "
              f"(grid then starts {offs[full] - sc.size[0] / 2:+.2f} m ahead of "
              f"the base — nothing under the feet is scanned)")
    print("=" * 74)

    if a.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(offs, 100 * cov, lw=2)
            ax.axvline(sc.forward_offset, ls="--", c="r",
                       label=f"current ({sc.forward_offset:g})")
            ax.set_xlabel("scandots.forward_offset [m]")
            ax.set_ylabel("scandots visible to depth camera [%]")
            ax.set_title(f"{a.robot}: teacher/student exteroception overlap "
                         f"(pitch {a.pitch:+.0f}°)")
            ax.grid(alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(a.plot, dpi=140)
            print(f"[plot] wrote {a.plot}")
        except ImportError:
            print("[plot] matplotlib unavailable; skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
