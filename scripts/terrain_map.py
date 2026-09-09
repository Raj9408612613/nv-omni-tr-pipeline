#!/usr/bin/env python3
"""
Training-ground map — plan view of the terrain curriculum (no Isaac needed)
===========================================================================
Draws the arena an `--robot` config actually builds: the row x col curriculum
grid, which sub-terrain each COLUMN holds, what DIFFICULTY each ROW applies,
and the resulting geometry per cell. Everything is computed from the config
plus Isaac's own placement rules, so the map cannot drift from the code.

Two rules do the work, and both are worth knowing before you retrain:

  COLUMN -> terrain TYPE.  With curriculum=True the generator assigns one
  sub-terrain per column by cumulative proportion:
      col c gets the first type whose cumulative proportion exceeds
      c/num_cols + 0.001
  A type with a small proportion relative to num_cols can therefore receive
  ZERO columns and never be generated at all, however it is configured.

  ROW -> DIFFICULTY.  difficulty = (row + U[0,1]) / num_rows, so row i spans
  [i/rows, (i+1)/rows). Difficulty then scales each terrain's active
  dimension (stair step height, obstacle height, ...).

Usage
-----
    python scripts/terrain_map.py --robot spot
    python scripts/terrain_map.py --chain          # the whole warm-start chain
    python scripts/terrain_map.py --robot spot --elevation
"""

from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from omni_spot.configs import get_experiment_cfg  # noqa: E402

# Isaac's built-in sub-terrains, in the order env_cfg._build_terrain inserts
# them, with the config field holding each one's proportion and the dimension
# that difficulty scales.
BUILTIN = [
    ("flat",               "flat_proportion",               None),
    ("rough",              "rough_proportion",              None),
    ("stairs_up",          "stairs_up_proportion",          "stair_step_height_range"),
    ("stairs_down",        "stairs_down_proportion",        "stair_step_height_range"),
    ("discrete_obstacles", "discrete_obstacles_proportion", "discrete_obstacle_height_range"),
    ("random_grid",        "random_grid_proportion",        "random_grid_height_range"),
    ("rails",              "rails_proportion",              "rail_height_range"),
    ("stepping_stones",    "stepping_stones_proportion",    None),
]

SHORT = {
    "flat": "flat", "rough": "rough", "stairs_up": "stairUP",
    "stairs_down": "stairDN", "discrete_obstacles": "boxes",
    "random_grid": "grid", "rails": "rails", "stepping_stones": "stones",
}

ICON = {
    "flat": "____", "rough": "^v^v", "stairs_up": "/‾/‾",
    "stairs_down": "\\_\\_", "discrete_obstacles": "▪ ▪▪",
    "random_grid": "▦▦▦▦", "rails": "|| |", "stepping_stones": "▫ ▫ ",
}


def active_subterrains(t) -> tuple[list[str], list[float], dict]:
    """(names, normalised proportions, name -> difficulty-scaled range field)."""
    names, props, scaled = [], [], {}
    for name, prop_field, range_field in BUILTIN:
        p = float(getattr(t, prop_field, 0.0))
        if p <= 0.0:
            continue
        names.append(name)
        props.append(p)
        scaled[name] = range_field
    total = sum(props) or 1.0
    return names, [p / total for p in props], scaled


def column_assignment(names: list[str], props: list[float], cols: int) -> list[str]:
    """Isaac's proportional column -> sub-terrain map (curriculum=True)."""
    cum, run = [], 0.0
    for p in props:
        run += p
        cum.append(run)
    out = []
    for c in range(cols):
        target = c / cols + 0.001
        idx = next((i for i, v in enumerate(cum) if target < v), len(names) - 1)
        out.append(names[idx])
    return out


def row_difficulty(row: int, rows: int) -> tuple[float, float, float]:
    """(low, mid, high) difficulty for a curriculum row."""
    return row / rows, (row + 0.5) / rows, (row + 1) / rows


def cell_geometry(name: str, t, difficulty: float, scaled: dict,
                  short: bool = False) -> str:
    """One-line description of what a cell contains at this difficulty."""
    rng_field = scaled.get(name)
    if name == "flat":
        return f"{100 * t.flat_noise_range[1]:.0f}cm" if short else \
               f"noise {100 * t.flat_noise_range[1]:.0f}cm"
    if name == "rough":
        return (f"{100 * t.rough_noise_range[0]:.0f}-"
                f"{100 * t.rough_noise_range[1]:.0f}cm") if short else \
               (f"noise {100 * t.rough_noise_range[0]:.0f}-"
                f"{100 * t.rough_noise_range[1]:.0f}cm")
    if rng_field is None:
        return "-"
    lo, hi = getattr(t, rng_field)
    v = lo + difficulty * (hi - lo)
    if name in ("stairs_up", "stairs_down"):
        run = max(0.0, (t.patch_size - t.stair_platform_width) / 2.0)
        n = int(run / t.stair_step_width) if t.stair_step_width > 0 else 0
        return f"{100 * v:.0f}cm x{n}" if short else \
               f"step {100 * v:.0f}cm x{n} = {n * v:.2f}m"
    if name == "discrete_obstacles":
        return f"{t.discrete_obstacle_num}x {100 * v:.0f}cm" if short else \
               f"{t.discrete_obstacle_num} boxes {100 * v:.0f}cm"
    if name == "random_grid":
        return f"h{100 * v:.0f}cm" if short else \
               f"cells {100 * t.random_grid_width:.0f}cm h{100 * v:.0f}cm"
    if name == "rails":
        return f"{100 * v:.0f}cm high" if short else f"rails {100 * v:.0f}cm high"
    return f"{100 * v:.0f}cm"


def draw_map(x, name: str, elevation: bool = False) -> None:
    t = x.terrain
    names, props, scaled = active_subterrains(t)
    cols = column_assignment(names, props, t.cols)
    arena_x = t.rows * t.patch_size
    arena_y = t.cols * t.patch_size

    print("=" * 78)
    print(f"TRAINING GROUND — {name}")
    print("=" * 78)
    if getattr(x, "course", None) is not None and x.course.enabled:
        c = x.course
        print(f"  COURSE MODE — the patch grid is REPLACED by course cells.")
        print(f"  {c.rows} rows x {c.n_variants} variants of "
              f"{c.cell_size:.0f} x {c.cell_size:.0f} m corridor courses "
              f"({c.rows * c.cell_size:.0f} x "
              f"{c.n_variants * c.cell_size:.0f} m total)")
        print(f"  shapes {tuple(c.shapes)}, lane {c.lane_width} m wide, "
              f"walls {c.wall_height} m, spawn at a RANDOM END, "
              f"carrot goal {c.lookahead} m ahead")
        print(f"  row scales harshness only: stairs "
              f"{100 * c.stair_height_range[0]:.0f}-"
              f"{100 * c.stair_height_range[1]:.0f}cm, rough "
              f"{100 * c.rough_noise_range[0]:.0f}-"
              f"{100 * c.rough_noise_range[1]:.0f}cm")
        print()
        return

    print(f"  Arena     {t.rows} rows x {t.cols} cols of "
          f"{t.patch_size:.0f} x {t.patch_size:.0f} m patches "
          f"= {arena_x:.0f} x {arena_y:.0f} m  (border {t.border_width} m)")
    print(f"  Placement one env per cell; ROW = difficulty, COL = terrain type")
    print(f"  Spawn     within +/-"
          f"{t.patch_half if x.goal.spawn_half is None else x.goal.spawn_half:.2f}"
          f" m of the patch centre, random yaw")
    print(f"  Goal      {x.goal.dist_range[0]:.1f}-{x.goal.dist_range[1]:.1f} m "
          f"away, random bearing, clamped to +/-{t.patch_half:.1f} m")
    print(f"  Episode   {x.goal.episode_len_steps} steps @ 50 Hz = "
          f"{x.goal.episode_len_steps * x.sim.control_dt:.0f} s")
    print()

    # Wide grids (parkour has 14 columns) need narrow cells to stay readable
    # in a terminal, so drop to short names and the bare key number.
    wide = t.cols > 6
    w = 13 if wide else 22
    bar = "        +" + "+".join("-" * w for _ in range(t.cols)) + "+"

    print("  PLAN VIEW   (hardest row on top; each cell is one terrain patch)")
    print("        " + "".join(f" col{c:<{w - 4}d}" for c in range(t.cols)))
    print(bar)
    for row in range(t.rows - 1, -1, -1):
        _, mid, _ = row_difficulty(row, t.rows)
        line1 = f"  r{row}    |"
        line2 = f"  d={mid:.2f}|"
        for c in range(t.cols):
            nm = cols[c]
            label = SHORT.get(nm, nm)[:w - 2] if wide else nm[:15]
            geo = cell_geometry(nm, t, mid, scaled, short=wide)
            line1 += f" {ICON.get(nm, '????')} {label:<{w - 7}}|" if not wide \
                else f" {label:<{w - 2}}|"
            line2 += f" {geo:<{w - 2}}|"
        print(line1)
        print(line2)
        print(bar)
    print(f"        {'^ easiest row is at the BOTTOM (row 0, difficulty ~0)':<20}")
    print()

    print("  COLUMN -> SUB-TERRAIN")
    for i, nm in enumerate(names):
        got = [c for c in range(t.cols) if cols[c] == nm]
        flag = "" if got else "   <-- NEVER GENERATED (no column)"
        print(f"    {nm:<20} proportion {props[i]:.3f}  -> "
              f"cols {got if got else '[]'}{flag}")
    print()

    print("  ROW -> DIFFICULTY   difficulty = (row + U[0,1]) / rows")
    for row in range(t.rows):
        lo, mid, hi = row_difficulty(row, t.rows)
        s_lo, s_hi = t.stair_step_height_range
        print(f"    row {row}  d in [{lo:.2f}, {hi:.2f})   "
              f"stair step {100 * (s_lo + lo * (s_hi - s_lo)):.0f}-"
              f"{100 * (s_lo + hi * (s_hi - s_lo)):.0f} cm")
    print()

    print("  ACCURACY")
    print("    Grid size, patch size, proportions, column assignment, spawn,")
    print("    goal and episode come straight from the config + Isaac's column")
    print("    rule, and are exact. Per-cell dimensions assume difficulty scales")
    print("    each terrain's range linearly (lo + d*(hi-lo)) — certain for the")
    print("    stair pyramids; verify the parkour tiles against your IsaacLab")
    print("    build before relying on those numbers:")
    print("      grep -n 'difficulty' $ISAACLAB/source/isaaclab/isaaclab/"
          "terrains/height_field/hf_terrains.py")
    print("    Flat/rough noise_range is NOT difficulty-scaled in the builds I")
    print("    know of — every row gets the same noise band.")
    print()

    if elevation and any(n.startswith("stairs") for n in names):
        draw_stair_elevation(t)


def draw_stair_elevation(t) -> None:
    """Side view of the pyramid a stairs_up cell builds at max difficulty."""
    lo, hi = t.stair_step_height_range
    run = max(0.0, (t.patch_size - t.stair_platform_width) / 2.0)
    n = int(run / t.stair_step_width) if t.stair_step_width > 0 else 0
    h = hi
    print(f"  SIDE ELEVATION — stairs_up at top row (step {100 * h:.0f} cm x "
          f"{t.stair_step_width:.2f} m tread, {n} steps each side)")
    print(f"  apex {n * h:.2f} m above the patch floor; "
          f"centre platform {t.stair_platform_width:.1f} m wide")
    print()
    # 1 character = one tread. Terrace at level lvl is
    # platform + 2*(n-lvl) treads wide, so the widest terrace is the FLOOR
    # and the narrowest is the apex platform.
    plat = max(1, int(round(t.stair_platform_width / t.stair_step_width)))
    step_rows = max(1, int(math.ceil(n / 14)))     # thin out very tall stacks
    for lvl in range(n, -1, -1):
        if lvl % step_rows and lvl not in (0, n):
            continue
        width = plat + 2 * (n - lvl)
        print(f"   {lvl * h:5.2f}m {' ' * lvl}{'#' * width}")
    print(f"          {'~' * (plat + 2 * n)}")
    print(f"          |<- {t.patch_size:.0f} m patch, 1 char = "
          f"{t.stair_step_width:.2f} m tread ->|")
    print(f"          Spawn is the CENTRE = the APEX platform, so the robot "
          f"starts on top;")
    print(f"          any goal off the platform means walking DOWN the "
          f"pyramid and possibly back up.")
    print()


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robot", default="spot")
    p.add_argument("--chain", action="store_true",
                   help="draw the whole warm-start chain in order")
    p.add_argument("--elevation", action="store_true",
                   help="also draw a side view of the stair pyramid")
    a = p.parse_args()

    robots = (["spot", "spot_robust", "spot_hard", "spot_parkour",
               "spot_parkour_robust", "spot_master", "spot_master_course"]
              if a.chain else [a.robot])
    for r in robots:
        draw_map(get_experiment_cfg(r), r, elevation=a.elevation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
