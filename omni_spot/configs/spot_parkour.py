"""
Spot — PARKOUR terrain stage (Axis-1 "next level" beyond stairs)
================================================================
Same robot / embodiment / networks as `spot` and `spot_hard` (identical obs and
action dims), so a `spot_hard` teacher — or a `spot_hard` PBT best.pt — warm-
starts straight in via --init_ckpt. This config INHERITS spot_hard's harder
stair/rough curriculum (so the policy keeps those skills) and ADDS parkour
sub-terrains as new curriculum columns:

    discrete_obstacles  scattered low boxes to step over / weave around
    random_grid         blocky uneven floor (generalises rough terrain)
    rails               thin raised rails to step over

Difficulty still rides the terrain ROWS (row 0 ~ trivial, top row = the
configured max), so each parkour skill ramps up automatically per-env, exactly
like the stairs did. Wide gaps / pits / tall boxes / stepping-stones are left
OFF here — a position-controlled Spot cannot truly leap or climb them, so they
would just collapse success; they are reserved for a later spot_parkour_hard
stage (flip on their proportions in TerrainCfg and re-balance).

    PYTHONPATH=. python -m omni_spot.train_pbt --robot spot_parkour \
        --init_ckpt omni_logs/<spot_hard_run>/best.pt --headless

Expect success to dip at first (new terrain distribution) — that dip is the
headroom the parkour columns add. If stepping-over is suppressed by the height
/ feet-air-time reward terms, those are the knobs to relax (not changed here).
"""

from __future__ import annotations

from .spot_hard import make_cfg as _spot_hard_make_cfg


def make_cfg():
    # Inherits spot_hard: taller stairs (0.10-0.30), rougher ground, rows=6.
    cfg = _spot_hard_make_cfg()

    t = cfg.terrain

    # Column mix across ALL active sub-terrains — must sum to 1.0. Keep flat
    # (recovery footing) and the stairs (retain that skill) while giving the
    # three parkour tiles a comparable share.
    t.flat_proportion = 0.14
    t.rough_proportion = 0.14
    t.stairs_up_proportion = 0.14
    t.stairs_down_proportion = 0.14
    t.discrete_obstacles_proportion = 0.14
    t.random_grid_proportion = 0.15
    t.rails_proportion = 0.15
    # sum = 1.00 over 7 active sub-terrains.

    # The curriculum assigns each COLUMN one sub-terrain by proportion, so we
    # need >= 1 column per type. 14 columns -> ~2 each, with margin so rounding
    # never starves a type. (rows stays at spot_hard's 6 difficulty levels.)
    t.cols = 14

    # Parkour difficulty envelopes (row 0 -> min, top row -> max), tuned to
    # what a position-controlled Spot can plausibly clear. Bump these in a
    # spot_parkour_hard stage once this one saturates.
    t.discrete_obstacle_height_range = (0.05, 0.18)
    t.discrete_obstacle_num = 10
    t.random_grid_height_range = (0.02, 0.12)
    t.rail_height_range = (0.05, 0.16)

    return cfg
