"""
Spot — STAGE 3: parkour terrain + get-up robustness (no leg failure yet)
========================================================================
Composes the two independent lineages (they touch disjoint config fields):
  terrain   <- spot_parkour   (hard stairs/rough + obstacle/grid/rail tiles)
  reward/dr <- spot_robust    (terminate_on_fall off, recovery shaping, pushes)

Warm-start from a converged spot_robust run:

    PYTHONPATH=. python -m omni_spot.train_pbt --robot spot_parkour_robust \
        --init_ckpt omni_logs/<spot_robust_run>/best.pt --headless

Chain: spot -> spot_robust -> spot_parkour_robust -> spot_master (-> course).
"""

from __future__ import annotations

from .spot_parkour import make_cfg as _parkour_make_cfg


def make_cfg():
    cfg = _parkour_make_cfg()

    # Robustness block — same values as spot_robust (kept in sync by hand;
    # the two lineages meet here).
    r = cfg.reward
    r.terminate_on_fall = False
    r.recover_w = 0.5
    r.ang_vel_w = -0.02
    r.upright_w = -0.5

    d = cfg.dr
    d.enabled = True
    d.push_robots = True

    # ── Round-2 improvements, ported from spot_robust ──────────────────────
    # (spot_master inherits this make_cfg, so the rest of the chain gets
    # these automatically.)
    # PBT searches the get-up gradient; the (lo>=hi)=pinned default would
    # otherwise hold recover_w at the hand value above.
    cfg.pbt.recover_w_range = (0.15, 1.5)

    # Rudin-scale traversal (see spot_robust for the full rationale).
    g = cfg.goal
    g.dist_range = (3.0, 7.0)
    g.episode_len_steps = 1200
    g.spawn_half = 0.0             # fixed start, standing on the platform

    t = cfg.terrain
    t.patch_size = 16.0
    t.patch_half = 7.5
    # Stair geometry re-derived for 16 m patches with spot_hard's HARD step
    # heights: 6.5 m slope run / 0.75 m tread ≈ 8.7 steps; height capped at
    # 0.26 keeps the pyramid apex <= ~2.3 m (0.30 m treads would stack ~5 m).
    t.stair_step_width = 0.75      # spot_hard had 0.30 on 8 m patches
    t.stair_step_height_range = (0.08, 0.26)  # spot_hard had (0.10, 0.30)
    # Parkour tiles: spawn_half=1.0 needs a >= 3 m clear center pad (default
    # platform is 2 m), and obstacle count scales with the 4x patch area
    # (10 boxes on an 8 m tile -> 24 on 16 m) so density is preserved.
    t.parkour_platform_width = 3.0
    t.discrete_obstacle_num = 24

    return cfg
