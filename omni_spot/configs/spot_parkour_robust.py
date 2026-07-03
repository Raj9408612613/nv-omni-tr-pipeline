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

    return cfg
