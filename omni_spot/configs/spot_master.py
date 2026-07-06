"""
Spot — STAGE 4: MASTER (all skills on) — the investor-demo teacher
==================================================================
Everything at once, in one training distribution / one network:
  * hard + parkour terrain curriculum   (from spot_parkour)
  * get-up recovery, no fall termination (from spot_robust)
  * per-episode one-leg disable          (from spot_robust_legfail)

Because every challenge is present simultaneously (different envs draw
different terrain/conditions each episode), nothing learned earlier is
forgotten — this run FORMS the joint policy. Warm-start from
spot_parkour_robust:

    PYTHONPATH=. python -m omni_spot.train_pbt --robot spot_master \
        --init_ckpt omni_logs/<spot_parkour_robust_run>/best.pt --headless

Its best.pt is the teacher to distill the deployable student from
(optionally after a spot_master_course polish round).
"""

from __future__ import annotations

from .spot_parkour_robust import make_cfg as _parkour_robust_make_cfg


def make_cfg():
    cfg = _parkour_robust_make_cfg()

    d = cfg.dr
    d.randomize_leg_failure = True
    d.leg_failure_prob = 0.15        # ~15% of episodes lose one leg
    d.leg_failure_strength = 0.0     # fully limp

    return cfg
