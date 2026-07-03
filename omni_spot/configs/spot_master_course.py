"""
Spot — STAGE 5 (polish): MASTER skills on shaped COURSE arenas
==============================================================
The final polish round (option A): the all-skills master teacher trained on
straight / L / T corridor courses with carrot goal-planning — end-to-end
traversal with direction changes, random spawn end, terrain changing along
the path, plus get-up recovery and one-leg failures throughout.

    PYTHONPATH=. python -m omni_spot.train_pbt --robot spot_master_course \
        --init_ckpt omni_logs/<spot_master_run>/best.pt --headless

Distill the deployable student from this run's best.pt. Note: on courses the
parkour PATCH terrain is replaced by the course cells (course mode swaps the
terrain builder); the parkour skills persist via warm-start + the course's
own stairs/rough segments. Keep this round short — it is polish, not the
main formation run.
"""

from __future__ import annotations

from .spot_master import make_cfg as _spot_master_make_cfg


def make_cfg():
    cfg = _spot_master_make_cfg()

    cfg.course.enabled = True
    cfg.obstacles.n_static = 0
    cfg.goal.episode_len_steps = 3000   # 60 s @ 50 Hz

    return cfg
