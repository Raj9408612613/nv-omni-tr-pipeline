"""
Spot — COURSE smoke config (shaped arenas + carrot goals, infra test)
=====================================================================
Straight / L / T corridor courses (see omni_spot/course.py) with the carrot
goal-planner, on top of the Round-1 robustness reward (get-up + recovery).
Use this to smoke-test the course terrain + carrot in sim BEFORE the real
polish stage (spot_master_course, which is the same idea on the master
teacher).

    PYTHONPATH=. python -m omni_spot.train_pbt --robot spot_course \
        --init_ckpt omni_logs/<spot_robust_run>/best.pt --headless

Box obstacles are removed and episodes lengthened so a full course
(~20-35 m of path incl. bends and hills at ~0.8 m/s) fits in one episode.
"""

from __future__ import annotations

from .spot_robust import make_cfg as _spot_robust_make_cfg


def make_cfg():
    cfg = _spot_robust_make_cfg()

    cfg.course.enabled = True
    cfg.obstacles.n_static = 0          # pure navigation on the courses
    cfg.goal.episode_len_steps = 3000   # 60 s @ 50 Hz — time to finish a course

    return cfg
