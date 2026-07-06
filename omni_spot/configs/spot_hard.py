"""
Spot — HARDER terrain curriculum (Axis-1 difficulty bump)
=========================================================
Same robot/embodiment as `spot` (identical joints, dims, networks), only the
terrain curriculum is harder: taller stairs, rougher ground, less flat, and
more difficulty rows so the curriculum has somewhere harder to promote good
envs into. Because the obs/action dims are unchanged, a `spot` teacher (or a
`spot` PBT best.pt) warm-starts straight into this via --init_ckpt.

    PYTHONPATH=. python -m omni_spot.train_pbt --robot spot_hard \
        --init_ckpt omni_logs/<spot_run>/best.pt --headless

Expect success to drop at first (distribution shift) — that is the point: it
gives PBT real headroom to optimize into, instead of a saturated ~0.985 ceiling.
"""

from __future__ import annotations

from .spot import make_cfg as _spot_make_cfg


def make_cfg():
    cfg = _spot_make_cfg()

    t = cfg.terrain
    # Taller stairs — this is the single most direct lever on the locomotion
    # ceiling (the ~1.5% the flat-terrain teacher fails are the hardest steps).
    t.stair_step_height_range = (0.10, 0.30)   # was (0.05, 0.23)
    # Rougher uneven ground.
    t.rough_noise_range = (0.05, 0.18)         # was (0.02, 0.10)
    # Shift the mix off flat toward stairs/rough (proportions must sum to 1.0).
    t.flat_proportion = 0.20                   # was 0.40
    t.rough_proportion = 0.30                  # was 0.20
    t.stairs_up_proportion = 0.25              # was 0.20
    t.stairs_down_proportion = 0.25            # was 0.20
    # More difficulty rows: with only 4 rows a 98.5%-success population has
    # already maxed the curriculum, so there is nowhere harder to promote to.
    t.rows = 6                                 # was 4
    # flat_row_max stays 1 so rows 0-1 remain flat (obstacles live there).

    return cfg
