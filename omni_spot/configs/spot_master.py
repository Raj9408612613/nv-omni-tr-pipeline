"""
Spot — MASTER: EVERYTHING ON (final teacher for the investor demo)
==================================================================
The last link in the progressive warm-start chain and the ⭐ PRIMARY OBJECTIVE:
ONE policy that does it all — walks hard/parkour terrain, RECOVERS instead of
falling (full get-up), and keeps going with ONE LEG DISABLED.

    spot            (base, exists)
     └─ spot_robust          get-up / recovery on the CURRENT terrain
         └─ spot_parkour_robust   add hard + parkour terrain, KEEP recovery
             └─ spot_master        THIS: add one-leg failure (everything on) ← FINAL
                 └─ distill → one student (depth-camera) for the investor demo

Composition (all three lineages, disjoint fields, nothing conflicts):
  * terrain  — inherited via spot_parkour_robust -> spot_parkour -> spot_hard:
    taller stairs, rougher ground, rows=6, + parkour columns.
  * recovery — spot_robust.apply_recovery: terminate_on_fall=False + get-up
    gradient + pushes.
  * leg-fail — spot_robust_legfail.apply_leg_failure: per-episode, ~15% of envs
    lose one random leg (limp), which the policy must infer from proprioception
    (NOT privileged obs) so the skill survives distillation to the student.

No catastrophic forgetting: every challenge is present SIMULTANEOUSLY in the mix
(different envs draw different terrain rows / push / leg-fail per episode), so
the network holds all skills at once instead of overwriting the last one.

Same obs/action dims as every spot config ⇒ warm-start from the stage-2 best.pt:

    PYTHONPATH=. python -m omni_spot.train_pbt --robot spot_master \
        --init_ckpt omni_logs/<spot_parkour_robust_run>/best.pt --headless

Then distill this teacher into ONE deployable depth-camera student (DAgger) and
verify the whole repertoire on the test arena:

    PYTHONPATH=. python student_overview.py --ckpt <student>.pt \
        --num_envs 64 --steps 1500 --headless --enable_cameras

Expect success to dip when leg-failures switch on — that dip is the final skill
forming. If it collapses toward "stand still", lower recover_w (see spot_robust)
or leg_failure_prob (see spot_robust_legfail).
"""

from __future__ import annotations

from .spot_parkour_robust import make_cfg as _spot_parkour_robust_make_cfg
from .spot_robust_legfail import apply_leg_failure


def make_cfg():
    # spot_parkour_robust already has terrain (hard + parkour) + recovery on.
    cfg = _spot_parkour_robust_make_cfg()
    # Add the last skill: one-leg disable (dr leg-failure fields only).
    return apply_leg_failure(cfg)
