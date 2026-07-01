"""
Spot — ROUND 2 ROBUSTNESS (one-leg disable, on top of get-up)
=============================================================
Inherits spot_robust (full get-up + recovery shaping + pushes) and ADDS a
per-episode actuator failure: with leg_failure_prob, ONE randomly chosen leg's
joints go limp (kp/kv -> leg_failure_strength), so the policy must walk and
recover on three working legs. The failure is deliberately NOT in the
privileged obs — the policy must infer it from proprioception, so the skill
survives distillation to the deployable student.

Warm-start from a converged spot_robust run:

    PYTHONPATH=. python -m omni_spot.train_pbt --robot spot_robust_legfail \
        --init_ckpt omni_logs/<spot_robust_run>/best.pt --headless

Expect success to dip when the failures switch on — that dip is the new skill.
Tune leg_failure_prob / leg_failure_strength to trade difficulty vs. stability.
"""

from __future__ import annotations

from .base import ExperimentCfg
from .spot_robust import make_cfg as _spot_robust_make_cfg


def apply_leg_failure(cfg: ExperimentCfg) -> ExperimentCfg:
    """Add the one-leg-disable failure to any config (in place).

    Edits ONLY the leg-failure fields of `cfg.dr`, so it layers on top of a
    recovery config (spot_robust) or a terrain+recovery config
    (spot_parkour_robust) without disturbing anything else. Single source of
    truth for the leg-failure knobs — spot_master reuses it. Assumes recovery
    is already applied (a limp leg only makes sense once falls don't terminate).
    """
    d = cfg.dr
    d.randomize_leg_failure = True
    d.leg_failure_prob = 0.15        # ~15% of episodes lose a leg
    d.leg_failure_strength = 0.0     # fully limp (0 = no torque on that leg)

    return cfg


def make_cfg():
    return apply_leg_failure(_spot_robust_make_cfg())
