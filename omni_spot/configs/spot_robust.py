"""
Spot — ROUND 1 ROBUSTNESS (rebalance instead of terminate / full get-up)
========================================================================
Same robot / embodiment / networks as `spot` (identical obs/action dims), so a
`spot` teacher (or PBT best.pt) warm-starts straight in via --init_ckpt. This
config does NOT change the terrain (staged: terrain difficulty comes later) —
it changes how FALLING is handled:

  * terminate_on_fall = False  -> a fall no longer ends the episode; the robot
    must get back up and keep going (full get-up).
  * recover_w / ang_vel_w      -> a get-up gradient (reward uprightness) plus a
    penalty on the base angular velocity that precedes a tip-over.
  * stronger upright penalty    -> proactively stay balanced.
  * pushes ON                   -> it must reject perturbations and recover.

Only the fall handling + recovery shaping change. Nice synergy with the
curriculum: once a stumble no longer counts as a terminal fall, envs stop being
demoted on every wobble, so they actually climb to the harder terrain rows.

    PYTHONPATH=. python -m omni_spot.train_pbt --robot spot_robust \
        --init_ckpt omni_logs/<spot_run>/best.pt --headless

Round 2 adds the one-leg-disable skill on top — see spot_robust_legfail.py.
These recovery weights are sensible starting points; PBT can search them.
"""

from __future__ import annotations

from .base import ExperimentCfg
from .spot import make_cfg as _spot_make_cfg


def apply_recovery(cfg: ExperimentCfg) -> ExperimentCfg:
    """Turn any config into a full-get-up / recovery config (in place).

    Edits ONLY `cfg.reward` and `cfg.dr` (fall handling + recovery shaping +
    pushes); it never touches terrain, so it composes cleanly with the terrain
    lineage (spot_hard / spot_parkour). This is the single source of truth for
    the recovery knobs — spot_parkour_robust and spot_master reuse it so the
    values can never drift out of sync. Leg failure is left OFF; layer it on
    with `spot_robust_legfail.apply_leg_failure` for the one-leg-disable skill.
    """
    r = cfg.reward
    # Full get-up: a fall does not end the episode.
    r.terminate_on_fall = False
    # Get-up gradient + tipping damping.
    r.recover_w = 0.5          # reward = recover_w * cos(tilt): + when upright
    r.ang_vel_w = -0.02        # penalize base angular velocity (anti-tip)
    r.upright_w = -0.5         # was -0.3: stronger proactive balance

    d = cfg.dr
    d.enabled = True
    d.push_robots = True       # must recover from perturbations
    # Leg failure stays OFF here (turned on in spot_robust_legfail / spot_master).

    return cfg


def make_cfg():
    return apply_recovery(_spot_make_cfg())
