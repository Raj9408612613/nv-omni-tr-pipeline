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

from .spot import make_cfg as _spot_make_cfg


def make_cfg():
    cfg = _spot_make_cfg()

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
    # Leg failure stays OFF this round (turned on in spot_robust_legfail).

    # ── PBT searches the get-up gradient ──────────────────────────────────
    # recover_w=0.5 plateaued (fall ~0.20, terr ~1.9 for 300+ updates); let
    # the population find the weight. Range spans "gentle" to "3x the hand
    # value"; fitness is weight-free so selection stays uncontaminated.
    cfg.pbt.recover_w_range = (0.15, 1.5)

    # ── Rudin-style traversal: longer trips on bigger patches ─────────────
    # legged_gym trains on 8 m patches but robots walk continuously for 20 s
    # (~10+ m/episode); our goal-terminated episodes were ~2.5 m / 2.5 s.
    # Bigger patches + farther goals restore that per-episode exposure.
    g = cfg.goal
    g.dist_range = (3.0, 7.0)      # was (1.5, 3.5)
    g.episode_len_steps = 1200     # 24 s: 7 m over stairs + get-ups fits
    # Fixed start at the patch CENTER. Spawn z is env_origin.z + init_height
    # and origin z is Isaac's max over a 2 m box at that centre — on the 3 m
    # stair platform that IS the terrain height there, so the robot is placed
    # standing on the surface. Any spread would put spawns off the platform on
    # 16 m stair rows, dropping (pyramid up) or burying (pyramid down) it by
    # up to the apex height.
    g.spawn_half = 0.0             # fixed start, standing on the platform

    t = cfg.terrain
    t.patch_size = 16.0            # was 8.0
    t.patch_half = 7.5             # usable half-width inside the patch
    # Stairs must NOT scale with the patch: at 16 m the slope run doubles
    # (2.5 m -> 6.5 m), and with the old 0.30 m treads the pyramid apex
    # would reach ~5 m — untraversable and a huge spawn/goal height gap.
    # Doubling the tread and trimming max step height keeps the apex at
    # ~2.2 m (vs ~1.8 m today) while per-step difficulty stays comparable.
    t.stair_step_width = 0.60      # was 0.30
    t.stair_step_height_range = (0.05, 0.20)  # was (0.05, 0.23)

    return cfg
