"""
Spot — PARKOUR terrain + RECOVERY (stage 2 of the investor chain)
=================================================================
This is the second link in the progressive warm-start chain toward the single
all-skills teacher:

    spot            (base, exists)
     └─ spot_robust          get-up / recovery on the CURRENT terrain
         └─ spot_parkour_robust   THIS: add hard + parkour terrain, KEEP recovery
             └─ spot_master        add one-leg failure (everything on)  ← FINAL

It COMPOSES two lineages that touch disjoint fields, so nothing conflicts:
  * terrain  — inherited from `spot_parkour` (which itself inherits `spot_hard`):
    taller stairs, rougher ground, rows=6, PLUS the parkour columns
    (discrete_obstacles / random_grid / rails).
  * recovery — `spot_robust.apply_recovery`: terminate_on_fall=False, the get-up
    gradient (recover_w / ang_vel_w / stronger upright_w), and pushes ON.

Because obs/action dims are identical to every other spot config, a `spot_robust`
best.pt (recovery already learned on easy terrain) or a `spot_parkour` best.pt
(parkour already learned) warm-starts straight in via --init_ckpt. Recommended:
seed from the recovery policy so the get-up skill is carried onto the harder
terrain rather than re-learned.

    PYTHONPATH=. python -m omni_spot.train_pbt --robot spot_parkour_robust \
        --init_ckpt omni_logs/<spot_robust_run>/best.pt --headless

Leg failure is deliberately OFF here — it is the last skill added, in
spot_master, so this stage can consolidate "recover on hard/parkour terrain"
before the three-legged challenge piles on.
"""

from __future__ import annotations

from .spot_parkour import make_cfg as _spot_parkour_make_cfg
from .spot_robust import apply_recovery


def make_cfg():
    # Start from the full terrain lineage (spot -> spot_hard -> spot_parkour):
    # hard stairs/rough + the parkour columns, curriculum by row.
    cfg = _spot_parkour_make_cfg()
    # Layer recovery (reward + dr only; terrain untouched). Leg failure stays OFF.
    return apply_recovery(cfg)
