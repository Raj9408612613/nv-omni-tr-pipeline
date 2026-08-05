# nv-omni-tr — Teacher-Student Quadruped Navigation (Isaac Lab)

Two-phase RMA / Extreme-Parkour-style pipeline for goal navigation on a
Spot quadruped over a terrain curriculum with obstacle avoidance.

- **Phase 1 — privileged PPO teacher** (no rendering): asymmetric
  actor-critic over scandot heightfield raycasts + privileged observations
  (friction, payload, CoM offset, motor strength, foot contact forces),
  with the adaptation module φ trained concurrently (ROA) from
  proprio-action history. Scales to ~32k parallel envs.
- **Phase 2 — depth distillation student** (rendering-bound): the teacher's
  scandot encoder is replaced by a CNN+GRU over 87x58 depth at 10 Hz while
  the policy runs at 50 Hz; trained by DAgger (student drives the sim, the
  frozen teacher labels every visited state, MSE action loss). φ is reused
  frozen — only exteroception is distilled.

## Layout

```
omni_spot/
├── configs/            per-robot dataclass configs (spot.py); --robot <name>
├── obs.py              observation assembly (proprio/scandots/priv/history)
├── networks.py         Teacher/Student policies (cross-loadable by name)
├── checkpoint.py       save/load + teacher->student named cross-load
├── ppo.py              Phase 1 asymmetric PPO + concurrent phi regression
├── dagger.py           Phase 2 streaming DAgger trainer
├── env_cfg.py          runtime Isaac Lab cfg builder (camera gating here)
├── nav_env.py          shared DirectRLEnv (scandots, DR, curriculum, depth)
├── reward.py           goal-nav reward terms (weights from config)
├── mock_env.py         CPU mock with the same obs interface (no Isaac)
└── train.py            single entrypoint: --phase {teacher,student}
tests/                  CPU-only unit + convergence tests (no Isaac needed)
scripts/                smoke tests, EC2 setup, monitors
models/                 Spot MJCF + converted USD
```

## Running

```bash
# CPU verification (any machine with torch; no Isaac required)
pip install torch pytest   # CPU wheel is fine
pytest tests/ -v           # or: python tests/test_shapes.py  etc.

# Phase 1 smoke (Isaac Lab machine): 256 envs x 50 updates + checks
bash scripts/smoke_phase1.sh
bash scripts/smoke_phase1.sh --probe32k     # 32768-env VRAM probe

# Phase 1 full training
PYTHONPATH=. python -m omni_spot.train --phase teacher --robot spot --headless

# Phase 2 smoke: 16 envs x 200 iters (uses a random teacher if none given)
TEACHER_CKPT=omni_logs/<run>/best.pt bash scripts/smoke_phase2.sh

# Phase 2 full distillation
PYTHONPATH=. python -m omni_spot.train --phase student --robot spot \
    --teacher_ckpt omni_logs/<run>/best.pt --headless
```

Logs/checkpoints land in `omni_logs/<run_id>/` (CSV + TensorBoard,
`ckpt_*.pt`, `best.pt`, `final.pt`); a `SUCCESS` marker is written on clean
completion. `[VRAM]` lines report both torch-allocator and nvidia-smi usage
(PhysX/RTX memory is outside the torch allocator).

## Training plan — one all-skills teacher

The goal is **one** deployable policy that does all three: walks hard/parkour
terrain, gets back up instead of ending the episode when it falls, and keeps
going with one leg disabled. Every stage below is a full **PBT population run**
(`train_pbt`), each warm-started from the previous stage's best member via
`--init_ckpt` — so the skills accumulate in a single network instead of living
in three separate policies.

| stage | config | course | get-up | leg-fail | parkour |
|---|---|---|---|---|---|
| 1 | `spot_robust` | – | ✅ | – | – |
| 2 | `spot_parkour_robust` | – | ✅ | – | ✅ |
| 3 | `spot_master` | – | ✅ | ✅ | ✅ |
| 4 (polish) | `spot_master_course` | ✅ | ✅ | ✅ | ✅ |
| smoke | `spot_course` | ✅ | ✅ | – | – |

```bash
# Stage 1 — get-up recovery on the base curriculum (seed: any spot teacher)
PYTHONPATH=. python -m omni_spot.train_pbt --robot spot_robust \
    --init_ckpt teacher_newreward_20260616.pt --headless

# Stage 2 — add the parkour terrain columns
PYTHONPATH=. python -m omni_spot.train_pbt --robot spot_parkour_robust \
    --init_ckpt omni_logs/<stage1_run>/best.pt --headless

# Stage 3 — add per-episode one-leg failure (the investor-demo teacher)
PYTHONPATH=. python -m omni_spot.train_pbt --robot spot_master \
    --init_ckpt omni_logs/<stage2_run>/best.pt --headless

# Stage 4 (polish) — the same skills on shaped straight/L/T courses
PYTHONPATH=. python -m omni_spot.train_pbt --robot spot_master_course \
    --init_ckpt omni_logs/<stage3_run>/best.pt --headless

# Off-chain infra smoke — course terrain + carrot goals, few envs, minutes
PYTHONPATH=. python -m omni_spot.train_pbt --robot spot_course \
    --init_ckpt omni_logs/<stage1_run>/best.pt --headless
```

Distill the deployable student from stage 3 or 4's `best.pt` (`--phase student`).

Notes on the table:

- **Stage 3 is where the joint policy actually forms.** Every challenge is
  present simultaneously — different envs draw different terrain rows and
  conditions each episode — so nothing learned earlier is forgotten.
- **Stage 4's parkour ✅ is inherited, not resident.** Enabling `course` swaps
  the terrain builder (`_build_course_terrain`), so the parkour patch tiles are
  replaced by course cells and their proportions go inert; the parkour skill
  persists through the warm-start plus the courses' own stairs/rough segments.
  Keep this round short — it is polish, not the formation run.
- **`spot_course` is off-chain**, an infra test of the course terrain and carrot
  goal-planner. Run it before committing to stage 4, not as a chain link.
- Every stage keeps identical obs/action dims, so any stage's checkpoint
  warm-starts into any other.

Per-update console columns: `terr` (curriculum row, should climb), `fall`
(fraction of steps fallen — the get-up metric, should drop), `rec` (recovery
reward, ceiling = `recover_w`), `succ` (goals / episodes ended this rollout),
`best_fit` (best member's fitness; `-inf` until PBT warmup ends).

## Adding a robot

Create `omni_spot/configs/<name>.py` with `make_cfg() -> ExperimentCfg`
(see `spot.py`) — joint names/limits/default pose, body names, actuator
gains, reward weights, DR ranges, sensors. Then `--robot <name>`. No
training-code changes; dims are derived from the config.

## Environment setup (EC2 / Isaac Lab install)

See `scripts/setup_ec2_isaac.sh`, `scripts/verify_setup.sh`, and
`agentcontext1.md` for the verified install order (Python 3.11,
Isaac Sim 5.x pip packages, Isaac Lab from source).

Monitors: `pip install psutil`, then `python3 gpu-monitor.py` /
`python3 cpu-monitor.py`.
