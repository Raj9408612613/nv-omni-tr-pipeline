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
