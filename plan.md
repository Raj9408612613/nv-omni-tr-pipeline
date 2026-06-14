# Two-Phase Teacher-Student Pipeline (RMA / Extreme Parkour style) for omni_spot

## Context

The repo trains a Spot quadruped on a **goal-navigation task** in Isaac Lab (DirectRLEnv + custom pure-PyTorch PPO, no rsl_rl): random goal 1.5–3.5 m away, 2 kinematic box obstacles on flat terrain rows, 4×4 terrain curriculum grid (flat/rough/stairs), depth from 3 RTX cameras at 120×160. The user wants this rebuilt as a **two-phase teacher-student pipeline**:

- **Phase 1**: privileged PPO teacher — scandot heightfield raycasts instead of cameras (zero rendering → 32k envs feasible), privileged obs (friction, payload, CoM, motor strength, foot contact forces), asymmetric critic, concurrent ROA adaptation module φ, domain randomization + terrain curriculum promotion.
- **Phase 2**: depth-camera student — deepcopy of teacher actor with scandot encoder swapped for CNN+GRU over 87×58 depth at 10 Hz (policy at 50 Hz), trained by DAgger (student drives sim, frozen teacher labels every visited state, MSE action loss), φ reused frozen.

### User decisions (from Q&A)
1. **Keep the goal-navigation task** — "commands" in proprio = goal direction (heading frame) + distance, replacing the spec's velocity commands. Obstacles, goal rewards, terrain curriculum stay.
2. **Refactor in place** — teacher-student becomes THE pipeline; old depth-PPO flow retired; mock env rewritten to the new interface.
3. **Deliver-only verification** — this sandbox has no GPU/torch/Isaac. I write runnable test scripts + exact commands; user executes on the RTX 6000 Pro (96 GB) and reports output. I syntax-check (`py_compile`) everything here.
4. **Python dataclass configs** — per-robot config modules (`omni_spot/configs/spot.py`), `--robot` arg → importlib lookup. Second robot = one new config file, zero code branching.

### Conflicts & findings flagged (spec vs repo — verified first-hand)
- Existing obs (37-dim) uses **world-frame** quat/lin-vel/goal-dir — not yaw-invariant. New proprio follows the spec (joint pos/vel, base ang vel, projected gravity, last action) + heading-frame goal command; old world-frame obs deleted.
- **The current depth CNN never trains**: `inference_step` is `@torch.no_grad()`, CNN features are cached, PPO updates only `head_forward` (`spot_actor_critic.py:146`, `ppo.py:141-270`). Redesign fixes this structurally: Phase 1 stores tiny raw scandots so PPO gradients reach all encoders; Phase 2 trains the depth CNN via supervised DAgger loss.
- **Obstacles are kinematic rigid bodies, invisible to RayCaster** (it only raycasts the static terrain mesh). Resolution: `scandots = max(terrain_raycast_z, analytic_box_top)` composited in torch from known box poses — keeps teacher exteroception aligned with what the student's depth camera sees.
- **Latent terrain bug**: height reward/fall-termination compare absolute world Z to fixed constants — wrong on stairs. Fix: height relative to terrain under base (center scandot rays).
- **Joint/body naming risks** (from MJCF/USD inspection): lower-leg bodies are `fl_lleg`/`fr_lleg`/`hl_lleg`/`hr_lleg`; USD root link is likely `body` (camera prim path evidence) though MJCF says `base_link`. All names go in config and are runtime-verified against `robot.body_names`/`find_joints` with hard-fail diagnostics. Joint-order remap (config order ↔ sim order via `find_joints(..., preserve_order=True)`) fixes a latent positional-indexing bug.
- `--n_steps 2048` default is navigation-scale; Phase 1 uses locomotion-scale `n_steps=24` and `num_minibatches=4` (replaces fixed `MINIBATCH_SZ=512`, which would mean ~12k grad steps/update at 786k samples). Flagged as semantic change.
- "Motor strength" DR with ImplicitActuator = per-env kp/kv gain scaling (`write_joint_stiffness/damping_to_sim`) — the closest analog since implicit PD torque can't be intercepted. Documented in config.

---

## Design

### 1. File tree

```
omni_spot/
├── __init__.py                  NEW (empty — package was implicit)
├── configs/
│   ├── __init__.py              NEW — get_experiment_cfg(robot) via importlib; clear error listing available configs
│   ├── base.py                  NEW — all dataclass schemas (stdlib only; importable with CPU torch, no Isaac)
│   └── spot.py                  NEW — make_cfg() -> ExperimentCfg (all values migrated from config.py)
├── env_cfg.py                   REFACTOR of spot_env_cfg.py — build_env_cfg(exp_cfg, phase): constructs
│                                Sim/Scene/DirectRLEnvCfg programmatically from dataclasses; keeps the
│                                dual-import (isaaclab / omni.isaac.lab) + ImportError-stub pattern
├── nav_env.py                   REFACTOR of spot_env.py — NavEnv(DirectRLEnv), robot-agnostic
├── obs.py                       NEW — pure torch: quat/frame helpers, HistoryBuffer (ring), 
│                                compose_scandots (terrain+box), proprio/priv/critic-obs builders + all scales
├── networks.py                  REFACTOR of spot_actor_critic.py — Actor, ScandotEncoder, PrivEncoder,
│                                AdaptationModule, DepthGRUEncoder, Critic, TeacherPolicy, StudentPolicy
│                                (gaussian_log_prob/entropy kept verbatim)
├── checkpoint.py                NEW — save/load + load_teacher_into_student (named cross-load)
├── ppo.py                       REFACTOR — asymmetric PPO + concurrent φ regression; GAE, return-norm EMA,
│                                KL early stop, LR anneal, NaN/grad-skip guards reused verbatim
├── dagger.py                    NEW — Phase 2 DAgger trainer
├── train.py                     REFACTOR — single entrypoint --phase {teacher,student} --robot spot;
│                                AppLauncher-before-imports, SimpleLogger/_Phase heartbeat reused
├── reward.py                    LIGHT ADAPT — weights passed as RewardWeightsCfg; terrain-relative height fix
├── mock_env.py                  REWRITE — MockEnv with new obs dict (synthetic depth only if camera.enabled), CPU-OK
├── diagnostics.py               KEEP + add adapt/DAgger rows;  physics_tuning.py KEEP (gains move to RobotCfg);
├── video_recorder.py            KEEP
├── config.py / spot_env.py / spot_env_cfg.py / spot_actor_critic.py   DELETE (git mv where renamed)
tests/
├── test_shapes.py               NEW — CPU-only unit tests (all modules, cross-load, GRU persistence)
└── test_dagger_mock.py          NEW — CPU-only DAgger convergence on MockEnv
scripts/
├── smoke_phase1.sh              NEW (256-env smoke + --probe32k VRAM variant)
├── smoke_phase2.sh              NEW (16-env smoke)
├── smoke_test.sh                UPDATE — Level 1 = pytest tests/; Level 2 delegates to smoke_phase1.sh
└── smoke_test.py                DELETE (superseded)
```

### 2. Config schema (`configs/base.py`, plain `@dataclass`, nested via `field(default_factory=...)`)

- **RobotCfg**: name, usd_path, `joint_names` (canonical policy order: fl_hx…hr_kn), joint_lower/upper/vel_limits, default_joint_pos (=STANDING_POSE), action_scale (None → (upper−lower)/2, preserves current behavior), base_body_name="body", contact_body_regex=".*_lleg", actuator stiffness=500/damping=40/effort=1000, init_height=0.5, cam_mount_body.
- **RewardWeightsCfg**: exact current values (goal_bonus=200 … vel_track_cap=1.5).
- **ScandotsCfg**: grid_x=17, grid_y=11, spacing=0.1 (→ 187 points, RayCaster size=(1.6, 1.0)), forward_offset=0.0, height_clip=1.0, attach_yaw_only=True.
- **CameraRigCfg**: enabled=False (Phase 1), width=87, height=58, update_period_s=0.1, h_fov=87°, min/max_depth, data_type="distance_to_camera", `mounts: tuple[CameraMountCfg,...]` (default 1 front cam; the old 3-cam 120×160 rig stays expressible).
- **DRCfg**: enabled + per-item bools; friction (0.4, 1.25), payload_kg (−1, 5), com_offset_m (±0.05), motor_strength (0.8, 1.2) as kp/kv scale, push_interval_s=8, push_max_vel_xy=0.5.
- **CurriculumCfg**: enabled, promote_on_goal, demote_on_fall, demote_progress_frac=0.5.
- **TerrainCfg / ObstacleCfg / GoalCfg / SimCfg**: current hardcoded generator values, box sizes (humanoid slot deleted — disabled today), goal range, dt/decimation.
- **PolicyCfg**: history_len=50, z_dim=8, extero_latent=64, MLP widths, gru_hidden=128, depth_fc=128, log_std init/min/max, all obs scales. Derived: `proprio_dim=45` (3 ang_vel + 3 gravity + 3 goal_cmd + 12 jp + 12 jv + 12 last_action), `priv_dim=18` (1 friction + 1 payload + 3 com + 1 motor + 12 contact forces), `history_feat=57`.
- **TeacherTrainCfg**: num_envs=32768, n_steps=24, gamma/λ/clip/coefs as today, n_epochs=5, num_minibatches=4, target_kl=0.03, adaptation_lr=1e-3, adapt_every=1.
- **StudentTrainCfg**: num_envs=256, total_iters=20000, lr=5e-4, `distill_extero_only=False` (default trains the full student actor per spec; True = depth-encoder-only variant), action_noise_std=0.
- **ExperimentCfg**: aggregates all of the above; the only object train/env/networks consume.

### 3. Networks + checkpoint cross-load (the namespace contract)

```
TeacherPolicy                                      StudentPolicy
├── actor.proprio_mlp     MLP 45→256→128 ─────────── identical names/shapes → copied
├── actor.extero_encoder  ScandotEncoder 187→256→128→64   DepthGRUEncoder (cnn 5x5s2/3x3s2/3x3s2 →
│                                                      fc 3456→128 → GRUCell(128) → out →64); flatten
│                                                      size computed by dummy forward (rig-agnostic)
├── actor.trunk           MLP (128+64+8)=200→256→128 ── copied
├── actor.head            Linear 128→12 (mean clamp ±2) ── copied        actor.log_std ── copied
├── priv_encoder          MLP 18→64→32→8 = z_t        (teacher-only; loaded into frozen teacher in Phase 2)
├── adaptation_module     embed 57→32 → Conv1d(32,32,k8,s4)→(k5)→(k5): 50→11→7→3 → fc 96→8 = ẑ_t
│                                                  ─── copied then frozen in student
└── critic                MLP 253→512→256→128→1     (teacher-only; critic input = cat(proprio 45,
                                                     scandots 187, priv 18, base_lin_vel 3) — raw, asymmetric)
```

`checkpoint.load_teacher_into_student`: filter `model_state_dict` by prefixes `("actor.proprio_mlp.", "actor.trunk.", "actor.head.", "actor.log_std", "adaptation_module.")`, `load_state_dict(strict=False)`, **assert** missing ⊆ `actor.extero_encoder.*` and unexpected == []. Checkpoint format extends the existing one: `{model_state_dict, optimizer_state_dict, adapt_optimizer_state_dict, ret_mean, ret_std, phase, robot}`.

`DepthGRUEncoder.step(depth, new_frame_mask, reset_mask)`: owns `_hidden (B,128)` + `_e_cache (B,64)`; GRU ticks only where new_frame_mask, holds e_cache otherwise, zeroes both on reset; hidden detached after every optimizer-visible use (TBPTT = 1 frame tick — matches Extreme Parkour).

### 4. Phase 1 flow (per iteration)

Obs dict at 50 Hz: `{"proprio": (B,45), "scandots": (B,187), "priv": (B,18), "history": (B,50,57), "critic_extras": (B,3)=base lin vel}`. Proprio is yaw-invariant (heading-frame goal cmd, projected gravity).

Rollout (existing loop skeleton): per step, no_grad → e=extero(scandots), z=priv_encoder(priv), mean=actor(proprio,e,z), sample+clamp action, value=critic(cat) — store `proprio, scandots, priv, critic_extras, action, log_prob, value, reward, done` (~0.4 GB at 32k×24; full critic obs recomposed at minibatch time, not stored). **Concurrent φ regression** every `adapt_every` steps with a separate Adam: `L_adapt = ‖φ(history) − z.detach()‖²` (ROA; avoids storing 9 GB of history in the rollout buffer; z.detach() guarantees zero interference with PPO).

Update: GAE → return-norm EMA → KL-early-stopped epochs — all reused verbatim; minibatch forward re-encodes scandots/priv so **PPO gradients now reach the encoders** (unlike the old cached-CNN design). Losses/guards unchanged.

### 5. Phase 2 flow (DAgger, student drives)

Frozen teacher (full TeacherPolicy from `--teacher_ckpt`); student warm-started via cross-load; φ frozen. Optimizer over `student.actor.parameters()` by default (spec-faithful full-actor distillation; `distill_extero_only` flag for the conservative variant).

Per step: `e_s = depth_encoder.step(depth/max_depth, new_frame, reset_mask=prev_done)` (grad ON) → `ẑ = φ(history)` (frozen) → `a_s = actor(proprio, e_s, ẑ)` mean → teacher labels same state no_grad: `a_t = teacher_actor(proprio, e_t(scandots), z(priv))` mean → `L = MSE(a_s, a_t)` → backward + step → `env.step(a_s.detach())`. **Streaming on-policy updates, no aggregation buffer** (256 parallel envs already decorrelate the batch; matches Extreme Parkour/rsl_rl distillation; keeps TBPTT trivially correct). Identical ±2 mean clamp on both (same `Actor` code path); identical obs scales (single source: `obs.py` + PolicyCfg).

Frame cadence: TiledCamera `update_period=0.1` (10 Hz) vs 50 Hz control → `depth_new_frame` from the camera frame counter (fallback: step modulo), forced True on first post-reset obs. Env always builds scandots+priv+history (teacher needs them); depth keys appear only when `camera.enabled`.

### 6. Env changes (nav_env.py / env_cfg.py)

1. `build_env_cfg(exp_cfg, phase)` builds scene programmatically (generalizes the existing `_build_obstacle_cfgs` + setattr pattern); cameras attached **only if** `camera.enabled` → Phase 1 constructs zero camera prims; train.py sets `args.enable_cameras` programmatically only for `--phase student`.
2. Joint-order hardening: `find_joints(cfg.joint_names, preserve_order=True)` → `_cfg2sim` gather/scatter; policy I/O in config order, sim writes in sim order.
3. `HistoryBuffer(B, 50, 57)` ring; pushed in `_pre_physics_step` (history at t covers ≤ t−1); zeroed in `_reset_idx`.
4. RayCaster: `RayCasterCfg(prim_path=.../base_body, attach_yaw_only=True, GridPatternCfg(resolution=0.1, size=(1.6,1.0)), mesh_prim_paths=["/World/ground"])`; heights sanitized; **box compositing**: AABB test of grid xy vs known box poses → `eff_z = max(terrain_z, box_top)`; obs = `clamp(root_z − eff_z − target_height, ±clip)` (also gives terrain-relative base height for the reward/termination fix).
5. ContactSensor on `.*_lleg` (12-dim net forces → priv); init-time assert regex resolves to exactly 4 bodies, print `robot.body_names` on failure.
6. DR in `_reset_idx` (+ startup pass), sampled values cached → priv obs: friction via physx-view `get/set_material_properties` (CPU index tensors!), payload via `set_masses`, CoM via `set_coms`, motor strength via `write_joint_stiffness/damping_to_sim`, pushes via per-env countdown + `write_root_velocity_to_sim` (fallback `write_root_com_velocity_to_sim`).
7. Curriculum: `_get_dones` caches goal_reached/fallen; `_reset_idx` computes progress fraction → `terrain.update_env_origins(env_ids, move_up, move_down)` **before** sampling new pose. Existing flat-row obstacle gating unchanged.
8. Last-action fix: `_last_policy_action` (normalized) for proprio/history, separate from denormalized `_prev_ctrl` used by the smoothness reward.

### 7. Key risks (mitigations built into plan)

- **Isaac Lab API variance** (RayCaster/patterns import paths, TiledCamera data_type names, `write_root_*velocity` rename, physx-view CPU tensors): extend the repo's existing dual-import/fallback idiom; config-driven data_type.
- **USD body/joint names unverified until first sim run**: smoke_phase1 prints `robot.body_names`/`joint_names` and hard-fails with diagnostics if config doesn't resolve.
- **32k-env VRAM**: PhysX dominates (est. 20–40 GB with current buffer caps); rollout+history+φ ≈ 2 GB. `--probe32k` (3 updates, prints `max_memory_allocated`/`reserved`) answers empirically; fallback 16384.
- **DAgger distribution shift early on**: mitigated by warm-started trunk/head + φ + held 10 Hz latents.
- **Behavior change flags**: n_steps 2048→24, minibatch 512→count-4, n_epochs 8→5, sparse goal bonus vs short horizon (dense progress/vel-track terms remain) — reward weights may need retuning after first real runs.

### 8. Milestones (commit + verification gate each; Phase 1 fully done before Phase 2)

| M | Content | Gate |
|---|---|---|
| M1 | `configs/`, `obs.py`, `networks.py`, `checkpoint.py`, `tests/test_shapes.py` | `pytest tests/test_shapes.py` green on CPU torch (user runs; I py_compile here). Covers: every encoder shape w/ fabricated batches, φ conv arithmetic, GRU persistence/hold/reset, teacher→student cross-load assertions |
| M2 | `ppo.py` refactor, `reward.py` adapt, `mock_env.py` rewrite, mock 3-update PPO test (finite losses, adapt_loss ↓ on synthetic priv mapping) | pytest green |
| M3 | `env_cfg.py`, `nav_env.py`, `train.py --phase teacher`, `scripts/smoke_phase1.sh`, delete old files | **User runs** `bash scripts/smoke_phase1.sh`: 256 envs × 50 iters headless → SUCCESS marker, finite + non-flat rew_mean (CSV std > 0), name asserts pass, VRAM printed; then `--probe32k`. Commit "Phase 1 complete" |
| M4 | `dagger.py`, depth encoder wiring, `train.py --phase student`, `tests/test_dagger_mock.py` (loss[180:200] < 0.5 × loss[0:20]), `scripts/smoke_phase2.sh` | pytest green + **user runs** smoke_phase2: 16 envs, 200 iters → dagger_loss ↓, depth_new_frame_rate ≈ 0.2, VRAM printed |
| M5 | README, retire superseded scripts, diagnostics polish | py_compile + final review |

### 9. CLI (single train.py, AppLauncher-before-imports preserved)

```
python -m omni_spot.train --phase teacher --robot spot [--num_envs] [--n_steps] [--total_updates] [--lr] [--seed] [--resume] [--profile N] --headless
python -m omni_spot.train --phase student --robot spot --teacher_ckpt PATH [--num_envs 256] [--total_iters] [--lr] [--resume] --headless
```

CLI values override matching cfg fields when provided. `--teacher_ckpt` required iff student (checked before AppLauncher). SimpleLogger reused with per-phase CSV field tuples (teacher: current fields − cnn_feat_* + adapt_loss, terrain_level_mean; student: dagger_loss(+EMA), depth_new_frame_rate, action_gap p50/p95, sps, vram_peak). Checkpoint cadence/best.pt/SUCCESS marker unchanged.

## Verification summary (who runs what)

- Here (sandbox): `python -m py_compile` over all files at every milestone; plan/code review.
- User's RTX 6000 Pro: `pytest tests/` (CPU torch sufficient), `bash scripts/smoke_phase1.sh` (+ `--probe32k`), `bash scripts/smoke_phase2.sh` — each gate's expected output documented in the script headers; user reports output back before the next milestone proceeds.
- All work on branch `claude/fervent-hypatia-x0qysq`, milestone commits pushed.
