# nv-omni-tr-pipeline — Agent Handoff Context

Paste this into a new agent to continue. It captures the full state as of the
last session. Read it fully before acting.

---

## ⭐ PRIMARY OBJECTIVE (read first — everything serves this)
Produce **ONE deployable policy that demonstrably does ALL of it** — walks
harder/parkour terrain, **recovers instead of falling** (full get-up), and keeps
going **with one leg disabled** — to show investors the robot's current
capabilities. The teacher must learn all of this in a single network, then be
**distilled into one student** (depth-camera, deployable) that shows the whole
repertoire.

Concretely the end state is the **`spot_master`** policy (all knobs on:
parkour terrain + get-up recovery + one-leg failure), reached via a progressive
warm-start chain, then distilled.

**How every training round runs: PBT (Population-Based Training).** Each round
below is NOT a single run — it is a **full PBT population run** launched with
`python -m omni_spot.train_pbt --robot <config> [--init_ckpt <prev best.pt>]`.
PBT trains a population of members in parallel and **evolves their
hyperparameters** during the run (the search space is `PBTCfg` in `base.py`; the
per-env reward-weight override in `nav_env` is how PBT applies per-member reward
knobs). `--init_ckpt` **warm-starts the ENTIRE population** from the previous
round's best member. `best.pt` = the population's best member, which seeds the
next round. So "the warm-start chain" = a sequence of PBT runs, each seeded by
the prior run's best.

Treat individual rounds (`spot_hard`,
`spot_parkour`, `spot_robust`, `spot_robust_legfail`) as **steps toward this one
combined policy**, not as separate deliverables. Bias every decision toward
shipping that single all-skills policy + its student. See §7 and §11.

Recommended order of training. 
spot                     (base, exists)
 └─ spot_robust          recovery / get-up on CURRENT terrain   ← do this FIRST
     └─ spot_parkour_robust   add hard + parkour terrain, KEEP recovery
         └─ spot_master        add one-leg failure (everything on)  ← FINAL
              └─ distill → one student for the investor demo
---

## 0. TL;DR
Spot quadruped **goal-navigation over varied terrain**, Isaac Lab, teacher→student
(PPO teacher with scandots → DAgger-distilled depth-camera student), trained with
**PBT**. Current focus: make the policy **robust** (recover instead of fall, walk on a
disabled leg) and capable on **harder/parkour terrain**, then distill a student that
demonstrates **all** capabilities in one policy (for investors).

Working branch: **`claude/quirky-dirac-9hwjld`** (repo `Raj9408612613/nv-omni-tr-pipeline`).
Develop, commit, and push there. The remote moves (other sessions/PR merges) — always
`git fetch origin <branch> && git rebase origin/<branch>` before pushing; push with
`git push -u origin claude/quirky-dirac-9hwjld` and retry on network errors.

---

## 1. Where things run (CRITICAL)
- **Training/sim runs on the USER's EC2 box** (`ubuntu@...`, `isaac` conda env,
  IsaacLab at `/home/ubuntu/IsaacLab/...`). The user runs commands there.
- **The agent's container only has the cloned repo** — **NO Isaac Lab, no `pxr`/USD,
  no GPU.** You CANNOT import `isaaclab` or run the sim here. `find / -name hf_terrains.py`
  returns nothing; `/home/ubuntu/IsaacLab` is not mounted.
- To inspect IsaacLab internals, **ask the user to paste `grep`/`sed` output** from their
  box (this worked well twice — for terrain classes and for the hf terrain API).
- A plain `python -c "import isaaclab..."` on their box fails with `No module named 'pxr'`
  unless run via the Isaac launcher: `cd /home/ubuntu/IsaacLab && ./isaaclab.sh -p -c "..."`.

### What you CAN validate in the agent container
- `python3 -m py_compile <file>` (syntax).
- Import configs: `omni_spot.configs` is **pure dataclasses** (no torch/Isaac) → loads fine.
  `get_experiment_cfg("<robot>")` works; `available_robots()` lists configs.
- Importing `omni_spot.env_cfg` works with `HAS_ISAAC=False` (Isaac branch not executed).
- **`reward.py` / `obs.py` are pure torch** → you CAN unit-test `compute_reward` on CPU
  with mock tensors (did this for the recovery reward).
- You CANNOT exercise `nav_env` sim paths, terrain generation, or networks-on-GPU.

---

## 2. Repo map (key files)
```
omni_spot/
  configs/
    base.py        # ALL dataclasses: RobotCfg, RewardWeightsCfg, GoalCfg, ObstacleCfg,
                   #   TerrainCfg, CurriculumCfg, ScandotsCfg, CameraMountCfg/CameraRigCfg,
                   #   DRCfg, SimCfg, PolicyDims, PBTCfg, ExperimentCfg
    __init__.py    # registry: any module with make_cfg() -> ExperimentCfg is auto-found;
                   #   select with --robot <module_name>
    spot.py        # base Spot teacher config
    spot_hard.py   # harder terrain (this session)
    spot_parkour.py# parkour terrain (this session)
    spot_robust.py # Round-1 robustness / get-up (this session)
    spot_robust_legfail.py # Round-2 one-leg disable (this session)
  env_cfg.py       # build_env_cfg(): SimCfg/SceneCfg/DirectRLEnvCfg from ExperimentCfg.
                   #   _build_terrain (curriculum grid generator), _build_robot,
                   #   _build_obstacles, _build_height_scanner, _build_cameras.
                   #   Dual import: isaaclab.* (2.x) with omni.isaac.lab.* (1.x) fallback.
                   #   Flags: HAS_ISAAC, HAS_TERRAIN, HAS_PARKOUR_TERRAIN.
  nav_env.py       # NavEnv(DirectRLEnv). _reset_idx (spawn+goal+curriculum+obstacles+DR),
                   #   _get_dones (check_termination), _get_rewards (compute_reward),
                   #   _apply_dr (friction/payload/com/motor/leg-fail), _apply_pushes,
                   #   _update_height_scan (compose_scandots).
  reward.py        # compute_reward(...), check_termination(...)
  obs.py           # compose_scandots(terrain raycast + analytic box compositing), build_critic_obs
  networks.py      # DepthGRUEncoder (CNN!), StudentPolicy, teacher actor/critic, ScandotEncoder
  ppo.py, pbt.py, pbt_vmap.py, train.py, train_pbt.py, dagger.py, checkpoint.py, mock_env.py
  diagnostics.py, video_recorder.py, physics_tuning.py
student_overview.py        # TWO-COURSE student test arena (carrot waypoints, green spheres, metrics)
student_test-hard.py       # harder test-course runner (from brave-edison)
student_track.py           # chase/tracking viz (from brave-edison)
teacher_train_overview.py  # teacher curriculum overview: 1 robot/patch, 32x32 patches, polished cam
scripts/isaac_run.sh       # setup+run (GPU-arch-aware torch install), install_isaaclab.sh
tests/                     # test_shapes.py, test_pbt_mock.py, test_pbt_reward.py
```

---

## 3. Architecture facts (the small details)
- **Control:** physics 200 Hz (`physics_dt=0.005`), `decimation=4` → policy at **50 Hz**
  (`control_dt=0.02 s`). `episode_len_steps` default **1000** → 20 s episode.
- **Actuators:** ImplicitActuator (position-target PD), `actuator_stiffness=500`,
  `actuator_damping=40`, `effort_limit=1000`. Action = `default_jp + action*action_scale`,
  `action_scale` = half joint range by default.
- **Joints (policy order, spot.py):** `fl_hx,fl_hy,fl_kn, fr_hx,fr_hy,fr_kn, hl_*, hr_*`
  → **legs are contiguous blocks of 3** in policy order (used by the leg-failure code:
  `jpl = action_dim // num_feet = 12//4 = 3`). `self._joint_ids` remaps policy→sim order.
- **Scandots (teacher exteroception):** GridPattern raycast, **17×11 = 187 pts @ 0.1 m**,
  `height_clip=1.0`. `obs.compose_scandots` = terrain raycast Z **+ analytic box-obstacle
  heights composited in torch** (raycaster only sees static terrain mesh).
- **Depth camera (student):** default **87×58**, `max_depth=10`, single forward cam.
  **THE DEPTH CNN IS HARD-WIRED TO 87×58** (`networks.DepthGRUEncoder`: the `fc` input size
  is measured by a dummy forward at `cam.height×cam.width` at construction). **A trained
  student checkpoint only loads at 87×58.** Higher-res depth ⇒ must **re-distill** a new
  student; you cannot just bump the cfg (load_state_dict shape mismatch). For tests, render
  high-res then downsample to 87×58 (the merged `student_overview.py` does this).
- **Base height is terrain-relative** (from the scandot raycast), NOT world Z — so fall
  detection / height reward work correctly on stairs/parkour. `robot.data.root_ang_vel_b`
  exists and is used (obs + new reward term).
- **Curriculum:** Isaac TerrainGenerator grid, `curriculum=True`, difficulty rises by ROW.
  Per-env promote/demote at reset (`CurriculumCfg`, Rudin-style with a "stay band":
  promote on goal or progress≥0.8; demote on fall or progress<0.25).

---

## 4. KEY FINDINGS from the test video (teacher policy)
Analyzed a 32 s overview MP4 (training success ~98.5%, but ~**60%** on the test arena):
1. **Dominant failure = TIP-OVERS on open/rough terrain**, NOT terrain it can't traverse.
   Robots reach the open expanse and fall on their sides/backs in large numbers.
2. **Stairs are mostly fine** — most robots climb the pyramids. Stairs are NOT the 40% loss.
3. **Tall box obstacles (0.8–1.0 m, taller than Spot; `ObstacleCfg.half_sizes` z=0.4/0.5)
   defeat it** — robots clamber onto / wedge against / get pinned behind them. Cause: local
   scandot patch only (~1.6×1.0 m), no global planning; reward pulls straight at the goal.
4. **Root cause of falls:** gait tuned to **rush** — `progress_w=50` (note: progress is
   DROPPED from the total; `vel_track_w=1.5` cap 1.5 m/s is the driver) + `goal_bonus=25`,
   with weak stability: `upright_w=-0.3`, `alive_bonus=0.05` (deliberately slashed),
   `fall_tilt_rad=π/3 (60°)` is permissive. Likely the curriculum also collapsed toward flat
   (demote-on-every-wobble), inflating the training number.
⇒ This justified the **axis-3 robustness work** (recover instead of terminate) as the
   highest-ROI fix, plus axis-1 terrain for PBT headroom. Axis-2 (longer nav goals) is fine
   already and was skipped.

---

## 5. WHAT WAS BUILT THIS SESSION (all pushed to `claude/quirky-dirac-9hwjld`)

### Configs are the SAME policy/network — they only change the training DISTRIBUTION.
They COMPOSE because they touch disjoint fields:
- terrain lineage (`spot_hard`, `spot_parkour`) edits **`cfg.terrain`** only.
- robustness lineage (`spot_robust`, `spot_robust_legfail`) edits **`cfg.reward` + `cfg.dr`** only.
Obs/action dims are identical across all of them ⇒ any best.pt **warm-starts** into any other
via `--init_ckpt`.

### `spot_hard.py` (terrain difficulty; warm-start from spot best)
- `stair_step_height_range=(0.10,0.30)` (was (0.05,0.23)), `rough_noise_range=(0.05,0.18)`.
- proportions: flat 0.20 / rough 0.30 / stairs_up 0.25 / stairs_down 0.25.
- `rows=6` (was 4). `flat_row_max` stays 1.

### `spot_parkour.py` (inherits spot_hard; warm-start from spot_hard best)
- Enables 3 parkour tiles; proportions sum to 1.0 over 7 active sub-terrains:
  flat .14 / rough .14 / stairs_up .14 / stairs_down .14 / **discrete_obstacles .14 /
  random_grid .15 / rails .15**. `cols=14` (so each tile gets ≥1 curriculum column).
- `discrete_obstacle_height_range=(0.05,0.18)`, `discrete_obstacle_num=10`,
  `random_grid_height_range=(0.02,0.12)`, `rail_height_range=(0.05,0.16)`,
  `parkour_platform_width=2.0`. Gaps/pits/tall-boxes/stepping-stones left OFF (too hard for
  position-controlled Spot) — fields exist in TerrainCfg (default 0.0) for a future
  `spot_parkour_hard`.

### Parkour terrain support (infra)
- `base.py TerrainCfg`: added parkour fields (all proportions default **0.0** ⇒ inert for
  spot/spot_hard) + `color_scheme: str = "none"` (from the brave-edison merge).
- `env_cfg.py _build_terrain`: **separately-guarded** parkour imports (`HAS_PARKOUR_TERRAIN`)
  so a missing class can't disable core terrain; builds `sub_terrains` dict, appends a parkour
  tile only when `HAS_PARKOUR_TERRAIN and proportion>0`; warns if parkour requested but
  unavailable, and warns if `cols < len(sub_terrains)` (curriculum would silently drop types).
  Then builds `terrain_gen` var + optional `color_scheme` (merged with brave-edison's refactor).
- Classes confirmed present in the user's IsaacLab: `HfDiscreteObstaclesTerrainCfg`,
  `HfSteppingStonesTerrainCfg`, `MeshRailsTerrainCfg`, `MeshRandomGridTerrainCfg` (+ Hf/Mesh
  pyramid/stairs/gap/pit/box/floating-ring/star/repeated-objects).

### Robustness (Round 1 + Round 2) — all backward-compatible (new knobs default OFF)
`reward.py compute_reward`:
- new optional kwarg `root_ang_vel: Tensor|None = None` (optional so `mock_env.py` + tests,
  which call via `**inp`, don't break).
- `r_recover = cos_tilt * w.recover_w` — `cos_tilt = 1 - 2(qx²+qy²)` = **+1 upright, 0 on side,
  −1 inverted**. With `recover_w>0` this is the **get-up gradient**.
- `r_ang_vel = clamp(sum(root_ang_vel²)*w.ang_vel_w, -2, 0)` — anti-tip penalty (0 if kwarg None).
- both added to `total` and to `info`.
NOTE the sign subtlety: `upright_w` is **negative** because it multiplies `tilt_rad`
(0=upright, grows when fallen) → penalty. `recover_w` is **positive** because it multiplies
`cos_tilt` (+1 upright). Both push toward upright. Do NOT flip `upright_w` positive.

`base.py RewardWeightsCfg` new fields: `terminate_on_fall=True`, `recover_w=0.0`, `ang_vel_w=0.0`.
`base.py DRCfg` new fields: `randomize_leg_failure=False`, `leg_failure_prob=0.0`,
`leg_failure_strength=0.0`.

`nav_env.py`:
- `_get_dones`: `terminated = at_goal | collided; if x.reward.terminate_on_fall: |= fallen`.
  So `terminate_on_fall=False` ⇒ a fall no longer ends the episode (full get-up). `_fallen` is
  still tracked for the alive-gating + curriculum demote signal.
- `_get_rewards`: passes `root_ang_vel=robot.data.root_ang_vel_b`.
- `_apply_dr`: motor block now runs if `randomize_motor_strength OR randomize_leg_failure`.
  Builds per-(env,joint) `scale_mat`; for the failed subset (`rand<leg_failure_prob`) it sets
  ONE random leg's joints to `leg_failure_strength` (vectorized via `joint_leg = arange(J)//jpl`).
  Writes kp/kv = `actuator_stiffness/damping * scale_mat`. Leg failure is **NOT** put in
  `self._motor` (privileged obs) on purpose — the policy must infer it (so it survives
  distillation to the student).

`spot_robust.py` (Round 1, inherits **spot** = current terrain):
- `reward.terminate_on_fall=False`, `recover_w=0.5`, `ang_vel_w=-0.02`, `upright_w=-0.5`
  (was -0.3). `dr.enabled=True`, `dr.push_robots=True`. Leg-failure OFF.

`spot_robust_legfail.py` (Round 2, inherits **spot_robust**):
- `dr.randomize_leg_failure=True`, `leg_failure_prob=0.15`, `leg_failure_strength=0.0` (limp).

### Reward CPU-validation done (passes): backward-compat (new terms 0 when off),
get-up gradient (upright > on-side > inverted), ang-vel penalty fires only when rotating.

---

## 6. THE brave-edison MERGE (already done, commit `ff0bd33`)
Merged `claude/brave-edison-8btmmy` (test tooling) INTO `quirky-dirac`. Resolutions:
- **`student_overview.py`** → took **brave-edison's** entirely (superset of mine: two-course
  arena + **moving "carrot" waypoint** [fixes my OOD bug of feeding a 25 m goal to a policy
  trained on 1.5–3.5 m] + green goal spheres + per-course metrics + plinth/pit stairs-down +
  depth render-then-downsample). My version discarded.
- **`env_cfg.py`, `base.py`** → hand-merged to keep BOTH parkour AND `color_scheme`.
- **`scripts/isaac_run.sh`** → kept their GPU-arch-aware `torch_arch_ok` (Stage 2) + my
  `ensure_torch` Stage-4 re-verify.
- **`teacher_overview.py`** → DELETED (superseded by `teacher_train_overview.py`:
  1 robot/patch, 32×32 m patches, polished cam). The duplicate `teacher_train-overview.py`
  (hyphen) was also deleted (exact copy).
- Preserved (mine): `spot_hard`, `spot_parkour`, parkour support, `pbt.py`/`train_pbt.py`
  (PBTCfg work), `install_isaaclab.sh`, `tests/test_pbt_mock.py`.
- Added (theirs): `student_test-hard.py`, `student_track.py`, `teacher_train_overview.py`.

---

## 7. STRATEGY & DECISIONS
- **Axes:** Axis-1 (terrain difficulty) + Axis-3 (robustness: recover-not-terminate +
  one-leg disable). **Skip Axis-2** (longer nav goals — already fine).
- **One policy that does EVERYTHING** (the investor goal): all configs are the same network;
  reach an all-skills policy via a **progressive warm-start chain of PBT runs** (each stage is
  a full `train_pbt` population run, `--init_ckpt`-seeded from the prior stage's best.pt, and
  keeps prior challenges while adding one), final config has all knobs on:
  `spot → spot_hard → spot_parkour → +recovery → +leg-fail`.
  Catastrophic-forgetting is avoided because the FINAL combined config keeps ALL terrain
  types in the mix WHILE adding recovery+leg-failure (everything present simultaneously;
  different envs get different conditions per episode).
- **Recommended next configs to BUILD (not done yet):**
  - `spot_parkour_robust` = spot_parkour terrain + spot_robust reward/dr (leg-fail off).
  - `spot_master` = everything on (parkour terrain + get-up + leg-fail). Final = investor demo.
  Composition is trivial (disjoint fields): start from one lineage's make_cfg(), then apply the
  other lineage's reward/dr (or terrain) edits.
- **PBT knobs:** open question — whether to add `recover_w`/`ang_vel_w` to `PBTCfg` search space
  so the population tunes them, or leave fixed for the first run.

---

## 8. DISCUSSED, NOT BUILT — terrain-shape redesign + path following
User wants training terrain that is **not one square box** but **elongated/shaped arenas**
(straight corridor of fixed width with terrain changing along the length; **L- or T-shaped**
so the robot must change direction midway), goal at the far end, **random spawn + random
spawn-end** (so it can't memorize "move forward").
- Confirmed: training ALREADY randomizes spawn yaw (`uniform(0,2π)`) and goal bearing — the
  policy is a true goal-seeker, not a forward-walker. (The straight-facing was test-only.)
- **Spot's policy is a reactive short-horizon goal-seeker** — it does NOT plan routes or follow
  geometric curves. To traverse an L/T/curve, feed it **WAYPOINTS (checkpoints)**: goal = next
  waypoint (one at each bend), advance on arrival; curve = dense waypoints. **The carrot/waypoint
  mechanism already exists** in `student_overview.py`, and path-following needs **NO retraining**
  (a waypoint is just a goal in the trained 1.5–3.5 m range).
- **Scan-and-plan** (robot perceives layout + plans its own route) is only needed for genuine
  route CHOICE (real T-junction) or unknown maps — deferred (medium/high effort: A* costmap →
  waypoints, or hierarchical RL).
- Building shaped arenas = **custom heightfield terrain** (same `@height_field_to_mesh`
  technique as the test course) + waypoint goals in nav_env. For TRAINING (not a demo),
  **randomize the layout per env** (segment order, bend direction, spawn end) for generalization.
- The hf terrain function contract (confirmed from the user's IsaacLab): a function decorated
  with `@height_field_to_mesh` (from `isaaclab.terrains.height_field.utils`) takes
  `(difficulty, cfg)` and returns an **int16 heightfield in units of `cfg.vertical_scale`**,
  shape `(x_px, y_px)` where `x_px=int(size[0]/horizontal_scale)`. The decorator builds the
  trimesh + origin.

---

## 9. GOTCHAS / not-yet-runtime-tested
- Everything new was validated only via **py_compile + config import + CPU reward math**.
  The **Isaac sim paths were NOT run** (no GPU/Isaac in the agent container). Likely first-run
  tweak spots if errors appear:
  - the custom hf terrain function `@height_field_to_mesh` convention (import path / dtype /
    array orientation) — version-sensitive.
  - the new `nav_env` leg-failure kp/kv write + `terminate_on_fall` path.
  Ask the user to paste any traceback; fixes are usually 1-liners.
- `student_overview.py` long-course note: depth stays 87×58 (CNN locked); the script renders
  higher-res then downsamples. Higher-res perception requires a re-distilled student.
- Leg-failure assumes joints are grouped by leg in policy order (true for Spot).

---

## 10. WARM-START CHAIN & COMMANDS (run on the EC2 box, `isaac` env, repo root)
```bash
# Round-1 robustness (get-up), warm-started from the current spot teacher best:
PYTHONPATH=. python -m omni_spot.train_pbt --robot spot_robust \
    --init_ckpt omni_logs/<spot_run>/best.pt --headless

# Round-2 (add one-leg disable), warm-started from the spot_robust best:
PYTHONPATH=. python -m omni_spot.train_pbt --robot spot_robust_legfail \
    --init_ckpt omni_logs/<spot_robust_run>/best.pt --headless

# Terrain headroom (independent track), warm-started from spot best:
PYTHONPATH=. python -m omni_spot.train_pbt --robot spot_hard \
    --init_ckpt omni_logs/<spot_run>/best.pt --headless
PYTHONPATH=. python -m omni_spot.train_pbt --robot spot_parkour \
    --init_ckpt omni_logs/<spot_hard_run>/best.pt --headless

# Test arena (student) — two designed courses, green-sphere goals, per-course metrics:
PYTHONPATH=. python student_overview.py --ckpt <student>.pt \
    --num_envs 64 --steps 1500 --headless --enable_cameras

# Distill the student (DAgger) once the robust teacher is ready (see dagger.py / train.py).
```
What to watch in robustness runs: `r_recover` up, falls→recoveries, terrain levels climbing
(fewer fall-demotions). Guard against a "stand still" local optimum (watch `r_vel_track`
stays positive); if it stalls, lower `recover_w`. Round-2 success dips when leg-failures
switch on — that's the new skill forming.

---

## 11. IMMEDIATE NEXT TASKS (suggested order)
1. (Optional) Build combined configs: `spot_parkour_robust`, `spot_master` (all-skills).
2. Decide PBT search space: add `recover_w`/`ang_vel_w` to `PBTCfg` or leave fixed.
3. If pursuing shaped arenas: spec + build a randomized corridor/L/T custom-heightfield
   terrain + waypoint goals in nav_env (keep per-env randomization for generalization).
4. Kick off Round-1 (`spot_robust`) on the box; iterate on any runtime traceback.
5. After robust teacher converges → distill student → re-run `student_overview.py` to verify.

## Conventions
- Commit trailers used this session:
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` and a `Claude-Session:` line.
- Do NOT put the raw model id in repo artifacts. Do NOT open PRs unless asked.
- Branch policy: develop/commit/push on `claude/quirky-dirac-9hwjld`; fetch+rebase before push.
