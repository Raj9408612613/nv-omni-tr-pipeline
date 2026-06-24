"""
Configuration Schemas — Teacher-Student Pipeline
=================================================
Plain stdlib dataclasses (NO torch, NO Isaac Lab imports) so configs are
importable everywhere: CPU unit tests, mock env, and the real sim.

All robot-specific values live in per-robot config modules (see spot.py).
Training code must never branch on robot name — it only reads these fields.
Adding a second robot = adding one new config module with make_cfg().
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ════════════════════════════════════════════════════════════════════════════
# Robot
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class RobotCfg:
    """Everything specific to one robot model."""
    name: str = "robot"
    usd_path: str = ""
    # Canonical POLICY-side joint order. Sim joint order may differ (USD
    # traversal); the env builds an index remap via find_joints() at startup.
    joint_names: tuple[str, ...] = ()
    joint_lower: tuple[float, ...] = ()
    joint_upper: tuple[float, ...] = ()
    joint_vel_limits: tuple[float, ...] = ()
    default_joint_pos: tuple[float, ...] = ()
    # None => half joint range per joint: (upper - lower) / 2
    # (preserves the original env behavior: target = default + action * half_range)
    action_scale: tuple[float, ...] | None = None
    # Body names — verified at runtime against robot.body_names (hard fail
    # with the discovered name list if they don't resolve).
    base_body_name: str = "body"
    contact_body_regex: str = ".*_lleg"
    num_feet: int = 4
    cam_mount_body: str = "body"
    # Actuators (ImplicitActuatorCfg)
    actuator_stiffness: float = 500.0
    actuator_damping: float = 40.0
    effort_limit: float = 1000.0
    # Spawn
    init_height: float = 0.5
    mass_kg: float = 0.0  # informational

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)


# ════════════════════════════════════════════════════════════════════════════
# Task: rewards / goals / obstacles / terrain
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class RewardWeightsCfg:
    """Weights for reward.compute_reward(). Values preserved from config.py."""
    goal_bonus: float = 25.0       # was 10.0 — arrival must actually pay
    goal_tol: float = 0.5          # m
    progress_w: float = 50.0
    collision_pen: float = -10.0
    near_coll_pen: float = -2.0
    near_coll_thresh: float = 0.35  # m
    upright_w: float = -0.3
    height_w: float = -1.0
    target_height: float = 0.5      # m, nominal base height over terrain
    energy_w: float = -0.0005       # 10x lower: a regularizer, not the dominant term
    smooth_w: float = -0.002
    alive_bonus: float = 0.05      # was 0.5 — killed survival-farming
    heading_w: float = 0.3
    vel_track_w: float = 1.5       # was 1.0 — stronger goal-seeking
    vel_track_cap: float = 1.5      # m/s
    # Termination thresholds (check_termination)
    fall_height: float = 0.2        # m over terrain
    fall_tilt_rad: float = 1.0472   # pi/3


@dataclass
class GoalCfg:
    dist_range: tuple[float, float] = (1.5, 3.5)  # m from spawn
    episode_len_steps: int = 1000


@dataclass
class ObstacleCfg:
    """Kinematic box obstacles, placed only on flat terrain rows."""
    n_static: int = 2
    half_sizes: tuple[tuple[float, float, float], ...] = (
        (0.30, 0.30, 0.50),
        (0.25, 0.40, 0.40),
    )
    collision_dist: float = 0.35    # m, analytic collision threshold
    edge_margin: float = 0.5        # m, keep-clear margin from patch edge


@dataclass
class TerrainCfg:
    """4x4 curriculum grid (rows = difficulty). Values from spot_env_cfg.py."""
    rows: int = 4                   # difficulty levels (row 0 = easiest)
    cols: int = 4
    patch_size: float = 8.0         # m
    patch_half: float = 3.5         # usable half-width inside a patch
    flat_row_max: int = 1           # rows 0..flat_row_max are flat -> obstacles
    border_width: float = 0.25
    horizontal_scale: float = 0.1
    vertical_scale: float = 0.005
    slope_threshold: float = 0.75
    seed: int = 0
    # Sub-terrain parameters (proportions sum to 1.0)
    flat_proportion: float = 0.4
    flat_noise_range: tuple[float, float] = (0.0, 0.01)
    flat_noise_step: float = 0.01
    rough_proportion: float = 0.2
    rough_noise_range: tuple[float, float] = (0.02, 0.10)
    rough_noise_step: float = 0.02
    stairs_up_proportion: float = 0.2
    stairs_down_proportion: float = 0.2
    stair_step_height_range: tuple[float, float] = (0.05, 0.23)
    stair_step_width: float = 0.30
    stair_platform_width: float = 3.0
    # ── Parkour sub-terrains ("next level" beyond stairs) ────────────────
    # Every proportion defaults to 0.0 -> the terrain is NOT added to the
    # generator, so `spot` / `spot_hard` build the exact original 4-terrain
    # mix. `spot_parkour` turns these on (and bumps `cols` so each active
    # type gets a curriculum column). Difficulty (terrain row) scales each
    # one's active dimension exactly like stair_step_height_range: row 0 =
    # easiest, top row = the configured max.
    parkour_platform_width: float = 2.0     # clear flat start patch (m)
    # Scattered low boxes -> hurdles to step over / weave around.
    discrete_obstacles_proportion: float = 0.0
    discrete_obstacle_height_range: tuple[float, float] = (0.05, 0.18)
    discrete_obstacle_width_range: tuple[float, float] = (0.25, 0.50)
    discrete_obstacle_num: int = 10
    # Grid of cells at random heights -> broken / uneven floor.
    random_grid_proportion: float = 0.0
    random_grid_width: float = 0.45
    random_grid_height_range: tuple[float, float] = (0.02, 0.12)
    # Thin raised rails -> narrow step-overs.
    rails_proportion: float = 0.0
    rail_thickness_range: tuple[float, float] = (0.05, 0.12)
    rail_height_range: tuple[float, float] = (0.05, 0.16)
    # Stepping stones over voids -> precise foot placement (HARD; off by
    # default, reserved for a later spot_parkour_hard stage).
    stepping_stones_proportion: float = 0.0
    stepping_stone_height_max: float = 0.10
    stepping_stone_width_range: tuple[float, float] = (0.30, 0.55)
    stepping_stone_distance_range: tuple[float, float] = (0.05, 0.18)


@dataclass
class CurriculumCfg:
    """Per-env terrain level promotion/demotion, applied at reset.

    Three-way (Rudin 2022 style) with a "stay band": promote only on a clear
    success (reached goal, or covered >= promote_progress_frac of the start
    distance), demote only on a clear failure (fell, or covered <
    demote_progress_frac). Episodes in between hold their level, so one
    mediocre rollout no longer bounces an env down to flat ground.
    """
    enabled: bool = True
    promote_on_goal: bool = True
    demote_on_fall: bool = True
    # Survived and covered >= this fraction of the start distance -> promote
    promote_progress_frac: float = 0.8
    # Covered < this fraction (and didn't fall) -> demote. Raised from 0.5,
    # which demoted nearly every episode and collapsed the curriculum to flat.
    demote_progress_frac: float = 0.25


# ════════════════════════════════════════════════════════════════════════════
# Sensors
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class ScandotsCfg:
    """Heightfield raycast grid around the base (teacher exteroception).

    Heights are composited with the known kinematic box obstacles in torch
    (the RayCaster only sees the static terrain mesh).
    """
    grid_x: int = 17                # points along base-forward axis
    grid_y: int = 11                # points along base-left axis
    spacing: float = 0.1            # m between points
    forward_offset: float = 0.0     # m, bias grid center forward of base
    height_clip: float = 1.0        # obs clamp, m
    attach_yaw_only: bool = True

    @property
    def n_points(self) -> int:
        return self.grid_x * self.grid_y

    @property
    def size(self) -> tuple[float, float]:
        """RayCaster GridPatternCfg size (extent between outermost rays)."""
        return ((self.grid_x - 1) * self.spacing, (self.grid_y - 1) * self.spacing)


@dataclass
class CameraMountCfg:
    """One depth camera rigidly attached to RobotCfg.cam_mount_body."""
    name: str = "cam_front_center"
    pos: tuple[float, float, float] = (0.45, 0.0, 0.0)   # body frame, m
    # Quaternion (w, x, y, z) in the mount-body frame, with `convention`
    # semantics from Isaac Lab CameraCfg.OffsetCfg. (0.5,-0.5,0.5,-0.5) in
    # "ros" convention = optical axis along body +X (looking forward).
    rot: tuple[float, float, float, float] = (0.5, -0.5, 0.5, -0.5)
    convention: str = "ros"


@dataclass
class CameraRigCfg:
    """Depth camera rig (student exteroception). Cameras are constructed only
    when enabled=True — Phase 1 must never create camera prims."""
    enabled: bool = False
    width: int = 87
    height: int = 58
    update_period_s: float = 0.1    # 10 Hz renders while policy runs at 50 Hz
    h_fov_deg: float = 87.0
    min_depth: float = 0.1
    max_depth: float = 10.0
    # Annotator name; some Isaac Lab versions use "distance_to_image_plane"
    # or "depth" — the env tries this first, then known fallbacks.
    data_type: str = "distance_to_camera"
    mounts: tuple[CameraMountCfg, ...] = (CameraMountCfg(),)

    @property
    def n_cams(self) -> int:
        return len(self.mounts)


# ════════════════════════════════════════════════════════════════════════════
# Domain randomization
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class DRCfg:
    """Per-env domain randomization ranges. Sampled values are exposed in the
    privileged observation. "Motor strength" with implicit PD actuators means
    scaling kp AND kv per env (torque output of the implicit PD cannot be
    intercepted directly)."""
    enabled: bool = True
    randomize_friction: bool = True
    friction_range: tuple[float, float] = (0.4, 1.25)      # static friction
    dynamic_friction_ratio: float = 0.8                     # dynamic = ratio * static
    randomize_payload: bool = True
    payload_range_kg: tuple[float, float] = (-1.0, 5.0)     # added to base body
    randomize_com: bool = True
    com_offset_range_m: tuple[float, float] = (-0.05, 0.05)  # per axis, base body
    randomize_motor_strength: bool = True
    motor_strength_range: tuple[float, float] = (0.8, 1.2)  # kp/kv scale
    push_robots: bool = True
    push_interval_s: float = 8.0
    push_max_vel_xy: float = 0.5                             # m/s impulse


# ════════════════════════════════════════════════════════════════════════════
# Simulation
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class SimCfg:
    physics_dt: float = 0.005       # 200 Hz PhysX
    decimation: int = 4             # control at 50 Hz

    @property
    def control_dt(self) -> float:
        return self.physics_dt * self.decimation


# ════════════════════════════════════════════════════════════════════════════
# Policy / network dimensions
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class PolicyCfg:
    # Latents
    z_dim: int = 8                  # privileged latent (priv encoder & phi)
    extero_latent_dim: int = 64     # e_t (scandot encoder & depth encoder)
    history_len: int = 50           # steps of (proprio, action) for phi
    # MLP widths
    proprio_mlp_hidden: tuple[int, ...] = (256, 128)
    scandot_hidden: tuple[int, ...] = (256, 128)
    priv_hidden: tuple[int, ...] = (64, 32)
    trunk_hidden: tuple[int, ...] = (256, 128)
    critic_hidden: tuple[int, ...] = (512, 256, 128)
    # Adaptation module (1D conv over history)
    adapt_embed_dim: int = 32
    # Depth encoder (student)
    gru_hidden_dim: int = 128
    depth_fc_dim: int = 128
    # Action distribution
    log_std_init: float = -1.0
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    action_mean_clip: float = 2.0
    # Observation scales (single source of truth — used by env, mock, deploy)
    ang_vel_scale: float = 0.1
    joint_pos_scale: float = 1.0    # applied to (q - q_default)
    joint_vel_scale: float = 0.05
    goal_dist_scale: float = 0.2
    contact_force_scale: float = 0.01
    contact_force_clip: float = 5.0
    obs_clip: float = 10.0


# ════════════════════════════════════════════════════════════════════════════
# Training
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class TeacherTrainCfg:
    """Phase 1: privileged PPO teacher (no rendering)."""
    num_envs: int = 32768
    n_steps: int = 24               # locomotion-scale rollout horizon
    total_updates: int = 1500
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.005         # lowered from 0.01 (action std was inflating)
    ent_coef_final: float = 0.0     # entropy bonus annealed to this over the run
    vf_coef: float = 1.0
    max_grad: float = 1.5
    lr: float = 3e-4
    n_epochs: int = 5
    num_minibatches: int = 4        # minibatch size = T*B / num_minibatches
    target_kl: float = 0.03
    # Concurrent adaptation-module regression (ROA)
    adaptation_lr: float = 1e-3
    adapt_every: int = 1            # train phi every Nth rollout step
    # Logging / checkpoints
    log_interval: int = 1
    save_interval: int = 50


@dataclass
class StudentTrainCfg:
    """Phase 2: DAgger depth distillation (rendering-bound)."""
    num_envs: int = 256
    total_iters: int = 20000        # env steps == optimizer iterations
    lr: float = 5e-4
    max_grad: float = 1.0
    # False (default): train the full student actor (spec: student is a
    # deepcopy of the teacher actor trained with MSE action loss).
    # True: train only the depth encoder, trunk/head stay at teacher weights.
    distill_extero_only: bool = False
    optimizer_step_every: int = 1
    action_noise_std: float = 0.0   # optional exploration noise on executed action
    log_interval: int = 50
    save_interval: int = 1000


# ════════════════════════════════════════════════════════════════════════════
# Population-Based Training (Phase-1 teacher search)
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class PBTCfg:
    """Per-robot PBT search space + schedule.

    Lives on ExperimentCfg so each embodiment declares its OWN knob ranges —
    reward magnitudes differ across robots, so a search space tuned for Spot
    must not be hardcoded into the trainer. train_pbt CLI flags override these
    when provided; otherwise these per-robot values are the source of truth.

    The 4 reward knobs are tiled per-env; the 3 PPO knobs are per-member. Ranges
    are (lo, hi) and are also the clamp bounds used after each perturbation.
    """
    # Population scale (VRAM-gated; see plan Phase 5).
    pop_size: int = 24
    envs_per_member: int = 2048
    # Schedule.
    pbt_interval: int = 50          # evolve every N updates
    pbt_warmup: int = 100           # no PBT before this update
    # Weight-free fitness = success_rate - fitness_dist_weight * mean_final_dist.
    fitness_dist_weight: float = 0.1
    # Reward-knob ranges (must stay within what reward.compute_reward expects).
    alive_bonus_range: tuple[float, float] = (0.0, 0.2)
    progress_w_range: tuple[float, float] = (10.0, 100.0)
    vel_track_w_range: tuple[float, float] = (0.5, 3.0)
    goal_bonus_range: tuple[float, float] = (10.0, 50.0)
    # PPO-knob ranges.
    clip_eps_range: tuple[float, float] = (0.1, 0.3)
    ent_coef_range: tuple[float, float] = (0.0, 0.02)
    lr_range: tuple[float, float] = (1e-5, 1e-3)
    # Exploration: each perturbed knob is multiplied by a random factor, clamped.
    perturb_factors: tuple[float, ...] = (0.8, 1.2)
    # Initial ent_coef floor: multiplicative perturbation cannot revive a knob
    # that started at exactly 0, so members are seeded strictly positive.
    ent_coef_min_init: float = 1e-3


# ════════════════════════════════════════════════════════════════════════════
# Aggregate
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class ExperimentCfg:
    """The single object every entry point consumes."""
    robot: RobotCfg = field(default_factory=RobotCfg)
    reward: RewardWeightsCfg = field(default_factory=RewardWeightsCfg)
    goal: GoalCfg = field(default_factory=GoalCfg)
    obstacles: ObstacleCfg = field(default_factory=ObstacleCfg)
    terrain: TerrainCfg = field(default_factory=TerrainCfg)
    curriculum: CurriculumCfg = field(default_factory=CurriculumCfg)
    scandots: ScandotsCfg = field(default_factory=ScandotsCfg)
    camera: CameraRigCfg = field(default_factory=CameraRigCfg)
    dr: DRCfg = field(default_factory=DRCfg)
    sim: SimCfg = field(default_factory=SimCfg)
    policy: PolicyCfg = field(default_factory=PolicyCfg)
    teacher: TeacherTrainCfg = field(default_factory=TeacherTrainCfg)
    student: StudentTrainCfg = field(default_factory=StudentTrainCfg)
    pbt: PBTCfg = field(default_factory=PBTCfg)

    # ── Derived dimensions (the only place they are defined) ─────────
    @property
    def action_dim(self) -> int:
        return self.robot.num_joints

    @property
    def proprio_dim(self) -> int:
        # ang_vel(3) + projected_gravity(3) + goal_cmd(3) + q(12) + qd(12) + last_action(12)
        return 9 + 3 * self.action_dim

    @property
    def priv_dim(self) -> int:
        # friction(1) + payload(1) + com(3) + motor_strength(1) + contact forces(3*feet)
        return 6 + 3 * self.robot.num_feet

    @property
    def history_feat_dim(self) -> int:
        return self.proprio_dim + self.action_dim

    @property
    def critic_obs_dim(self) -> int:
        # proprio + scandots + priv + ground-truth base lin vel (3)
        return self.proprio_dim + self.scandots.n_points + self.priv_dim + 3
