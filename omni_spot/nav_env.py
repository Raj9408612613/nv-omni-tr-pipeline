"""
Isaac Lab DirectRLEnv — Goal Navigation, teacher-student pipeline
==================================================================
One environment shared by both phases:

  Phase 1 (teacher): obs = {proprio, scandots, priv, history, critic_extras}.
      Scandots come from a RayCaster grid (heightmap queries — no rendering)
      composited with the known kinematic box obstacles; priv carries the
      sampled DR values + foot contact forces.
  Phase 2 (student): cameras are enabled via config, adding
      {depth, depth_new_frame}. Scandots/priv stay available (the frozen
      teacher needs them for DAgger labels).

Robot-agnostic: every robot-specific value comes from ExperimentCfg. Joint
order is remapped between the config (policy) order and the sim (USD
traversal) order via find_joints at startup; body/joint names are verified
against the articulation with hard failures listing what was found.
"""

from __future__ import annotations

import math

import torch

from . import obs as obs_utils
from .configs.base import ExperimentCfg
from .env_cfg import HAS_ISAAC, NavEnvCfg
from .reward import check_termination, compute_reward

if HAS_ISAAC:
    try:
        from isaaclab.envs import DirectRLEnv
    except ImportError:
        from omni.isaac.lab.envs import DirectRLEnv


if HAS_ISAAC:

    class NavEnv(DirectRLEnv):
        """Batched goal-navigation env over the terrain curriculum."""

        cfg: NavEnvCfg

        def __init__(self, cfg: NavEnvCfg, exp_cfg: ExperimentCfg, **kwargs):
            self._x = exp_cfg
            self._warned: set[str] = set()
            super().__init__(cfg, **kwargs)

            x = self._x
            robot = self.scene["robot"]
            J = x.action_dim

            # ── Joint order: config (policy) order -> sim order ──────
            # Build the mapping by name so it is independent of both the
            # USD traversal order and find_joints version differences.
            sim_index = {name: i for i, name in enumerate(robot.joint_names)}
            missing = [
                nm for nm in x.robot.joint_names if nm not in sim_index
            ]
            if missing:
                raise RuntimeError(
                    f"Config joint_names {missing} not found in the "
                    f"articulation. Sim joint names: {robot.joint_names}"
                )
            joint_ids = [sim_index[nm] for nm in x.robot.joint_names]
            self._joint_ids = joint_ids                       # list[int]
            self._joint_ids_t = torch.tensor(
                joint_ids, device=self.device, dtype=torch.long
            )

            base_ids, _ = robot.find_bodies(x.robot.base_body_name)
            if len(base_ids) != 1:
                raise RuntimeError(
                    f"base_body_name '{x.robot.base_body_name}' matched "
                    f"{len(base_ids)} bodies. Robot bodies: {robot.body_names}"
                )
            self._base_body_idx = base_ids[0]

            # ── Action mapping (policy order) ─────────────────────────
            lower = torch.tensor(
                x.robot.joint_lower, device=self.device
            )
            upper = torch.tensor(
                x.robot.joint_upper, device=self.device
            )
            if x.robot.action_scale is None:
                self._action_scale = (upper - lower) / 2.0
            else:
                self._action_scale = torch.tensor(
                    x.robot.action_scale, device=self.device
                )
            self._default_jp = torch.tensor(
                x.robot.default_joint_pos, device=self.device
            )

            # ── Task state ────────────────────────────────────────────
            B = self.num_envs
            self._goal = torch.zeros(B, 2, device=self.device)
            self._prev_dist = torch.zeros(B, device=self.device)
            self._init_dist = torch.ones(B, device=self.device)
            self._ctrl = self._default_jp.repeat(B, 1)
            self._prev_ctrl = self._default_jp.repeat(B, 1)
            self._last_policy_action = torch.zeros(B, J, device=self.device)
            self._last_proprio = torch.zeros(
                B, x.proprio_dim, device=self.device
            )
            self._history = obs_utils.HistoryBuffer(
                B, x.policy.history_len, x.history_feat_dim, self.device
            )

            K = x.obstacles.n_static
            self._obs_keys = [f"obs_static_{i:02d}" for i in range(K)]
            self._obs_pos = torch.full(
                (B, K, 3), 100.0, device=self.device
            )
            self._box_half = torch.tensor(
                x.obstacles.half_sizes, device=self.device
            )[:K]

            # ── Termination / curriculum flags ────────────────────────
            self._fallen = torch.zeros(B, dtype=torch.bool, device=self.device)
            self._at_goal = torch.zeros(B, dtype=torch.bool, device=self.device)
            self._just_reset = torch.ones(
                B, dtype=torch.bool, device=self.device
            )
            # Optional per-env reward-weight override (PBT). None => every env
            # uses the scalar self._x.reward, i.e. the original teacher
            # behavior. When set by train_pbt it is a RewardWeightsCfg whose 4
            # PBT knobs are (B,) tensors tiled across each member's env slice.
            self._reward_weights = None
            self._base_height = torch.full(
                (B,), x.reward.target_height, device=self.device
            )
            self._heights = torch.zeros(
                B, x.scandots.n_points, device=self.device
            )

            # ── DR state (neutral until sampled) ──────────────────────
            self._friction = torch.ones(B, device=self.device)
            self._payload = torch.zeros(B, device=self.device)
            self._com = torch.zeros(B, 3, device=self.device)
            self._motor = torch.ones(B, device=self.device)
            self._push_timer = torch.zeros(
                B, dtype=torch.long, device=self.device
            )
            self._default_masses = None
            self._default_coms = None

            # ── Sensor bookkeeping ────────────────────────────────────
            self._scan_checked = False
            self._contact_checked = False
            self._global_step = 0
            self._render_every = max(
                1, round(x.camera.update_period_s / x.sim.control_dt)
            )
            self._cam_frame_cache: dict[str, torch.Tensor] = {}

        # ── Small helpers ─────────────────────────────────────────────
        def _warn_once(self, key: str, msg: str):
            if key not in self._warned:
                self._warned.add(key)
                print(f"[NavEnv][WARN] {msg}", flush=True)

        def _joint_pos_policy(self) -> torch.Tensor:
            return self.scene["robot"].data.joint_pos[:, self._joint_ids_t]

        def _joint_vel_policy(self) -> torch.Tensor:
            return self.scene["robot"].data.joint_vel[:, self._joint_ids_t]

        # ── Height scan (terrain raycast + analytic box compositing) ──
        def _update_height_scan(self):
            x = self._x
            scanner = self.scene["height_scanner"]
            hits = scanner.data.ray_hits_w
            if not self._scan_checked:
                if hits.shape[1] != x.scandots.n_points:
                    raise RuntimeError(
                        f"RayCaster produced {hits.shape[1]} rays, config "
                        f"expects {x.scandots.n_points} "
                        f"({x.scandots.grid_x}x{x.scandots.grid_y}). Adjust "
                        f"ScandotsCfg grid/spacing."
                    )
                self._scan_checked = True
            root_pos = self.scene["robot"].data.root_pos_w
            heights, base_h = obs_utils.compose_scandots(
                hits, root_pos,
                self._obs_pos if self._box_half.shape[0] > 0 else None,
                self._box_half if self._box_half.shape[0] > 0 else None,
                x.scandots.height_clip, x.reward.target_height,
            )
            self._heights = heights
            self._base_height = base_h

        # ── Reset ─────────────────────────────────────────────────────
        def _reset_idx(self, env_ids: torch.Tensor):
            x = self._x
            n = len(env_ids)
            robot = self.scene["robot"]

            # Curriculum BEFORE pose sampling (it moves env origins).
            # Uses the flags cached by _get_dones for these envs.
            if x.curriculum.enabled:
                progress = torch.clamp(
                    (self._init_dist[env_ids] - self._prev_dist[env_ids])
                    / self._init_dist[env_ids].clamp(min=1e-6),
                    0.0, 1.0,
                )
                at_goal = self._at_goal[env_ids]
                fallen = self._fallen[env_ids]
                cur = x.curriculum
                # Promote on a clear success; demote only on a fall or on
                # genuinely poor progress. Episodes between the two thresholds
                # hold their level (the "stay band"), so a single bad rollout
                # no longer bounces an env down to flat ground.
                move_up = (at_goal & cur.promote_on_goal) | (
                    ~fallen & (progress >= cur.promote_progress_frac)
                )
                move_down = (fallen & cur.demote_on_fall) | (
                    ~at_goal & ~fallen
                    & (progress < cur.demote_progress_frac)
                )
                try:
                    self.scene.terrain.update_env_origins(
                        env_ids, move_up, move_down
                    )
                except (AttributeError, TypeError) as e:
                    self._warn_once(
                        "curriculum",
                        f"terrain curriculum unavailable ({e}); "
                        "envs stay at fixed origins",
                    )

            super()._reset_idx(env_ids)

            env_origins = self.scene.env_origins[env_ids]
            half = x.terrain.patch_half

            # ── Robot pose ────────────────────────────────────────────
            local_xy = torch.empty(n, 2, device=self.device).uniform_(
                -half, half
            )
            yaw = torch.empty(n, device=self.device).uniform_(
                0.0, 2 * math.pi
            )
            root_state = robot.data.default_root_state[env_ids].clone()
            root_state[:, 0] = env_origins[:, 0] + local_xy[:, 0]
            root_state[:, 1] = env_origins[:, 1] + local_xy[:, 1]
            root_state[:, 2] = env_origins[:, 2] + x.robot.init_height
            root_state[:, 3] = torch.cos(yaw / 2)
            root_state[:, 4] = 0.0
            root_state[:, 5] = 0.0
            root_state[:, 6] = torch.sin(yaw / 2)
            root_state[:, 7:] = 0.0
            self._write_root_state(robot, root_state, env_ids)

            robot.write_joint_state_to_sim(
                self._default_jp.unsqueeze(0).expand(n, -1),
                torch.zeros(n, x.action_dim, device=self.device),
                joint_ids=self._joint_ids,
                env_ids=env_ids,
            )

            # ── Goal ──────────────────────────────────────────────────
            d = torch.empty(n, device=self.device).uniform_(
                *x.goal.dist_range
            )
            ang = torch.empty(n, device=self.device).uniform_(
                0.0, 2 * math.pi
            )
            goal_local = (local_xy + torch.stack(
                [d * torch.cos(ang), d * torch.sin(ang)], dim=-1
            )).clamp(-half, half)
            goal_world = env_origins[:, :2] + goal_local
            self._goal[env_ids] = goal_world
            dist0 = torch.linalg.norm(
                goal_world - (env_origins[:, :2] + local_xy), dim=-1
            )
            self._prev_dist[env_ids] = dist0
            self._init_dist[env_ids] = dist0.clamp(min=1e-6)

            # ── Obstacles (flat terrain rows only) ────────────────────
            K = self._box_half.shape[0]
            if K > 0:
                try:
                    levels = self.scene.terrain.terrain_levels[env_ids]
                    is_flat = levels <= x.terrain.flat_row_max
                except (AttributeError, TypeError, KeyError):
                    is_flat = torch.ones(
                        n, dtype=torch.bool, device=self.device
                    )
                lim = half - x.obstacles.edge_margin
                box_xy = torch.empty(n, K, 2, device=self.device).uniform_(
                    -lim, lim
                )
                box_z = self._box_half[:, 2].expand(n, K).unsqueeze(-1)
                new_pos = torch.cat([box_xy, box_z], dim=-1) \
                    + env_origins.unsqueeze(1)
                off_scene = torch.tensor(
                    [1000.0, 0.0, 0.5], device=self.device
                )
                flat3 = is_flat.unsqueeze(-1).expand(-1, 3)
                for k in range(K):
                    new_pos[:, k] = torch.where(
                        flat3, new_pos[:, k], off_scene
                    )
                self._obs_pos[env_ids] = new_pos
                self._write_obstacle_poses(env_ids)

            # ── Policy-side state ─────────────────────────────────────
            self._ctrl[env_ids] = self._default_jp
            self._prev_ctrl[env_ids] = self._default_jp
            self._last_policy_action[env_ids] = 0.0
            self._last_proprio[env_ids] = 0.0
            self._history.reset_idx(env_ids)
            self._just_reset[env_ids] = True

            # ── Domain randomization ──────────────────────────────────
            self._apply_dr(env_ids)
            if x.dr.push_robots:
                interval = max(
                    1, round(x.dr.push_interval_s / x.sim.control_dt)
                )
                self._push_timer[env_ids] = torch.randint(
                    interval // 2, interval + interval // 2, (n,),
                    device=self.device,
                )

        def _write_root_state(self, asset, root_state, env_ids):
            try:
                asset.write_root_state_to_sim(root_state, env_ids)
            except AttributeError:
                asset.write_root_pose_to_sim(root_state[:, :7], env_ids)
                asset.write_root_velocity_to_sim(root_state[:, 7:], env_ids)

        def _write_obstacle_poses(self, env_ids: torch.Tensor):
            n = len(env_ids)
            for k, key in enumerate(self._obs_keys):
                state = torch.zeros(n, 13, device=self.device)
                state[:, :3] = self._obs_pos[env_ids, k]
                state[:, 3] = 1.0  # identity quat
                self._write_root_state(self.scene[key], state, env_ids)

        # ── Domain randomization ──────────────────────────────────────
        def _apply_dr(self, env_ids: torch.Tensor):
            x = self._x
            dr = x.dr
            if not dr.enabled:
                return
            n = len(env_ids)
            robot = self.scene["robot"]
            ids_cpu = env_ids.to("cpu")

            if dr.randomize_friction:
                try:
                    view = robot.root_physx_view
                    mats = view.get_material_properties()
                    f = torch.empty(n).uniform_(*dr.friction_range)
                    mats[ids_cpu, :, 0] = f.unsqueeze(-1)
                    mats[ids_cpu, :, 1] = (
                        f * dr.dynamic_friction_ratio
                    ).unsqueeze(-1)
                    view.set_material_properties(mats, ids_cpu)
                    self._friction[env_ids] = f.to(self.device)
                except Exception as e:  # noqa: BLE001 — version variance
                    self._warn_once("friction", f"friction DR failed: {e}")

            if dr.randomize_payload or dr.randomize_com:
                try:
                    view = robot.root_physx_view
                    bi = self._base_body_idx
                    if self._default_masses is None:
                        self._default_masses = view.get_masses().clone()
                        self._default_coms = view.get_coms().clone()
                    if dr.randomize_payload:
                        masses = view.get_masses()
                        payload = torch.empty(n).uniform_(
                            *dr.payload_range_kg
                        )
                        masses[ids_cpu, bi] = (
                            self._default_masses[ids_cpu, bi] + payload
                        )
                        view.set_masses(masses, ids_cpu)
                        self._payload[env_ids] = payload.to(self.device)
                    if dr.randomize_com:
                        coms = view.get_coms()
                        off = torch.empty(n, 3).uniform_(
                            *dr.com_offset_range_m
                        )
                        coms[ids_cpu, bi, :3] = (
                            self._default_coms[ids_cpu, bi, :3] + off
                        )
                        view.set_coms(coms, ids_cpu)
                        self._com[env_ids] = off.to(self.device)
                except Exception as e:  # noqa: BLE001
                    self._warn_once("payload", f"payload/CoM DR failed: {e}")

            if dr.randomize_motor_strength:
                try:
                    scale = torch.empty(n, device=self.device).uniform_(
                        *dr.motor_strength_range
                    )
                    J = x.action_dim
                    kp = (x.robot.actuator_stiffness * scale).unsqueeze(
                        -1
                    ).expand(n, J)
                    kv = (x.robot.actuator_damping * scale).unsqueeze(
                        -1
                    ).expand(n, J)
                    robot.write_joint_stiffness_to_sim(
                        kp, joint_ids=self._joint_ids, env_ids=env_ids
                    )
                    robot.write_joint_damping_to_sim(
                        kv, joint_ids=self._joint_ids, env_ids=env_ids
                    )
                    self._motor[env_ids] = scale
                except Exception as e:  # noqa: BLE001
                    self._warn_once("motor", f"motor-strength DR failed: {e}")

        def _apply_pushes(self):
            dr = self._x.dr
            self._push_timer -= 1
            ids = torch.nonzero(self._push_timer <= 0).squeeze(-1)
            if len(ids) == 0:
                return
            robot = self.scene["robot"]
            try:
                vel = robot.data.root_vel_w[ids].clone()
                vel[:, :2] += torch.empty(
                    len(ids), 2, device=self.device
                ).uniform_(-dr.push_max_vel_xy, dr.push_max_vel_xy)
                try:
                    robot.write_root_velocity_to_sim(vel, env_ids=ids)
                except AttributeError:
                    robot.write_root_com_velocity_to_sim(vel, env_ids=ids)
            except Exception as e:  # noqa: BLE001
                self._warn_once("push", f"random push failed: {e}")
            interval = max(
                1, round(dr.push_interval_s / self._x.sim.control_dt)
            )
            self._push_timer[ids] = interval

        # ── Actions ───────────────────────────────────────────────────
        def _pre_physics_step(self, actions: torch.Tensor):
            actions = torch.clamp(actions, -1.0, 1.0)
            actions = torch.where(
                torch.isfinite(actions), actions, torch.zeros_like(actions)
            )
            # History records the (proprio, action) pair the policy used
            self._history.push(
                torch.cat([self._last_proprio, actions], dim=-1)
            )
            self._last_policy_action = actions.clone()
            self._ctrl = self._default_jp + actions * self._action_scale
            self._global_step += 1
            if self._x.dr.enabled and self._x.dr.push_robots:
                self._apply_pushes()

        def _apply_action(self):
            self.scene["robot"].set_joint_position_target(
                self._ctrl, joint_ids=self._joint_ids
            )

        # ── Termination (also caches height scan + curriculum flags) ──
        def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
            x = self._x
            robot = self.scene["robot"]
            self._update_height_scan()
            fallen, at_goal, timeout = check_termination(
                x.reward,
                robot_pos=robot.data.root_pos_w,
                robot_quat=robot.data.root_quat_w,
                goal_pos=self._goal,
                base_height=self._base_height,
                step_count=self.episode_length_buf,
                max_steps=self.max_episode_length,
            )
            # Collision with an obstacle ends the episode. Without this,
            # leaning on a box forever (paying collision_pen but collecting
            # the alive bonus) is a stable attractor, and the unbounded
            # -10/step tails dominate the return distribution.
            if self._box_half.shape[0] > 0:
                obs_d = torch.linalg.norm(
                    self._obs_pos[:, :, :2]
                    - robot.data.root_pos_w[:, None, :2], dim=-1
                )
                collided = (
                    obs_d.min(dim=-1).values < x.obstacles.collision_dist
                )
            else:
                collided = torch.zeros_like(fallen)
            self._fallen = fallen | collided  # demotes via the curriculum
            self._at_goal = at_goal
            terminated = fallen | at_goal | collided
            truncated = timeout & ~terminated
            return terminated, truncated

        # ── Rewards ───────────────────────────────────────────────────
        def _get_rewards(self) -> torch.Tensor:
            x = self._x
            robot = self.scene["robot"]
            root_pos = robot.data.root_pos_w

            if self._box_half.shape[0] > 0:
                dists = torch.linalg.norm(
                    self._obs_pos[:, :, :2] - root_pos[:, None, :2], dim=-1
                )
                min_obs_dist = dists.min(dim=-1).values
            else:
                min_obs_dist = torch.full(
                    (self.num_envs,), 1e6, device=self.device
                )

            # Per-env reward weights (PBT) when set, else the scalar x.reward.
            # check_termination (in _get_dones) keeps the scalar thresholds —
            # fall_height / fall_tilt / goal_tol are not PBT knobs.
            reward_w = (
                self._reward_weights if self._reward_weights is not None
                else x.reward
            )
            reward, info, new_dist = compute_reward(
                reward_w,
                robot_pos=root_pos,
                robot_quat=robot.data.root_quat_w,
                goal_pos=self._goal,
                root_lin_vel=robot.data.root_lin_vel_w,
                joint_vel=self._joint_vel_policy(),
                action=self._ctrl,
                prev_action=self._prev_ctrl,
                min_obs_dist=min_obs_dist,
                has_collision=min_obs_dist < x.obstacles.collision_dist,
                prev_dist_goal=self._prev_dist,
                base_height=self._base_height,
            )
            self._prev_dist = new_dist
            self._prev_ctrl = self._ctrl.clone()

            # Forward reward components to extras for the trainer's logging
            for k, v in info.items():
                if isinstance(v, torch.Tensor):
                    self.extras[k] = v
            try:
                self.extras["terrain_level"] = (
                    self.scene.terrain.terrain_levels.float()
                )
            except (AttributeError, TypeError):
                pass
            return reward

        # ── Observations ──────────────────────────────────────────────
        def _read_depth(self) -> tuple[torch.Tensor, torch.Tensor]:
            """Returns (depth (B, n_cams, H, W), new_frame (B,) bool)."""
            cam_cfg = self._x.camera
            imgs = []
            new_frame = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            frame_counter_ok = True
            for mount in cam_cfg.mounts:
                cam = self.scene[mount.name]
                out = cam.data.output
                d = None
                for key in (cam_cfg.data_type, "distance_to_camera",
                            "distance_to_image_plane", "depth"):
                    if key in out:
                        d = out[key]
                        break
                if d is None:
                    # Pre-first-render (or wrong data_type): substitute a
                    # far-plane image instead of crashing during reset.
                    self._warn_once(
                        f"depth_{mount.name}",
                        f"no depth annotator on {mount.name} yet; available:"
                        f" {list(out.keys())} — using max_depth dummy",
                    )
                    d = torch.full(
                        (self.num_envs, 1, cam_cfg.height, cam_cfg.width),
                        cam_cfg.max_depth, device=self.device,
                    )
                if d.ndim == 4 and d.shape[-1] == 1:
                    d = d.permute(0, 3, 1, 2)
                elif d.ndim == 3:
                    d = d.unsqueeze(1)
                d = torch.nan_to_num(
                    d, nan=cam_cfg.max_depth, posinf=cam_cfg.max_depth,
                    neginf=cam_cfg.min_depth,
                )
                imgs.append(
                    torch.clamp(d, cam_cfg.min_depth, cam_cfg.max_depth)
                )

                frames = getattr(cam, "frame", None)
                if isinstance(frames, torch.Tensor):
                    cached = self._cam_frame_cache.get(mount.name)
                    if cached is not None:
                        new_frame |= (frames != cached).to(self.device)
                    else:
                        new_frame |= torch.ones_like(new_frame)
                    self._cam_frame_cache[mount.name] = frames.clone()
                else:
                    frame_counter_ok = False

            if not frame_counter_ok:
                # Fallback: assume renders land every Nth control step
                tick = (self._global_step % self._render_every) == 0
                new_frame |= torch.full_like(new_frame, tick)
            return torch.cat(imgs, dim=1), new_frame

        def _get_observations(self) -> dict:
            x = self._x
            robot = self.scene["robot"]
            quat = robot.data.root_quat_w
            root_pos = robot.data.root_pos_w

            # Recompute heights post-reset; sensor data is one step stale
            # for envs that just reset, so force nominal flat ground there.
            self._update_height_scan()
            if self._just_reset.any():
                self._heights[self._just_reset] = 0.0
                self._base_height[self._just_reset] = x.reward.target_height

            contact = self.scene["contact_sensor"].data.net_forces_w
            if not self._contact_checked:
                if contact.shape[1] != x.robot.num_feet:
                    raise RuntimeError(
                        f"contact_body_regex '{x.robot.contact_body_regex}' "
                        f"matched {contact.shape[1]} bodies, expected "
                        f"{x.robot.num_feet}. Robot bodies: "
                        f"{robot.body_names}"
                    )
                self._contact_checked = True

            priv = obs_utils.build_priv(
                x.dr, x.policy,
                friction=self._friction,
                payload_kg=self._payload,
                com_offset=self._com,
                motor_scale=self._motor,
                contact_forces=contact,
            )
            proprio = obs_utils.build_proprio(
                x.policy,
                ang_vel_b=robot.data.root_ang_vel_b,
                projected_gravity_b=robot.data.projected_gravity_b,
                root_quat_w=quat,
                root_pos_xy=root_pos[:, :2],
                goal_xy=self._goal,
                joint_pos=self._joint_pos_policy(),
                joint_vel=self._joint_vel_policy(),
                default_joint_pos=self._default_jp,
                last_action=self._last_policy_action,
            )
            self._last_proprio = proprio

            out = {
                "proprio": proprio,
                "scandots": self._heights,
                "priv": priv,
                "history": self._history.get(),
                "critic_extras": robot.data.root_lin_vel_b.clone(),
            }
            if x.camera.enabled:
                depth, new_frame = self._read_depth()
                out["depth"] = depth
                out["depth_new_frame"] = new_frame | self._just_reset
            self._just_reset[:] = False
            return out

else:
    # Stub when Isaac Lab is not available
    class NavEnv:  # noqa: D401
        """Stub — install Isaac Lab to use this environment."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "NavEnv requires NVIDIA Isaac Lab. "
                "Install Isaac Lab: https://isaac-sim.github.io/IsaacLab/"
            )
