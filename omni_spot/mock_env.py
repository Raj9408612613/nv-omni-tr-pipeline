"""
Mock Environment — pure PyTorch, no Isaac Lab
==============================================
Implements the exact observation/step interface of NavEnv so the trainers,
networks, and obs assembly can be exercised on CPU:

    obs = {proprio, scandots, priv, history, critic_extras
           [, depth, depth_new_frame  when cfg.camera.enabled]}
    step(action) -> (obs, reward, terminated, truncated, info)   # gym 5-tuple
    reset()      -> (obs, info)

Dynamics are a toy kinematic model (action -> base twist + first-order joint
tracking); rewards/termination/obs go through the REAL reward.py and obs.py
so this doubles as an integration test of those modules. Terrain is flat
(z=0) with the configured box obstacles; depth is synthesized from the known
box poses so depth correlates with scandots — enough signal for the DAgger
convergence test.

Auto-reset matches Isaac Lab: done envs are reset before the next obs.
"""

from __future__ import annotations

import math

import torch

from . import obs as obs_utils
from .configs.base import ExperimentCfg
from .reward import check_termination, compute_reward


class MockEnv:
    def __init__(self, cfg: ExperimentCfg, num_envs: int, device: str = "cpu"):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = torch.device(device)
        J = cfg.action_dim
        K = cfg.obstacles.n_static

        self._default_jp = torch.tensor(
            cfg.robot.default_joint_pos, device=self.device
        )
        lower = torch.tensor(cfg.robot.joint_lower, device=self.device)
        upper = torch.tensor(cfg.robot.joint_upper, device=self.device)
        if cfg.robot.action_scale is None:
            self._action_scale = (upper - lower) / 2.0
        else:
            self._action_scale = torch.tensor(
                cfg.robot.action_scale, device=self.device
            )
        self._vel_limits = torch.tensor(
            cfg.robot.joint_vel_limits, device=self.device
        )

        B = num_envs
        self._pos = torch.zeros(B, 3, device=self.device)
        self._yaw = torch.zeros(B, device=self.device)
        self._lin_vel_b = torch.zeros(B, 3, device=self.device)
        self._ang_vel_b = torch.zeros(B, 3, device=self.device)
        self._joint_pos = self._default_jp.repeat(B, 1)
        self._joint_vel = torch.zeros(B, J, device=self.device)
        self._last_action = torch.zeros(B, J, device=self.device)
        self._prev_ctrl = self._default_jp.repeat(B, 1)
        self._goal = torch.zeros(B, 2, device=self.device)
        self._prev_dist = torch.zeros(B, device=self.device)
        self._step_count = torch.zeros(B, dtype=torch.long, device=self.device)
        self._box_pos = torch.zeros(B, K, 3, device=self.device)
        self._box_half = torch.tensor(
            cfg.obstacles.half_sizes, device=self.device
        )[:K]
        # DR samples (exposed via priv obs)
        self._friction = torch.ones(B, device=self.device)
        self._payload = torch.zeros(B, device=self.device)
        self._com = torch.zeros(B, 3, device=self.device)
        self._motor = torch.ones(B, device=self.device)

        self._history = obs_utils.HistoryBuffer(
            B, cfg.policy.history_len, cfg.history_feat_dim, self.device
        )
        self._last_proprio = torch.zeros(
            B, cfg.proprio_dim, device=self.device
        )

        # Scandot grid offsets in the heading frame (N, 2), x-major like the
        # RayCaster grid pattern
        sc = cfg.scandots
        xs = torch.linspace(
            -sc.size[0] / 2, sc.size[0] / 2, sc.grid_x, device=self.device
        ) + sc.forward_offset
        ys = torch.linspace(
            -sc.size[1] / 2, sc.size[1] / 2, sc.grid_y, device=self.device
        )
        gx, gy = torch.meshgrid(xs, ys, indexing="ij")
        self._grid_b = torch.stack([gx.flatten(), gy.flatten()], dim=-1)

        self._global_step = 0
        self._render_every = max(
            1, round(cfg.camera.update_period_s / cfg.sim.control_dt)
        )
        self._just_reset = torch.ones(B, dtype=torch.bool, device=self.device)
        self._reward_info: dict = {}
        self._base_height = torch.full(
            (B,), cfg.reward.target_height, device=self.device
        )
        # Termination flags, mirrored from NavEnv so fitness/eval code can read
        # them identically on either env (PBT success-rate uses _at_goal).
        self._at_goal = torch.zeros(B, dtype=torch.bool, device=self.device)
        self._fallen = torch.zeros(B, dtype=torch.bool, device=self.device)
        # Optional per-env reward-weight override (PBT). None => use the scalar
        # cfg.reward for every env, i.e. the original single-policy behavior.
        # A RewardWeightsCfg whose 4 PBT knobs are (B,) tensors when set.
        self._reward_weights = None

    # ── Reset ─────────────────────────────────────────────────────────
    def _reset_idx(self, env_ids: torch.Tensor):
        n = len(env_ids)
        if n == 0:
            return
        cfg = self.cfg
        half = cfg.terrain.patch_half
        dev = self.device

        self._pos[env_ids, :2] = torch.empty(n, 2, device=dev).uniform_(
            -half, half
        )
        self._pos[env_ids, 2] = cfg.reward.target_height
        self._yaw[env_ids] = torch.empty(n, device=dev).uniform_(
            0, 2 * math.pi
        )
        self._lin_vel_b[env_ids] = 0.0
        self._ang_vel_b[env_ids] = 0.0
        self._joint_pos[env_ids] = self._default_jp
        self._joint_vel[env_ids] = 0.0
        self._last_action[env_ids] = 0.0
        self._prev_ctrl[env_ids] = self._default_jp
        self._step_count[env_ids] = 0

        d = torch.empty(n, device=dev).uniform_(*cfg.goal.dist_range)
        ang = torch.empty(n, device=dev).uniform_(0, 2 * math.pi)
        goal = self._pos[env_ids, :2] + torch.stack(
            [d * torch.cos(ang), d * torch.sin(ang)], dim=-1
        )
        self._goal[env_ids] = goal.clamp(-half, half)
        self._prev_dist[env_ids] = torch.linalg.norm(
            self._goal[env_ids] - self._pos[env_ids, :2], dim=-1
        )

        K = cfg.obstacles.n_static
        lim = half - cfg.obstacles.edge_margin
        box_xy = torch.empty(n, K, 2, device=dev).uniform_(-lim, lim)
        self._box_pos[env_ids] = torch.cat(
            [box_xy, self._box_half[:, 2].expand(n, K).unsqueeze(-1)], dim=-1
        )

        dr = cfg.dr
        self._friction[env_ids] = torch.empty(n, device=dev).uniform_(
            *dr.friction_range
        )
        self._payload[env_ids] = torch.empty(n, device=dev).uniform_(
            *dr.payload_range_kg
        )
        self._com[env_ids] = torch.empty(n, 3, device=dev).uniform_(
            *dr.com_offset_range_m
        )
        self._motor[env_ids] = torch.empty(n, device=dev).uniform_(
            *dr.motor_strength_range
        )

        self._history.reset_idx(env_ids)
        self._just_reset[env_ids] = True

    def reset(self) -> tuple[dict, dict]:
        self._reset_idx(torch.arange(self.num_envs, device=self.device))
        return self._build_obs(), {}

    # ── Helpers ───────────────────────────────────────────────────────
    def _quat(self) -> torch.Tensor:
        """Yaw-only orientation (mock body stays level)."""
        half = self._yaw / 2
        q = torch.zeros(self.num_envs, 4, device=self.device)
        q[:, 0] = torch.cos(half)
        q[:, 3] = torch.sin(half)
        return q

    def _lin_vel_w(self) -> torch.Tensor:
        c, s = torch.cos(self._yaw), torch.sin(self._yaw)
        vx, vy = self._lin_vel_b[:, 0], self._lin_vel_b[:, 1]
        return torch.stack([c * vx - s * vy, s * vx + c * vy,
                            torch.zeros_like(vx)], dim=-1)

    def _scandot_hits(self) -> torch.Tensor:
        """Fabricated RayCaster hits: grid around base, flat terrain z=0."""
        c, s = torch.cos(self._yaw), torch.sin(self._yaw)
        gx, gy = self._grid_b[:, 0], self._grid_b[:, 1]
        wx = self._pos[:, 0:1] + c.unsqueeze(1) * gx - s.unsqueeze(1) * gy
        wy = self._pos[:, 1:2] + s.unsqueeze(1) * gx + c.unsqueeze(1) * gy
        return torch.stack([wx, wy, torch.zeros_like(wx)], dim=-1)

    def _synth_depth(self) -> torch.Tensor:
        """Depth from known box poses: vertical strips at each box bearing."""
        cam = self.cfg.camera
        B, W = self.num_envs, cam.width
        img = torch.full(
            (B, cam.n_cams, cam.height, W), cam.max_depth, device=self.device
        )
        fov = math.radians(cam.h_fov_deg)
        cols = torch.arange(W, device=self.device, dtype=torch.float32)
        c, s = torch.cos(self._yaw), torch.sin(self._yaw)
        for k in range(self._box_pos.shape[1]):
            rel = self._box_pos[:, k, :2] - self._pos[:, :2]
            rx = c * rel[:, 0] + s * rel[:, 1]
            ry = -s * rel[:, 0] + c * rel[:, 1]
            dist = torch.sqrt(rx**2 + ry**2).clamp(min=cam.min_depth)
            bearing = torch.atan2(ry, rx)
            visible = (rx > 0) & (bearing.abs() < fov / 2) & (
                dist < cam.max_depth
            )
            center = (0.5 - bearing / fov) * W
            half_w = torch.clamp(
                torch.atan2(self._box_half[k, 0], dist) / fov * W, 1.0, W / 2
            )
            strip = (
                (cols.unsqueeze(0) - center.unsqueeze(1)).abs()
                <= half_w.unsqueeze(1)
            ) & visible.unsqueeze(1)                       # (B, W)
            d_img = torch.where(
                strip,
                dist.unsqueeze(1).expand(-1, W),
                torch.full((B, W), cam.max_depth, device=self.device),
            )
            img = torch.minimum(img, d_img[:, None, None, :].expand_as(img))
        return img.clamp(cam.min_depth, cam.max_depth)

    def _build_obs(self) -> dict:
        cfg = self.cfg
        quat = self._quat()
        proprio = obs_utils.build_proprio(
            cfg.policy,
            ang_vel_b=self._ang_vel_b,
            projected_gravity_b=obs_utils.projected_gravity(quat),
            root_quat_w=quat,
            root_pos_xy=self._pos[:, :2],
            goal_xy=self._goal,
            joint_pos=self._joint_pos,
            joint_vel=self._joint_vel,
            default_joint_pos=self._default_jp,
            last_action=self._last_action,
        )
        self._last_proprio = proprio

        scandots, base_h = obs_utils.compose_scandots(
            self._scandot_hits(), self._pos, self._box_pos, self._box_half,
            cfg.scandots.height_clip, cfg.reward.target_height,
        )
        self._base_height = base_h

        # Fabricated foot contact forces ~ weight on feet + noise
        feet = cfg.robot.num_feet
        contact = torch.randn(self.num_envs, feet, 3, device=self.device) * 5.0
        contact[..., 2] += (
            (cfg.robot.mass_kg + self._payload).unsqueeze(-1) * 9.81 / feet
        )
        priv = obs_utils.build_priv(
            cfg.dr, cfg.policy,
            friction=self._friction,
            payload_kg=self._payload,
            com_offset=self._com,
            motor_scale=self._motor,
            contact_forces=contact,
        )

        out = {
            "proprio": proprio,
            "scandots": scandots,
            "priv": priv,
            "history": self._history.get(),
            "critic_extras": self._lin_vel_b.clone(),
        }
        if cfg.camera.enabled:
            new_frame = self._just_reset | (
                self._global_step % self._render_every == 0
            )
            out["depth"] = self._synth_depth()
            out["depth_new_frame"] = new_frame.clone()
            self._just_reset[:] = False
        return out

    # ── Step ──────────────────────────────────────────────────────────
    def step(self, action: torch.Tensor):
        cfg = self.cfg
        dt = cfg.sim.control_dt
        action = torch.clamp(action, -1.0, 1.0)
        action = torch.where(
            torch.isfinite(action), action, torch.zeros_like(action)
        )

        # History records the (proprio, action) pair the policy just used
        self._history.push(torch.cat([self._last_proprio, action], dim=-1))
        self._last_action = action

        # First-order joint tracking, scaled by motor strength
        ctrl = self._default_jp + action * self._action_scale
        new_jp = self._joint_pos + (ctrl - self._joint_pos) * (
            0.5 * self._motor.unsqueeze(-1)
        )
        self._joint_vel = torch.clamp(
            (new_jp - self._joint_pos) / dt, -self._vel_limits, self._vel_limits
        )
        self._joint_pos = new_jp

        # Toy base twist from action thirds
        J = cfg.action_dim
        third = max(1, J // 3)
        vx = torch.tanh(action[:, :third].mean(-1)) * 1.0
        vy = torch.tanh(action[:, third:2 * third].mean(-1)) * 0.3
        wz = torch.tanh(action[:, 2 * third:].mean(-1)) * 1.5
        self._lin_vel_b = torch.stack(
            [vx, vy, torch.zeros_like(vx)], dim=-1
        )
        self._ang_vel_b = torch.stack(
            [torch.zeros_like(wz), torch.zeros_like(wz), wz], dim=-1
        )
        self._yaw = self._yaw + wz * dt
        self._pos[:, :2] = self._pos[:, :2] + self._lin_vel_w()[:, :2] * dt
        self._step_count += 1
        self._global_step += 1

        # Reward / termination via the real modules
        quat = self._quat()
        dists = torch.linalg.norm(
            self._box_pos[:, :, :2] - self._pos[:, None, :2], dim=-1
        )
        min_obs_dist = dists.min(dim=-1).values
        # Per-env reward weights (PBT) when set, else the scalar cfg.reward.
        # Only compute_reward sees per-env weights; check_termination keeps the
        # scalar thresholds (fall_height / fall_tilt / goal_tol are not knobs).
        reward_w = (
            self._reward_weights if self._reward_weights is not None
            else cfg.reward
        )
        reward, self._reward_info, new_dist = compute_reward(
            reward_w,
            robot_pos=self._pos,
            robot_quat=quat,
            goal_pos=self._goal,
            root_lin_vel=self._lin_vel_w(),
            joint_vel=self._joint_vel,
            action=ctrl,
            prev_action=self._prev_ctrl,
            min_obs_dist=min_obs_dist,
            has_collision=min_obs_dist < cfg.obstacles.collision_dist,
            prev_dist_goal=self._prev_dist,
            base_height=self._pos[:, 2],
        )
        self._prev_dist = new_dist
        self._prev_ctrl = ctrl

        fallen, at_goal, timeout = check_termination(
            cfg.reward,
            robot_pos=self._pos,
            robot_quat=quat,
            goal_pos=self._goal,
            base_height=self._pos[:, 2],
            step_count=self._step_count,
            max_steps=cfg.goal.episode_len_steps,
        )
        self._at_goal = at_goal
        self._fallen = fallen
        terminated = fallen | at_goal
        truncated = timeout & ~terminated

        # Isaac-style auto-reset before building the next obs
        done_ids = torch.nonzero(terminated | truncated).squeeze(-1)
        self._reset_idx(done_ids)

        info = dict(self._reward_info)
        return self._build_obs(), reward, terminated, truncated, info
