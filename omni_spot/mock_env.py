"""
Mock Environment for Testing Without Isaac Sim
=================================================
Provides a lightweight Gym-style environment that mimics SpotNavEnv's
interface using simple rigid-body-ish physics in pure PyTorch.

This lets you validate the ENTIRE training pipeline (network, PPO,
rewards, diagnostics, video recording) on any GPU — no Omniverse needed.

The mock physics is intentionally simple:
  - Spot is a point mass with orientation (no articulated dynamics)
  - Joint commands move the robot via a simple kinematic model
  - Depth images are synthetic (procedurally generated from obstacle positions)
  - Collisions use sphere-box distance checks (same as real env)

Usage:
    from omni_spot.mock_env import MockSpotEnv
    env = MockSpotEnv(num_envs=64, device="cuda")
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F

from .config import (
    JOINT_LOWER, JOINT_UPPER, STANDING_POSE, TARGET_HEIGHT,
    N_CAMS, CAM_H, CAM_W, MIN_DEPTH, MAX_DEPTH,
    N_OBS, N_STATIC, N_DYNAMIC, HUMANOID_MOCAP_IDX,
    ROOM_HALF, HUMANOID_OBSTACLE, PROPRIO_DIM, ACTION_DIM,
    OBS_HALF_SIZES,
)
from .reward import compute_reward, check_termination


class MockSpotEnv:
    """
    Mock Spot environment for pipeline testing.

    Mimics SpotNavEnv's API without Isaac Lab:
      - obs = {"depth": (B,5,120,160), "proprio": (B,37)}
      - action = (B, 12)
      - reward = (B,)
    """

    def __init__(
        self,
        num_envs: int = 64,
        device: str = "cuda",
        render_depth: bool = True,
    ):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.render_depth = render_depth

        # Joint limits
        self._joint_lower = torch.tensor(JOINT_LOWER, device=self.device)
        self._joint_upper = torch.tensor(JOINT_UPPER, device=self.device)
        self._joint_mid = (self._joint_upper + self._joint_lower) / 2.0
        self._joint_range = (self._joint_upper - self._joint_lower) / 2.0

        # State tensors
        self._root_pos = torch.zeros(num_envs, 3, device=self.device)
        self._root_quat = torch.zeros(num_envs, 4, device=self.device)
        self._root_quat[:, 0] = 1.0  # identity quaternion [w,x,y,z]
        self._root_linvel = torch.zeros(num_envs, 3, device=self.device)
        self._root_angvel = torch.zeros(num_envs, 3, device=self.device)
        self._joint_pos = torch.tensor(
            STANDING_POSE, device=self.device
        ).unsqueeze(0).expand(num_envs, -1).clone()
        self._joint_vel = torch.zeros(num_envs, ACTION_DIM, device=self.device)

        # Goal, tracking, obstacles
        self._goal_pos = torch.zeros(num_envs, 2, device=self.device)
        self._prev_dist = torch.zeros(num_envs, device=self.device)
        self._prev_action = torch.zeros(num_envs, ACTION_DIM, device=self.device)
        self._prev_prev_action = torch.zeros(num_envs, ACTION_DIM, device=self.device)
        self._prev_root_pos = torch.zeros(num_envs, 3, device=self.device)
        self._step_count = torch.zeros(num_envs, device=self.device, dtype=torch.int32)
        #self._root_lin_vel = torch.zeros(B, 3)
        self._obs_pos = torch.zeros(num_envs, N_OBS, 3, device=self.device)
        self._human_pos = torch.zeros(num_envs, 2, device=self.device)
        self._human_wp_idx = torch.zeros(num_envs, device=self.device, dtype=torch.int32)

        # For video recording: store trajectory
        self.trajectory_log = []
        self._logging = False

    def reset(self, seed=None):
        """Reset all environments."""
        env_ids = torch.arange(self.num_envs, device=self.device)
        self._reset_idx(env_ids)
        obs = self._get_observations()
        return obs, {}

    def _reset_idx(self, env_ids: torch.Tensor):
        n = len(env_ids)

        # Robot at random position
        robot_xy = torch.empty(n, 2, device=self.device).uniform_(-ROOM_HALF, ROOM_HALF)
        robot_yaw = torch.empty(n, device=self.device).uniform_(0, 2 * math.pi)

        self._root_pos[env_ids, 0] = robot_xy[:, 0]
        self._root_pos[env_ids, 1] = robot_xy[:, 1]
        self._root_pos[env_ids, 2] = TARGET_HEIGHT
        self._root_quat[env_ids, 0] = torch.cos(robot_yaw / 2)
        self._root_quat[env_ids, 1] = 0.0
        self._root_quat[env_ids, 2] = 0.0
        self._root_quat[env_ids, 3] = torch.sin(robot_yaw / 2)
        self._root_linvel[env_ids] = 0.0
        self._root_angvel[env_ids] = 0.0

        self._joint_pos[env_ids] = torch.tensor(
            STANDING_POSE, device=self.device
        ).unsqueeze(0).expand(n, -1)
        self._joint_vel[env_ids] = 0.0

        # Goal 2-6m away
        goal_dist = torch.empty(n, device=self.device).uniform_(2.0, 6.0)
        goal_ang = torch.empty(n, device=self.device).uniform_(0, 2 * math.pi)
        goal_xy = robot_xy + torch.stack([
            goal_dist * torch.cos(goal_ang),
            goal_dist * torch.sin(goal_ang),
        ], dim=-1)
        goal_xy = torch.clamp(goal_xy, -ROOM_HALF, ROOM_HALF)
        self._goal_pos[env_ids] = goal_xy
        self._prev_dist[env_ids] = torch.linalg.norm(goal_xy - robot_xy, dim=-1)
        self._prev_action[env_ids] = 0.0
        self._prev_prev_action[env_ids] = 0.0
        self._prev_root_pos[env_ids] = self._root_pos[env_ids].clone()
        self._step_count[env_ids] = 0

        # Obstacles
        obs_xy = torch.empty(n, N_OBS, 2, device=self.device).uniform_(-ROOM_HALF, ROOM_HALF)
        obs_z = torch.full((n, N_OBS, 1), 0.5, device=self.device)
        obs_pos = torch.cat([obs_xy, obs_z], dim=-1)

        n_active = torch.randint(2, 7, (n,), device=self.device)
        n_non_human = N_OBS - 1
        indices = torch.arange(n_non_human, device=self.device)
        mask = indices.unsqueeze(0) < n_active.unsqueeze(1)
        OFF = torch.tensor([100.0, 0.0, 0.5], device=self.device)
        obs_pos[:, :n_non_human] = torch.where(mask.unsqueeze(-1), obs_pos[:, :n_non_human], OFF)

        # Humanoid disabled — send its slot off-scene so it doesn't affect distances
        obs_pos[:, HUMANOID_MOCAP_IDX] = torch.tensor(
            [1000.0, 0.0, 0.5], device=self.device
        )
        self._obs_pos[env_ids] = obs_pos
        self._human_pos[env_ids] = 1000.0
        self._human_wp_idx[env_ids] = 0

    def step(self, action: torch.Tensor):
        """Step all environments."""
        action = torch.clamp(action, -1.0, 1.0)
        action = torch.where(torch.isfinite(action), action, torch.zeros_like(action))

        # Save previous state
        self._prev_root_pos = self._root_pos.clone()
        self._prev_prev_action = self._prev_action.clone()

        # Simple kinematic model:
        # Joint targets -> joint positions (with some lag)
        ctrl = self._joint_mid + action * self._joint_range
        self._joint_vel = (ctrl - self._joint_pos) * 10.0  # spring dynamics
        self._joint_pos = self._joint_pos + self._joint_vel * 0.02

        # Robot moves toward goal based on front leg extension difference
        # (very simplified — just to produce meaningful reward signals)
        front_ext = (action[:, 1] + action[:, 4]) / 2.0  # front leg hy
        rear_ext = (action[:, 7] + action[:, 10]) / 2.0   # rear leg hy
        drive = (front_ext - rear_ext) * 0.3  # forward drive from leg diff

        yaw = 2 * torch.atan2(self._root_quat[:, 3], self._root_quat[:, 0])
        hip_diff = (action[:, 0] - action[:, 3] + action[:, 6] - action[:, 9]) * 0.05
        yaw = yaw + hip_diff

        dx = drive * torch.cos(yaw) * 0.02
        dy = drive * torch.sin(yaw) * 0.02

        self._root_pos[:, 0] += dx
        self._root_pos[:, 1] += dy
        # Height wobble from leg extension
        avg_knee = (action[:, 2] + action[:, 5] + action[:, 8] + action[:, 11]) / 4.0
        self._root_pos[:, 2] = TARGET_HEIGHT + avg_knee * 0.05

        self._root_pos[:, 0].clamp_(-ROOM_HALF, ROOM_HALF)
        self._root_pos[:, 1].clamp_(-ROOM_HALF, ROOM_HALF)

        self._root_quat[:, 0] = torch.cos(yaw / 2)
        self._root_quat[:, 3] = torch.sin(yaw / 2)
        self._root_linvel[:, 0] = dx / 0.02
        self._root_linvel[:, 1] = dy / 0.02
        self._prev_action = ctrl

        # Humanoid patrol
        if HUMANOID_OBSTACLE["enabled"]:
            self._update_humanoid_patrol()

        # Rewards
        self._obs_pos[:, HUMANOID_MOCAP_IDX, 0] = self._human_pos[:, 0]
        self._obs_pos[:, HUMANOID_MOCAP_IDX, 1] = self._human_pos[:, 1]
        self._obs_pos[:, HUMANOID_MOCAP_IDX, 2] = HUMANOID_OBSTACLE["mocap_z"]

        robot_xy = self._root_pos[:, :2]
        obs_xy = self._obs_pos[:, :, :2]
        dists = torch.linalg.norm(obs_xy - robot_xy.unsqueeze(1), dim=-1)
        min_obs_dist = dists.min(dim=-1).values
        has_collision = min_obs_dist < 0.35
        root_lin_vel = torch.zeros((self.num_envs, 3), device=self.device)

        reward, self._reward_info, new_dist = compute_reward(
            robot_pos=self._root_pos,
            robot_quat=self._root_quat,
            goal_pos=self._goal_pos,
            prev_robot_pos=self._prev_root_pos,
            joint_vel=self._joint_vel,
            root_lin_vel=root_lin_vel,
            action=self._prev_action,
            prev_action=self._prev_prev_action,
            min_obs_dist=min_obs_dist,
            has_collision=has_collision,
            prev_dist_goal=self._prev_dist,
        )
        self._prev_dist = new_dist
        self._step_count += 1

        # Termination
        terminated = check_termination(
            self._root_pos, self._root_quat,
            self._goal_pos, self._step_count,
        )
        truncated = self._step_count >= 1000
        terminated = terminated & ~truncated
        done = terminated | truncated

        # Auto-reset done envs
        done_ids = torch.where(done)[0]
        if len(done_ids) > 0:
            self._reset_idx(done_ids)

        # Log trajectory for video
        if self._logging:
            self.trajectory_log.append({
                "root_pos": self._root_pos[0].cpu().clone(),
                "root_quat": self._root_quat[0].cpu().clone(),
                "joint_pos": self._joint_pos[0].cpu().clone(),
                "goal_pos": self._goal_pos[0].cpu().clone(),
                "obs_pos": self._obs_pos[0].cpu().clone(),
                "reward": reward[0].item(),
                "done": done[0].item(),
            })

        obs = self._get_observations()
        return obs, reward, terminated, truncated, {}

    def _get_observations(self) -> dict:
        """Generate observations (synthetic depth + proprio)."""
        # Proprio (same as real env)
        goal_diff = self._goal_pos - self._root_pos[:, :2]
        goal_dist = torch.linalg.norm(goal_diff, dim=-1, keepdim=True)
        goal_dir = goal_diff / (goal_dist + 1e-8)

        proprio = torch.cat([
            self._joint_pos / 3.14,
            self._joint_vel / 20.0,
            self._root_quat,
            self._root_linvel / 5.0,
            self._root_angvel / 10.0,
            goal_dir,
            goal_dist / 5.0,
        ], dim=-1)
        proprio = torch.clamp(proprio, -10.0, 10.0)

        # Synthetic depth (procedural from obstacle distances)
        if self.render_depth:
            depth = self._render_synthetic_depth()
        else:
            depth = torch.full(
                (self.num_envs, N_CAMS, CAM_H, CAM_W),
                MAX_DEPTH, device=self.device,
            )

        return {"depth": depth, "proprio": proprio}

    def _render_synthetic_depth(self) -> torch.Tensor:
        """Generate approximate depth images from obstacle positions.

        Not physically accurate — just produces structured depth that
        gives the CNN something meaningful to learn from.
        """
        B = self.num_envs
        depth = torch.full((B, N_CAMS, CAM_H, CAM_W), MAX_DEPTH, device=self.device)

        robot_xy = self._root_pos[:, :2]
        yaw = 2 * torch.atan2(self._root_quat[:, 3], self._root_quat[:, 0])

        # Camera angles (relative to robot heading)
        cam_angles = torch.tensor(
            [0.0, 1.2217, -1.2217, 2.5307, -2.5307], device=self.device
        )
        hfov_rad = math.radians(87.0 / 2)

        for obs_idx in range(N_OBS):
            obs_xy = self._obs_pos[:, obs_idx, :2]
            diff = obs_xy - robot_xy
            dist = torch.linalg.norm(diff, dim=-1)
            angle_to_obs = torch.atan2(diff[:, 1], diff[:, 0])

            hs = OBS_HALF_SIZES[obs_idx]
            obs_width = max(hs[0], hs[1]) * 2
            obs_height = hs[2] * 2

            for cam_idx in range(N_CAMS):
                cam_world_angle = yaw + cam_angles[cam_idx]
                rel_angle = angle_to_obs - cam_world_angle
                rel_angle = torch.atan2(torch.sin(rel_angle), torch.cos(rel_angle))

                in_fov = rel_angle.abs() < hfov_rad
                if not in_fov.any():
                    continue

                # Pixel column from angle
                u_frac = (rel_angle / hfov_rad + 1.0) / 2.0
                u = (u_frac * CAM_W).long().clamp(0, CAM_W - 1)

                # Obstacle angular width in pixels
                angular_width = torch.atan2(
                    torch.tensor(obs_width, device=self.device),
                    dist.clamp(min=0.1)
                )
                pixel_half_w = (angular_width / hfov_rad * CAM_W / 2).long().clamp(1, CAM_W // 4)

                # Obstacle vertical extent
                obs_top = self._obs_pos[:, obs_idx, 2] + hs[2]
                obs_bot = self._obs_pos[:, obs_idx, 2] - hs[2]
                cam_z = self._root_pos[:, 2] + 0.05

                v_top = ((cam_z - obs_top) / (dist.clamp(min=0.1)) * CAM_H / 2 + CAM_H / 2).long().clamp(0, CAM_H - 1)
                v_bot = ((cam_z - obs_bot) / (dist.clamp(min=0.1)) * CAM_H / 2 + CAM_H / 2).long().clamp(0, CAM_H - 1)

                for b in range(min(B, 8)):  # limit per-pixel ops to first 8 envs for speed
                    if not in_fov[b]:
                        continue
                    d = dist[b].item()
                    if d < MIN_DEPTH or d > MAX_DEPTH:
                        continue
                    hw = max(1, pixel_half_w[b].item())
                    u_start = max(0, u[b].item() - hw)
                    u_end = min(CAM_W, u[b].item() + hw)
                    v_start = min(v_top[b].item(), v_bot[b].item())
                    v_end = max(v_top[b].item(), v_bot[b].item())
                    v_start = max(0, v_start)
                    v_end = min(CAM_H, v_end)
                    if v_start < v_end and u_start < u_end:
                        depth[b, cam_idx, v_start:v_end, u_start:u_end] = min(
                            depth[b, cam_idx, v_start:v_end, u_start:u_end].min().item(),
                            d
                        )

        # For envs > 8, just copy depth from env 0 with noise
        if B > 8:
            noise = torch.randn(B - 8, N_CAMS, CAM_H, CAM_W, device=self.device) * 0.1
            depth[8:] = depth[0:1] + noise

        return depth.clamp(MIN_DEPTH, MAX_DEPTH)

    def _update_humanoid_patrol(self):
        patrol_r = HUMANOID_OBSTACLE["patrol_radius"]
        wp0 = torch.stack([
            torch.clamp(self._goal_pos[:, 0] + patrol_r, -ROOM_HALF, ROOM_HALF),
            torch.clamp(self._goal_pos[:, 1], -ROOM_HALF, ROOM_HALF),
        ], dim=-1)
        wp1 = torch.stack([
            torch.clamp(self._goal_pos[:, 0] - patrol_r, -ROOM_HALF, ROOM_HALF),
            torch.clamp(self._goal_pos[:, 1], -ROOM_HALF, ROOM_HALF),
        ], dim=-1)
        waypoints = torch.stack([wp0, wp1], dim=1)

        env_idx = torch.arange(self.num_envs, device=self.device)
        target = waypoints[env_idx, self._human_wp_idx]

        diff = target - self._human_pos
        dist_h = torch.linalg.norm(diff, dim=-1, keepdim=True)
        dir_h = diff / (dist_h + 1e-8)

        speed = HUMANOID_OBSTACLE["speed"]
        self._human_pos = torch.clamp(
            self._human_pos + dir_h * speed * 0.02,
            -ROOM_HALF, ROOM_HALF,
        )
        self._human_wp_idx = torch.where(
            dist_h.squeeze(-1) < HUMANOID_OBSTACLE["wp_switch_dist"],
            1 - self._human_wp_idx,
            self._human_wp_idx,
        )

    def start_logging(self):
        self._logging = True
        self.trajectory_log = []

    def stop_logging(self):
        self._logging = False
        return self.trajectory_log
