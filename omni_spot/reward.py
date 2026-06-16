"""
PyTorch Reward Function — goal navigation
==========================================
Formulas preserved from the original reward.py with two changes:
  1. Weights come from RewardWeightsCfg (per-robot config), not constants.
  2. Height reward / fall termination use base height OVER TERRAIN
     (from the scandot raycast) instead of absolute world Z — the old
     absolute check was wrong on stairs and elevated terrain.
"""

from __future__ import annotations

import torch

from .configs.base import RewardWeightsCfg


def compute_reward(
    w: RewardWeightsCfg,
    *,
    robot_pos: torch.Tensor,       # (B, 3) world
    robot_quat: torch.Tensor,      # (B, 4)  w, x, y, z
    goal_pos: torch.Tensor,        # (B, 2) world
    root_lin_vel: torch.Tensor,    # (B, 3) world
    joint_vel: torch.Tensor,       # (B, J)
    action: torch.Tensor,          # (B, J) joint targets
    prev_action: torch.Tensor,     # (B, J)
    min_obs_dist: torch.Tensor,    # (B,)
    has_collision: torch.Tensor,   # (B,) bool
    prev_dist_goal: torch.Tensor,  # (B,)
    base_height: torch.Tensor,     # (B,) height over terrain under the base
) -> tuple[torch.Tensor, dict, torch.Tensor]:
    """Fully batched reward. Returns (total, info dict, new_dist_goal)."""
    robot_xy = robot_pos[:, :2]

    # ── 1. Progress ─────────────────────────────────────────────────────
    dist_goal = torch.linalg.norm(goal_pos - robot_xy, dim=-1)
    progress = prev_dist_goal - dist_goal
    r_progress = torch.clamp(progress * w.progress_w, -5.0, 5.0)

    # ── 2. Goal reached bonus ───────────────────────────────────────────
    goal_reached = dist_goal < w.goal_tol
    r_goal = torch.where(goal_reached, w.goal_bonus, 0.0)

    # ── 3. Collision penalties ──────────────────────────────────────────
    r_collision = torch.where(has_collision, w.collision_pen, 0.0)
    r_near = torch.where(
        min_obs_dist < w.near_coll_thresh, w.near_coll_pen, 0.0
    )

    # ── 4. Upright (tilt from quaternion) ───────────────────────────────
    x, y = robot_quat[:, 1], robot_quat[:, 2]
    cos_tilt = 1.0 - 2.0 * (x**2 + y**2)
    tilt_rad = torch.arccos(torch.clamp(cos_tilt, -1.0, 1.0))
    r_upright = tilt_rad * w.upright_w

    # ── 5. Height deviation (terrain-relative) ──────────────────────────
    r_height = torch.abs(base_height - w.target_height) * w.height_w

    # ── 6. Energy ───────────────────────────────────────────────────────
    r_energy = torch.clamp(
        torch.sum(joint_vel**2, dim=-1) * w.energy_w, -2.0, 0.0
    )

    # ── 7. Smoothness ───────────────────────────────────────────────────
    r_smooth = torch.clamp(
        torch.sum((action - prev_action) ** 2, dim=-1) * w.smooth_w, -2.0, 0.0
    )

    # ── 8. Alive bonus (only while upright and at height) ───────────────
    # Gated on not-fallen so it rewards survival rather than acting as a
    # constant per-step offset (it used to be paid even on the fall step).
    fallen = (base_height < w.fall_height) | (tilt_rad > w.fall_tilt_rad)
    r_alive = torch.where(
        fallen,
        torch.zeros_like(base_height),
        torch.full_like(base_height, w.alive_bonus),
    )

    # ── 9. Heading (face toward goal) ───────────────────────────────────
    goal_dir = goal_pos - robot_xy
    goal_dir_norm = goal_dir / (
        torch.linalg.norm(goal_dir, dim=-1, keepdim=True) + 1e-8
    )
    qw, qz = robot_quat[:, 0], robot_quat[:, 3]
    fwd_x = 1.0 - 2.0 * (y**2 + qz**2)
    fwd_y = 2.0 * (x * y + qw * qz)
    fwd_norm = torch.stack([fwd_x, fwd_y], dim=-1)
    fwd_norm = fwd_norm / (
        torch.linalg.norm(fwd_norm, dim=-1, keepdim=True) + 1e-8
    )
    r_heading = torch.sum(fwd_norm * goal_dir_norm, dim=-1) * w.heading_w

    # ── 10. Velocity tracking toward goal ───────────────────────────────
    vel_toward_goal = torch.sum(root_lin_vel[:, :2] * goal_dir_norm, dim=-1)
    r_vel_track = (
        torch.clamp(vel_toward_goal, -w.vel_track_cap, w.vel_track_cap)
        * w.vel_track_w
    )

    # ── Total ───────────────────────────────────────────────────────────
    # r_progress and r_vel_track were redundant (both reward velocity toward
    # the goal). vel_track is the better-shaped term, so progress is dropped
    # from the objective and kept in `info` for monitoring only.
    total = (r_progress + r_goal + r_collision + r_near
             + r_upright + r_height + r_energy + r_smooth
             + r_alive + r_heading + r_vel_track)
    total = torch.where(torch.isfinite(total), total, torch.zeros_like(total))
    total = torch.clamp(total, -20.0, w.goal_bonus + 10.0)

    info = {
        "r_progress": r_progress,
        "r_goal": r_goal,
        "r_collision": r_collision,
        "r_near": r_near,
        "r_upright": r_upright,
        "r_height": r_height,
        "r_energy": r_energy,
        "r_smooth": r_smooth,
        "r_alive": r_alive,
        "r_heading": r_heading,
        "r_vel_track": r_vel_track,
        "dist_goal": dist_goal,
    }
    return total, info, dist_goal


def check_termination(
    w: RewardWeightsCfg,
    *,
    robot_pos: torch.Tensor,    # (B, 3) world (xy used for goal distance)
    robot_quat: torch.Tensor,   # (B, 4)
    goal_pos: torch.Tensor,     # (B, 2)
    base_height: torch.Tensor,  # (B,) height over terrain
    step_count: torch.Tensor,   # (B,) int
    max_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (fallen, at_goal, timeout) — each (B,) bool. The env combines
    them (fallen | at_goal -> terminated, timeout -> truncated) and the
    curriculum uses them individually."""
    x, y = robot_quat[:, 1], robot_quat[:, 2]
    cos_tilt = 1.0 - 2.0 * (x**2 + y**2)
    tilt = torch.arccos(torch.clamp(cos_tilt, -1.0, 1.0))

    fallen = (base_height < w.fall_height) | (tilt > w.fall_tilt_rad)
    at_goal = (
        torch.linalg.norm(goal_pos - robot_pos[:, :2], dim=-1) < w.goal_tol
    )
    timeout = step_count >= max_steps
    return fallen, at_goal, timeout
