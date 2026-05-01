"""
PyTorch Reward Function
========================
Ported from jax_reward.py. All ops are batched PyTorch tensor math on GPU.
All clipping guards preserved to prevent inf/NaN values.
"""

import torch

from .config import (
    GOAL_BONUS, GOAL_TOL, PROGRESS_W, COLLISION_PEN,
    NEAR_COLL_PEN, NEAR_COLL_THRESH, UPRIGHT_W, HEIGHT_W,
    TARGET_HEIGHT, ENERGY_W, SMOOTH_W, ALIVE_BONUS, HEADING_W, 
    VEL_TRACK_CAP, VEL_TRACK_W,
)


def compute_reward(
    robot_pos:       torch.Tensor,   # (B, 3)
    robot_quat:      torch.Tensor,   # (B, 4)  w, x, y, z
    goal_pos:        torch.Tensor,   # (B, 2)
    prev_robot_pos:  torch.Tensor,   # (B, 3)
    root_lin_vel:    torch.Tensor,   # (B, 3)
    joint_vel:       torch.Tensor,   # (B, 12)
    action:          torch.Tensor,   # (B, 12)
    prev_action:     torch.Tensor,   # (B, 12)
    min_obs_dist:    torch.Tensor,   # (B,)
    has_collision:   torch.Tensor,   # (B,) bool
    prev_dist_goal:  torch.Tensor,   # (B,)
) -> tuple[torch.Tensor, dict, torch.Tensor]:
    """
    Fully batched reward computation.

    Returns:
        total:         (B,) float32, clipped to [-20, 210]
        info:          dict of (B,) reward components
        new_dist_goal: (B,) updated distance to goal
    """
    robot_xy = robot_pos[:, :2]
    goal_xy  = goal_pos

    # ── 1. Progress ─────────────────────────────────────────────────────
    dist_goal = torch.linalg.norm(goal_xy - robot_xy, dim=-1)
    progress  = prev_dist_goal - dist_goal
    r_progress = torch.clamp(progress * PROGRESS_W, -5.0, 5.0)

    # ── 2. Goal reached bonus ───────────────────────────────────────────
    goal_reached = dist_goal < GOAL_TOL
    r_goal = torch.where(goal_reached, GOAL_BONUS, 0.0)

    # ── 3. Collision penalties ──────────────────────────────────────────
    r_collision = torch.where(has_collision, COLLISION_PEN, 0.0)
    r_near = torch.where(min_obs_dist < NEAR_COLL_THRESH, NEAR_COLL_PEN, 0.0)

    # ── 4. Upright (tilt from quaternion) ───────────────────────────────
    w, x, y, z = robot_quat[:, 0], robot_quat[:, 1], robot_quat[:, 2], robot_quat[:, 3]
    cos_tilt = 1.0 - 2.0 * (x ** 2 + y ** 2)
    tilt_rad = torch.arccos(torch.clamp(cos_tilt, -1.0, 1.0))
    r_upright = tilt_rad * UPRIGHT_W

    # ── 5. Height deviation ─────────────────────────────────────────────
    height_dev = torch.abs(robot_pos[:, 2] - TARGET_HEIGHT)
    r_height = height_dev * HEIGHT_W

    # ── 6. Energy (joint velocity magnitude) ────────────────────────────
    r_energy = torch.clamp(torch.sum(joint_vel ** 2, dim=-1) * ENERGY_W, -2.0, 0.0)

    # ── 7. Smoothness (action change) ───────────────────────────────────
    r_smooth = torch.clamp(
        torch.sum((action - prev_action) ** 2, dim=-1) * SMOOTH_W, -2.0, 0.0
    )

    # ── 8. Alive bonus ──────────────────────────────────────────────────
    r_alive = torch.full((robot_pos.shape[0],), ALIVE_BONUS,
                         device=robot_pos.device, dtype=robot_pos.dtype)

    # ── 9. Heading reward (face toward goal) ────────────────────────────
    goal_dir = goal_xy - robot_xy
    goal_dir_norm = goal_dir / (torch.linalg.norm(goal_dir, dim=-1, keepdim=True) + 1e-8)
    fwd_x = 1.0 - 2.0 * (y ** 2 + z ** 2)
    fwd_y = 2.0 * (x * y + w * z)
    fwd_norm = torch.stack([fwd_x, fwd_y], dim=-1)
    fwd_norm = fwd_norm / (torch.linalg.norm(fwd_norm, dim=-1, keepdim=True) + 1e-8)
    heading_dot = torch.sum(fwd_norm * goal_dir_norm, dim=-1)
    r_heading = heading_dot * HEADING_W
    # ── 10. Velocity tracking (smooth forward-progress signal) ─────────
    # Project linear velocity onto direction-to-goal.
    # Positive = moving toward goal, negative = moving away.
    vel_xy = root_lin_vel[:, :2]
    vel_toward_goal = torch.sum(vel_xy * goal_dir_norm, dim=-1)  # (B,) in m/s
    # Saturate so we don't reward pathological fast glitches
    vel_toward_goal_capped = torch.clamp(vel_toward_goal, -VEL_TRACK_CAP, VEL_TRACK_CAP)
    r_vel_track = vel_toward_goal_capped * VEL_TRACK_W
    # ── Total ───────────────────────────────────────────────────────────
    total = (r_progress + r_goal + r_collision + r_near
             + r_upright + r_height + r_energy + r_smooth
             + r_alive + r_heading + r_vel_track)

    # Guard: replace NaN/inf with 0 and clip to finite range
    total = torch.where(torch.isfinite(total), total, torch.zeros_like(total))
    total = torch.clamp(total, -20.0, 210.0)

    info = {
        "r_progress":  r_progress,
        "r_goal":      r_goal,
        "r_collision": r_collision,
        "r_near":      r_near,
        "r_upright":   r_upright,
        "r_height":    r_height,
        "r_energy":    r_energy,
        "r_smooth":    r_smooth,
        "r_alive":     r_alive,
        "r_heading":   r_heading,
        "dist_goal":   dist_goal,
        "r_vel_track": r_vel_track,
    }
    return total, info, dist_goal


def check_termination(
    robot_pos:   torch.Tensor,   # (B, 3)
    robot_quat:  torch.Tensor,   # (B, 4)
    goal_pos:    torch.Tensor,   # (B, 2)
    step_count:  torch.Tensor,   # (B,) int32
    max_steps:   int = 1000,
    min_height:  float = 0.2,
    max_tilt:    float = 1.0472,   # pi/3
) -> torch.Tensor:
    """
    Returns terminated (B,) bool.
    Fallen: height < min_height or tilt > max_tilt.
    Goal:   dist < GOAL_TOL.
    Timeout: step_count >= max_steps.
    """
    w, x, y, z = robot_quat[:, 0], robot_quat[:, 1], robot_quat[:, 2], robot_quat[:, 3]
    cos_tilt = 1.0 - 2.0 * (x ** 2 + y ** 2)
    tilt = torch.arccos(torch.clamp(cos_tilt, -1.0, 1.0))

    fallen  = (robot_pos[:, 2] < min_height) | (tilt > max_tilt)
    at_goal = torch.linalg.norm(goal_pos - robot_pos[:, :2], dim=-1) < GOAL_TOL
    timeout = step_count >= max_steps
    return fallen | at_goal | timeout
