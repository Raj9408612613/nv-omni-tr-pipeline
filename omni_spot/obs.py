"""
Observation Assembly — pure PyTorch
====================================
Single source of truth for every observation the pipeline produces:
proprioception, scandots (terrain + obstacle compositing), privileged
vector, history buffer, critic input. Used identically by NavEnv (Isaac),
MockEnv (CPU), unit tests, and deployment — no Isaac Lab imports here.

All scaling constants come from PolicyCfg / DRCfg; nothing is hardcoded.
"""

from __future__ import annotations

import torch

from .configs.base import DRCfg, PolicyCfg


# ════════════════════════════════════════════════════════════════════════════
# Quaternion / frame helpers  (quats are (w, x, y, z))
# ════════════════════════════════════════════════════════════════════════════

def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate world vector v into the frame described by quaternion q.

    Args:
        q: (B, 4) unit quaternions (w, x, y, z)
        v: (B, 3) vectors in world frame
    Returns:
        (B, 3) vectors in body frame
    """
    q_w = q[:, 0:1]
    q_vec = q[:, 1:4]
    a = v * (2.0 * q_w**2 - 1.0)
    b = torch.cross(q_vec, v, dim=-1) * q_w * 2.0
    c = q_vec * torch.sum(q_vec * v, dim=-1, keepdim=True) * 2.0
    return a - b + c


def yaw_from_quat(q: torch.Tensor) -> torch.Tensor:
    """Yaw angle (rad) of quaternion q (w, x, y, z). Returns (B,)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y**2 + z**2))


def projected_gravity(q: torch.Tensor) -> torch.Tensor:
    """Unit gravity vector expressed in the body frame. Returns (B, 3)."""
    g = torch.zeros_like(q[:, :3])
    g[:, 2] = -1.0
    return quat_rotate_inverse(q, g)


def goal_command(
    root_quat_w: torch.Tensor,   # (B, 4)
    root_pos_xy: torch.Tensor,   # (B, 2) world
    goal_xy: torch.Tensor,       # (B, 2) world
    dist_scale: float,
) -> torch.Tensor:
    """Goal direction in the robot HEADING frame (yaw-invariant) + distance.

    Returns (B, 3): [unit_dir_x, unit_dir_y, dist * dist_scale].
    This replaces the spec's velocity command for the goal-navigation task.
    """
    diff = goal_xy - root_pos_xy
    dist = torch.linalg.norm(diff, dim=-1, keepdim=True)
    yaw = yaw_from_quat(root_quat_w)
    cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
    # World -> heading frame: R(-yaw) @ diff
    dir_x = cos_y * diff[:, 0] + sin_y * diff[:, 1]
    dir_y = -sin_y * diff[:, 0] + cos_y * diff[:, 1]
    dir_b = torch.stack([dir_x, dir_y], dim=-1) / (dist + 1e-8)
    return torch.cat([dir_b, dist * dist_scale], dim=-1)


def sanitize(t: torch.Tensor, clip: float) -> torch.Tensor:
    """Replace non-finite values with 0 and clamp to +/- clip."""
    t = torch.where(torch.isfinite(t), t, torch.zeros_like(t))
    return torch.clamp(t, -clip, clip)


# ════════════════════════════════════════════════════════════════════════════
# History buffer (input to the adaptation module phi)
# ════════════════════════════════════════════════════════════════════════════

class HistoryBuffer:
    """Ring buffer of the last H (proprio, action) frames per env.

    get() returns a contiguous (B, H, F) tensor ordered oldest -> newest.
    Reset envs read as zeros until they refill — phi is trained under the
    same convention, so deployment behavior matches.
    """

    def __init__(self, num_envs: int, horizon: int, feat_dim: int, device):
        self.horizon = horizon
        self._buf = torch.zeros(num_envs, horizon, feat_dim, device=device)
        self._ptr = 0

    def push(self, feats: torch.Tensor):
        """feats: (B, F) — the (proprio_t, action_t) frame."""
        self._buf[:, self._ptr] = feats
        self._ptr = (self._ptr + 1) % self.horizon

    def reset_idx(self, env_ids: torch.Tensor):
        self._buf[env_ids] = 0.0

    def get(self) -> torch.Tensor:
        return torch.cat(
            [self._buf[:, self._ptr:], self._buf[:, : self._ptr]], dim=1
        )


# ════════════════════════════════════════════════════════════════════════════
# Scandots: terrain raycast heights composited with known box obstacles
# ════════════════════════════════════════════════════════════════════════════

def compose_scandots(
    ray_hits_w: torch.Tensor,        # (B, N, 3) RayCaster hit points, world
    root_pos: torch.Tensor,          # (B, 3) base position, world
    box_pos: torch.Tensor | None,    # (B, K, 3) obstacle centers, world (or None)
    box_half_sizes: torch.Tensor | None,  # (K, 3)
    height_clip: float,
    target_height: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Heightfield observation around the base.

    The RayCaster only sees the static terrain mesh; the kinematic box
    obstacles are composited analytically (their poses are known) so the
    teacher's exteroception covers everything the student's depth camera
    will see: eff_z = max(terrain_z, box_top at that grid point).

    Returns:
        heights: (B, N) clamped to [-clip, clip].
                 0 = nominal ground at target_height below the base;
                 negative = surface higher / closer to the base.
        base_height: (B,) base height over the terrain (box-free mean) —
                 used for terrain-relative height reward and fall check.
    """
    root_z = root_pos[:, 2]
    terrain_z = ray_hits_w[..., 2]
    # Missed rays (inf/nan) -> assume nominal ground under the base
    nominal = (root_z - target_height).unsqueeze(1).expand_as(terrain_z)
    terrain_z = torch.where(torch.isfinite(terrain_z), terrain_z, nominal)

    eff_z = terrain_z
    if box_pos is not None and box_half_sizes is not None and box_pos.shape[1] > 0:
        p = ray_hits_w[..., :2]                              # (B, N, 2)
        c = box_pos[:, None, :, :2]                          # (B, 1, K, 2)
        hs = box_half_sizes[None, None, :, :2]               # (1, 1, K, 2)
        # NaN/inf grid points compare False -> excluded automatically
        inside = (torch.abs(p.unsqueeze(2) - c) <= hs).all(dim=-1)   # (B, N, K)
        box_top = (box_pos[..., 2] + box_half_sizes[None, :, 2])     # (B, K)
        cand = torch.where(
            inside, box_top.unsqueeze(1).expand_as(inside).to(terrain_z.dtype),
            torch.full_like(inside, float("-inf"), dtype=terrain_z.dtype),
        )
        eff_z = torch.maximum(terrain_z, cand.max(dim=-1).values)

    heights = torch.clamp(
        root_z.unsqueeze(1) - eff_z - target_height, -height_clip, height_clip
    )
    base_height = root_z - terrain_z.mean(dim=1)
    return heights, base_height


# ════════════════════════════════════════════════════════════════════════════
# Proprioception (actor input — deployable on hardware)
# ════════════════════════════════════════════════════════════════════════════

def build_proprio(
    pc: PolicyCfg,
    *,
    ang_vel_b: torch.Tensor,          # (B, 3) base angular velocity, body frame
    projected_gravity_b: torch.Tensor,  # (B, 3)
    root_quat_w: torch.Tensor,        # (B, 4) for the heading-frame goal command
    root_pos_xy: torch.Tensor,        # (B, 2) world
    goal_xy: torch.Tensor,            # (B, 2) world
    joint_pos: torch.Tensor,          # (B, J) policy joint order
    joint_vel: torch.Tensor,          # (B, J)
    default_joint_pos: torch.Tensor,  # (J,) or (B, J)
    last_action: torch.Tensor,        # (B, J) normalized [-1, 1]
) -> torch.Tensor:
    """Returns (B, 9 + 3J). Yaw-invariant: no world-frame quantities."""
    goal_cmd = goal_command(root_quat_w, root_pos_xy, goal_xy, pc.goal_dist_scale)
    proprio = torch.cat(
        [
            ang_vel_b * pc.ang_vel_scale,
            projected_gravity_b,
            goal_cmd,
            (joint_pos - default_joint_pos) * pc.joint_pos_scale,
            joint_vel * pc.joint_vel_scale,
            last_action,
        ],
        dim=-1,
    )
    return sanitize(proprio, pc.obs_clip)


# ════════════════════════════════════════════════════════════════════════════
# Privileged observation (teacher-only; never deployed)
# ════════════════════════════════════════════════════════════════════════════

def build_priv(
    dr: DRCfg,
    pc: PolicyCfg,
    *,
    friction: torch.Tensor,        # (B,) sampled static friction
    payload_kg: torch.Tensor,      # (B,) sampled payload delta
    com_offset: torch.Tensor,      # (B, 3) sampled CoM offset
    motor_scale: torch.Tensor,     # (B,) sampled kp/kv scale
    contact_forces: torch.Tensor,  # (B, feet, 3) or (B, 3*feet) net forces, N
) -> torch.Tensor:
    """Returns (B, 6 + 3*feet), each block normalized by its DR range."""

    def _norm(x: torch.Tensor, rng: tuple[float, float]) -> torch.Tensor:
        mid = 0.5 * (rng[0] + rng[1])
        half = max(0.5 * (rng[1] - rng[0]), 1e-6)
        return (x - mid) / half

    contact = contact_forces.reshape(contact_forces.shape[0], -1)
    contact = torch.clamp(
        contact * pc.contact_force_scale,
        -pc.contact_force_clip, pc.contact_force_clip,
    )
    priv = torch.cat(
        [
            _norm(friction, dr.friction_range).unsqueeze(-1),
            _norm(payload_kg, dr.payload_range_kg).unsqueeze(-1),
            _norm(com_offset, dr.com_offset_range_m),
            _norm(motor_scale, dr.motor_strength_range).unsqueeze(-1),
            contact,
        ],
        dim=-1,
    )
    return sanitize(priv, pc.obs_clip)


# ════════════════════════════════════════════════════════════════════════════
# Critic input (asymmetric: everything + ground-truth base lin vel)
# ════════════════════════════════════════════════════════════════════════════

def build_critic_obs(
    proprio: torch.Tensor,
    scandots: torch.Tensor,
    priv: torch.Tensor,
    root_lin_vel_b: torch.Tensor,  # (B, 3) ground truth, body frame
) -> torch.Tensor:
    return torch.cat([proprio, scandots, priv, root_lin_vel_b], dim=-1)
