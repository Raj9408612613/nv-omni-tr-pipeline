"""
Unit tests: shapes + behavior of every module with fabricated batches.
CPU-only (no Isaac Lab, no GPU). Run with either:
    pytest tests/test_shapes.py -v
    python tests/test_shapes.py
"""

from __future__ import annotations

import math
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from omni_spot.configs import get_experiment_cfg
from omni_spot import obs as obs_utils
from omni_spot.networks import (
    Actor,
    AdaptationModule,
    Critic,
    DepthGRUEncoder,
    PrivEncoder,
    ScandotEncoder,
    StudentPolicy,
    TeacherPolicy,
    gaussian_entropy,
    gaussian_log_prob,
)
from omni_spot.checkpoint import (
    SHARED_PREFIXES,
    load_teacher_into_student,
)

B = 4
torch.manual_seed(0)


def _cfg():
    return get_experiment_cfg("spot")


# ════════════════════════════════════════════════════════════════════════════
# Config / derived dimensions
# ════════════════════════════════════════════════════════════════════════════

def test_config_derived_dims():
    cfg = _cfg()
    assert cfg.action_dim == 12
    assert cfg.proprio_dim == 45                    # 9 + 3*12
    assert cfg.priv_dim == 18                       # 6 + 3*4
    assert cfg.history_feat_dim == 57               # 45 + 12
    assert cfg.scandots.n_points == 17 * 11 == 187
    assert cfg.critic_obs_dim == 45 + 187 + 18 + 3 == 253
    assert cfg.scandots.size == (1.6, 1.0)
    assert cfg.camera.n_cams == 1
    assert abs(cfg.sim.control_dt - 0.02) < 1e-9


def test_config_registry_rejects_unknown():
    try:
        get_experiment_cfg("not_a_robot")
    except ValueError as e:
        assert "spot" in str(e)
    else:
        raise AssertionError("expected ValueError for unknown robot")


# ════════════════════════════════════════════════════════════════════════════
# Observation utilities
# ════════════════════════════════════════════════════════════════════════════

def test_history_buffer_order_and_reset():
    buf = obs_utils.HistoryBuffer(B, horizon=5, feat_dim=2, device="cpu")
    for t in range(7):  # wrap around
        buf.push(torch.full((B, 2), float(t)))
    h = buf.get()
    assert h.shape == (B, 5, 2)
    # Oldest -> newest after 7 pushes into 5 slots: 2, 3, 4, 5, 6
    assert torch.equal(h[0, :, 0], torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0]))
    buf.reset_idx(torch.tensor([1]))
    assert torch.all(buf.get()[1] == 0)
    assert torch.all(buf.get()[0] != 0)


def test_compose_scandots_flat_box_offscene_clip():
    cfg = _cfg()
    n = cfg.scandots.n_points
    root_pos = torch.tensor([[0.0, 0.0, 0.5]]).repeat(B, 1)
    # Fabricated grid: hits on flat ground z=0 at xy in [-0.8, 0.8] x [-0.5, 0.5]
    xs = torch.linspace(-0.8, 0.8, cfg.scandots.grid_x)
    ys = torch.linspace(-0.5, 0.5, cfg.scandots.grid_y)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    hits = torch.stack([gx.flatten(), gy.flatten(),
                        torch.zeros(n)], dim=-1).repeat(B, 1, 1)

    # env 0: box at origin (top z=1.0); env 1: box off-scene at x=1000
    box_pos = torch.tensor([[[0.0, 0.0, 0.5]],
                            [[1000.0, 0.0, 0.5]],
                            [[0.0, 0.0, 0.5]],
                            [[1000.0, 0.0, 0.5]]])
    box_half = torch.tensor([[0.3, 0.3, 0.5]])

    heights, base_h = obs_utils.compose_scandots(
        hits, root_pos, box_pos, box_half,
        height_clip=cfg.scandots.height_clip,
        target_height=cfg.reward.target_height,
    )
    assert heights.shape == (B, n) and base_h.shape == (B,)
    # Flat ground at target height -> 0 everywhere the box doesn't cover
    assert torch.allclose(base_h, torch.full((B,), 0.5))
    assert torch.all(heights[1] == 0)  # off-scene box ignored
    # Box top at z=1.0, root at 0.5 -> 0.5 - 1.0 - 0.5 = -1.0 (== -clip)
    center_covered = heights[0].min()
    assert center_covered.item() == -cfg.scandots.height_clip
    assert (heights[0] < 0).any() and (heights[1] == 0).all()

    # Missed rays (inf) -> nominal ground -> obs 0
    hits_missed = hits.clone()
    hits_missed[2, :50, 2] = float("inf")
    h2, _ = obs_utils.compose_scandots(
        hits_missed, root_pos, None, None, 1.0, 0.5
    )
    assert torch.all(h2[2, :50] == 0) and torch.isfinite(h2).all()


def test_goal_command_yaw_invariance():
    pc = _cfg().policy
    pos = torch.zeros(2, 2)
    # Robot 0 faces +x with goal ahead; robot 1 faces +y with goal rotated too
    quat0 = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    yaw = math.pi / 2
    quat1 = torch.tensor([[math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]])
    goal0 = torch.tensor([[2.0, 0.0]])
    goal1 = torch.tensor([[0.0, 2.0]])
    cmd0 = obs_utils.goal_command(quat0, pos[:1], goal0, pc.goal_dist_scale)
    cmd1 = obs_utils.goal_command(quat1, pos[1:], goal1, pc.goal_dist_scale)
    assert torch.allclose(cmd0, cmd1, atol=1e-5)
    assert torch.allclose(cmd0[0, :2], torch.tensor([1.0, 0.0]), atol=1e-5)
    assert abs(cmd0[0, 2].item() - 2.0 * pc.goal_dist_scale) < 1e-5


def test_build_proprio_and_priv():
    cfg = _cfg()
    J = cfg.action_dim
    proprio = obs_utils.build_proprio(
        cfg.policy,
        ang_vel_b=torch.randn(B, 3),
        projected_gravity_b=torch.randn(B, 3),
        root_quat_w=torch.tensor([[1.0, 0, 0, 0]]).repeat(B, 1),
        root_pos_xy=torch.randn(B, 2),
        goal_xy=torch.randn(B, 2),
        joint_pos=torch.randn(B, J),
        joint_vel=torch.randn(B, J) * 30,  # exercise the clamp
        default_joint_pos=torch.tensor(cfg.robot.default_joint_pos),
        last_action=torch.rand(B, J) * 2 - 1,
    )
    assert proprio.shape == (B, cfg.proprio_dim)
    assert torch.isfinite(proprio).all()
    assert proprio.abs().max() <= cfg.policy.obs_clip

    # Mid-range DR samples normalize to ~0
    dr = cfg.dr
    priv = obs_utils.build_priv(
        dr, cfg.policy,
        friction=torch.full((B,), sum(dr.friction_range) / 2),
        payload_kg=torch.full((B,), sum(dr.payload_range_kg) / 2),
        com_offset=torch.zeros(B, 3),
        motor_scale=torch.full((B,), sum(dr.motor_strength_range) / 2),
        contact_forces=torch.randn(B, cfg.robot.num_feet, 3) * 100,
    )
    assert priv.shape == (B, cfg.priv_dim)
    assert torch.isfinite(priv).all()
    assert torch.allclose(priv[:, :2], torch.zeros(B, 2), atol=1e-6)
    assert priv[:, 6:].abs().max() <= cfg.policy.contact_force_clip


# ════════════════════════════════════════════════════════════════════════════
# Networks
# ════════════════════════════════════════════════════════════════════════════

def test_encoder_shapes():
    cfg = _cfg()
    pc = cfg.policy
    e = ScandotEncoder(cfg)(torch.randn(B, cfg.scandots.n_points))
    assert e.shape == (B, pc.extero_latent_dim)
    z = PrivEncoder(cfg)(torch.randn(B, cfg.priv_dim))
    assert z.shape == (B, pc.z_dim)
    z_hat = AdaptationModule(cfg)(
        torch.randn(B, pc.history_len, cfg.history_feat_dim)
    )
    assert z_hat.shape == (B, pc.z_dim)


def test_actor_critic_shapes():
    cfg = _cfg()
    pc = cfg.policy
    actor = Actor(cfg, ScandotEncoder(cfg))
    mean, log_std = actor.forward_with_latent(
        torch.randn(B, cfg.proprio_dim),
        torch.randn(B, pc.extero_latent_dim),
        torch.randn(B, pc.z_dim),
    )
    assert mean.shape == (B, cfg.action_dim)
    assert mean.abs().max() <= pc.action_mean_clip
    assert log_std.shape == (cfg.action_dim,)
    assert log_std.min() >= pc.log_std_min and log_std.max() <= pc.log_std_max

    value = Critic(cfg)(torch.randn(B, cfg.critic_obs_dim))
    assert value.shape == (B,)

    lp = gaussian_log_prob(mean, log_std, torch.clamp(mean, -1, 1))
    ent = gaussian_entropy(log_std)
    assert lp.shape == (B,) and torch.isfinite(lp).all()
    assert torch.isfinite(ent).all()


def test_teacher_policy_forward():
    cfg = _cfg()
    teacher = TeacherPolicy(cfg)
    proprio = torch.randn(B, cfg.proprio_dim)
    scandots = torch.randn(B, cfg.scandots.n_points)
    priv = torch.randn(B, cfg.priv_dim)
    critic_obs = obs_utils.build_critic_obs(
        proprio, scandots, priv, torch.randn(B, 3)
    )
    assert critic_obs.shape == (B, cfg.critic_obs_dim)
    mean, log_std, value = teacher.evaluate(proprio, scandots, priv, critic_obs)
    assert mean.shape == (B, cfg.action_dim) and value.shape == (B,)
    assert teacher.act_mean(proprio, scandots, priv).shape == (B, cfg.action_dim)


def test_depth_gru_encoder_persistence_hold_reset_grad():
    cfg = _cfg()
    enc = DepthGRUEncoder(cfg)
    cam = cfg.camera
    depth = torch.rand(B, cam.n_cams, cam.height, cam.width)
    tick = torch.ones(B, dtype=torch.bool)
    hold = torch.zeros(B, dtype=torch.bool)

    e1 = enc.step(depth, tick)
    assert e1.shape == (B, cfg.policy.extero_latent_dim)
    assert e1.requires_grad, "gradient must flow on tick steps"

    # Hold: same latent values, no grad path (cached)
    e_hold = enc.step(depth, hold)
    assert torch.allclose(e_hold, e1.detach())
    assert not e_hold.requires_grad

    # Second tick with identical input -> different latent (hidden evolved)
    e2 = enc.step(depth, tick)
    assert not torch.allclose(e2.detach(), e1.detach())

    # Reset zeroes state: post-reset tick reproduces the first-tick latent
    reset = torch.ones(B, dtype=torch.bool)
    e3 = enc.step(depth, tick, reset_mask=reset)
    assert torch.allclose(e3.detach(), e1.detach(), atol=1e-6)

    # Backward through a tick reaches the CNN
    enc.zero_grad()
    enc.step(depth, tick).sum().backward()
    g = next(enc.cnn.parameters()).grad
    assert g is not None and torch.isfinite(g).all()


def test_depth_encoder_alternate_rig():
    # The old 3-cam 120x160 rig must work from config alone
    cfg = _cfg()
    from omni_spot.configs.base import CameraMountCfg, CameraRigCfg
    cfg.camera = CameraRigCfg(
        enabled=True, width=160, height=120,
        mounts=tuple(CameraMountCfg(name=f"cam{i}") for i in range(3)),
    )
    enc = DepthGRUEncoder(cfg)
    e = enc.step(torch.rand(B, 3, 120, 160), torch.ones(B, dtype=torch.bool))
    assert e.shape == (B, cfg.policy.extero_latent_dim)


# ════════════════════════════════════════════════════════════════════════════
# Checkpoint cross-loading
# ════════════════════════════════════════════════════════════════════════════

def test_cross_load_teacher_to_student():
    cfg = _cfg()
    teacher = TeacherPolicy(cfg)
    student = StudentPolicy(cfg)
    state = teacher.state_dict()

    missing, unexpected = load_teacher_into_student(student, state)
    assert unexpected == []
    assert missing and all(m.startswith("actor.extero_encoder.") for m in missing)

    # Every shared parameter is byte-identical
    s_state = student.state_dict()
    shared = [k for k in state if k.startswith(SHARED_PREFIXES)]
    assert shared, "no shared keys found"
    for k in shared:
        assert torch.equal(state[k], s_state[k]), f"mismatch on {k}"

    # phi is frozen in the student
    assert all(not p.requires_grad
               for p in student.adaptation_module.parameters())
    # ... and the rest of the actor still trains
    assert any(p.requires_grad for p in student.actor.parameters())

    # Student forward works end-to-end after the load
    cam = cfg.camera
    a = student.act_mean(
        torch.randn(B, cfg.proprio_dim),
        torch.rand(B, cam.n_cams, cam.height, cam.width),
        torch.ones(B, dtype=torch.bool),
        torch.randn(B, cfg.policy.history_len, cfg.history_feat_dim),
    )
    assert a.shape == (B, cfg.action_dim) and torch.isfinite(a).all()


# ════════════════════════════════════════════════════════════════════════════
# Standalone runner (no pytest required)
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    failures = 0
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:  # noqa: BLE001 — report and continue
            failures += 1
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
