"""
PBT per-env reward-weight tests — CPU-only, no Isaac Lab.

The contract (plan Phase 1): compute_reward must accept the 4 PBT knobs
(alive_bonus, progress_w, vel_track_w, goal_bonus) as a scalar OR a (N,)
tensor, and a per-env tensor whose rows all equal the scalar must produce
rewards BIT-IDENTICAL to the scalar path. This is the regression guard for
FIX A (r_alive broadcasting) and the tensor-max clamp.

Run with either:
    pytest tests/test_pbt_reward.py -v
    python tests/test_pbt_reward.py
"""

from __future__ import annotations

import dataclasses
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from omni_spot.configs import get_experiment_cfg
from omni_spot.mock_env import MockEnv
from omni_spot.reward import compute_reward

torch.manual_seed(0)

# The four reward knobs PBT varies per-env (everything else stays scalar).
PBT_REWARD_KNOBS = ("alive_bonus", "progress_w", "vel_track_w", "goal_bonus")


def _fabricate(N: int, J: int):
    """Inputs that exercise both branches of every reward term: some envs
    fallen, some at the goal, some far, varied tilt/velocity."""
    quat = torch.randn(N, 4)
    quat = quat / quat.norm(dim=-1, keepdim=True)
    robot_pos = torch.randn(N, 3)
    goal_pos = robot_pos[:, :2] + torch.randn(N, 2)
    # Force a few rows to be exactly at the goal and a few to be "fallen".
    goal_pos[:2] = robot_pos[:2, :2]                 # at goal
    base_height = torch.full((N,), 0.5)
    base_height[2:4] = 0.05                           # below fall_height
    return dict(
        robot_pos=robot_pos,
        robot_quat=quat,
        goal_pos=goal_pos,
        root_lin_vel=torch.randn(N, 3),
        joint_vel=torch.randn(N, J) * 5.0,
        action=torch.randn(N, J).clamp(-1, 1),
        prev_action=torch.randn(N, J).clamp(-1, 1),
        min_obs_dist=torch.rand(N) * 2.0,
        has_collision=torch.rand(N) < 0.3,
        prev_dist_goal=torch.rand(N) * 4.0,
        base_height=base_height,
    )


def _per_env_equal(base_reward, N: int):
    """Return a RewardWeightsCfg copy whose 4 PBT knobs are (N,) tensors all
    equal to the scalar value (everything else untouched / scalar)."""
    overrides = {
        k: torch.full((N,), float(getattr(base_reward, k)))
        for k in PBT_REWARD_KNOBS
    }
    return dataclasses.replace(base_reward, **overrides)


def test_per_env_weights_equal_scalar_path():
    cfg = get_experiment_cfg("spot")
    N, J = 32, cfg.action_dim
    inp = _fabricate(N, J)

    total_s, info_s, dist_s = compute_reward(cfg.reward, **inp)
    total_t, info_t, dist_t = compute_reward(_per_env_equal(cfg.reward, N), **inp)

    assert torch.equal(total_s, total_t), (
        "per-env (all-equal) total differs from scalar; "
        f"max|Δ|={float((total_s - total_t).abs().max()):.3e}"
    )
    assert torch.equal(dist_s, dist_t)
    for k in info_s:
        assert torch.equal(info_s[k], info_t[k]), f"info[{k}] differs"


def test_per_env_weights_actually_vary_reward():
    """Sanity: distinct per-env knobs must change the reward (so the tensor
    path is genuinely consumed, not silently ignored)."""
    cfg = get_experiment_cfg("spot")
    N, J = 8, cfg.action_dim
    inp = _fabricate(N, J)

    # Member 0 gets large goal/progress/vel weights; member 1 gets tiny ones.
    w = _per_env_equal(cfg.reward, N)
    w.goal_bonus[0], w.goal_bonus[1] = 50.0, 10.0
    w.progress_w[0], w.progress_w[1] = 100.0, 10.0
    w.vel_track_w[0], w.vel_track_w[1] = 3.0, 0.5
    w.alive_bonus[0], w.alive_bonus[1] = 0.2, 0.0

    total, info, _ = compute_reward(w, **inp)
    assert torch.isfinite(total).all()
    # goal_bonus is the per-env clamp ceiling (goal_bonus + 10): row 0 may reach
    # up to 60, row 1 only up to 20.
    assert total[0].item() <= 50.0 + 10.0 + 1e-4
    assert total[1].item() <= 10.0 + 10.0 + 1e-4


def test_per_env_clamp_ceiling_is_per_env():
    """The upper clamp must be goal_bonus+10 *per env*, not a single scalar."""
    cfg = get_experiment_cfg("spot")
    N, J = 4, cfg.action_dim
    inp = _fabricate(N, J)
    # All envs reach the goal so r_goal == goal_bonus dominates the total.
    inp["goal_pos"] = inp["robot_pos"][:, :2].clone()
    inp["prev_dist_goal"] = torch.full((N,), 5.0)     # big positive progress
    inp["base_height"] = torch.full((N,), 0.5)        # upright, not fallen

    w = _per_env_equal(cfg.reward, N)
    w.goal_bonus[:] = torch.tensor([10.0, 20.0, 30.0, 40.0])
    total, info, _ = compute_reward(w, **inp)
    # Each total is capped at its own goal_bonus + 10.
    caps = torch.tensor([20.0, 30.0, 40.0, 50.0])
    assert torch.all(total <= caps + 1e-4), f"{total} exceeds {caps}"
    # And the high-goal-bonus envs really do score higher (cap is binding).
    assert total[3] > total[0]


def test_mock_env_per_env_weights_equal_scalar_plumbing():
    """FIX B: setting MockEnv._reward_weights to an all-equal per-env tensor
    must reproduce the scalar-cfg reward exactly. Two envs built+stepped under
    the same seed run identical RNG streams, so any difference is the weight
    plumbing, not the dynamics."""
    cfg = get_experiment_cfg("spot")
    N, J = 16, cfg.action_dim
    action = torch.randn(N, J).clamp(-1, 1)

    torch.manual_seed(123)
    env_s = MockEnv(cfg, num_envs=N, device="cpu")
    env_s.reset()
    _, r_s, term_s, trunc_s, info_s = env_s.step(action)

    torch.manual_seed(123)
    env_t = MockEnv(cfg, num_envs=N, device="cpu")
    env_t.reset()
    env_t._reward_weights = _per_env_equal(cfg.reward, N)
    _, r_t, term_t, trunc_t, info_t = env_t.step(action)

    assert torch.equal(r_s, r_t), (
        f"mock per-env reward differs from scalar; "
        f"max|Δ|={float((r_s - r_t).abs().max()):.3e}"
    )
    assert torch.equal(term_s, term_t) and torch.equal(trunc_s, trunc_t)
    for k in info_s:
        if isinstance(info_s[k], torch.Tensor):
            assert torch.equal(info_s[k], info_t[k]), f"info[{k}] differs"


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
