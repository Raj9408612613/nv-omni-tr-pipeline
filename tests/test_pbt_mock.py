"""
PBT population tests on the mock env — CPU-only, no Isaac Lab.

Exercises the structural changes the plan calls "must work":
  - per-env reward-weight tiling across member slices,
  - shared-step rollout producing one correct per-member batch,
  - per-member PPO update applying that member's PPO knobs,
  - weight-free fitness + exploit/explore (copy + perturb + re-tile),
  - population checkpoint save/load roundtrip.

Run with either:
    pytest tests/test_pbt_mock.py -v
    python tests/test_pbt_mock.py
"""

from __future__ import annotations

import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from omni_spot.configs import get_experiment_cfg
from omni_spot.mock_env import MockEnv
from omni_spot.pbt import (
    ALL_KNOB_RANGES,
    REWARD_KNOBS,
    Population,
)

torch.manual_seed(0)

N_MEMBERS = 4
ENVS_PER_MEMBER = 4
N_STEPS = 4


def _cfg():
    cfg = get_experiment_cfg("spot")
    cfg.teacher.n_steps = N_STEPS
    cfg.teacher.num_minibatches = 2
    cfg.teacher.n_epochs = 2
    cfg.teacher.total_updates = 10
    return cfg


def _make_pop_env(seed: int = 0):
    cfg = _cfg()
    pop = Population(
        cfg, n_members=N_MEMBERS, envs_per_member=ENVS_PER_MEMBER,
        device="cpu", seed=seed,
    )
    env = MockEnv(cfg, num_envs=pop.total_envs, device="cpu")
    env._reward_weights = pop.reward_weights
    return cfg, pop, env


# ════════════════════════════════════════════════════════════════════════════
# Layout / tiling
# ════════════════════════════════════════════════════════════════════════════

def test_reward_weight_tiling_matches_member_knobs():
    _, pop, env = _make_pop_env()
    assert pop.total_envs == N_MEMBERS * ENVS_PER_MEMBER
    # env consumes the SAME object the population mutates in place.
    assert env._reward_weights is pop.reward_weights
    for knob in REWARD_KNOBS:
        tensor = getattr(pop.reward_weights, knob)
        assert tensor.shape == (pop.total_envs,)
        for m in pop.members:
            sl = pop.member_slice(m.id)
            assert torch.allclose(
                tensor[sl], torch.full((ENVS_PER_MEMBER,), m.knobs[knob])
            ), f"member {m.id} knob {knob} not tiled"


def test_members_have_distinct_knobs_in_range():
    _, pop, _ = _make_pop_env(seed=1)
    for knob, (lo, hi) in ALL_KNOB_RANGES.items():
        vals = [m.knobs[knob] for m in pop.members]
        assert all(lo <= v <= hi for v in vals), f"{knob} out of range: {vals}"
    # At least one knob actually varies across members (search needs diversity).
    varied = any(
        len({round(m.knobs[k], 9) for m in pop.members}) > 1
        for k in ALL_KNOB_RANGES
    )
    assert varied, "members are not diversified"


# ════════════════════════════════════════════════════════════════════════════
# Rollout + update
# ════════════════════════════════════════════════════════════════════════════

def test_collect_rollouts_shapes_and_update():
    cfg, pop, env = _make_pop_env()
    obs, _ = env.reset()

    obs, batches, stats = pop.collect_rollouts(env, obs, N_STEPS)
    assert len(batches) == N_MEMBERS and len(stats) == N_MEMBERS

    n = ENVS_PER_MEMBER * N_STEPS
    for b, st in zip(batches, stats):
        assert b.proprio.shape == (n, cfg.proprio_dim)
        assert b.scandots.shape == (n, cfg.scandots.n_points)
        assert b.action.shape == (n, cfg.action_dim)
        assert b.advantage.shape == (n,)
        for k in ("rew_mean", "rew_min", "rew_max"):
            v = st[k]
            assert v == v and abs(v) < 1e9
        assert "reward_components" in st["_diag"]

    infos = pop.update_members(batches)
    assert len(infos) == N_MEMBERS
    for info in infos:
        for k in ("policy_loss", "value_loss", "entropy", "grad_norm"):
            v = info[k]
            assert v == v and abs(v) < 1e9


def test_update_applies_per_member_ppo_knobs():
    _, pop, env = _make_pop_env()
    obs, _ = env.reset()
    obs, batches, _ = pop.collect_rollouts(env, obs, N_STEPS)
    pop.update_members(batches)
    for m in pop.members:
        t = m.trainer
        assert abs(t._clip_eps - m.knobs["clip_eps"]) < 1e-12
        assert abs(t._cur_ent_coef - m.knobs["ent_coef"]) < 1e-12
        lr = t.optimizer.param_groups[0]["lr"]
        assert abs(lr - m.knobs["lr"]) < 1e-12


def test_per_member_ret_stats_are_independent():
    """ret_mean/ret_std are per member (each PPOTrainer keeps its own)."""
    _, pop, env = _make_pop_env()
    obs, _ = env.reset()
    pop.collect_rollouts(env, obs, N_STEPS)
    means = [m.trainer._ret_mean for m in pop.members]
    # Bootstrapped from each member's own returns -> not all identical.
    assert len({round(x, 6) for x in means}) > 1


# ════════════════════════════════════════════════════════════════════════════
# Exploit / explore
# ════════════════════════════════════════════════════════════════════════════

def _set_fitness_inputs(m, success: float, dist: float, ep: int = 40):
    m.ep_count = ep
    m.goal_count = int(round(success * ep))
    m.final_dist_sum = dist * ep


def test_evolve_exploits_best_and_perturbs():
    _, pop, env = _make_pop_env(seed=3)
    # Rank deterministically: member 2 best, member 1 worst.
    _set_fitness_inputs(pop.members[0], success=0.5, dist=2.0)
    _set_fitness_inputs(pop.members[1], success=0.0, dist=4.0)   # worst
    _set_fitness_inputs(pop.members[2], success=1.0, dist=0.0)   # best
    _set_fitness_inputs(pop.members[3], success=0.6, dist=1.5)

    best = pop.members[2]
    worst = pop.members[1]
    best_state_before = {k: v.clone() for k, v in best.trainer.net.state_dict().items()}
    best_knobs_before = dict(best.knobs)

    events = pop.evolve()
    assert len(events) == max(1, N_MEMBERS // 4)
    ev = events[0]
    assert ev["source"] == best.id and ev["target"] == worst.id

    # Exploit: worst now carries the donor's (pre-perturb) weights byte-for-byte.
    ws = worst.trainer.net.state_dict()
    for k, v in best_state_before.items():
        assert torch.equal(ws[k], v), f"weights not copied on {k}"
    # ret stats copied too.
    assert worst.trainer._ret_mean == best.trainer._ret_mean
    assert worst.trainer._ret_std == best.trainer._ret_std

    # Explore: each knob is donor_knob * {0.8,1.2} then clamped, in range.
    for knob, (lo, hi) in ALL_KNOB_RANGES.items():
        new = worst.knobs[knob]
        assert lo <= new <= hi
        base = best_knobs_before[knob]
        expected = {
            min(max(base * 0.8, lo), hi),
            min(max(base * 1.2, lo), hi),
        }
        assert any(abs(new - e) < 1e-9 for e in expected), (
            f"{knob}: {new} not a perturbation of donor {base}"
        )

    # Re-tile: the worst member's env slice reflects its NEW knobs.
    for knob in REWARD_KNOBS:
        sl = pop.member_slice(worst.id)
        assert torch.allclose(
            getattr(pop.reward_weights, knob)[sl],
            torch.full((ENVS_PER_MEMBER,), worst.knobs[knob]),
        )

    # Accumulators reset for the next interval.
    assert all(m.ep_count == 0 for m in pop.members)


def test_fitness_is_weight_free():
    """Fitness must not depend on goal_bonus: two members with identical
    behavior (same success/dist) but different goal_bonus must score equal."""
    _, pop, _ = _make_pop_env()
    pop.members[0].knobs["goal_bonus"] = 50.0
    pop.members[1].knobs["goal_bonus"] = 10.0
    _set_fitness_inputs(pop.members[0], success=0.7, dist=1.0)
    _set_fitness_inputs(pop.members[1], success=0.7, dist=1.0)
    f0 = pop.compute_fitness(pop.members[0])
    f1 = pop.compute_fitness(pop.members[1])
    assert abs(f0 - f1) < 1e-12, "fitness leaked the goal_bonus knob"


# ════════════════════════════════════════════════════════════════════════════
# Checkpoint
# ════════════════════════════════════════════════════════════════════════════

def test_population_save_load_roundtrip():
    _, pop, env = _make_pop_env(seed=5)
    obs, _ = env.reset()
    obs, batches, _ = pop.collect_rollouts(env, obs, N_STEPS)
    pop.update_members(batches)

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "pop.pt")
        pop.save(path)

        other = Population(
            _cfg(), n_members=N_MEMBERS, envs_per_member=ENVS_PER_MEMBER,
            device="cpu", seed=999,   # different seed -> different init
        )
        other.load(path)

    for m1, m2 in zip(pop.members, other.members):
        assert m1.knobs == m2.knobs
        assert m1.trainer._ret_mean == m2.trainer._ret_mean
        s1 = m1.trainer.net.state_dict()
        s2 = m2.trainer.net.state_dict()
        for k in s1:
            assert torch.equal(s1[k], s2[k]), f"member {m1.id} weights differ on {k}"
    # Re-tiling restored on load.
    for knob in REWARD_KNOBS:
        for m in other.members:
            sl = other.member_slice(m.id)
            assert torch.allclose(
                getattr(other.reward_weights, knob)[sl],
                torch.full((ENVS_PER_MEMBER,), m.knobs[knob]),
            )


def test_knob_fitness_correlation_runs():
    _, pop, _ = _make_pop_env()
    for i, m in enumerate(pop.members):
        _set_fitness_inputs(m, success=0.2 * i, dist=1.0)
    for m in pop.members:
        m.fitness = pop.compute_fitness(m)
    corr = pop.knob_fitness_correlations()
    assert set(corr) == set(ALL_KNOB_RANGES)
    for v in corr.values():
        assert (v != v) or (-1.0 - 1e-9 <= v <= 1.0 + 1e-9)


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
            import traceback
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
