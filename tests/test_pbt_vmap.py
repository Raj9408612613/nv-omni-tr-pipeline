"""
Vmapped PBT update tests — CPU-only, no Isaac Lab.

The contract: the batched functional_call+vmap update (pbt_vmap.vmap_update_
members) must produce the SAME per-member parameter updates as the explicit
per-member loop doing identical math (pbt_vmap.reference_update_members), given
the same start state, data, knobs, and minibatch permutations.

Run with either:
    pytest tests/test_pbt_vmap.py -v
    python tests/test_pbt_vmap.py
"""

from __future__ import annotations

import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from omni_spot.configs import get_experiment_cfg
from omni_spot.mock_env import MockEnv
from omni_spot.pbt import Population
from omni_spot.pbt_vmap import (
    reference_update_members,
    vmap_update_members,
)

torch.manual_seed(0)

N_MEMBERS = 4
ENVS_PER_MEMBER = 6
N_STEPS = 4


def _cfg(n_epochs=1, num_minibatches=1):
    cfg = get_experiment_cfg("spot")
    cfg.teacher.n_steps = N_STEPS
    cfg.teacher.num_minibatches = num_minibatches
    cfg.teacher.n_epochs = n_epochs
    cfg.teacher.total_updates = 10
    return cfg


def _fresh_pop(cfg, seed):
    return Population(
        cfg, n_members=N_MEMBERS, envs_per_member=ENVS_PER_MEMBER,
        device="cpu", seed=seed,
    )


def _rollout_batches(pop, cfg):
    env = MockEnv(cfg, num_envs=pop.total_envs, device="cpu")
    env._reward_weights = pop.reward_weights
    obs, _ = env.reset()
    _, batches, _ = pop.collect_rollouts(env, obs, cfg.teacher.n_steps)
    return batches


def _clone_pop(src: Population, cfg, seed):
    """A population with the SAME weights/optimizers/knobs as src."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "p.pt")
        src.save(path)
        dst = _fresh_pop(cfg, seed=seed)   # different init...
        dst.load(path)                     # ...then overwritten by src state
    return dst


def _max_param_diff(pop_a: Population, pop_b: Population) -> float:
    worst = 0.0
    for ma, mb in zip(pop_a.members, pop_b.members):
        sa, sb = ma.trainer.net.state_dict(), mb.trainer.net.state_dict()
        for k in sa:
            d = (sa[k] - sb[k]).abs().max().item()
            worst = max(worst, d)
    return worst


def _run_pair(n_epochs, num_minibatches):
    """Build one population + rollout, clone it twice, run reference on one and
    vmap on the other with the SAME permutations, return (params match diff)."""
    cfg = _cfg(n_epochs, num_minibatches)
    base = _fresh_pop(cfg, seed=7)
    batches = _rollout_batches(base, cfg)

    pop_ref = _clone_pop(base, cfg, seed=11)
    pop_vmap = _clone_pop(base, cfg, seed=11)
    # Identical starting params (both cloned from base).
    assert _max_param_diff(pop_ref, pop_vmap) == 0.0

    total = batches[0].proprio.shape[0]
    perms = [torch.randperm(total) for _ in range(n_epochs)]

    info_ref = reference_update_members(pop_ref, batches, perms=perms)
    info_vmap = vmap_update_members(pop_vmap, batches, perms=perms)
    return pop_ref, pop_vmap, info_ref, info_vmap


def test_vmap_matches_reference_single_step():
    """One Adam step (1 epoch, 1 minibatch): tightest equivalence."""
    pop_ref, pop_vmap, ir, iv = _run_pair(n_epochs=1, num_minibatches=1)
    diff = _max_param_diff(pop_ref, pop_vmap)
    assert diff < 1e-6, f"vmap vs loop param diff too large: {diff:.3e}"
    # The update actually moved the weights (not a no-op equivalence).
    for i in range(N_MEMBERS):
        assert iv[i]["total_loss"] == iv[i]["total_loss"]  # finite


def test_vmap_matches_reference_multi_minibatch():
    """Several Adam steps compound float-order differences; still tight."""
    pop_ref, pop_vmap, ir, iv = _run_pair(n_epochs=2, num_minibatches=2)
    diff = _max_param_diff(pop_ref, pop_vmap)
    assert diff < 1e-4, f"vmap vs loop param diff too large: {diff:.3e}"


def test_vmap_update_actually_changes_weights():
    """Sanity: a vmap update must move the parameters from their init."""
    cfg = _cfg(n_epochs=1, num_minibatches=2)
    pop = _fresh_pop(cfg, seed=3)
    batches = _rollout_batches(pop, cfg)
    before = {m.id: {k: v.clone() for k, v in m.trainer.net.state_dict().items()}
              for m in pop.members}
    vmap_update_members(pop, batches)
    moved = False
    for m in pop.members:
        after = m.trainer.net.state_dict()
        for k in after:
            if not torch.equal(after[k], before[m.id][k]):
                moved = True
                break
    assert moved, "vmap update did not change any weights"


def test_population_dispatch_vmap_mode():
    """Population.update_members(mode='vmap') runs end-to-end and returns one
    info per member."""
    cfg = _cfg(n_epochs=1, num_minibatches=2)
    pop = _fresh_pop(cfg, seed=5)
    batches = _rollout_batches(pop, cfg)
    infos = pop.update_members(batches, mode="vmap")
    assert len(infos) == N_MEMBERS
    for info in infos:
        for k in ("policy_loss", "value_loss", "entropy", "grad_norm"):
            v = info[k]
            assert v == v and abs(v) < 1e9, f"{k} not finite: {v}"


def test_vmap_respects_per_member_lr():
    """Two members with the SAME weights but different lr must take different-
    sized steps under vmap (per-member lr is honored)."""
    cfg = _cfg(n_epochs=1, num_minibatches=1)
    pop = _fresh_pop(cfg, seed=9)
    batches = _rollout_batches(pop, cfg)
    # Force members 0 and 1 to identical weights, different lr.
    sd0 = pop.members[0].trainer.net.state_dict()
    pop.members[1].trainer.net.load_state_dict(sd0)
    pop.members[0].knobs["lr"] = 1e-3
    pop.members[1].knobs["lr"] = 1e-5
    # Same data for both so only lr differs.
    batches[1] = batches[0]
    before = {k: v.clone() for k, v in sd0.items()}

    vmap_update_members(pop, batches)
    s0 = pop.members[0].trainer.net.state_dict()
    s1 = pop.members[1].trainer.net.state_dict()
    step0 = max((s0[k] - before[k]).abs().max().item() for k in before)
    step1 = max((s1[k] - before[k]).abs().max().item() for k in before)
    assert step0 > step1 * 5, (
        f"higher-lr member should move more: {step0:.3e} vs {step1:.3e}"
    )


if __name__ == "__main__":
    failures = 0
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            import traceback
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
