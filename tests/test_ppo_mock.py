"""
Phase 1 trainer tests on the mock env — CPU-only, no Isaac Lab.
Run with either:
    pytest tests/test_ppo_mock.py -v
    python tests/test_ppo_mock.py
"""

from __future__ import annotations

import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from omni_spot.configs import get_experiment_cfg
from omni_spot.mock_env import MockEnv
from omni_spot.networks import AdaptationModule, PrivEncoder
from omni_spot.ppo import PPOTrainer

torch.manual_seed(0)


def _small_cfg():
    cfg = get_experiment_cfg("spot")
    cfg.teacher.num_envs = 8
    cfg.teacher.n_steps = 8
    cfg.teacher.num_minibatches = 2
    cfg.teacher.n_epochs = 2
    cfg.teacher.total_updates = 3
    return cfg


def test_phi_regression_decreases():
    """phi must be able to regress z from a history that encodes priv."""
    cfg = _small_cfg()
    B = 256
    priv_enc = PrivEncoder(cfg)
    phi = AdaptationModule(cfg)
    opt = torch.optim.Adam(phi.parameters(), lr=1e-3)

    priv = torch.randn(B, cfg.priv_dim)
    with torch.no_grad():
        z_target = priv_enc(priv)
    history = torch.zeros(B, cfg.policy.history_len, cfg.history_feat_dim)
    history[:, :, : cfg.priv_dim] = priv.unsqueeze(1)
    history += 0.05 * torch.randn_like(history)

    losses = []
    for _ in range(200):
        loss = torch.mean(torch.sum((phi(history) - z_target) ** 2, dim=-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))

    first = sum(losses[:20]) / 20
    last = sum(losses[-20:]) / 20
    assert last < 0.5 * first, f"phi loss did not decrease: {first:.4f} -> {last:.4f}"


def test_ppo_three_updates_on_mock_env():
    cfg = _small_cfg()
    env = MockEnv(cfg, num_envs=cfg.teacher.num_envs, device="cpu")
    trainer = PPOTrainer(cfg, device="cpu")

    obs, _ = env.reset()
    assert obs["proprio"].shape == (8, cfg.proprio_dim)
    assert obs["scandots"].shape == (8, cfg.scandots.n_points)
    assert obs["priv"].shape == (8, cfg.priv_dim)
    assert obs["history"].shape == (8, cfg.policy.history_len,
                                    cfg.history_feat_dim)
    assert obs["critic_extras"].shape == (8, 3)
    assert "depth" not in obs, "Phase 1 obs must not contain depth"

    for update in range(1, cfg.teacher.total_updates + 1):
        trainer.anneal_lr(update)
        obs, batch, stats = trainer.collect_rollout(env, obs)
        info = trainer.update(batch)

        n = cfg.teacher.num_envs * cfg.teacher.n_steps
        assert batch.proprio.shape == (n, cfg.proprio_dim)
        assert batch.scandots.shape == (n, cfg.scandots.n_points)
        assert batch.action.shape == (n, cfg.action_dim)
        assert batch.advantage.shape == (n,)

        for k in ("rew_mean", "rew_min", "rew_max", "adapt_loss"):
            v = stats[k]
            assert v == v and abs(v) < 1e9, f"{k} not finite: {v}"
        for k in ("policy_loss", "value_loss", "entropy", "running_kl",
                  "grad_norm"):
            v = info[k]
            assert v == v and abs(v) < 1e9, f"{k} not finite: {v}"
        assert "reward_components" in stats["_diag"]
        assert "r_progress" in stats["_diag"]["reward_components"]


def test_checkpoint_save_load_roundtrip():
    cfg = _small_cfg()
    trainer = PPOTrainer(cfg, device="cpu")
    trainer._ret_mean, trainer._ret_std = 3.5, 2.0

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pt")
        trainer.save(path)

        other = PPOTrainer(cfg, device="cpu")
        other.load(path)

    assert other._ret_mean == 3.5 and other._ret_std == 2.0
    for (k1, p1), (k2, p2) in zip(
        trainer.net.state_dict().items(), other.net.state_dict().items()
    ):
        assert k1 == k2 and torch.equal(p1, p2), f"mismatch on {k1}"


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
