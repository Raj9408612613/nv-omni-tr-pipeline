"""
Phase 2 DAgger tests on the mock env — CPU-only, no Isaac Lab.
Verifies: depth render cadence, GRU state persistence across steps,
DAgger loss DECREASES over ~200 iterations, phi stays frozen.
Run with either:
    pytest tests/test_dagger_mock.py -v
    python tests/test_dagger_mock.py
"""

from __future__ import annotations

import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from omni_spot.checkpoint import save_checkpoint
from omni_spot.configs import get_experiment_cfg
from omni_spot.dagger import DaggerTrainer
from omni_spot.mock_env import MockEnv
from omni_spot.networks import TeacherPolicy

torch.manual_seed(0)


def _cfg():
    cfg = get_experiment_cfg("spot")
    cfg.camera.enabled = True
    cfg.student.num_envs = 16
    cfg.student.lr = 1e-3
    return cfg


def _make_trainer(cfg, tmpdir):
    """A random-init teacher is a perfectly good fixed labeling function
    for verifying that distillation MACHINERY converges."""
    teacher = TeacherPolicy(cfg)
    path = os.path.join(tmpdir, "teacher.pt")
    save_checkpoint(path, model_state=teacher.state_dict(),
                    phase="teacher", robot=cfg.robot.name)
    return DaggerTrainer(cfg, path, device="cpu")


def test_dagger_loss_decreases_over_200_iters():
    cfg = _cfg()
    env = MockEnv(cfg, cfg.student.num_envs, device="cpu")
    with tempfile.TemporaryDirectory() as d:
        trainer = _make_trainer(cfg, d)

        # phi must be frozen in the student
        assert all(not p.requires_grad
                   for p in trainer.student.adaptation_module.parameters())
        assert all(not p.requires_grad
                   for p in trainer.teacher.parameters())

        obs, _ = env.reset()
        assert obs["depth"].shape == (
            cfg.student.num_envs, cfg.camera.n_cams,
            cfg.camera.height, cfg.camera.width,
        )

        losses, frame_fracs = [], []
        for _ in range(200):
            action, m = trainer.step(obs)
            assert action.shape == (cfg.student.num_envs, cfg.action_dim)
            assert torch.isfinite(action).all()
            obs, _r, term, trunc, _i = env.step(action)
            trainer.update_dones(term, trunc)
            losses.append(m["dagger_loss"])
            frame_fracs.append(m["new_frame_frac"])

    first = sum(losses[:20]) / 20
    last = sum(losses[-20:]) / 20
    assert last < 0.5 * first, (
        f"DAgger loss did not decrease: first20={first:.5f} last20={last:.5f}"
    )
    print(f"  dagger loss first20={first:.5f} -> last20={last:.5f} "
          f"(ratio {last / first:.3f})")

    # Depth renders at 10 Hz while the policy runs at 50 Hz -> ~0.2 of steps
    # carry a fresh frame (slightly more due to forced post-reset frames).
    rate = sum(frame_fracs) / len(frame_fracs)
    assert 0.15 <= rate <= 0.5, f"depth_new_frame rate {rate:.3f} off 0.2"
    print(f"  depth_new_frame rate = {rate:.3f} (expected ~0.2)")


def test_gru_state_bridges_between_renders():
    """Between renders the latent is HELD (GRU state bridging); a new render
    changes it; an env reset clears it."""
    cfg = _cfg()
    env = MockEnv(cfg, 4, device="cpu")
    cfg.student.num_envs = 4
    with tempfile.TemporaryDirectory() as d:
        trainer = _make_trainer(cfg, d)
        obs, _ = env.reset()

        enc = trainer.student.actor.extero_encoder
        zero_action = torch.zeros(4, cfg.action_dim)
        latents, ticks = [], []
        for _ in range(11):
            _a, m = trainer.step(obs)
            latents.append(enc._e_cache.clone())
            ticks.append(obs["depth_new_frame"].clone())
            obs, _r, term, trunc, _i = env.step(zero_action)
            trainer.update_dones(term, trunc)

        for t in range(1, 11):
            held = ~ticks[t]
            if held.any():
                assert torch.allclose(
                    latents[t][held], latents[t - 1][held]
                ), f"latent changed without a new frame at step {t}"
        changed_on_tick = any(
            ticks[t].all() and not torch.allclose(latents[t], latents[t - 1])
            for t in range(1, 11)
        )
        assert changed_on_tick, "latent never changed despite new frames"


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
