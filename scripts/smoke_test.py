"""
Smoke Test: Train Spot briefly + record evaluation video
==========================================================
Runs a short training session (10 updates) on Isaac Lab, then records
a few episodes of the (partially trained) policy navigating to goals.

Usage (inside Isaac Sim Python):
    python scripts/smoke_test.py --num_envs 64 --n_steps 128 --updates 10

For Colab:
    See notebooks/colab_spot_test.ipynb
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)


def parse_args():
    p = argparse.ArgumentParser(description="Spot RL Smoke Test")
    p.add_argument("--num_envs", type=int, default=64,
                   help="Number of parallel environments (lower for testing)")
    p.add_argument("--n_steps", type=int, default=128,
                   help="Rollout steps per update")
    p.add_argument("--updates", type=int, default=10,
                   help="Number of PPO updates to run")
    p.add_argument("--record_episodes", type=int, default=3,
                   help="Number of evaluation episodes to record")
    p.add_argument("--output_dir", type=str, default="smoke_test_output",
                   help="Directory for outputs (videos, checkpoints)")
    p.add_argument("--headless", action="store_true", default=True,
                   help="Run in headless mode (no GUI)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("  Spot RL Smoke Test — Isaac Lab")
    print(f"  Envs: {args.num_envs}  Steps: {args.n_steps}  Updates: {args.updates}")
    print("=" * 60)

    # ── GPU info ──────────────────────────────────────────────────
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        print(f"  GPU: {gpu.name} ({gpu.total_mem / 1e9:.1f} GB)")
    else:
        print("  WARNING: No CUDA GPU detected!")

    # ── Import Isaac Lab components ───────────────────────────────
    print("\n[1/5] Importing Isaac Lab...")
    t0 = time.time()
    try:
        from omni_spot.spot_env_cfg import SpotNavEnvCfg
        from omni_spot.spot_env import SpotNavEnv
        from omni_spot.ppo import PPOTrainer
        from omni_spot.physics_tuning import apply_tuning
        from omni_spot.video_recorder import record_episodes
        from omni_spot.diagnostics import print_diagnostics
    except ImportError as e:
        print(f"  ERROR: {e}")
        print("  Make sure Isaac Lab is installed and you're running")
        print("  with Isaac Sim's Python interpreter.")
        sys.exit(1)
    print(f"  Imports OK ({time.time() - t0:.1f}s)")

    # ── Create environment ────────────────────────────────────────
    print("\n[2/5] Creating environment...")
    t0 = time.time()

    env_cfg = SpotNavEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    apply_tuning(env_cfg.sim, env_cfg.scene)

    env = SpotNavEnv(cfg=env_cfg)
    print(f"  Environment created ({time.time() - t0:.1f}s)")
    print(f"  Observation space: depth=(B,5,120,160) + proprio=(B,37)")
    print(f"  Action space: (B, 12)")

    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        print(f"  GPU memory after env: {alloc:.2f} GB")

    # ── Create trainer ────────────────────────────────────────────
    print("\n[3/5] Creating PPO trainer...")
    trainer = PPOTrainer(
        n_envs=args.num_envs,
        n_steps=args.n_steps,
        lr=3e-4,
    )
    n_params = sum(p.numel() for p in trainer.net.parameters())
    print(f"  Network parameters: {n_params:,}")

    # ── Train ─────────────────────────────────────────────────────
    print(f"\n[4/5] Training for {args.updates} updates...")
    obs, _ = env.reset()

    train_start = time.time()
    for update in range(1, args.updates + 1):
        t_roll = time.time()
        obs, batch, stats = trainer.collect_rollout(
            env, obs, profile=(update == 1)
        )
        rollout_sec = time.time() - t_roll

        t_upd = time.time()
        update_info = trainer.update(batch)
        update_sec = time.time() - t_upd

        sps = (args.num_envs * args.n_steps) / (rollout_sec + update_sec)
        print(f"  [{update:>3d}/{args.updates}] "
              f"rew={stats['rew_mean']:>8.3f}  "
              f"eps={stats['ep_count']:>4d}  "
              f"roll={rollout_sec:.1f}s  upd={update_sec:.1f}s  "
              f"SPS={sps:,.0f}")

        # Print full diagnostics on first and last update
        if update == 1 or update == args.updates:
            diag = stats.get("_diag", {})
            if diag:
                print_diagnostics(update, diag, update_info)

    train_time = time.time() - train_start
    total_ts = args.num_envs * args.n_steps * args.updates
    print(f"\n  Training done: {total_ts:,} timesteps in {train_time:.1f}s "
          f"({total_ts / train_time:,.0f} SPS)")

    # Save checkpoint
    ckpt_path = os.path.join(args.output_dir, "smoke_test.pt")
    trainer.save(ckpt_path)
    print(f"  Checkpoint saved: {ckpt_path}")

    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"  Peak GPU memory: {peak:.2f} GB")

    # ── Record evaluation episodes ────────────────────────────────
    print(f"\n[5/5] Recording {args.record_episodes} evaluation episodes...")
    video_path = os.path.join(args.output_dir, "spot_eval.mp4")

    try:
        result = record_episodes(
            env=env,
            trainer=trainer,
            n_episodes=args.record_episodes,
            output_path=video_path,
            max_steps=500,
            fps=25,
        )
        if result:
            print(f"  Video saved: {result}")
        else:
            print("  Video recording returned no result (camera may not be configured)")
            print("  Trying viewport-based recording...")
            _try_viewport_record(args.output_dir)
    except Exception as e:
        print(f"  Video recording failed: {e}")
        print("  This is expected if no RGB camera is in the scene.")
        print("  See Task 2 below for rendered Spot visualization.")

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SMOKE TEST COMPLETE")
    print("=" * 60)
    print(f"  Total timesteps: {total_ts:,}")
    print(f"  Final mean reward: {stats['rew_mean']:.3f}")
    print(f"  Output directory: {args.output_dir}/")
    print()
    print("  Note: With only 10 updates, the policy is barely trained.")
    print("  Spot won't navigate well yet — this just verifies the")
    print("  environment, physics, and training loop work end-to-end.")


def _try_viewport_record(output_dir):
    """Fallback: try recording from Isaac Sim's viewport."""
    try:
        from omni_spot.video_recorder import record_from_viewport
        vp_path = os.path.join(output_dir, "spot_viewport.mp4")
        record_from_viewport(output_path=vp_path, n_frames=150, fps=30)
    except Exception as e:
        print(f"  Viewport recording also failed: {e}")


if __name__ == "__main__":
    main()
