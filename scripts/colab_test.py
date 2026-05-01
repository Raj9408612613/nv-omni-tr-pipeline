"""
Colab Smoke Test — Train Spot RL + Visualize
==============================================
Runs the full training pipeline with MockSpotEnv (no Isaac Sim needed).
Produces:
  1. Training curve plot
  2. Episode video (top-down 2D visualization)
  3. 3D Spot mesh render from MJCF OBJ files
  4. Diagnostic printout

Usage:
    python scripts/colab_test.py --num_envs 64 --updates 20 --output_dir /content/output
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num_envs", type=int, default=64)
    p.add_argument("--n_steps", type=int, default=128)
    p.add_argument("--updates", type=int, default=20)
    p.add_argument("--output_dir", type=str, default="colab_output")
    p.add_argument("--eval_episodes", type=int, default=3)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("  Spot RL — Colab Smoke Test (Mock Environment)")
    print(f"  Envs: {args.num_envs}  Steps: {args.n_steps}  Updates: {args.updates}")
    print("=" * 60)

    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        print(f"  GPU: {gpu.name} ({gpu.total_mem / 1e9:.1f} GB)")
    else:
        print("  Running on CPU")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Imports ───────────────────────────────────────────────────
    from omni_spot.mock_env import MockSpotEnv
    from omni_spot.ppo import PPOTrainer
    from omni_spot.diagnostics import print_diagnostics
    from omni_spot.config import ACTION_DIM

    # ── Create environment ────────────────────────────────────────
    print("\n[1/5] Creating mock environment...")
    env = MockSpotEnv(num_envs=args.num_envs, device=device, render_depth=True)
    print(f"  Mock env: {args.num_envs} envs, depth={env.render_depth}")

    # ── Create trainer ────────────────────────────────────────────
    print("[2/5] Creating PPO trainer...")
    trainer = PPOTrainer(
        n_envs=args.num_envs,
        n_steps=args.n_steps,
        lr=3e-4,
        device=device,
    )
    n_params = sum(p.numel() for p in trainer.net.parameters())
    print(f"  Network: {n_params:,} parameters")

    # ── Train ─────────────────────────────────────────────────────
    print(f"\n[3/5] Training for {args.updates} updates...")
    obs, _ = env.reset()
    train_start = time.time()

    rewards_history = []
    loss_history = []

    for update in range(1, args.updates + 1):
        t0 = time.time()
        obs, batch, stats = trainer.collect_rollout(env, obs, profile=(update == 1))
        rollout_sec = time.time() - t0

        t0 = time.time()
        update_info = trainer.update(batch)
        update_sec = time.time() - t0

        sps = (args.num_envs * args.n_steps) / (rollout_sec + update_sec)
        rewards_history.append(stats["rew_mean"])
        loss_history.append(update_info.get("total_loss", 0))

        print(f"  [{update:>3d}/{args.updates}] "
              f"rew={stats['rew_mean']:>8.3f}  "
              f"eps={stats['ep_count']:>4d}  "
              f"loss={update_info.get('total_loss', 0):.4f}  "
              f"SPS={sps:,.0f}")

        if update == 1 or update == args.updates:
            diag = stats.get("_diag", {})
            if diag:
                print_diagnostics(update, diag, update_info)

    train_time = time.time() - train_start
    total_ts = args.num_envs * args.n_steps * args.updates
    print(f"\n  Done: {total_ts:,} timesteps in {train_time:.1f}s "
          f"({total_ts / train_time:,.0f} SPS)")

    ckpt_path = os.path.join(args.output_dir, "mock_train.pt")
    trainer.save(ckpt_path)
    print(f"  Checkpoint: {ckpt_path}")

    # ── Plot training curve ───────────────────────────────────────
    print("\n[4/5] Generating plots...")
    _plot_training_curve(rewards_history, loss_history, args.output_dir)

    # ── Evaluate + record episodes ────────────────────────────────
    print(f"\n[5/5] Recording {args.eval_episodes} evaluation episodes...")
    _record_eval_episodes(env, trainer, args.eval_episodes, args.output_dir, device)

    # ── Render Spot mesh ──────────────────────────────────────────
    print("\nBonus: Rendering Spot 3D mesh...")
    _render_spot_3d(args.output_dir)

    print(f"\n{'=' * 60}")
    print(f"  SMOKE TEST COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Output directory: {args.output_dir}/")
    for f in sorted(os.listdir(args.output_dir)):
        fpath = os.path.join(args.output_dir, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    {f} ({size_kb:.0f} KB)")
    print()
    print(f"  Final mean reward: {rewards_history[-1]:.3f}")
    print(f"  Reward trend: {rewards_history[0]:.3f} -> {rewards_history[-1]:.3f}")
    if rewards_history[-1] > rewards_history[0]:
        print(f"  Policy IS learning (reward increased)")
    else:
        print(f"  Policy may need more updates to show clear improvement")


def _plot_training_curve(rewards, losses, output_dir):
    """Plot reward and loss curves."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(rewards, "b-", linewidth=2)
        ax1.set_xlabel("Update")
        ax1.set_ylabel("Mean Reward")
        ax1.set_title("Training Reward Curve")
        ax1.grid(True, alpha=0.3)

        ax2.plot(losses, "r-", linewidth=2)
        ax2.set_xlabel("Update")
        ax2.set_ylabel("Total Loss")
        ax2.set_title("PPO Loss")
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        path = os.path.join(output_dir, "training_curve.png")
        fig.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved: {path}")
    except ImportError:
        print("  matplotlib not available, skipping plot")


def _record_eval_episodes(env, trainer, n_episodes, output_dir, device):
    """Record top-down 2D visualization of evaluation episodes."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, Rectangle, FancyArrowPatch
        import imageio
    except ImportError:
        print("  matplotlib/imageio not available, skipping video")
        return

    trainer.net.eval()
    frames = []
    episodes_done = 0
    max_steps = 300

    obs, _ = env.reset()
    env.start_logging()

    with torch.no_grad():
        for step in range(n_episodes * max_steps):
            action, _, _, _ = trainer.sample_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = (terminated | truncated)
            if done[0]:
                episodes_done += 1
                if episodes_done >= n_episodes:
                    break

    log = env.stop_logging()
    if not log:
        print("  No trajectory data captured")
        return

    # Generate frames from trajectory log
    print(f"  Rendering {len(log)} frames...")
    for i, frame_data in enumerate(log):
        if i % 2 != 0:  # Skip every other frame for speed
            continue
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        _draw_topdown_frame(ax, frame_data)
        fig.canvas.draw()

        # Convert to numpy array
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(h, w, 3)
        frames.append(buf)
        plt.close(fig)

    if frames:
        video_path = os.path.join(output_dir, "spot_eval_topdown.mp4")
        imageio.mimsave(video_path, frames, fps=25)
        print(f"  Video saved: {video_path} ({len(frames)} frames)")

        # Also save a single frame as PNG
        still_path = os.path.join(output_dir, "spot_eval_frame.png")
        imageio.imwrite(still_path, frames[len(frames) // 2])
        print(f"  Still saved: {still_path}")


def _draw_topdown_frame(ax, data):
    """Draw a single top-down frame of the environment."""
    from matplotlib.patches import Circle, Rectangle, FancyArrowPatch

    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(-5.5, 5.5)
    ax.set_aspect("equal")
    ax.set_facecolor("#e8e8e0")

    # Room walls
    for x, y, w, h in [(-5, -5, 10, 0.2), (-5, 4.8, 10, 0.2),
                        (-5, -5, 0.2, 10), (4.8, -5, 0.2, 10)]:
        ax.add_patch(Rectangle((x, y), w, h, color="#888888"))

    # Obstacles
    obs_pos = data["obs_pos"].numpy()
    from omni_spot.config import OBS_HALF_SIZES, N_STATIC, N_DYNAMIC
    for j in range(len(obs_pos)):
        ox, oy, oz = obs_pos[j]
        if ox > 50:  # off-scene
            continue
        hs = OBS_HALF_SIZES[j] if j < len(OBS_HALF_SIZES) else [0.3, 0.3, 0.5]
        if j < N_STATIC:
            color = "#cc6633"
            ax.add_patch(Rectangle(
                (ox - hs[0], oy - hs[1]), hs[0] * 2, hs[1] * 2,
                color=color, alpha=0.7
            ))
        elif j < N_STATIC + N_DYNAMIC:
            ax.add_patch(Circle((ox, oy), hs[0], color="#3399cc", alpha=0.7))
        else:
            ax.add_patch(Circle((ox, oy), 0.3, color="#cc3333", alpha=0.8))
            ax.annotate("H", (ox, oy), ha="center", va="center",
                        fontsize=8, fontweight="bold", color="white")

    # Goal
    gx, gy = data["goal_pos"].numpy()
    ax.add_patch(Circle((gx, gy), 0.5, color="#22cc22", alpha=0.3))
    ax.plot(gx, gy, "g*", markersize=15)

    # Robot
    rx, ry = data["root_pos"][0].item(), data["root_pos"][1].item()
    qw, _, _, qz = data["root_quat"].numpy()
    yaw = 2 * math.atan2(qz, qw)

    # Robot body (rectangle showing orientation)
    body_l, body_w = 0.6, 0.3
    corners = np.array([
        [-body_l / 2, -body_w / 2],
        [body_l / 2, -body_w / 2],
        [body_l / 2, body_w / 2],
        [-body_l / 2, body_w / 2],
    ])
    R = np.array([[math.cos(yaw), -math.sin(yaw)],
                  [math.sin(yaw), math.cos(yaw)]])
    corners = corners @ R.T + np.array([rx, ry])
    from matplotlib.patches import Polygon
    ax.add_patch(Polygon(corners, closed=True, color="#1a1a1a", alpha=0.9))

    # Heading arrow
    hx = rx + 0.5 * math.cos(yaw)
    hy = ry + 0.5 * math.sin(yaw)
    ax.annotate("", xy=(hx, hy), xytext=(rx, ry),
                arrowprops=dict(arrowstyle="->", color="yellow", lw=2))

    # Info text
    dist_to_goal = math.sqrt((rx - gx) ** 2 + (ry - gy) ** 2)
    rew = data["reward"]
    ax.set_title(
        f"Dist: {dist_to_goal:.2f}m | Rew: {rew:.2f} | "
        f"{'DONE' if data['done'] else 'running'}",
        fontsize=10,
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")


def _render_spot_3d(output_dir):
    """Render Spot's OBJ meshes as a 3D visualization."""
    import glob
    mesh_dir = os.path.join(PROJECT_ROOT, "models", "assets")
    obj_files = sorted(glob.glob(os.path.join(mesh_dir, "*.obj")))

    if not obj_files:
        print("  No OBJ files found in models/assets/")
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError:
        print("  matplotlib 3D not available")
        return

    print(f"  Loading {len(obj_files)} OBJ meshes...")

    all_verts = []
    all_faces = []
    body_colors = []

    for obj_path in obj_files:
        name = os.path.basename(obj_path)
        verts, faces = _load_obj(obj_path)
        if verts is None:
            continue

        # Color based on part name
        if "body" in name:
            color = (0.15, 0.15, 0.15, 0.9)
        elif "hip" in name:
            color = (0.2, 0.2, 0.2, 0.9)
        elif "upper" in name:
            if "wrap" in name or "_1" in name:
                color = (0.88, 0.67, 0.23, 0.9)  # gold wrap
            else:
                color = (0.15, 0.15, 0.15, 0.9)
        elif "lower" in name:
            color = (0.15, 0.15, 0.15, 0.9)
        elif "collision" in name:
            continue  # skip collision meshes
        else:
            color = (0.3, 0.3, 0.3, 0.9)

        all_verts.append(verts)
        all_faces.append(faces)
        body_colors.append(color)

    if not all_verts:
        print("  Could not parse any OBJ meshes")
        return

    # Render from multiple angles
    angles = [
        ("front_3q", 25, -45),
        ("side", 20, 0),
        ("top", 80, -45),
    ]

    for view_name, elev, azim in angles:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        for verts, faces, color in zip(all_verts, all_faces, body_colors):
            # Subsample faces for speed
            step = max(1, len(faces) // 2000)
            for face in faces[::step]:
                try:
                    tri = [verts[idx] for idx in face]
                    poly = Poly3DCollection([tri], alpha=color[3])
                    poly.set_facecolor(color[:3])
                    poly.set_edgecolor((0.1, 0.1, 0.1, 0.2))
                    ax.add_collection3d(poly)
                except (IndexError, ValueError):
                    continue

        # Set limits
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.3, 0.3)
        ax.set_zlim(-0.4, 0.2)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"Spot Robot — {view_name}")
        ax.set_facecolor("white")

        path = os.path.join(output_dir, f"spot_3d_{view_name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")


def _load_obj(path):
    """Load vertices and faces from an OBJ file."""
    verts = []
    faces = []
    try:
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                if parts[0] == "v" and len(parts) >= 4:
                    verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif parts[0] == "f":
                    # OBJ face indices are 1-based, may have v/vt/vn format
                    face = []
                    for p in parts[1:]:
                        idx = int(p.split("/")[0]) - 1
                        face.append(idx)
                    if len(face) >= 3:
                        faces.append(face[:3])  # triangulate
    except Exception:
        return None, None

    if not verts:
        return None, None
    return verts, faces


if __name__ == "__main__":
    main()
