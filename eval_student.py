"""
eval_student.py — roll out a trained StudentPolicy, print navigation metrics,
and record an MP4. Headless-friendly (no GUI needed).

Run from the repo root, in the `isaac` conda env:

    PYTHONPATH=. python eval_student.py \
        --ckpt student_best_20260615.pt \
        --num_envs 64 --steps 1000 --headless --enable_cameras

Outputs (under --out_dir, default ./eval_out):
    metrics.txt           goal / fall / timeout counts + success rate
    student_pov_depth.mp4  what the policy sees (depth, guaranteed to render)
    student_chase_rgb.mp4  third-person RGB (best-effort; may be absent)

WHAT THE NUMBERS MEAN
    success_rate = goals / episodes_ended. This is the real verdict, NOT the
    distillation loss. A low loss only means the student copied the teacher's
    outputs; this measures whether those outputs actually navigate.
"""

from __future__ import annotations

import argparse
import os
import sys

# ── Step 1: launch Isaac Sim BEFORE importing isaaclab submodules ────────────
try:
    from isaaclab.app import AppLauncher
except ImportError:
    from omni.isaac.lab.app import AppLauncher  # type: ignore

_p = argparse.ArgumentParser(description="Eval a trained student policy + record video")
_p.add_argument("--ckpt", required=True, help="Path to student checkpoint (.pt)")
_p.add_argument("--robot", default="spot")
_p.add_argument("--num_envs", type=int, default=64)
_p.add_argument("--steps", type=int, default=1000, help="Rollout length (env steps)")
_p.add_argument("--out_dir", default="eval_out")
_p.add_argument("--fps", type=int, default=25)
_p.add_argument("--seed", type=int, default=0)
_p.add_argument("--no_rgb", action="store_true", help="Skip the RGB chase cam, depth video only")
AppLauncher.add_app_launcher_args(_p)
args = _p.parse_args()
args.enable_cameras = True  # student needs cameras; eval video needs them too

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
print("[INIT] Isaac Sim launched.", flush=True)

# ── Step 2: now safe to import sim-dependent modules ─────────────────────────
import numpy as np  # noqa: E402
import torch  # noqa: E402

from omni_spot.configs import get_experiment_cfg  # noqa: E402
from omni_spot.networks import StudentPolicy  # noqa: E402
from omni_spot.checkpoint import load_checkpoint  # noqa: E402

try:
    import imageio.v3 as iio
except ImportError:
    import imageio as iio  # type: ignore


# ── Resolve Isaac Lab camera classes (same version-fallback as env_cfg) ──────
def _camera_classes():
    try:
        import isaaclab.sim as sim_utils
        from isaaclab.sensors import CameraCfg
        try:
            from isaaclab.sensors import TiledCameraCfg as Cam
        except ImportError:
            Cam = CameraCfg
    except ImportError:
        import omni.isaac.lab.sim as sim_utils  # type: ignore
        from omni.isaac.lab.sensors import CameraCfg  # type: ignore
        try:
            from omni.isaac.lab.sensors import TiledCameraCfg as Cam  # type: ignore
        except ImportError:
            Cam = CameraCfg
    return sim_utils, Cam


def _attach_rgb_cam(env_cfg, cfg):
    """Attach a third-person RGB chase camera to env 0's robot base body.

    LOW CONFIDENCE: optical-axis quaternion + camera class vary by Isaac Lab
    version. If this produces no frames the script falls back to depth video.
    """
    sim_utils, Cam = _camera_classes()
    import math
    cam = Cam(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{cfg.robot.cam_mount_body}/video_cam",
        update_period=cfg.sim.control_dt,  # render every control step for smooth video
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=1.0,
            horizontal_aperture=2.0 * math.tan(math.radians(90.0 / 2)),
            clipping_range=(0.05, 50.0),
        ),
        # Behind + above the body, looking forward (body +X). Same ros optical
        # convention the depth cam uses.
        offset=Cam.OffsetCfg(
            pos=(-1.4, 0.0, 0.7),
            rot=(0.5, -0.5, 0.5, -0.5),
            convention="ros",
        ),
    )
    setattr(env_cfg.scene, "video_cam", cam)


def _depth_to_rgb(depth_img: torch.Tensor, max_depth: float) -> np.ndarray:
    """(H, W) depth -> (H, W, 3) uint8 grayscale (near=white, far=black)."""
    d = depth_img.detach().float().cpu().numpy()
    d = np.clip(d / max_depth, 0.0, 1.0)
    g = ((1.0 - d) * 255).astype(np.uint8)
    return np.stack([g, g, g], axis=-1)


def main() -> int:
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = get_experiment_cfg(args.robot)
    cfg.camera.enabled = True  # build the depth rig (student exteroception)

    from omni_spot.env_cfg import build_env_cfg
    from omni_spot.nav_env import NavEnv

    env_cfg = build_env_cfg(cfg, args.num_envs)
    env_cfg.seed = args.seed

    want_rgb = not args.no_rgb
    if want_rgb:
        try:
            _attach_rgb_cam(env_cfg, cfg)
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] RGB cam attach failed ({e}); depth video only", flush=True)
            want_rgb = False

    print("[INIT] Building env...", flush=True)
    env = NavEnv(env_cfg, cfg)
    device = env.device

    # ── Load student ─────────────────────────────────────────────────────────
    ckpt = load_checkpoint(args.ckpt, device)
    print(f"[INIT] ckpt phase={ckpt.get('phase')} robot={ckpt.get('robot')}", flush=True)
    student = StudentPolicy(cfg).to(device)
    student.load_state_dict(ckpt["model_state_dict"])
    student.eval()
    student.requires_grad_(False)

    obs, _ = env.reset()
    prev_done = torch.ones(args.num_envs, dtype=torch.bool, device=device)

    goals = falls = timeouts = episodes = 0
    depth_frames: list[np.ndarray] = []
    rgb_frames: list[np.ndarray] = []

    print(f"[RUN] {args.steps} steps x {args.num_envs} envs...", flush=True)
    with torch.no_grad():
        for it in range(1, args.steps + 1):
            depth = obs["depth"] / cfg.camera.max_depth
            action = student.act_mean(
                obs["proprio"], depth, obs["depth_new_frame"],
                obs["history"], reset_mask=prev_done,
            )
            obs, _reward, terminated, truncated, _info = env.step(action)
            prev_done = terminated | truncated

            # Metrics: split goal vs fall via the env's cached flags.
            goals += int(env._at_goal.sum())
            falls += int(env._fallen.sum())
            timeouts += int((truncated & ~terminated).sum())
            episodes += int((terminated | truncated).sum())

            # Depth POV of env 0 (cam dim 0): guaranteed available
            depth_frames.append(_depth_to_rgb(obs["depth"][0, 0], cfg.camera.max_depth))

            # RGB chase of env 0: best-effort
            if want_rgb:
                try:
                    rgb = env.scene["video_cam"].data.output["rgb"]
                    f = rgb[0].detach().cpu().numpy()
                    if f.shape[-1] == 4:
                        f = f[..., :3]
                    rgb_frames.append(f.astype(np.uint8))
                except (KeyError, AttributeError, RuntimeError, IndexError):
                    pass

            if it % 100 == 0:
                sr = goals / max(1, episodes)
                print(f"  [{it}/{args.steps}] episodes={episodes} "
                      f"goals={goals} falls={falls} timeouts={timeouts} "
                      f"success_rate={sr:.3f}", flush=True)

    # ── Write metrics ────────────────────────────────────────────────────────
    sr = goals / max(1, episodes)
    lines = [
        f"checkpoint     : {args.ckpt}",
        f"phase          : {ckpt.get('phase')}",
        f"envs x steps   : {args.num_envs} x {args.steps}",
        f"episodes ended : {episodes}",
        f"goals reached  : {goals}",
        f"falls/collide  : {falls}",
        f"timeouts       : {timeouts}",
        f"SUCCESS RATE   : {sr:.3f}   (goals / episodes_ended)",
    ]
    report = "\n".join(lines)
    print("\n" + report, flush=True)
    with open(os.path.join(args.out_dir, "metrics.txt"), "w") as f:
        f.write(report + "\n")

    # ── Write videos ─────────────────────────────────────────────────────────
    if depth_frames:
        p = os.path.join(args.out_dir, "student_pov_depth.mp4")
        iio.imwrite(p, np.stack(depth_frames), fps=args.fps, codec="h264")
        print(f"[VIDEO] {p}  ({len(depth_frames)} frames)", flush=True)
    if rgb_frames:
        p = os.path.join(args.out_dir, "student_chase_rgb.mp4")
        iio.imwrite(p, np.stack(rgb_frames), fps=args.fps, codec="h264")
        print(f"[VIDEO] {p}  ({len(rgb_frames)} frames)", flush=True)
    elif want_rgb:
        print("[VIDEO] RGB chase cam produced no frames — see depth video.", flush=True)

    env.close()
    return 0


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stderr.flush()
    finally:
        simulation_app.close()
    sys.exit(code)
