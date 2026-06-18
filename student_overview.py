"""
student_overview.py — run the trained STUDENT policy on the terrain/parkour
curriculum, record a fixed-angle oblique overhead MP4 (the "wall of robots"
overview), AND print navigation metrics. Headless.

This is the STUDENT counterpart of teacher_overview.py: same world overview
camera, but the policy is the deployable depth-CNN+GRU StudentPolicy (cameras
ON), so the video shows what the distilled student actually does on the grid.

Run from the repo root in the `isaac` conda env:

    PYTHONPATH=. python student_overview.py \
        --ckpt student_best_20260615.pt \
        --num_envs 256 --steps 800 --headless --enable_cameras

Outputs under --out_dir (default ./student_overview_out):
    overview.mp4   fixed world camera across the env grid (robots seeking goals)
    metrics.txt    goals / falls / timeouts + success rate

NOTES
    * The student needs the depth rig, so cameras are forced ON; this is a
      short throwaway viz run (rendering-bound), NOT a training run.
    * --enable_cameras is set automatically; keep it for clarity.
    * If VRAM is tight, lower --num_envs (per-env depth cams + the 1280x720
      overview cam both render every step). Tune the framing with --cam_pos /
      --cam_look if the grid grows with more envs.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

try:
    from isaaclab.app import AppLauncher
except ImportError:
    from omni.isaac.lab.app import AppLauncher  # type: ignore

_p = argparse.ArgumentParser(description="Student overview video + eval")
_p.add_argument("--ckpt", required=True, help="Student checkpoint (.pt)")
_p.add_argument("--robot", default="spot")
_p.add_argument("--num_envs", type=int, default=256)
_p.add_argument("--steps", type=int, default=800)
_p.add_argument("--out_dir", default="student_overview_out")
_p.add_argument("--fps", type=int, default=25)
_p.add_argument("--seed", type=int, default=0)
# Camera placement (world meters). Defaults aim across the terrain grid from
# high and to the side, matching the oblique overview screenshot.
_p.add_argument("--cam_pos", type=float, nargs=3, default=[-12.0, -12.0, 9.0])
_p.add_argument("--cam_look", type=float, nargs=3, default=[6.0, 6.0, 0.0])
AppLauncher.add_app_launcher_args(_p)
args = _p.parse_args()
args.enable_cameras = True  # student needs depth cams; overview also renders

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
print("[INIT] Isaac Sim launched.", flush=True)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from omni_spot.configs import get_experiment_cfg  # noqa: E402
from omni_spot.networks import StudentPolicy  # noqa: E402
from omni_spot.checkpoint import load_checkpoint  # noqa: E402

try:
    import imageio.v3 as iio
except ImportError:
    import imageio as iio  # type: ignore


def _look_at_quat(eye, target, up=(0.0, 0.0, 1.0)):
    """Quaternion (w,x,y,z) for a camera at `eye` looking at `target`.
    USD/Isaac camera convention (-Z forward, +Y up) via the 'world' offset."""
    eye = np.asarray(eye, float)
    target = np.asarray(target, float)
    up = np.asarray(up, float)
    fwd = target - eye
    fwd /= (np.linalg.norm(fwd) + 1e-9)
    right = np.cross(fwd, up)
    right /= (np.linalg.norm(right) + 1e-9)
    up2 = np.cross(right, fwd)
    R = np.stack([right, up2, -fwd], axis=1)  # camera: X=right, Y=up2, Z=-fwd
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        s = 0.5 / math.sqrt(t + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * math.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s; x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * math.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * math.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    q = np.array([w, x, y, z])
    return tuple(float(v) for v in q / (np.linalg.norm(q) + 1e-9))


def _camera_classes():
    try:
        import isaaclab.sim as su
        from isaaclab.sensors import CameraCfg
        try:
            from isaaclab.sensors import TiledCameraCfg as Cam
        except ImportError:
            Cam = CameraCfg
    except ImportError:
        import omni.isaac.lab.sim as su  # type: ignore
        from omni.isaac.lab.sensors import CameraCfg  # type: ignore
        try:
            from omni.isaac.lab.sensors import TiledCameraCfg as Cam  # type: ignore
        except ImportError:
            Cam = CameraCfg
    return su, Cam


def _attach_overview_cam(env_cfg, args):
    """One FIXED camera in world space (NOT per-env, NOT robot-attached)."""
    su, Cam = _camera_classes()
    quat = _look_at_quat(args.cam_pos, args.cam_look)
    cam = Cam(
        prim_path="/World/overview_cam",   # fixed world prim, no ENV_REGEX_NS
        update_period=0.0,                 # render every step
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=su.PinholeCameraCfg(
            focal_length=1.2,
            horizontal_aperture=2.0 * math.tan(math.radians(70.0 / 2)),
            clipping_range=(0.1, 200.0),
        ),
        offset=Cam.OffsetCfg(
            pos=tuple(float(v) for v in args.cam_pos),
            rot=quat,
            convention="world",
        ),
    )
    setattr(env_cfg.scene, "overview_cam", cam)


def main() -> int:
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = get_experiment_cfg(args.robot)
    cfg.camera.enabled = True  # student exteroception = depth rig

    from omni_spot.env_cfg import build_env_cfg
    from omni_spot.nav_env import NavEnv

    env_cfg = build_env_cfg(cfg, args.num_envs)
    env_cfg.seed = args.seed
    try:
        _attach_overview_cam(env_cfg, args)
        have_cam = True
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] overview cam attach failed ({e}); metrics only", flush=True)
        have_cam = False

    print(f"[INIT] Building env ({args.num_envs} envs + depth rig + overview "
          f"cam)...", flush=True)
    env = NavEnv(env_cfg, cfg)
    device = env.device

    ckpt = load_checkpoint(args.ckpt, device)
    print(f"[INIT] ckpt phase={ckpt.get('phase')} robot={ckpt.get('robot')}",
          flush=True)
    if ckpt.get("phase") != "student":
        print(f"[WARN] expected phase=student, got {ckpt.get('phase')} — "
              f"loading anyway", flush=True)
    student = StudentPolicy(cfg).to(device)
    student.load_state_dict(ckpt["model_state_dict"])
    student.eval()
    student.requires_grad_(False)

    obs, _ = env.reset()
    prev_done = torch.ones(args.num_envs, dtype=torch.bool, device=device)
    goals = falls = timeouts = episodes = 0
    frames: list[np.ndarray] = []

    print(f"[RUN] {args.steps} steps x {args.num_envs} envs...", flush=True)
    with torch.no_grad():
        for it in range(1, args.steps + 1):
            # Student inference: depth normalized by max_depth, GRU state reset
            # for envs that ended last step (same convention as eval_student).
            depth = obs["depth"] / cfg.camera.max_depth
            action = student.act_mean(
                obs["proprio"], depth, obs["depth_new_frame"],
                obs["history"], reset_mask=prev_done,
            )
            obs, _r, terminated, truncated, _i = env.step(action)
            prev_done = terminated | truncated

            goals += int(env._at_goal.sum())
            falls += int(env._fallen.sum())
            timeouts += int((truncated & ~terminated).sum())
            episodes += int((terminated | truncated).sum())

            if have_cam:
                try:
                    rgb = env.scene["overview_cam"].data.output["rgb"]
                    f = rgb[0].detach().cpu().numpy()
                    if f.shape[-1] == 4:
                        f = f[..., :3]
                    frames.append(f.astype(np.uint8))
                except (KeyError, AttributeError, RuntimeError, IndexError):
                    pass

            if it % 100 == 0:
                sr = goals / max(1, episodes)
                print(f"  [{it}/{args.steps}] episodes={episodes} goals={goals} "
                      f"falls={falls} timeouts={timeouts} success={sr:.3f}",
                      flush=True)

    sr = goals / max(1, episodes)
    report = "\n".join([
        f"checkpoint     : {args.ckpt}",
        f"phase          : {ckpt.get('phase')}  (STUDENT eval)",
        f"envs x steps   : {args.num_envs} x {args.steps}",
        f"episodes ended : {episodes}",
        f"goals reached  : {goals}",
        f"falls/collide  : {falls}",
        f"timeouts       : {timeouts}",
        f"SUCCESS RATE   : {sr:.3f}   (goals / episodes_ended)",
    ])
    print("\n" + report, flush=True)
    with open(os.path.join(args.out_dir, "metrics.txt"), "w") as f:
        f.write(report + "\n")

    if frames:
        p = os.path.join(args.out_dir, "overview.mp4")
        iio.imwrite(p, np.stack(frames), fps=args.fps, codec="h264")
        print(f"[VIDEO] {p}  ({len(frames)} frames)", flush=True)
    else:
        print("[VIDEO] no frames captured — check cam placement", flush=True)

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
