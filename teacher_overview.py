"""
teacher_overview.py — run the TEACHER policy on the terrain grid, record a
fixed-angle overview MP4 (screenshot-style wall of robots), AND print
navigation metrics. Headless.

Run from repo root in the `isaac` conda env:

    PYTHONPATH=. python teacher_overview.py \
        --ckpt trained-pol-2.pt \
        --num_envs 256 --steps 800 --headless

Outputs under --out_dir (default ./teacher_out):
    overview.mp4   fixed world camera across the env grid
    metrics.txt    goals / falls / timeouts + success rate

WHY THIS MATTERS
    This is the teacher eval you've deferred. If the teacher's success rate is
    ~0.36 like the student, the student is faithfully copying a mediocre
    teacher -> fix Phase 1. If the teacher is ~0.8, distillation is the lossy
    step -> fix the student/depth path.

NOTE
    The teacher trains with cameras OFF (that's why it scales to 32k envs).
    This run forces rendering on for the video, so it is a short throwaway
    viz run, NOT your real training run.
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

_p = argparse.ArgumentParser(description="Teacher overview video + eval")
_p.add_argument("--ckpt", required=True, help="Teacher checkpoint (.pt)")
_p.add_argument("--robot", default="spot")
_p.add_argument("--num_envs", type=int, default=256)
_p.add_argument("--steps", type=int, default=800)
_p.add_argument("--out_dir", default="teacher_out")
_p.add_argument("--fps", type=int, default=25)
_p.add_argument("--seed", type=int, default=0)
# Camera placement (world meters). Defaults aim across the terrain grid from
# high and to the side, matching the screenshot's oblique overview.
# The terrain is a 4x4 grid of 8 m patches centered on the origin (~±16 m),
# and that extent is fixed regardless of --num_envs (extra envs share the 16
# terrain cells), so this framing — pulled back to the -x/-y corner and
# elevated, looking at the grid center — captures the whole wall of robots.
_p.add_argument("--cam_pos", type=float, nargs=3, default=[-22.0, -22.0, 16.0])
_p.add_argument("--cam_look", type=float, nargs=3, default=[0.0, 0.0, 0.0])
AppLauncher.add_app_launcher_args(_p)
args = _p.parse_args()
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
print("[INIT] Isaac Sim launched.", flush=True)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from omni_spot.configs import get_experiment_cfg  # noqa: E402
from omni_spot.networks import TeacherPolicy  # noqa: E402
from omni_spot.checkpoint import load_checkpoint  # noqa: E402

try:
    import imageio.v3 as iio
except ImportError:
    import imageio as iio  # type: ignore


def _look_at_quat(eye, target, up=(0.0, 0.0, 1.0)):
    """Quaternion (w,x,y,z) for a camera at `eye` looking at `target`.
    Uses USD/Isaac camera convention (-Z forward, +Y up) via 'world' offset."""
    eye = np.asarray(eye, float)
    target = np.asarray(target, float)
    up = np.asarray(up, float)
    fwd = target - eye
    fwd /= (np.linalg.norm(fwd) + 1e-9)
    right = np.cross(fwd, up)
    right /= (np.linalg.norm(right) + 1e-9)
    up2 = np.cross(right, fwd)
    # camera: X=right, Y=up2, Z=-fwd
    R = np.stack([right, up2, -fwd], axis=1)
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
            # _look_at_quat builds the rotation in the OpenGL camera frame
            # (-Z forward, +Y up). Isaac Lab's "world" convention is +X
            # forward / +Z up — a DIFFERENT axis — so tagging this "world"
            # aimed the cam off into the sky (the all-gray/all-black render).
            convention="opengl",
        ),
    )
    setattr(env_cfg.scene, "overview_cam", cam)


def _attach_overview_light(env_cfg):
    """Add scene lighting so the RGB overview isn't pure black.

    The pipeline never spawns a light — depth cameras (geometric z-buffer),
    scandot raycasts, and physics all need none — so an RGB render of the
    scene is completely unlit (black). A dome gives uniform ambient fill from
    all directions (guarantees everything is visible); a distant light adds a
    key from above for shadows/3D definition. Both are single static world
    prims (no {ENV_REGEX_NS}), so they spawn once and never enter the per-env
    reset path.
    """
    try:
        import isaaclab.sim as su
        from isaaclab.assets import AssetBaseCfg
    except ImportError:  # Isaac Lab 1.x fallback
        import omni.isaac.lab.sim as su  # type: ignore
        from omni.isaac.lab.assets import AssetBaseCfg  # type: ignore
    env_cfg.scene.overview_dome = AssetBaseCfg(
        prim_path="/World/overview_dome",
        spawn=su.DomeLightCfg(intensity=1500.0, color=(0.9, 0.9, 0.9)),
    )
    env_cfg.scene.overview_sun = AssetBaseCfg(
        prim_path="/World/overview_sun",
        spawn=su.DistantLightCfg(intensity=2000.0, color=(1.0, 1.0, 0.95)),
    )


def main() -> int:
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = get_experiment_cfg(args.robot)
    cfg.camera.enabled = False  # teacher uses scandots, not depth cams

    from omni_spot.env_cfg import build_env_cfg
    from omni_spot.nav_env import NavEnv
    from omni_spot.obs import build_critic_obs  # noqa: F401  (kept for parity)

    env_cfg = build_env_cfg(cfg, args.num_envs)
    env_cfg.seed = args.seed
    try:
        _attach_overview_light(env_cfg)
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] light attach failed ({e}); RGB overview may be dark",
              flush=True)
    try:
        _attach_overview_cam(env_cfg, args)
        have_cam = True
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] overview cam attach failed ({e}); metrics only", flush=True)
        have_cam = False

    print("[INIT] Building env (256 envs + overview cam)...", flush=True)
    env = NavEnv(env_cfg, cfg)
    device = env.device

    if have_cam:
        # overview_cam is a SINGLE fixed world prim (no {ENV_REGEX_NS}), so it
        # has exactly 1 instance — but InteractiveScene.reset(env_ids) indexes
        # EVERY scene sensor's buffers with the full per-env env_ids (size
        # num_envs), a CUDA out-of-bounds write on this cam's size-1 buffers
        # (device-side assert on env.reset()). Render/update never takes
        # external env_ids, so neutralize only this sensor's per-env reset.
        env.scene["overview_cam"].reset = lambda env_ids=None: None

    ckpt = load_checkpoint(args.ckpt, device)
    print(f"[INIT] ckpt phase={ckpt.get('phase')} robot={ckpt.get('robot')}", flush=True)
    if ckpt.get("phase") != "teacher":
        print(f"[WARN] expected phase=teacher, got {ckpt.get('phase')} — "
              f"loading anyway", flush=True)
    teacher = TeacherPolicy(cfg).to(device)
    teacher.load_state_dict(ckpt["model_state_dict"])
    teacher.eval()
    teacher.requires_grad_(False)

    obs, _ = env.reset()

    # ── Auto-aim the overview cam at where the robots ACTUALLY are ────────
    # The hardcoded cam_pos/cam_look guess where the grid is; instead read the
    # real per-env world origins (known only after reset) and frame their
    # centroid. set_world_poses_from_view uses Isaac Lab's own camera
    # convention, so it cannot get the -Z/+X axis mix-up wrong. This also
    # adapts the distance to however large the env grid turns out to be.
    if have_cam:
        try:
            origins = env.scene.env_origins  # (num_envs, 3) world positions
            center = origins.mean(dim=0)
            span_xy = (origins[:, :2].max(dim=0).values
                       - origins[:, :2].min(dim=0).values)
            radius = 0.5 * float(torch.linalg.norm(span_xy)) + 8.0  # +pad
            dist = max(radius / math.tan(math.radians(35.0)), 14.0)
            d = torch.tensor([-1.0, -1.0, 0.9], device=center.device)
            d = d / torch.linalg.norm(d)
            eye = center + d * dist
            target = center + torch.tensor(
                [0.0, 0.0, 0.3], device=center.device
            )
            print(f"[CAM] grid center="
                  f"{[round(v, 2) for v in center.tolist()]} "
                  f"radius={radius:.1f}m  eye="
                  f"{[round(v, 2) for v in eye.tolist()]}", flush=True)
            env.scene["overview_cam"].set_world_poses_from_view(
                eye.unsqueeze(0), target.unsqueeze(0)
            )
            print("[CAM] aimed via set_world_poses_from_view (auto-framed)",
                  flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[CAM][WARN] auto-frame failed ({e}); using cfg pose "
                  f"(--cam_pos/--cam_look)", flush=True)

    goals = falls = timeouts = episodes = 0
    frames: list[np.ndarray] = []

    print(f"[RUN] {args.steps} steps x {args.num_envs} envs...", flush=True)
    with torch.no_grad():
        for it in range(1, args.steps + 1):
            action = teacher.act_mean(obs["proprio"], obs["scandots"], obs["priv"])
            obs, _r, terminated, truncated, _i = env.step(action)

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
        f"phase          : {ckpt.get('phase')}  (TEACHER eval)",
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
