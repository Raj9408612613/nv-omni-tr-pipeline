"""
teacher_train_overview.py — TEACHER training-curriculum overview video
======================================================================
Runs the trained TEACHER policy on the REAL terrain-curriculum generator
(flat / rough / stairs-up / stairs-down) but laid out so that **each robot
gets its own terrain patch** — no more "wall of robots" sharing one patch.
Records a fixed-angle oblique overview MP4 and prints navigation metrics.

This is the teacher counterpart to student_overview.py: same polished look
(oblique world camera with the correct OpenGL convention, dome + sun lights,
per-patch terrain coloring, small green goal spheres), but over the actual
training terrain types rather than the hand-built student courses.

KEY DIFFERENCES vs teacher_overview.py
    * ONE robot per terrain patch (env_origins pinned 1:1 to the generator
      grid; curriculum promotion disabled so robots stay put for the clip).
    * Bigger patches (default 32x32 m, was 8x8) so a full pyramid-up / inverted
      pyramid-down actually fits in a patch and the robot traverses it.
    * Rows read as difficulty (row 0 easiest), exactly like training.
    * Camera/lighting/coloring/goal-sphere treatment ported from
      student_test-hard.py (fixes the sky-cam + black-render bugs).

The teacher trains with cameras OFF (scandots only), which is why it scales to
32k envs. This run forces rendering on for the video, so it is a short
throwaway viz run, NOT a training run.

Run from repo root in the `isaac` conda env:

    PYTHONPATH=. python teacher_train_overview.py \
        --ckpt trained-pol-2.pt \
        --steps 800 --headless --enable_cameras

Outputs under --out_dir (default ./teacher_train_out):
    overview.mp4   fixed world camera over the curriculum grid
    metrics.txt    goals / falls / timeouts + success rate
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

_p = argparse.ArgumentParser(description="Teacher training-curriculum overview")
_p.add_argument("--ckpt", required=True, help="Teacher checkpoint (.pt)")
_p.add_argument("--robot", default="spot")
_p.add_argument("--num_envs", type=int, default=0,
                help="Robots on screen. 0 => rows*cols (one per patch).")
_p.add_argument("--steps", type=int, default=800)
_p.add_argument("--out_dir", default="teacher_train_out")
_p.add_argument("--fps", type=int, default=25)
_p.add_argument("--seed", type=int, default=0)
_p.add_argument("--patch_m", type=float, default=32.0,
                help="Side length of each terrain patch (m). Bigger => full "
                     "stairs fit and the robot traverses more terrain.")
_p.add_argument("--color_scheme", default="random",
                choices=["random", "height", "none"],
                help="Per-patch terrain coloring (viz only).")
# Overview camera (world meters). Auto-framed from the grid when left at None.
_p.add_argument("--cam_pos", type=float, nargs=3, default=None)
_p.add_argument("--cam_look", type=float, nargs=3, default=None)
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


def _isaac_imports():
    try:
        import isaaclab.sim as su
        from isaaclab.assets import AssetBaseCfg
        from isaaclab.sensors import CameraCfg
        try:
            from isaaclab.sensors import TiledCameraCfg as Cam
        except ImportError:
            Cam = CameraCfg
    except ImportError:
        import omni.isaac.lab.sim as su  # type: ignore
        from omni.isaac.lab.assets import AssetBaseCfg  # type: ignore
        from omni.isaac.lab.sensors import CameraCfg  # type: ignore
        try:
            from omni.isaac.lab.sensors import TiledCameraCfg as Cam  # type: ignore
        except ImportError:
            Cam = CameraCfg
    return su, AssetBaseCfg, Cam


def _look_at_quat(eye, target, up=(0.0, 0.0, 1.0)):
    """Quaternion (w,x,y,z) for a camera at `eye` looking at `target`, built in
    the OpenGL camera frame (-Z forward, +Y up) -> attach with convention
    'opengl' (NOT 'world', which aims it at the sky)."""
    eye = np.asarray(eye, float)
    target = np.asarray(target, float)
    up = np.asarray(up, float)
    fwd = target - eye
    fwd /= (np.linalg.norm(fwd) + 1e-9)
    right = np.cross(fwd, up)
    right /= (np.linalg.norm(right) + 1e-9)
    up2 = np.cross(right, fwd)
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


def _attach_overview_cam(env_cfg, cam_pos, cam_look):
    su, _AssetBaseCfg, Cam = _isaac_imports()
    quat = _look_at_quat(cam_pos, cam_look)
    cam = Cam(
        prim_path="/World/overview_cam",   # fixed world prim, not per-env
        update_period=0.0,                 # render every step
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=su.PinholeCameraCfg(
            focal_length=1.2,
            horizontal_aperture=2.0 * math.tan(math.radians(70.0 / 2)),
            clipping_range=(0.1, 600.0),
        ),
        offset=Cam.OffsetCfg(
            pos=tuple(float(v) for v in cam_pos), rot=quat,
            convention="opengl",
        ),
    )
    setattr(env_cfg.scene, "overview_cam", cam)


def _attach_lights(env_cfg):
    """A bare ground carries no lighting; without a light the robots and
    terrain render black. Dome gives uniform ambient fill; the distant 'sun'
    adds a key from above for shading/3D relief on the stairs."""
    su, AssetBaseCfg, _Cam = _isaac_imports()
    env_cfg.scene.dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=su.DomeLightCfg(intensity=1500.0, color=(0.9, 0.9, 0.95)),
    )
    env_cfg.scene.sun_light = AssetBaseCfg(
        prim_path="/World/SunLight",
        spawn=su.DistantLightCfg(intensity=2000.0, color=(1.0, 1.0, 0.95)),
    )


def _make_goal_markers(num_envs):
    """One small green sphere per env, repositioned to each robot's goal every
    step via VisualizationMarkers. Returns the markers handle or None."""
    try:
        try:
            import isaaclab.sim as su
            from isaaclab.markers import (
                VisualizationMarkers, VisualizationMarkersCfg,
            )
        except ImportError:
            import omni.isaac.lab.sim as su  # type: ignore
            from omni.isaac.lab.markers import (  # type: ignore
                VisualizationMarkers, VisualizationMarkersCfg,
            )
        cfg = VisualizationMarkersCfg(
            prim_path="/World/goal_markers",
            markers={
                "goal": su.SphereCfg(
                    radius=0.25,
                    visual_material=su.PreviewSurfaceCfg(
                        diffuse_color=(0.05, 0.90, 0.18),
                        emissive_color=(0.02, 0.45, 0.06),
                    ),
                )
            },
        )
        return VisualizationMarkers(cfg)
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] goal markers unavailable ({e}); skipping spheres",
              flush=True)
        return None


def _pin_one_robot_per_patch(env, rows, cols):
    """Override the terrain importer so env i sits alone on grid cell
    (row = i // cols  -> difficulty, col = i % cols). Returns True on success."""
    try:
        terrain = env.scene.terrain
        origins_grid = terrain.terrain_origins        # (rows, cols, 3)
        dev = env.device
        n = env.num_envs
        idx = torch.arange(n, device=dev)
        levels = torch.remainder(idx // cols, rows)
        types = torch.remainder(idx, cols)
        terrain.terrain_levels[:] = levels
        terrain.terrain_types[:] = types
        terrain.env_origins[:] = origins_grid[levels, types]
        return True
    except (AttributeError, TypeError, IndexError, RuntimeError) as e:
        print(f"[WARN] could not pin 1 robot/patch ({e}); using importer "
              "defaults", flush=True)
        return False


def main() -> int:
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = get_experiment_cfg(args.robot)
    cfg.camera.enabled = False              # teacher uses scandots, not depth
    cfg.curriculum.enabled = False          # fixed grid; robots stay on patch

    rows, cols = cfg.terrain.rows, cfg.terrain.cols
    num_envs = args.num_envs if args.num_envs > 0 else rows * cols

    # Bigger patches so full stairs fit, and widen the spawn/goal footprint so
    # the robot actually crosses the patch instead of sitting on the apex.
    cfg.terrain.patch_size = float(args.patch_m)
    cfg.terrain.patch_half = max(3.5, args.patch_m / 2.0 - 2.0)
    cfg.terrain.color_scheme = args.color_scheme
    cfg.goal.dist_range = (
        0.35 * cfg.terrain.patch_half, 0.90 * cfg.terrain.patch_half
    )

    from omni_spot.env_cfg import build_env_cfg
    from omni_spot.nav_env import NavEnv

    env_cfg = build_env_cfg(cfg, num_envs)
    env_cfg.seed = args.seed
    _attach_lights(env_cfg)

    # Auto-frame the oblique cam over the whole grid (terrain is centered at the
    # world origin). Span ~ grid extent; sit back along -Y, slightly elevated.
    span = max(rows, cols) * cfg.terrain.patch_size
    half = span / 2.0
    y_back = half / math.tan(math.radians(33.0)) + half + 4.0
    cam_pos = args.cam_pos or [0.0, -y_back, 0.62 * span + 3.0]
    cam_look = args.cam_look or [0.0, 0.0, 0.0]
    print(f"[CAM] pos={[round(v, 1) for v in cam_pos]} "
          f"look={[round(v, 1) for v in cam_look]}", flush=True)

    have_cam = True
    try:
        _attach_overview_cam(env_cfg, cam_pos, cam_look)
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] overview cam attach failed ({e}); metrics only",
              flush=True)
        have_cam = False

    print(f"[INIT] Building env: {num_envs} robots, "
          f"{rows}x{cols} patches @ {cfg.terrain.patch_size:.0f} m...",
          flush=True)
    env = NavEnv(env_cfg, cfg)
    device = env.device

    # The overview cam is a SINGLE global instance; InteractiveScene.reset()
    # resets every sensor with env_ids = arange(num_envs), and indexing its
    # size-1 buffer with [0..N-1] triggers a CUDA device-side assert. Reset its
    # one instance regardless of the per-env indices passed in.
    if have_cam:
        try:
            _ov = env.scene["overview_cam"]
            _ov_reset = _ov.reset
            _ov.reset = lambda env_ids=None, _r=_ov_reset: _r(None)
        except (KeyError, AttributeError):
            pass

    _pin_one_robot_per_patch(env, rows, cols)
    markers = _make_goal_markers(num_envs)

    ckpt = load_checkpoint(args.ckpt, device)
    print(f"[INIT] ckpt phase={ckpt.get('phase')} robot={ckpt.get('robot')}",
          flush=True)
    if ckpt.get("phase") != "teacher":
        print(f"[WARN] expected phase=teacher, got {ckpt.get('phase')} — "
              f"loading anyway", flush=True)
    teacher = TeacherPolicy(cfg).to(device)
    teacher.load_state_dict(ckpt["model_state_dict"])
    teacher.eval()
    teacher.requires_grad_(False)

    obs, _ = env.reset()
    goals = falls = timeouts = episodes = 0
    frames: list[np.ndarray] = []

    def _update_markers():
        if markers is None:
            return
        try:
            z = env.scene.env_origins[:, 2] + 0.4
            pos = torch.cat([env._goal, z.unsqueeze(-1)], dim=-1)
            markers.visualize(translations=pos)
        except Exception:  # noqa: BLE001
            pass

    print(f"[RUN] {args.steps} steps x {num_envs} envs...", flush=True)
    with torch.no_grad():
        for it in range(1, args.steps + 1):
            action = teacher.act_mean(
                obs["proprio"], obs["scandots"], obs["priv"]
            )
            obs, _r, terminated, truncated, _i = env.step(action)
            _update_markers()

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
                print(f"  [{it}/{args.steps}] episodes={episodes} "
                      f"goals={goals} falls={falls} timeouts={timeouts} "
                      f"success={sr:.3f}", flush=True)

    sr = goals / max(1, episodes)
    report = "\n".join([
        f"checkpoint     : {args.ckpt}",
        f"phase          : {ckpt.get('phase')}  (TEACHER training overview)",
        f"layout         : {num_envs} robots, {rows}x{cols} patches @ "
        f"{cfg.terrain.patch_size:.0f} m (1 robot/patch)",
        f"envs x steps   : {num_envs} x {args.steps}",
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
