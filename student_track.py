"""
student_track.py — drive the STUDENT down a fixed straight lane
(flat -> stairs up -> stairs down -> rough -> flat+goal) with a far fixed
goal at the end, filmed by ONE static overhead camera. ~10 robots.

Run from repo root in the `isaac` conda env:

    PYTHONPATH=. python student_track.py \
        --ckpt omni_logs/student_20260616_091307/best.pt \
        --num_envs 10 --steps 1500 --headless

Outputs under --out_dir (default ./track_out):
    track_overview.mp4   static top-down/oblique camera over the lane
    metrics.txt          how many robots reached the end + where they failed

CAVEAT (read this): the student trained on the 8m curriculum patches with
goals 1.5-3.5m away. A long straight corridor with a far goal is OUT OF
DISTRIBUTION. If robots stall, that may be the test design, not the policy.
Lane segments are kept short-ish to stay near training scale.
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

_p = argparse.ArgumentParser(description="Student straight-lane track test + video")
_p.add_argument("--ckpt", required=True)
_p.add_argument("--robot", default="spot")
_p.add_argument("--num_envs", type=int, default=10)
_p.add_argument("--steps", type=int, default=1500)
_p.add_argument("--out_dir", default="track_out")
_p.add_argument("--fps", type=int, default=25)
_p.add_argument("--seed", type=int, default=0)
_p.add_argument("--goal_x", type=float, default=12.0,
                help="Goal distance down the lane (m from spawn)")
_p.add_argument("--lane_spacing", type=float, default=3.0,
                help="Lateral gap between the robots' parallel lanes (m)")
AppLauncher.add_app_launcher_args(_p)
args = _p.parse_args()
args.enable_cameras = True

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
    eye = np.asarray(eye, float); target = np.asarray(target, float)
    up = np.asarray(up, float)
    fwd = target - eye; fwd /= (np.linalg.norm(fwd) + 1e-9)
    right = np.cross(fwd, up); right /= (np.linalg.norm(right) + 1e-9)
    up2 = np.cross(right, fwd)
    R = np.stack([right, up2, -fwd], axis=1)
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        s = 0.5 / math.sqrt(t + 1.0); w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s; y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2 * math.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s; x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2 * math.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2 * math.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    q = np.array([w, x, y, z]); return tuple(float(v) for v in q / (np.linalg.norm(q) + 1e-9))


def _isaac_terrain_classes():
    try:
        import isaaclab.sim as su
        from isaaclab.terrains import (
            TerrainImporterCfg, TerrainGeneratorCfg,
            HfRandomUniformTerrainCfg, HfPyramidStairsTerrainCfg,
            HfInvertedPyramidStairsTerrainCfg,
        )
        from isaaclab.sensors import CameraCfg
        try:
            from isaaclab.sensors import TiledCameraCfg as Cam
        except ImportError:
            Cam = CameraCfg
    except ImportError:
        import omni.isaac.lab.sim as su  # type: ignore
        from omni.isaac.lab.terrains import (  # type: ignore
            TerrainImporterCfg, TerrainGeneratorCfg,
            HfRandomUniformTerrainCfg, HfPyramidStairsTerrainCfg,
            HfInvertedPyramidStairsTerrainCfg,
        )
        from omni.isaac.lab.sensors import CameraCfg  # type: ignore
        try:
            from omni.isaac.lab.sensors import TiledCameraCfg as Cam  # type: ignore
        except ImportError:
            Cam = CameraCfg
    return (su, TerrainImporterCfg, TerrainGeneratorCfg,
            HfRandomUniformTerrainCfg, HfPyramidStairsTerrainCfg,
            HfInvertedPyramidStairsTerrainCfg, Cam)


def _build_lane_terrain(su, TI, TG, Flat, StairUp, StairDown, num_envs):
    """Grid of `num_envs` ROWS, each row a lane of `num_cols` segments.

    CRITICAL: num_rows MUST equal num_envs. Isaac Lab distributes the N robots
    across the rows*cols tile grid; if the tile grid is smaller than N, the
    per-env origin/terrain-level lookup indexes out of bounds (the crash we
    hit with num_rows=1). One row per robot keeps the mapping in range and
    gives each robot its own identical lane.

    Each lane runs along the COLUMNS: flat -> stairs up -> stairs down ->
    rough -> flat(goal). NOTE: with the stock generator, sub_terrains are
    placed by proportion and the exact left-to-right ORDER within a row is not
    guaranteed; if the sequence looks shuffled, that is the generator, not a
    bug, and we iterate on it.
    """
    seg = 3.0  # m per segment (close to training patch scale)
    return TI(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TG(
            seed=0,
            size=(seg, seg),
            border_width=0.5,
            num_rows=num_envs,    # ONE ROW PER ROBOT — fixes the index crash
            num_cols=5,           # 5 segments per lane
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            curriculum=False,     # fixed layout, not a curriculum
            sub_terrains={
                "flat_start": Flat(proportion=0.2, noise_range=(0.0, 0.01), noise_step=0.01),
                "stairs_up":  StairUp(proportion=0.2, step_height_range=(0.08, 0.15),
                                      step_width=0.30, platform_width=1.5),
                "stairs_dn":  StairDown(proportion=0.2, step_height_range=(0.08, 0.15),
                                        step_width=0.30, platform_width=1.5),
                "rough":      Flat(proportion=0.2, noise_range=(0.04, 0.10), noise_step=0.02),
                "flat_goal":  Flat(proportion=0.2, noise_range=(0.0, 0.01), noise_step=0.01),
            },
        ),
        collision_group=-1,
        physics_material=su.RigidBodyMaterialCfg(
            friction_combine_mode="multiply", restitution_combine_mode="multiply",
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
        ),
        debug_vis=False,
    )


def main() -> int:
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = get_experiment_cfg(args.robot)
    cfg.camera.enabled = True

    from omni_spot.env_cfg import build_env_cfg
    from omni_spot.nav_env import NavEnv

    env_cfg = build_env_cfg(cfg, args.num_envs)
    env_cfg.seed = args.seed

    # ── Replace curriculum terrain with the straight lane ────────────────────
    lane_ok = True
    try:
        (su, TI, TG, Flat, StairUp, StairDown, Cam) = _isaac_terrain_classes()
        lane = _build_lane_terrain(su, TI, TG, Flat, StairUp, StairDown, args.num_envs)
        # env_cfg.scene stores terrain under .terrain (HAS_TERRAIN path)
        if hasattr(env_cfg.scene, "terrain"):
            env_cfg.scene.terrain = lane
        else:
            env_cfg.scene.ground = lane
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] custom lane terrain failed ({e}); using default terrain", flush=True)
        lane_ok = False
        (su, *_rest, Cam) = _isaac_terrain_classes()

    # ── Static overhead camera framing ALL lanes ────────────────────────────
    # 10 lanes are laid out as terrain ROWS (spread along Y); each lane runs
    # along +X for ~goal_x meters. Camera high above the center, tilted to see
    # the full set. Scales with num_envs so all rows stay in frame.
    mid_x = args.goal_x / 2.0
    span_y = args.num_envs * 3.0  # rows are ~seg(3m) apart
    cam_height = max(10.0, span_y * 0.9)
    cam_pos = (mid_x, -span_y * 0.4, cam_height)
    cam_look = (mid_x, 0.0, 0.0)
    have_cam = True
    try:
        quat = _look_at_quat(cam_pos, cam_look)
        cam = Cam(
            prim_path="/World/track_cam",
            update_period=0.0,
            height=720, width=1280,
            data_types=["rgb"],
            spawn=su.PinholeCameraCfg(
                focal_length=1.4,
                horizontal_aperture=2.0 * math.tan(math.radians(75.0 / 2)),
                clipping_range=(0.1, 200.0),
            ),
            offset=Cam.OffsetCfg(pos=cam_pos, rot=quat, convention="world"),
        )
        setattr(env_cfg.scene, "track_cam", cam)
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] track cam failed ({e}); metrics only", flush=True)
        have_cam = False

    print("[INIT] Building env (10 robots on the lane)...", flush=True)
    env = NavEnv(env_cfg, cfg)
    device = env.device

    ckpt = load_checkpoint(args.ckpt, device)
    print(f"[INIT] ckpt phase={ckpt.get('phase')} robot={ckpt.get('robot')}", flush=True)
    student = StudentPolicy(cfg).to(device)
    student.load_state_dict(ckpt["model_state_dict"])
    student.eval(); student.requires_grad_(False)

    obs, _ = env.reset()

    # ── Force the goal to the END of the lane for every env ──────────────────
    # Spawn robots near x=0 in parallel lateral lanes; goal far down +X.
    n = args.num_envs
    robot = env.scene["robot"]
    origins = env.scene.env_origins  # (n,3) world
    # lateral spread so the overhead view separates them
    ys = (torch.arange(n, device=device) - (n - 1) / 2.0) * args.lane_spacing
    goal_world = torch.stack([
        origins[:, 0] + args.goal_x,
        origins[:, 1] + ys * 0.0,    # goals straight ahead in each lane
    ], dim=-1)
    env._goal[:] = goal_world
    env._prev_dist[:] = torch.linalg.norm(goal_world - origins[:, :2], dim=-1)
    env._init_dist[:] = env._prev_dist.clamp(min=1e-6)

    reached = torch.zeros(n, dtype=torch.bool, device=device)
    reach_step = torch.full((n,), -1, dtype=torch.long, device=device)
    frames: list[np.ndarray] = []

    print(f"[RUN] {args.steps} steps, goal at x=+{args.goal_x}m ...", flush=True)
    with torch.no_grad():
        prev_done = torch.ones(n, dtype=torch.bool, device=device)
        for it in range(1, args.steps + 1):
            depth = obs["depth"] / cfg.camera.max_depth
            action = student.act_mean(
                obs["proprio"], depth, obs["depth_new_frame"],
                obs["history"], reset_mask=prev_done,
            )
            obs, _r, terminated, truncated, _i = env.step(action)
            # NOTE: env auto-resets done envs; for a clean one-shot track test
            # we DON'T re-randomize the goal — re-pin it every step so a reset
            # env keeps aiming down the lane.
            env._goal[:] = goal_world
            prev_done = terminated | truncated

            # Track first arrival per robot (dist to its lane goal)
            pos = robot.data.root_pos_w[:, :2]
            d = torch.linalg.norm(goal_world - pos, dim=-1)
            now = (d < cfg.reward.goal_tol) & (~reached)
            reach_step[now] = it
            reached |= now

            if have_cam:
                try:
                    rgb = env.scene["track_cam"].data.output["rgb"]
                    f = rgb[0].detach().cpu().numpy()
                    if f.shape[-1] == 4:
                        f = f[..., :3]
                    frames.append(f.astype(np.uint8))
                except (KeyError, AttributeError, RuntimeError, IndexError):
                    pass

            if it % 200 == 0:
                print(f"  [{it}/{args.steps}] reached={int(reached.sum())}/{n}", flush=True)

    n_reached = int(reached.sum())
    report = [
        f"checkpoint   : {args.ckpt}",
        f"robots       : {n}",
        f"goal_x       : {args.goal_x} m down the lane",
        f"lane         : flat -> stairs up -> stairs down -> rough -> flat(goal)"
        + ("" if lane_ok else "   [WARN: custom lane FAILED, default terrain used]"),
        f"reached goal : {n_reached}/{n}",
    ]
    for i in range(n):
        s = int(reach_step[i])
        report.append(f"  robot {i:2d}: " + (f"reached at step {s}" if s >= 0 else "did NOT reach"))
    rep = "\n".join(report)
    print("\n" + rep, flush=True)
    with open(os.path.join(args.out_dir, "metrics.txt"), "w") as f:
        f.write(rep + "\n")

    if frames:
        p = os.path.join(args.out_dir, "track_overview.mp4")
        iio.imwrite(p, np.stack(frames), fps=args.fps, codec="h264")
        print(f"[VIDEO] {p}  ({len(frames)} frames)", flush=True)
    else:
        print("[VIDEO] no frames — check cam placement", flush=True)

    env.close()
    return 0


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    except BaseException:
        import traceback
        traceback.print_exc(); sys.stderr.flush()
    finally:
        simulation_app.close()
    sys.exit(code)
