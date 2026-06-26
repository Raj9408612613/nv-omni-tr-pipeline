"""
student_overview.py — run the trained STUDENT policy on a DESIGNED TEST ARENA
of two end-to-end "courses", record a fixed oblique overview MP4, and print
per-course navigation metrics. Headless.

ARENA (real height-field terrain, NOT the random curriculum grid)
    Course 1 (terrain row 0):  flat -> rough -> up pyramid stairs -> flat (GOAL)
    Course 2 (terrain row 1):  down stairs -> rough -> flat (GOAL)
Each robot spawns at the START of its lane and must walk the length to a GOAL
marked by a GREEN SPHERE at the far end. Box obstacles, the terrain curriculum,
and DR pushes are all OFF — this is a pure "can the current policy navigate the
course end-to-end?" test.

DEPTH RESOLUTION NOTE
    The student's depth CNN is hard-wired to the resolution it was distilled at
    (cfg.camera width/height); the checkpoint only loads at that size. So depth
    stays at the trained resolution here. To actually test a higher-res depth
    rig you must RE-DISTILL a student at that resolution — a training change,
    not a flag on this eval.

Run from the repo root in the `isaac` conda env:

    PYTHONPATH=. python student_overview.py \
        --ckpt student_best_20260615.pt \
        --num_envs 256 --steps 3000 --headless --enable_cameras

Outputs under --out_dir (default ./student_overview_out):
    overview.mp4   fixed world camera across the two courses
    metrics.txt    goals / falls / timeouts + success rate, split per course

NOTES
    * The student needs the depth rig, so cameras are forced ON.
    * --steps must be long enough to walk a course (~course_len metres at
      ~1 m/s); the default is sized for the default course length.
    * Tune framing with --cam_pos / --cam_look if the arena grows.
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

_p = argparse.ArgumentParser(description="Student course-arena video + eval")
_p.add_argument("--ckpt", required=True, help="Student checkpoint (.pt)")
_p.add_argument("--robot", default="spot")
_p.add_argument("--num_envs", type=int, default=256)
_p.add_argument("--steps", type=int, default=3000)
_p.add_argument("--out_dir", default="student_overview_out")
_p.add_argument("--fps", type=int, default=25)
_p.add_argument("--seed", type=int, default=0)
# ── Course arena geometry ──────────────────────────────────────────────
_p.add_argument("--lanes", type=int, default=8,
                help="parallel lanes per course (terrain columns)")
_p.add_argument("--course_len", type=float, default=28.0,
                help="length of each course (m), walked along +x")
_p.add_argument("--lane_width", type=float, default=4.0,
                help="width of each lane (m)")
_p.add_argument("--episode_s", type=float, default=60.0,
                help="episode length (s); must exceed course walk time")
_p.add_argument("--frame_stride", type=int, default=2,
                help="capture 1 overview frame every N steps (caps video size)")
# Camera placement (world m). If left as None they are auto-framed from the
# arena extent; pass explicit values to override.
_p.add_argument("--cam_pos", type=float, nargs=3, default=None)
_p.add_argument("--cam_look", type=float, nargs=3, default=None)
AppLauncher.add_app_launcher_args(_p)
args = _p.parse_args()
args.enable_cameras = True  # student needs depth cams; overview also renders

# Auto-frame the camera from the arena extent unless overridden. Grid is
# num_rows(=2 courses) * course_len along x, lanes * lane_width along y,
# centered on the origin by the terrain generator.
_ext_x = 2.0 * args.course_len
_ext_y = args.lanes * args.lane_width
if args.cam_pos is None:
    args.cam_pos = [-(0.5 * _ext_x + 8.0), -(0.5 * _ext_y + 8.0),
                    0.60 * max(_ext_x, _ext_y)]
if args.cam_look is None:
    args.cam_look = [0.0, 0.0, 0.0]

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
print("[INIT] Isaac Sim launched.", flush=True)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from omni_spot.configs import get_experiment_cfg  # noqa: E402
from omni_spot.networks import StudentPolicy  # noqa: E402
from omni_spot.checkpoint import load_checkpoint  # noqa: E402

import imageio  # noqa: E402  (streaming MP4 writer; needs imageio-ffmpeg)


# ════════════════════════════════════════════════════════════════════════
# Designed course terrain (real height field)
# ════════════════════════════════════════════════════════════════════════
def _terrain_imports():
    """Import the height-field terrain API (2.x with 1.x fallback)."""
    try:
        import isaaclab.sim as su
        from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
        from isaaclab.terrains.height_field import HfTerrainBaseCfg
        from isaaclab.terrains.height_field.utils import height_field_to_mesh
        from isaaclab.utils import configclass
    except ImportError:  # Isaac Lab 1.x
        import omni.isaac.lab.sim as su  # type: ignore
        from omni.isaac.lab.terrains import (  # type: ignore
            TerrainGeneratorCfg, TerrainImporterCfg,
        )
        from omni.isaac.lab.terrains.height_field import HfTerrainBaseCfg  # type: ignore
        from omni.isaac.lab.terrains.height_field.utils import (  # type: ignore
            height_field_to_mesh,
        )
        from omni.isaac.lab.utils import configclass  # type: ignore
    return (su, TerrainGeneratorCfg, TerrainImporterCfg, HfTerrainBaseCfg,
            height_field_to_mesh, configclass)


def _build_course_terrain(args):
    """A 2-row terrain: row 0 = course 1, row 1 = course 2, baked by difficulty.

    Each (row, col) cell is a `course_len x lane_width` rectangle; the robot
    walks the long (x) axis. Returns a TerrainImporterCfg ready to drop onto
    env_cfg.scene.terrain.
    """
    (su, TerrainGeneratorCfg, TerrainImporterCfg, HfTerrainBaseCfg,
     height_field_to_mesh, configclass) = _terrain_imports()

    HS, VS = 0.10, 0.005  # horizontal / vertical scale (m per pixel / unit)

    # NOTE (version-sensitive): the @height_field_to_mesh contract is that the
    # function returns an int16 height field in units of cfg.vertical_scale,
    # shape (x_pixels, y_pixels), and the decorator builds the mesh + origin.
    # If your IsaacLab build differs, this function is the only thing to tweak.
    @height_field_to_mesh
    def course_terrain(difficulty, cfg):
        hs = float(cfg.horizontal_scale)
        vs = float(cfg.vertical_scale)
        nx = max(2, int(round(float(cfg.size[0]) / hs)))   # along course (x)
        ny = max(2, int(round(float(cfg.size[1]) / hs)))   # across lane (y)
        hf = np.zeros((nx, ny), dtype=np.int32)

        def u(m):                      # metres -> discrete units
            return int(round(m / vs))

        step_h = max(1, u(cfg.step_height))
        step_w = max(1, int(round(cfg.step_width / hs)))
        plat_w = max(1, int(round(cfg.stair_platform / hs)))
        n_step = int(cfg.stair_n_steps)
        nlo, nhi = u(cfg.rough_noise[0]), u(cfg.rough_noise[1])
        rng = np.random.default_rng(7 + int(round(difficulty * 1000)))

        # Segment layout (kind, length-fraction). difficulty>=0.5 -> course 2.
        if difficulty >= 0.5:
            segs = [("down_stairs", 0.32), ("rough", 0.30), ("flat", 0.38)]
        else:
            segs = [("flat", 0.22), ("rough", 0.26),
                    ("up_stairs", 0.30), ("flat", 0.22)]

        acc = 0
        for k, (kind, frac) in enumerate(segs):
            x0 = acc
            x1 = nx if k == len(segs) - 1 else min(nx, acc + int(round(frac * nx)))
            acc = x1
            seg = x1 - x0
            if seg <= 0:
                continue
            if kind == "flat":
                hf[x0:x1, :] = 0
            elif kind == "rough":
                hf[x0:x1, :] = rng.integers(nlo, nhi + 1, size=(seg, ny))
            elif kind in ("up_stairs", "down_stairs"):
                sgn = 1 if kind == "up_stairs" else -1
                ramp = n_step * step_w
                total = 2 * ramp + plat_w
                s0 = x0 + max(0, (seg - total) // 2)   # centered, flat run-up
                # up ramp
                for i in range(n_step):
                    a, b = s0 + i * step_w, min(x1, s0 + (i + 1) * step_w)
                    hf[a:b, :] = sgn * (i + 1) * step_h
                # platform
                a, b = s0 + ramp, min(x1, s0 + ramp + plat_w)
                hf[a:b, :] = sgn * n_step * step_h
                # down ramp
                for i in range(n_step):
                    a = s0 + ramp + plat_w + i * step_w
                    b = min(x1, a + step_w)
                    if a >= x1:
                        break
                    hf[a:b, :] = sgn * (n_step - (i + 1)) * step_h
        return hf.astype(np.int16)

    @configclass
    class HfCourseCfg(HfTerrainBaseCfg):
        function = course_terrain
        rough_noise: tuple[float, float] = (0.02, 0.10)
        step_height: float = 0.13          # per step (m) — moderate for Spot
        step_width: float = 0.32           # tread depth (m)
        stair_n_steps: int = 5             # peak ~= n_steps * step_height
        stair_platform: float = 0.8        # top platform length (m)

    gen = TerrainGeneratorCfg(
        seed=args.seed,
        size=(args.course_len, args.lane_width),
        border_width=1.0,
        num_rows=2,                        # row 0 = course 1, row 1 = course 2
        num_cols=args.lanes,
        horizontal_scale=HS,
        vertical_scale=VS,
        slope_threshold=0.75,
        curriculum=True,                   # difficulty = row/(rows-1): 0 / 1
        sub_terrains={"course": HfCourseCfg(proportion=1.0)},
    )
    importer = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=gen,
        max_init_terrain_level=1,          # populate BOTH rows (both courses)
        collision_group=-1,
        physics_material=su.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
        ),
        debug_vis=False,
    )
    return importer


# ════════════════════════════════════════════════════════════════════════
# Overview camera + lighting (preserved from the original script)
# ════════════════════════════════════════════════════════════════════════
def _look_at_quat(eye, target, up=(0.0, 0.0, 1.0)):
    """Quaternion (w,x,y,z) for a camera at `eye` looking at `target`."""
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
        prim_path="/World/overview_cam",
        update_period=0.0,
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=su.PinholeCameraCfg(
            focal_length=1.2,
            horizontal_aperture=2.0 * math.tan(math.radians(70.0 / 2)),
            clipping_range=(0.1, 400.0),
        ),
        offset=Cam.OffsetCfg(
            pos=tuple(float(v) for v in args.cam_pos),
            rot=quat,
            # _look_at_quat builds the rotation in the OpenGL camera frame
            # (-Z forward, +Y up). Isaac Lab's "world" convention is +X
            # forward / +Z up — a DIFFERENT axis — so tagging this "world"
            # aims the cam off into the sky (the all-gray/all-black render).
            convention="opengl",
        ),
    )
    setattr(env_cfg.scene, "overview_cam", cam)


def _attach_overview_light(env_cfg):
    """Scene lighting so the RGB overview isn't pure black."""
    try:
        import isaaclab.sim as su
        from isaaclab.assets import AssetBaseCfg
    except ImportError:
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


def _make_goal_markers():
    """Green sphere markers placed at every robot's goal (course end)."""
    try:
        import isaaclab.sim as su
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
    except ImportError:
        import omni.isaac.lab.sim as su  # type: ignore
        from omni.isaac.lab.markers import (  # type: ignore
            VisualizationMarkers, VisualizationMarkersCfg,
        )
    cfg = VisualizationMarkersCfg(
        prim_path="/World/goal_markers",
        markers={
            "goal": su.SphereCfg(
                radius=0.30,
                visual_material=su.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 1.0, 0.0),
                    emissive_color=(0.0, 0.8, 0.0),
                ),
            )
        },
    )
    return VisualizationMarkers(cfg)


# ════════════════════════════════════════════════════════════════════════
# Course env: pin spawn to course start, goal to course end
# ════════════════════════════════════════════════════════════════════════
def _make_course_env_class():
    from omni_spot.nav_env import NavEnv

    class CourseNavEnv(NavEnv):
        """NavEnv variant for the linear test arena.

        Each env's cell spans `course_len` along x and `lane_width` along y,
        centered on its terrain origin. We spawn the robot at the -x end facing
        +x, and pin the goal at the +x end. Same-cell robots are spread across
        the lane (and slightly in x) so they don't spawn on top of each other.
        """

        def set_course(self, course_len, lane_width, margin=1.5):
            self._course_len = float(course_len)
            self._lane_half = 0.5 * float(lane_width)
            self._course_margin = float(margin)

        def _reset_idx(self, env_ids):
            super()._reset_idx(env_ids)   # DR / history / sensors (spawn+goal overwritten below)
            # Some Isaac Lab versions reset envs inside __init__, before
            # set_course() runs — fall back to the default reset until configured.
            if getattr(self, "_course_len", None) is None:
                return
            origins = self.scene.env_origins[env_ids]
            n = len(env_ids)
            half_len = 0.5 * self._course_len - self._course_margin
            lane = max(0.2, self._lane_half - 0.4)

            # spawn: -x end, spread across lane width + small x jitter
            sx = origins[:, 0] - half_len
            jitter = torch.empty(n, device=self.device).uniform_(0.0, 2.0)
            ly = torch.empty(n, device=self.device).uniform_(-lane, lane)
            robot = self.scene["robot"]
            root = robot.data.default_root_state[env_ids].clone()
            root[:, 0] = sx + jitter
            root[:, 1] = origins[:, 1] + ly
            root[:, 2] = origins[:, 2] + self._x.robot.init_height
            root[:, 3] = 1.0           # quat (w,x,y,z) = identity -> face +x
            root[:, 4:7] = 0.0
            root[:, 7:] = 0.0
            self._write_root_state(robot, root, env_ids)

            # goal: +x end of the lane, straight ahead of the spawn
            gx = origins[:, 0] + half_len
            gy = origins[:, 1] + ly
            self._goal[env_ids, 0] = gx
            self._goal[env_ids, 1] = gy
            d = (gx - (sx + jitter)).abs().clamp(min=1e-6)
            self._prev_dist[env_ids] = d
            self._init_dist[env_ids] = d

    return CourseNavEnv


def main() -> int:
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = get_experiment_cfg(args.robot)
    cfg.camera.enabled = True               # student exteroception = depth rig

    # ── Arena policy knobs: pure navigation, no distractors ──────────────
    cfg.obstacles.n_static = 0              # remove ALL box obstacles
    cfg.curriculum.enabled = False         # fixed courses, no promotion/demotion
    cfg.dr.push_robots = False             # don't knock robots over mid-course
    control_dt = cfg.sim.physics_dt * cfg.sim.decimation
    cfg.goal.episode_len_steps = max(1, int(round(args.episode_s / control_dt)))

    from omni_spot.env_cfg import build_env_cfg

    env_cfg = build_env_cfg(cfg, args.num_envs)
    env_cfg.seed = args.seed
    # Replace the random curriculum grid with the designed two-course terrain.
    try:
        env_cfg.scene.terrain = _build_course_terrain(args)
        print("[INIT] course terrain attached (2 courses x "
              f"{args.lanes} lanes).", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[FATAL] course terrain build failed ({e})", flush=True)
        raise

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

    print(f"[INIT] Building arena ({args.num_envs} envs + depth rig + overview "
          f"cam)...", flush=True)
    CourseNavEnv = _make_course_env_class()
    env = CourseNavEnv(env_cfg, cfg)
    env.set_course(args.course_len, args.lane_width)
    device = env.device

    if have_cam:
        # Single fixed world cam has size-1 buffers; the per-env reset hook
        # would index them with the full env_ids (CUDA OOB). Make it a no-op.
        env.scene["overview_cam"].reset = lambda env_ids=None: None

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

    # ── Auto-aim the overview cam at where the robots ACTUALLY are ────────
    # The cfg pose (--cam_pos) is a guess from the arena extent; the real
    # per-env origins are known only after reset. Frame their centroid via
    # set_world_poses_from_view, which uses Isaac Lab's own camera convention
    # (so it can't get the -Z/+X axis mix-up wrong) and adapts the distance to
    # the actual grid size.
    if have_cam:
        try:
            origins = env.scene.env_origins                      # (num_envs, 3)
            center = origins.mean(dim=0)
            span_xy = (origins[:, :2].max(dim=0).values
                       - origins[:, :2].min(dim=0).values)
            radius = 0.5 * float(torch.linalg.norm(span_xy)) + 8.0
            dist = max(radius / math.tan(math.radians(35.0)), 14.0)
            d = torch.tensor([-1.0, -1.0, 0.9], device=center.device)
            d = d / torch.linalg.norm(d)
            eye = center + d * dist
            target = center + torch.tensor([0.0, 0.0, 0.3], device=center.device)
            print(f"[CAM] grid center={[round(v, 2) for v in center.tolist()]} "
                  f"radius={radius:.1f}m eye={[round(v, 2) for v in eye.tolist()]}",
                  flush=True)
            env.scene["overview_cam"].set_world_poses_from_view(
                eye.unsqueeze(0), target.unsqueeze(0)
            )
            print("[CAM] aimed via set_world_poses_from_view (auto-framed)",
                  flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[CAM][WARN] auto-frame failed ({e}); using cfg pose "
                  f"(--cam_pos/--cam_look)", flush=True)

    # Per-course masks (terrain row 0 = course 1, row 1 = course 2).
    try:
        levels = env.scene.terrain.terrain_levels.to(device)
        course1 = levels == 0
        course2 = levels == 1
    except (AttributeError, RuntimeError):
        course1 = torch.ones(args.num_envs, dtype=torch.bool, device=device)
        course2 = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
        print("[WARN] terrain_levels unavailable; per-course split disabled",
              flush=True)

    # Green goal spheres at every robot's goal (z just above local ground).
    markers = None
    try:
        markers = _make_goal_markers()
        gz = env.scene.env_origins[:, 2] + 0.30
        goal_xyz = torch.cat([env._goal, gz.unsqueeze(-1)], dim=-1)
        markers.visualize(translations=goal_xyz)
        print("[INIT] goal markers placed.", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] goal markers failed ({e}); continuing without", flush=True)

    prev_done = torch.ones(args.num_envs, dtype=torch.bool, device=device)
    g = f = t = ep = 0
    g1 = g2 = ep1 = ep2 = 0

    # Stream frames straight to disk so RAM stays flat (3000x 720p in a list
    # would be ~8 GB and OOM the host at the end).
    vid_path = os.path.join(args.out_dir, "overview.mp4")
    writer = n_frames = None
    if have_cam:
        try:
            writer = imageio.get_writer(
                vid_path, fps=args.fps, codec="h264", macro_block_size=None,
            )
            n_frames = 0
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] video writer init failed ({e}); metrics only",
                  flush=True)
            writer = None

    print(f"[RUN] {args.steps} steps x {args.num_envs} envs...", flush=True)
    with torch.no_grad():
        for it in range(1, args.steps + 1):
            depth = obs["depth"] / cfg.camera.max_depth
            action = student.act_mean(
                obs["proprio"], depth, obs["depth_new_frame"],
                obs["history"], reset_mask=prev_done,
            )
            obs, _r, terminated, truncated, _i = env.step(action)
            prev_done = terminated | truncated

            atg = env._at_goal
            done = terminated | truncated
            g += int(atg.sum()); f += int(env._fallen.sum())
            t += int((truncated & ~terminated).sum()); ep += int(done.sum())
            g1 += int((atg & course1).sum()); ep1 += int((done & course1).sum())
            g2 += int((atg & course2).sum()); ep2 += int((done & course2).sum())

            if markers is not None:
                try:
                    markers.visualize(translations=goal_xyz)
                except Exception:  # noqa: BLE001
                    pass

            if writer is not None and it % max(1, args.frame_stride) == 0:
                try:
                    rgb = env.scene["overview_cam"].data.output["rgb"]
                    fr = rgb[0].detach().cpu().numpy()
                    if fr.shape[-1] == 4:
                        fr = fr[..., :3]
                    writer.append_data(fr.astype(np.uint8))
                    n_frames += 1
                except (KeyError, AttributeError, RuntimeError, IndexError):
                    pass

            if it % 100 == 0:
                sr = g / max(1, ep)
                print(f"  [{it}/{args.steps}] episodes={ep} goals={g} "
                      f"falls={f} timeouts={t} success={sr:.3f}", flush=True)

    sr = g / max(1, ep)
    sr1 = g1 / max(1, ep1)
    sr2 = g2 / max(1, ep2)
    report = "\n".join([
        f"checkpoint     : {args.ckpt}",
        f"phase          : {ckpt.get('phase')}  (STUDENT eval)",
        f"arena          : 2 courses x {args.lanes} lanes, "
        f"{args.course_len:.0f}m each",
        f"envs x steps   : {args.num_envs} x {args.steps}",
        f"episodes ended : {ep}",
        f"goals reached  : {g}",
        f"falls/collide  : {f}",
        f"timeouts       : {t}",
        f"SUCCESS RATE   : {sr:.3f}   (goals / episodes_ended)",
        f"  course 1 (flat->rough->up-stairs->flat): {sr1:.3f}  "
        f"({g1}/{ep1})",
        f"  course 2 (down-stairs->rough->flat)    : {sr2:.3f}  "
        f"({g2}/{ep2})",
    ])
    print("\n" + report, flush=True)
    with open(os.path.join(args.out_dir, "metrics.txt"), "w") as fp:
        fp.write(report + "\n")

    if writer is not None:
        writer.close()
        if n_frames:
            print(f"[VIDEO] {vid_path}  ({n_frames} frames)", flush=True)
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
