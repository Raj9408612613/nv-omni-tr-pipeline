"""
teacher-ckpt-strgt.py — TEST-COURSE runner for a TEACHER checkpoint
===================================================================
Same two hand-built straight courses, carrot waypoints, green goal spheres,
overview camera and metrics as student_overview.py — but drives the TEACHER
policy (scandots + privileged obs) instead of the depth-camera student, so a
mid-training PBT best.pt can be watched on the courses long before any
student has been distilled.

HOW THE TEACHER "SEES" THE COURSE (the one real difference vs the student)
    The student perceives the course cuboids through its depth camera. The
    teacher's exteroception is a scandot heightfield raycast — and the
    RayCaster only sees the static terrain mesh (/World/ground), NOT the
    course cuboids. Training has the same limitation for box obstacles and
    solves it analytically: obs.compose_scandots() composites known box
    poses over the raycast (eff_z = max(terrain_z, box_top)). This script
    reuses that exact path by registering EVERY course cuboid as an
    "obstacle" in the env's box buffers — the teacher then sees the stairs,
    rough cells, and pits in its scandots exactly the way it sees boxes in
    training. (Rewards also see the cuboids as collisions, but this is an
    eval — rewards are ignored; the course _get_dones never uses collision.)

    No depth cameras are built at all (cfg.camera.enabled=False), so this
    runs much lighter than the student script.

Run from the repo root in the `isaac` conda env (works mid-training — copy
best.pt first so a checkpoint save can't be read half-written):

    RUN=$(ls -td omni_logs/pbt_* | head -1) && cp "$RUN/best.pt" /tmp/snap.pt
    PYTHONPATH=. python teacher-chpt-strgt.py \
        --ckpt /tmp/snap.pt --robot spot_robust \
        --num_envs 64 --steps 1500 --headless --enable_cameras

Outputs under --out_dir (default ./teacher_course_out):
    overview.mp4    fixed world camera over both courses
    metrics.txt     per-course goals / falls / timeouts + success rate
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

_p = argparse.ArgumentParser(description="Teacher test-course overview + eval")
_p.add_argument("--ckpt", required=True, help="Teacher checkpoint (.pt)")
_p.add_argument("--robot", default="spot_robust",
                help="Config the checkpoint was trained with")
_p.add_argument("--num_envs", type=int, default=64,
                help="Total robots, split evenly across the two courses")
_p.add_argument("--steps", type=int, default=1500)
_p.add_argument("--out_dir", default="teacher_course_out")
_p.add_argument("--fps", type=int, default=25)
_p.add_argument("--seed", type=int, default=0)
# Course / navigation geometry (identical to student_overview.py).
_p.add_argument("--seg_len", type=float, default=4.0, help="Length of each terrain segment (m)")
_p.add_argument("--lane_width", type=float, default=3.0, help="Walkable lane width (m)")
_p.add_argument("--lookahead", type=float, default=2.0, help="Carrot waypoint lookahead (m)")
_p.add_argument("--episode_s", type=float, default=45.0, help="Max seconds per course attempt")
_p.add_argument("--domain_rand", action="store_true", help="Enable DR + random pushes (off by default for a clean test)")
# Overview camera placement (world meters). Auto-framed from the course if None.
_p.add_argument("--cam_pos", type=float, nargs=3, default=None)
_p.add_argument("--cam_look", type=float, nargs=3, default=None)
AppLauncher.add_app_launcher_args(_p)
args = _p.parse_args()
args.enable_cameras = True  # overview cam only (no depth rig for the teacher)

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
print("[INIT] Isaac Sim launched.", flush=True)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from omni_spot.configs import get_experiment_cfg  # noqa: E402
from omni_spot.networks import TeacherPolicy  # noqa: E402
from omni_spot.checkpoint import load_checkpoint  # noqa: E402
from omni_spot.reward import check_termination  # noqa: E402

try:
    import imageio.v3 as iio
except ImportError:
    import imageio as iio  # type: ignore


# ════════════════════════════════════════════════════════════════════════════
# Course geometry — identical to student_overview.py.
# ════════════════════════════════════════════════════════════════════════════

PLINTH = 0.5          # flat walkable surface height (m)
PYR_HEIGHT = 0.45     # pyramid apex rise above the plinth (m)
PIT_DEPTH = 0.40      # stairs-down pit depth below the plinth (m)
STEP_DEPTH = 0.30     # x-extent of one stair strip (m)
ROUGH_CELL = 0.50     # rough-terrain pillar footprint (m)
ROUGH_AMP = 0.12      # rough-terrain max bump above the plinth (m)
LANE_GAP = 2.0        # clear gap between the two lanes (m)

COL_START = (0.25, 0.75, 0.45)
COL_PLAIN = (0.60, 0.60, 0.68)
COL_ROUGH = (0.85, 0.55, 0.20)
COL_UP = (0.20, 0.55, 0.95)
COL_DOWN = (0.80, 0.30, 0.80)

COURSE_SEGMENTS = {
    0: ["plain", "rough", "stairs_up", "plain"],
    1: ["stairs_down", "rough", "plain"],
}


def _seg_cuboids(kind, sx, L, yc, W, rng):
    out = []
    if kind in ("plain", "start"):
        col = COL_START if kind == "start" else COL_PLAIN
        out.append(((sx + L / 2, yc, PLINTH / 2), (L, W, PLINTH), col))
    elif kind == "rough":
        nx = max(1, int(round(L / ROUGH_CELL)))
        ny = max(1, int(round(W / ROUGH_CELL)))
        cx_sz, cy_sz = L / nx, W / ny
        for i in range(nx):
            for j in range(ny):
                h = PLINTH + float(rng.uniform(0.0, ROUGH_AMP))
                cx = sx + (i + 0.5) * cx_sz
                cy = yc - W / 2 + (j + 0.5) * cy_sz
                out.append(((cx, cy, h / 2), (cx_sz, cy_sz, h), COL_ROUGH))
    elif kind in ("stairs_up", "stairs_down"):
        nx = max(2, int(round(L / STEP_DEPTH)))
        strip = L / nx
        half = L / 2
        for i in range(nx):
            dx = (i + 0.5) * strip
            frac = max(0.0, 1.0 - abs(dx - half) / half)
            if kind == "stairs_up":
                h = PLINTH + PYR_HEIGHT * frac
                col = COL_UP
            else:
                h = max(0.06, PLINTH - PIT_DEPTH * frac)
                col = COL_DOWN
            out.append(((sx + dx, yc, h / 2), (strip, W, h), col))
    return out


def build_courses(num_envs, seg_len, lane_width, lookahead, seed):
    rng = np.random.default_rng(seed)
    W = lane_width
    yc_by_course = {0: (lane_width + LANE_GAP) / 2.0,
                    1: -(lane_width + LANE_GAP) / 2.0}

    n0 = (num_envs + 1) // 2
    course_id = np.array([0] * n0 + [1] * (num_envs - n0), dtype=np.int64)

    lat = max(1, int(lane_width / 0.7))
    slot_w = lane_width / lat
    row_sp = 0.7

    terrain, markers = [], []
    start_xy = np.zeros((num_envs, 2), dtype=np.float32)
    start_z = np.full(num_envs, PLINTH, dtype=np.float32)
    yaw0 = np.zeros(num_envs, dtype=np.float32)
    end_x = np.zeros(num_envs, dtype=np.float32)
    yc_arr = np.zeros(num_envs, dtype=np.float32)
    total_len = 0.0

    for c, segs in COURSE_SEGMENTS.items():
        yc = yc_by_course[c]
        idxs = np.where(course_id == c)[0]
        ng = len(idxs)
        rows = max(1, int(math.ceil(ng / lat)))
        pad_len = max(3.0, rows * row_sp + 1.0)

        terrain += _seg_cuboids("start", 0.0, pad_len, yc, W, rng)
        sx = pad_len
        for kind in segs:
            terrain += _seg_cuboids(kind, sx, seg_len, yc, W, rng)
            sx += seg_len
        course_end_x = sx - 0.6
        total_len = max(total_len, sx)

        markers.append(((course_end_x, yc, PLINTH + 0.40), 0.35))

        for g, e in enumerate(idxs):
            row, col = g // lat, g % lat
            start_xy[e, 0] = 0.5 + row * row_sp
            start_xy[e, 1] = yc + (col - (lat - 1) / 2.0) * slot_w
            yc_arr[e] = yc
            end_x[e] = course_end_x

    per_env = dict(course_id=course_id, start_xy=start_xy, start_z=start_z,
                   yaw0=yaw0, end_x=end_x, yc=yc_arr)
    meta = dict(total_len=float(total_len),
                y_extent=float(abs(yc_by_course[0]) + lane_width / 2))
    return terrain, markers, per_env, meta


# ════════════════════════════════════════════════════════════════════════════
# Isaac Lab cfg helpers — identical to student_overview.py.
# ════════════════════════════════════════════════════════════════════════════

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
    """Quaternion (w,x,y,z) for a camera at `eye` looking at `target`, built
    in the OpenGL camera frame (-Z forward, +Y up)."""
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


def _ground_plane(su, AssetBaseCfg):
    return AssetBaseCfg(
        prim_path="/World/ground",
        spawn=su.GroundPlaneCfg(
            size=(400.0, 400.0),
            physics_material=su.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
            ),
        ),
    )


def _cuboid_cfg(su, AssetBaseCfg, name, pos, size, color):
    return AssetBaseCfg(
        prim_path=f"/World/course/{name}",
        spawn=su.CuboidCfg(
            size=tuple(float(v) for v in size),
            collision_props=su.CollisionPropertiesCfg(),
            physics_material=su.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
            ),
            visual_material=su.PreviewSurfaceCfg(diffuse_color=tuple(color)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=tuple(float(v) for v in pos)
        ),
    )


def _goal_sphere_cfg(su, AssetBaseCfg, name, pos, radius):
    return AssetBaseCfg(
        prim_path=f"/World/markers/{name}",
        spawn=su.SphereCfg(
            radius=float(radius),
            visual_material=su.PreviewSurfaceCfg(
                diffuse_color=(0.05, 0.90, 0.18),
                emissive_color=(0.02, 0.45, 0.06),
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=tuple(float(v) for v in pos)
        ),
    )


def _attach_overview_cam(env_cfg, cam_pos, cam_look):
    su, AssetBaseCfg, Cam = _isaac_imports()
    quat = _look_at_quat(cam_pos, cam_look)
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
            pos=tuple(float(v) for v in cam_pos), rot=quat,
            convention="opengl",
        ),
    )
    setattr(env_cfg.scene, "overview_cam", cam)


def main() -> int:
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = get_experiment_cfg(args.robot)

    cfg.camera.enabled = False             # teacher: scandots, no depth rig
    cfg.obstacles.n_static = 0             # no box field on the test courses
    cfg.curriculum.enabled = False         # fixed courses, no level shuffling
    if not args.domain_rand:
        cfg.dr.enabled = False             # clean traversal test by default

    # ── Build the two courses ────────────────────────────────────────────
    terrain, markers, per_env, meta = build_courses(
        args.num_envs, args.seg_len, args.lane_width, args.lookahead, args.seed
    )
    print(f"[COURSE] {args.num_envs} robots | course-1 segs={COURSE_SEGMENTS[0]} "
          f"| course-2 segs={COURSE_SEGMENTS[1]} | length~{meta['total_len']:.1f} m "
          f"| {len(terrain)} terrain prims", flush=True)

    from omni_spot.env_cfg import build_env_cfg

    env_cfg = build_env_cfg(cfg, args.num_envs)
    env_cfg.seed = args.seed
    env_cfg.episode_length_s = float(args.episode_s)

    su, AssetBaseCfg, _ = _isaac_imports()
    # Replace the curriculum terrain importer with a plain ground plane; the
    # height-scanner raycasts /World/ground and the course cuboids are
    # composited analytically (see module docstring).
    if hasattr(env_cfg.scene, "terrain"):
        env_cfg.scene.terrain = _ground_plane(su, AssetBaseCfg)
    else:
        env_cfg.scene.ground = _ground_plane(su, AssetBaseCfg)
    for k, (pos, size, color) in enumerate(terrain):
        setattr(env_cfg.scene, f"course_{k:04d}",
                _cuboid_cfg(su, AssetBaseCfg, f"c{k:04d}", pos, size, color))
    for k, (pos, radius) in enumerate(markers):
        setattr(env_cfg.scene, f"goal_{k}",
                _goal_sphere_cfg(su, AssetBaseCfg, f"goal_{k}", pos, radius))
    env_cfg.scene.dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=su.DomeLightCfg(intensity=1500.0, color=(0.9, 0.9, 0.95)),
    )
    env_cfg.scene.sun_light = AssetBaseCfg(
        prim_path="/World/SunLight",
        spawn=su.DistantLightCfg(intensity=2000.0, color=(1.0, 1.0, 0.95)),
    )

    # Auto-frame the overview camera over the whole arena if not overridden.
    cx = meta["total_len"] / 2.0
    half = meta["total_len"] / 2.0
    y_back = half / math.tan(math.radians(33.0)) + meta["y_extent"] + 2.0
    cam_pos = args.cam_pos or [cx, -y_back, 0.5 * half + 3.0]
    cam_look = args.cam_look or [cx, 0.0, 0.5]
    print(f"[CAM] overview pos={[round(v, 1) for v in cam_pos]} "
          f"look={[round(v, 1) for v in cam_look]}", flush=True)
    try:
        _attach_overview_cam(env_cfg, cam_pos, cam_look)
        have_cam = True
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] overview cam attach failed ({e}); metrics only", flush=True)
        have_cam = False

    print(f"[INIT] Building course env ({args.num_envs} robots, scandots, "
          f"overview cam)...", flush=True)
    env = CourseNavEnv(env_cfg, cfg)
    device = env.device
    env.set_course_layout(per_env, lookahead=args.lookahead)

    # ── Teacher exteroception: course cuboids -> analytic scandot boxes ──
    # The RayCaster only sees /World/ground; register every course cuboid in
    # the env's box buffers so obs.compose_scandots() composites them exactly
    # like training composites box obstacles. Set AFTER construction (the
    # build-time resets ran with the empty K=0 buffers) and BEFORE reset().
    box_pos = torch.tensor([p for p, _s, _c in terrain],
                           dtype=torch.float32, device=device)      # (K, 3)
    box_half = torch.tensor([s for _p, s, _c in terrain],
                            dtype=torch.float32, device=device) / 2  # (K, 3)
    env._obs_pos = box_pos.unsqueeze(0).expand(
        args.num_envs, -1, -1
    ).contiguous()
    env._box_half = box_half
    print(f"[SCAN] {box_pos.shape[0]} course cuboids composited into the "
          f"teacher's scandots", flush=True)

    # The overview cam is a SINGLE global instance; InteractiveScene.reset()
    # resets every sensor with env_ids = arange(num_envs), and indexing its
    # size-1 buffer with [0..N-1] triggers a CUDA device-side assert. Reset
    # its one instance regardless of the per-env indices passed in.
    if have_cam:
        try:
            _ov = env.scene["overview_cam"]
            _ov_reset = _ov.reset
            _ov.reset = lambda env_ids=None, _r=_ov_reset: _r(None)
        except (KeyError, AttributeError):
            pass

    # ── Load the teacher ─────────────────────────────────────────────────
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
    course_id = env._course_id  # (B,) on device

    # Sanity: where did the robots actually land?
    rp = env.scene["robot"].data.root_pos_w
    print(f"[DBG] robot xyz: x[{rp[:,0].min():.1f},{rp[:,0].max():.1f}] "
          f"y[{rp[:,1].min():.1f},{rp[:,1].max():.1f}] "
          f"z[{rp[:,2].min():.1f},{rp[:,2].max():.1f}]", flush=True)

    if have_cam:
        try:
            ctr = rp.mean(dim=0)
            ctr[2] = 0.4
            span = (rp[:, :2].max(dim=0).values
                    - rp[:, :2].min(dim=0).values)
            radius = 0.5 * float(torch.linalg.norm(span)) + 6.0
            dist = max(radius / math.tan(math.radians(35.0)), 14.0)
            d = torch.tensor([-0.35, -1.0, 0.85], device=ctr.device)
            d = d / torch.linalg.norm(d)
            eye = ctr + d * dist
            print(f"[CAM] auto-frame center="
                  f"{[round(v, 1) for v in ctr.tolist()]} radius={radius:.1f} "
                  f"eye={[round(v, 1) for v in eye.tolist()]}", flush=True)
            env.scene["overview_cam"].set_world_poses_from_view(
                eye.unsqueeze(0), ctr.unsqueeze(0)
            )
        except Exception as e:  # noqa: BLE001
            print(f"[CAM][WARN] auto-frame failed ({e}); using cfg pose "
                  f"(tune --cam_pos/--cam_look)", flush=True)

    # Per-course tallies. NOTE: with terminate_on_fall=False, episodes only
    # end at the goal (or the --episode_s timeout, step episode_s*50) — run
    # at least ~2.5x episode_s*50 steps if you want every attempt scored.
    goals = torch.zeros(2, device=device)
    fall_events = torch.zeros(2, device=device)   # rising edges, not steps
    timeouts = torch.zeros(2, device=device)
    episodes = torch.zeros(2, device=device)
    prev_fallen = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
    best_frac = torch.zeros(args.num_envs, device=device)  # per-attempt max

    start_x = env._course_start_xy[:, 0]
    span_x = (env._course_end_x - start_x).clamp(min=1e-6)

    frames: list[np.ndarray] = []

    print(f"[RUN] {args.steps} steps x {args.num_envs} robots "
          f"(timeout at step {int(args.episode_s * 50)})...", flush=True)
    with torch.no_grad():
        for it in range(1, args.steps + 1):
            action = teacher.act_mean(
                obs["proprio"], obs["scandots"], obs["priv"]
            )
            obs, _r, terminated, truncated, _i = env.step(action)

            done = terminated | truncated
            fell_now = env._fallen & ~prev_fallen      # new falls this step
            prev_fallen = env._fallen.clone()
            px = env.scene["robot"].data.root_pos_w[:, 0]
            frac = ((px - start_x) / span_x).clamp(0.0, 1.0)
            best_frac = torch.maximum(best_frac, frac)
            best_frac[done] = 0.0                      # new attempt starts
            for c in (0, 1):
                m = course_id == c
                goals[c] += (env._at_goal & m).sum()
                fall_events[c] += (fell_now & m).sum()
                timeouts[c] += (truncated & ~terminated & m).sum()
                episodes[c] += (done & m).sum()

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
                g, e = int(goals.sum()), int(episodes.sum())
                sr = g / max(1, e)
                down = int(env._fallen.sum())
                fr = [float(frac[course_id == c].mean()) for c in (0, 1)]
                print(f"  [{it}/{args.steps}] episodes={e} goals={g} "
                      f"falls={int(fall_events.sum())} down_now={down} "
                      f"timeouts={int(timeouts.sum())} success={sr:.3f} "
                      f"| progress c1={fr[0]:.2f} c2={fr[1]:.2f}", flush=True)

    # ── Report ───────────────────────────────────────────────────────────
    def _line(label, c):
        e = int(episodes[c])
        sr = int(goals[c]) / max(1, e)
        m = course_id == c
        unfinished = m & (best_frac > 0.0)
        bf = best_frac[unfinished]
        prog = (f"unfinished: n={int(unfinished.sum())} "
                f"best_frac mean={float(bf.mean()):.2f} "
                f"min={float(bf.min()):.2f} max={float(bf.max()):.2f}"
                if int(unfinished.sum()) > 0 else "unfinished: none")
        return (f"  {label:9s}: episodes={e:4d} goals={int(goals[c]):4d} "
                f"fall_events={int(fall_events[c]):4d} "
                f"timeouts={int(timeouts[c]):4d} success={sr:.3f}\n"
                f"             {prog}")

    tot_e = int(episodes.sum())
    tot_sr = int(goals.sum()) / max(1, tot_e)
    down_end = int(env._fallen.sum())
    report = "\n".join([
        f"checkpoint     : {args.ckpt}",
        f"phase          : {ckpt.get('phase')}  (TEACHER test-course eval)",
        f"robots x steps : {args.num_envs} x {args.steps} "
        f"(timeout at step {int(args.episode_s * 50)})",
        f"exteroception  : scandots + analytic course compositing "
        f"({len(terrain)} cuboids)",
        f"domain_rand    : {bool(args.domain_rand)}",
        f"course 1 segs  : {COURSE_SEGMENTS[0]}",
        f"course 2 segs  : {COURSE_SEGMENTS[1]}",
        "per-course (success = reached the green goal at the course end;",
        " best_frac = furthest point reached along the course, this attempt):",
        _line("course 1", 0),
        _line("course 2", 1),
        f"fallen at end  : {down_end} robots still down when the run ended",
        f"OVERALL        : episodes={tot_e} SUCCESS RATE={tot_sr:.3f}",
    ])
    print("\n" + report, flush=True)
    with open(os.path.join(args.out_dir, "metrics.txt"), "w") as f:
        f.write(report + "\n")

    if frames:
        p = os.path.join(args.out_dir, "overview.mp4")
        iio.imwrite(p, np.stack(frames), fps=args.fps, codec="h264")
        print(f"[VIDEO] {p}  ({len(frames)} frames)", flush=True)
    else:
        print("[VIDEO] no overview frames — check cam placement", flush=True)

    env.close()
    return 0


# ════════════════════════════════════════════════════════════════════════════
# CourseNavEnv — identical to student_overview.py: places robots at the start
# of their course, drives them with a moving carrot waypoint, and scores
# success at the true course end.
# ════════════════════════════════════════════════════════════════════════════

from omni_spot.nav_env import NavEnv  # noqa: E402


class CourseNavEnv(NavEnv):
    """NavEnv variant for the deterministic test courses."""

    def set_course_layout(self, per_env, lookahead):
        dev = self.device
        self._course_id = torch.as_tensor(
            per_env["course_id"], device=dev, dtype=torch.long
        )
        self._course_start_xy = torch.as_tensor(
            per_env["start_xy"], device=dev, dtype=torch.float32
        )
        self._course_start_z = torch.as_tensor(
            per_env["start_z"], device=dev, dtype=torch.float32
        )
        self._course_yaw0 = torch.as_tensor(
            per_env["yaw0"], device=dev, dtype=torch.float32
        )
        self._course_end_x = torch.as_tensor(
            per_env["end_x"], device=dev, dtype=torch.float32
        )
        self._course_yc = torch.as_tensor(
            per_env["yc"], device=dev, dtype=torch.float32
        )
        self._lookahead = float(lookahead)
        self._final_tol = 0.6  # m, distance to the true course end = success
        self._course_ready = True
        # NOTE: do NOT call _reset_idx here — main()'s env.reset() applies the
        # placement after this (forcing one earlier trips a sensor assert).

    # ── Moving carrot waypoint (keeps the goal input in-distribution) ────
    def _update_carrot(self):
        px = self.scene["robot"].data.root_pos_w[:, 0]
        wp_x = torch.minimum(px + self._lookahead, self._course_end_x)
        self._goal[:, 0] = wp_x
        self._goal[:, 1] = self._course_yc

    def _reset_idx(self, env_ids):
        if not getattr(self, "_course_ready", False):
            return super()._reset_idx(env_ids)

        x = self._x
        n = len(env_ids)
        robot = self.scene["robot"]

        # Grandparent (DirectRLEnv) buffer reset — SKIP NavEnv's curriculum
        # and random pose/goal sampling (they assume the training grid), and
        # crucially its obstacle resampling (our box buffers ARE the course).
        direct_rl = next(
            c for c in type(self).__mro__ if c.__name__ == "DirectRLEnv"
        )
        direct_rl._reset_idx(self, env_ids)

        # ── Robot pose: at the course start pad, facing +X ──────────────
        yaw = self._course_yaw0[env_ids] + torch.empty(
            n, device=self.device
        ).uniform_(-0.08, 0.08)
        root_state = robot.data.default_root_state[env_ids].clone()
        root_state[:, 0] = self._course_start_xy[env_ids, 0]
        root_state[:, 1] = self._course_start_xy[env_ids, 1]
        root_state[:, 2] = self._course_start_z[env_ids] + x.robot.init_height
        root_state[:, 3] = torch.cos(yaw / 2)
        root_state[:, 4] = 0.0
        root_state[:, 5] = 0.0
        root_state[:, 6] = torch.sin(yaw / 2)
        root_state[:, 7:] = 0.0
        self._write_root_state(robot, root_state, env_ids)
        robot.write_joint_state_to_sim(
            self._default_jp.unsqueeze(0).expand(n, -1),
            torch.zeros(n, x.action_dim, device=self.device),
            joint_ids=self._joint_ids, env_ids=env_ids,
        )

        # ── Goal: initial carrot waypoint for these envs ─────────────────
        start_xy = self._course_start_xy[env_ids]
        wp_x = torch.minimum(
            start_xy[:, 0] + self._lookahead, self._course_end_x[env_ids]
        )
        goal = torch.stack([wp_x, self._course_yc[env_ids]], dim=-1)
        self._goal[env_ids] = goal
        end_xy = torch.stack(
            [self._course_end_x[env_ids], self._course_yc[env_ids]], dim=-1
        )
        dist0 = torch.linalg.norm(end_xy - start_xy, dim=-1)
        self._prev_dist[env_ids] = torch.linalg.norm(goal - start_xy, dim=-1)
        self._init_dist[env_ids] = dist0.clamp(min=1e-6)

        # ── Policy-side state ─────────────────────────────────────────────
        self._ctrl[env_ids] = self._default_jp
        self._prev_ctrl[env_ids] = self._default_jp
        self._last_policy_action[env_ids] = 0.0
        self._last_proprio[env_ids] = 0.0
        self._history.reset_idx(env_ids)
        self._just_reset[env_ids] = True
        # NOTE: do NOT clear self._fallen / self._at_goal here — the eval
        # loop reads them post-auto-reset (see student_overview.py).

        # ── Domain randomization (only if enabled) ────────────────────────
        self._apply_dr(env_ids)
        if x.dr.enabled and x.dr.push_robots:
            interval = max(1, round(x.dr.push_interval_s / x.sim.control_dt))
            self._push_timer[env_ids] = torch.randint(
                interval // 2, interval + interval // 2, (n,), device=self.device,
            )

    def _get_dones(self):
        if not getattr(self, "_course_ready", False):
            return super()._get_dones()

        x = self._x
        robot = self.scene["robot"]
        self._update_carrot()           # advance the waypoint before scoring
        self._update_height_scan()

        # Reuse the standard fall/timeout test; ignore its (carrot) at_goal.
        fallen, _wp_at_goal, timeout = check_termination(
            x.reward,
            robot_pos=robot.data.root_pos_w,
            robot_quat=robot.data.root_quat_w,
            goal_pos=self._goal,
            base_height=self._base_height,
            step_count=self.episode_length_buf,
            max_steps=self.max_episode_length,
        )
        # Success = reached the TRUE course end (not the moving waypoint).
        end_xy = torch.stack([self._course_end_x, self._course_yc], dim=-1)
        final_dist = torch.linalg.norm(
            end_xy - robot.data.root_pos_w[:, :2], dim=-1
        )
        at_goal = final_dist < self._final_tol
        self._fallen = fallen
        self._fallen_raw = fallen
        self._at_goal = at_goal
        # Respect the checkpoint's training rules: with terminate_on_fall off
        # (spot_robust) a fall does NOT end the attempt — the robot gets the
        # rest of the episode to get up and keep going, exactly like training.
        terminated = at_goal
        if x.reward.terminate_on_fall:
            terminated = terminated | fallen
        truncated = timeout & ~terminated
        return terminated, truncated


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
