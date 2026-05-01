"""
Render Spot Robot in Isaac Sim
================================
Spawns the Spot USD in Isaac Sim and captures high-quality rendered
frames from multiple camera angles. Outputs PNG stills and an MP4 turntable.

Usage:
    # Inside Isaac Sim Python:
    python scripts/render_spot.py

    # With custom output:
    python scripts/render_spot.py --output_dir renders/ --turntable --n_frames 120
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)


def parse_args():
    p = argparse.ArgumentParser(description="Render Spot in Isaac Sim")
    p.add_argument("--output_dir", type=str, default="renders",
                   help="Directory for rendered outputs")
    p.add_argument("--turntable", action="store_true", default=True,
                   help="Render 360-degree turntable video")
    p.add_argument("--n_frames", type=int, default=120,
                   help="Number of turntable frames (120 = 4s at 30fps)")
    p.add_argument("--resolution", type=int, nargs=2, default=[1080, 1920],
                   help="Render resolution (height width)")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--with_obstacles", action="store_true",
                   help="Also spawn some obstacles in the scene")
    p.add_argument("--standing_anim", action="store_true",
                   help="Animate Spot performing a standing pose")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("  Spot Robot — Isaac Sim Renderer")
    print("=" * 60)

    # ── Import Isaac Sim ──────────────────────────────────────────
    try:
        import omni.isaac.core.utils.stage as stage_utils
        from omni.isaac.core import World
        from omni.isaac.core.utils.prims import create_prim
        import omni.isaac.lab.sim as sim_utils
        from omni.isaac.sensor import Camera
    except ImportError:
        print("ERROR: This script requires Isaac Sim.")
        print("Run with: ~/.local/share/ov/pkg/isaac-sim-*/python.sh scripts/render_spot.py")
        sys.exit(1)

    from omni_spot.config import STANDING_POSE, TARGET_HEIGHT

    # ── Create world ──────────────────────────────────────────────
    print("[render] Creating Isaac Sim world...")
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane(
        z_position=0.0,
        name="ground",
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
    )

    # ── Lighting ──────────────────────────────────────────────────
    print("[render] Setting up lighting...")
    # Dome light for ambient
    create_prim(
        "/World/DomeLight",
        "DomeLight",
        attributes={
            "inputs:intensity": 500.0,
            "inputs:color": (0.95, 0.95, 1.0),
        },
    )
    # Key light
    create_prim(
        "/World/KeyLight",
        "DistantLight",
        position=(5.0, -5.0, 8.0),
        attributes={
            "inputs:intensity": 3000.0,
            "inputs:angle": 2.0,
            "inputs:color": (1.0, 0.98, 0.95),
        },
    )

    # ── Spawn Spot ────────────────────────────────────────────────
    usd_path = os.path.join(PROJECT_ROOT, "models", "spot_scene.usd")
    if not os.path.isfile(usd_path):
        print(f"[render] WARNING: {usd_path} not found!")
        print("[render] You need to run the MJCF-to-USD converter first:")
        print("  python omni_spot/convert_mjcf_to_usd.py")
        print("[render] Attempting to run converter now...")
        _try_convert()
        if not os.path.isfile(usd_path):
            print("[render] Conversion failed. Cannot render Spot.")
            sys.exit(1)

    print(f"[render] Loading Spot from {usd_path}")
    create_prim(
        "/World/Spot",
        "Xform",
        usd_path=usd_path,
        position=(0.0, 0.0, TARGET_HEIGHT),
    )

    # ── Optionally add obstacles ──────────────────────────────────
    if args.with_obstacles:
        print("[render] Adding sample obstacles...")
        _add_sample_obstacles()

    # ── Setup render cameras ──────────────────────────────────────
    h, w = args.resolution
    print(f"[render] Setting up cameras ({w}x{h})...")

    # Camera angles for still renders
    camera_poses = {
        "front_3q": {
            "pos": (2.5, -1.5, 1.2),
            "target": (0.0, 0.0, 0.3),
        },
        "side": {
            "pos": (0.0, -3.0, 1.0),
            "target": (0.0, 0.0, 0.3),
        },
        "rear_3q": {
            "pos": (-2.0, -1.5, 1.5),
            "target": (0.0, 0.0, 0.3),
        },
        "top_down": {
            "pos": (0.0, 0.0, 4.0),
            "target": (0.0, 0.0, 0.0),
        },
    }

    # ── Initialize world ──────────────────────────────────────────
    print("[render] Initializing physics...")
    world.reset()

    # Let physics settle (standing pose)
    for _ in range(100):
        world.step(render=True)

    # ── Render still frames ───────────────────────────────────────
    print("[render] Capturing still renders...")
    for name, pose in camera_poses.items():
        cam = Camera(
            prim_path=f"/World/RenderCam_{name}",
            resolution=(w, h),
            position=np.array(pose["pos"]),
            orientation=_look_at_quat(
                np.array(pose["pos"]),
                np.array(pose["target"]),
            ),
        )
        world.scene.add(cam)
        world.reset()

        # Wait for RTX to converge (path tracing needs frames to denoise)
        for _ in range(30):
            world.step(render=True)

        rgb = cam.get_rgba()
        if rgb is not None:
            out_path = os.path.join(args.output_dir, f"spot_{name}.png")
            _save_png(rgb[:, :, :3], out_path)
            print(f"  Saved: {out_path}")

    # ── Turntable video ───────────────────────────────────────────
    if args.turntable:
        print(f"[render] Rendering turntable ({args.n_frames} frames)...")
        frames = []
        radius = 3.0
        cam_z = 1.2
        target = np.array([0.0, 0.0, 0.3])

        turntable_cam = Camera(
            prim_path="/World/TurntableCam",
            resolution=(w, h),
        )
        world.scene.add(turntable_cam)

        for i in range(args.n_frames):
            angle = 2 * math.pi * i / args.n_frames
            cam_pos = np.array([
                radius * math.cos(angle),
                radius * math.sin(angle),
                cam_z,
            ])

            turntable_cam.set_world_pose(
                position=cam_pos,
                orientation=_look_at_quat(cam_pos, target),
            )

            # Step physics + render
            world.step(render=True)

            rgb = turntable_cam.get_rgba()
            if rgb is not None:
                frames.append(rgb[:, :, :3].copy())

            if (i + 1) % 30 == 0:
                print(f"  Frame {i+1}/{args.n_frames}")

        if frames:
            video_path = os.path.join(args.output_dir, "spot_turntable.mp4")
            try:
                import imageio.v3 as iio
                iio.imwrite(video_path, np.stack(frames), fps=args.fps, codec="h264")
                print(f"  Turntable video saved: {video_path}")
            except ImportError:
                print("  imageio not available — saving individual PNGs instead")
                for i, frame in enumerate(frames):
                    _save_png(frame, os.path.join(args.output_dir, f"turntable_{i:04d}.png"))

    # ── Standing pose animation ───────────────────────────────────
    if args.standing_anim:
        print("[render] Rendering standing animation (100 frames)...")
        _render_standing_animation(world, args.output_dir, args.fps, (w, h))

    print("\n[render] Done! Output directory:", args.output_dir)
    print("Files:")
    for f in sorted(os.listdir(args.output_dir)):
        fpath = os.path.join(args.output_dir, f)
        size_mb = os.path.getsize(fpath) / 1e6
        print(f"  {f} ({size_mb:.1f} MB)")


def _look_at_quat(eye: np.ndarray, target: np.ndarray, up=None):
    """Compute quaternion [w,x,y,z] for camera looking from eye to target."""
    if up is None:
        up = np.array([0.0, 0.0, 1.0])
    fwd = target - eye
    fwd = fwd / (np.linalg.norm(fwd) + 1e-8)
    right = np.cross(fwd, up)
    right = right / (np.linalg.norm(right) + 1e-8)
    up2 = np.cross(right, fwd)

    # Rotation matrix (camera convention: -Z forward, X right, Y up)
    R = np.stack([right, up2, -fwd], axis=1)

    # Matrix to quaternion
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def _save_png(rgb_array: np.ndarray, path: str):
    """Save RGB numpy array as PNG."""
    try:
        import imageio.v3 as iio
        iio.imwrite(path, rgb_array.astype(np.uint8))
    except ImportError:
        from PIL import Image
        Image.fromarray(rgb_array.astype(np.uint8)).save(path)


def _add_sample_obstacles():
    """Add a few obstacles to the render scene."""
    from omni.isaac.core.utils.prims import create_prim

    obstacles = [
        {"name": "box1", "pos": (2.0, 1.0, 0.5), "size": (0.6, 0.6, 1.0), "color": (0.8, 0.4, 0.2)},
        {"name": "box2", "pos": (-1.5, 2.0, 0.4), "size": (0.5, 0.8, 0.8), "color": (0.7, 0.5, 0.3)},
        {"name": "cyl1", "pos": (1.0, -1.5, 0.85), "size": (0.5, 0.5, 1.7), "color": (0.2, 0.6, 0.8)},
    ]

    for obs in obstacles:
        create_prim(
            f"/World/Obstacles/{obs['name']}",
            "Cube",
            position=obs["pos"],
            scale=obs["size"],
            attributes={"primvars:displayColor": [obs["color"]]},
        )


def _render_standing_animation(world, output_dir, fps, resolution):
    """Render Spot settling into standing pose from a slight drop."""
    try:
        import imageio.v3 as iio
    except ImportError:
        print("  imageio not available, skipping animation")
        return

    from omni.isaac.sensor import Camera

    w, h = resolution
    cam = Camera(
        prim_path="/World/StandingCam",
        resolution=(w, h),
        position=np.array([2.5, -1.5, 1.0]),
        orientation=_look_at_quat(
            np.array([2.5, -1.5, 1.0]),
            np.array([0.0, 0.0, 0.3]),
        ),
    )
    world.scene.add(cam)
    world.reset()

    frames = []
    for i in range(100):
        world.step(render=True)
        rgb = cam.get_rgba()
        if rgb is not None:
            frames.append(rgb[:, :, :3].copy())

    if frames:
        path = os.path.join(output_dir, "spot_standing.mp4")
        iio.imwrite(path, np.stack(frames), fps=fps, codec="h264")
        print(f"  Standing animation: {path}")


def _try_convert():
    """Try to run MJCF-to-USD conversion."""
    try:
        from omni_spot.convert_mjcf_to_usd import convert, MJCF_PATH, DEFAULT_OUTPUT
        convert(MJCF_PATH, DEFAULT_OUTPUT)
    except Exception as e:
        print(f"  Auto-conversion failed: {e}")


if __name__ == "__main__":
    main()
