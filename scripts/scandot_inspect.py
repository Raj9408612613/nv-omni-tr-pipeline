#!/usr/bin/env python3
"""
Scandot Inspector — see exactly what the network sees (Isaac Lab required)
==========================================================================
Three panes over one live rollout, so geometry bugs and ENCODING bugs are
both visible:

  Pane A  the composited `heights` tensor as a live 17x11 heatmap — the
          actual float vector handed to ScandotEncoder, not a picture of
          the scene. Catches clip saturation, sign flips, stale frames and
          missing box compositing, none of which appear in a 3D viewport.
  Pane B  3D markers drawn FROM THAT TENSOR (one sphere per grid point,
          coloured by value). Unlike RayCasterCfg.debug_vis — which draws
          raycaster hits only — these include the analytic box-obstacle
          compositing from obs.compose_scandots, i.e. the custom path.
  Pane C  the depth image the student will consume (when --camera), plus
          the measured overlap between the two exteroception sources.

It also verifies the grid FLATTEN ORDER against the live sensor instead of
assuming it, because a wrong reshape silently transposes every spatial
conclusion drawn from Pane A.

Usage (on the Isaac box, repo root)
-----------------------------------
    # geometry + encoding, teacher-side only, no rendering
    PYTHONPATH=. python scripts/scandot_inspect.py --robot spot --headless

    # add the depth pane and the teacher/student overlap measurement
    PYTHONPATH=. python scripts/scandot_inspect.py --robot spot \
        --camera --headless

    # watch it live with 3D markers, driven by a trained teacher
    PYTHONPATH=. python scripts/scandot_inspect.py --robot spot \
        --markers --ckpt omni_logs/<run>/best.pt

Run with a frozen/random policy and num_envs=1: this validates plumbing,
not learning.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

# ── Isaac Sim must launch before any isaaclab submodule import ──────────
try:
    from isaaclab.app import AppLauncher
except ImportError:
    try:
        from omni.isaac.lab.app import AppLauncher
    except ImportError:
        print("[ERROR] Cannot import AppLauncher — is Isaac Lab installed?")
        raise SystemExit(1)

_p = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
_p.add_argument("--robot", default="spot")
_p.add_argument("--num_envs", type=int, default=1)
_p.add_argument("--steps", type=int, default=200)
_p.add_argument("--every", type=int, default=20, help="print a pane every K steps")
_p.add_argument("--seed", type=int, default=42)
_p.add_argument("--ckpt", default=None, help="teacher checkpoint to drive with")
_p.add_argument("--camera", action="store_true",
                help="enable the depth rig (Pane C + overlap measurement)")
_p.add_argument("--markers", action="store_true",
                help="draw 3D scandot markers from the composited tensor")
_p.add_argument("--forward_offset", type=float, default=None,
                help="override scandots.forward_offset for this run")
_p.add_argument("--save_dir", default=None,
                help="write heights/depth npz + PNGs here")
AppLauncher.add_app_launcher_args(_p)
args = _p.parse_args()
if args.camera:
    args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ── Now safe to import Isaac Lab submodules and torch ───────────────────
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from omni_spot.configs import get_experiment_cfg  # noqa: E402
from omni_spot.scandot_probe import (  # noqa: E402
    ascii_heatmap,
    frustum_mask,
    grid_offsets,
    infer_grid_order,
    saturation_report,
    to_heatmap,
)


# ════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════

def yaw_roll_pitch(q: np.ndarray) -> tuple[float, float, float]:
    """(yaw, roll, pitch) in radians from a (w, x, y, z) quaternion.

    Sign convention matches scandot_probe._rpy_body_to_yaw: positive pitch is
    nose DOWN, positive roll is right side down.
    """
    w, x, y, z = (float(v) for v in q)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    return yaw, roll, pitch


def hits_to_yaw_frame(hits_w: np.ndarray, root_pos: np.ndarray,
                      yaw: float) -> np.ndarray:
    """(N,3) world ray hits -> the yaw-aligned base frame."""
    rel = hits_w - root_pos[None, :]
    c, s = math.cos(-yaw), math.sin(-yaw)
    out = rel.copy()
    out[:, 0] = c * rel[:, 0] - s * rel[:, 1]
    out[:, 1] = s * rel[:, 0] + c * rel[:, 1]
    return out


def ascii_depth(img: np.ndarray, max_depth: float, cols: int = 58,
                rows: int = 20) -> str:
    """Downsample a depth image to text. Near = dark ink, far = blank."""
    h, w = img.shape
    ys = np.linspace(0, h - 1, rows).astype(int)
    xs = np.linspace(0, w - 1, cols).astype(int)
    small = img[np.ix_(ys, xs)]
    ramp = "@%#*+=-:. "
    t = np.clip(small / max(max_depth, 1e-6), 0.0, 1.0)
    idx = np.round(t * (len(ramp) - 1)).astype(int)
    return "\n".join("".join(ramp[i] for i in row) for row in idx)


def build_markers(n: int, n_colors: int = 9):
    """VisualizationMarkers with a blue->red sphere ramp, or None."""
    try:
        try:
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
            import isaaclab.sim as sim_utils
        except ImportError:
            from omni.isaac.lab.markers import (  # type: ignore
                VisualizationMarkers, VisualizationMarkersCfg,
            )
            import omni.isaac.lab.sim as sim_utils  # type: ignore
        protos = {}
        for k in range(n_colors):
            t = k / max(1, n_colors - 1)
            protos[f"c{k}"] = sim_utils.SphereCfg(
                radius=0.018,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(t, 0.15, 1.0 - t)
                ),
            )
        return VisualizationMarkers(VisualizationMarkersCfg(
            prim_path="/Visuals/scandots", markers=protos
        )), n_colors
    except Exception as e:  # noqa: BLE001 — viz is optional
        print(f"[inspect][WARN] 3D markers unavailable ({e}); "
              f"panes A and C still work", flush=True)
        return None, 0


# ════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════

def main() -> int:
    torch.manual_seed(args.seed)
    x = get_experiment_cfg(args.robot)
    if args.forward_offset is not None:
        x.scandots.forward_offset = args.forward_offset
    if args.camera:
        x.camera.enabled = True

    from omni_spot.env_cfg import build_env_cfg
    from omni_spot.nav_env import NavEnv

    env_cfg = build_env_cfg(x, args.num_envs)
    env_cfg.seed = args.seed
    print(f"[INIT] building env: {args.num_envs} envs, "
          f"scandots {x.scandots.grid_x}x{x.scandots.grid_y} @ "
          f"{x.scandots.spacing} m, forward_offset="
          f"{x.scandots.forward_offset:+.3f}, cameras="
          f"{'ON' if x.camera.enabled else 'OFF'}", flush=True)
    env = NavEnv(env_cfg, x)
    device = env.device

    policy = None
    if args.ckpt:
        from omni_spot.checkpoint import load_checkpoint
        from omni_spot.networks import TeacherPolicy
        policy = TeacherPolicy(x).to(device)
        policy.load_state_dict(load_checkpoint(args.ckpt, device)["model_state_dict"])
        policy.eval()
        print(f"[INIT] driving with teacher {args.ckpt}")
    else:
        print("[INIT] driving with zero actions (plumbing check)")

    obs, _ = env.reset()
    markers, n_colors = build_markers(x.scandots.n_points) if args.markers else (None, 0)

    order = None
    sat_acc: list[float] = []
    cov_acc: list[float] = []
    saved: dict[str, list] = {"heights": [], "depth": []}

    for step in range(1, args.steps + 1):
        if policy is not None:
            with torch.no_grad():
                action = policy.act_mean(
                    obs["proprio"], obs["scandots"], obs["priv"]
                ).clamp(-1.0, 1.0)
        else:
            action = torch.zeros(args.num_envs, x.action_dim, device=device)
        obs, _r, _te, _tr, _info = env.step(action)

        heights = obs["scandots"][0].detach().cpu().numpy()
        robot = env.scene["robot"]
        root_pos = robot.data.root_pos_w[0].detach().cpu().numpy()
        quat = robot.data.root_quat_w[0].detach().cpu().numpy()
        yaw, roll, pitch = yaw_roll_pitch(quat)

        hits_w = env.scene["height_scanner"].data.ray_hits_w[0].detach().cpu().numpy()
        local = hits_to_yaw_frame(hits_w, root_pos, yaw)

        # ── Verify the flatten order ONCE against the live sensor ───────
        if order is None:
            finite = np.isfinite(local[:, :2]).all(axis=1)
            if finite.sum() < 0.5 * local.shape[0]:
                print("[inspect][WARN] most rays missed; cannot verify grid "
                      "order this step — will retry")
            else:
                order = infer_grid_order(local[:, :2], x.scandots)
                ref, _, _ = grid_offsets(x.scandots, order=order)
                err = float(np.abs(ref[finite] - local[finite, :2]).max())
                print(f"\n[VERIFY] grid flatten order = '{order}' "
                      f"(max XY mismatch {err * 1000:.1f} mm)")
                print(f"[VERIFY] measured grid X span "
                      f"{local[finite, 0].min():+.2f} .. "
                      f"{local[finite, 0].max():+.2f} m, Y span "
                      f"{local[finite, 1].min():+.2f} .. "
                      f"{local[finite, 1].max():+.2f} m")
                if order != "ij":
                    print("[VERIFY][WARN] order is NOT 'ij' — pass "
                          f"--order {order} to scripts/scandot_coverage.py")

        sat = saturation_report(heights, x.scandots.height_clip)
        sat_acc.append(sat["frac_saturated"])

        # ── Measured teacher/student exteroception overlap ──────────────
        cov = None
        if x.camera.enabled:
            pts = local.copy()
            pts[:, 2] = hits_w[:, 2] - root_pos[2]      # height rel. to base
            good = np.isfinite(pts).all(axis=1)
            m = np.zeros(pts.shape[0], dtype=bool)
            if good.any():
                m[good] = frustum_mask(
                    pts[good], x.camera, x.camera.mounts[0],
                    pitch_deg=math.degrees(pitch), roll_deg=math.degrees(roll),
                )
            cov = float(m.mean())
            cov_acc.append(cov)

        # ── 3D markers from the COMPOSITED tensor (Pane B) ──────────────
        if markers is not None and order is not None:
            eff_z = root_pos[2] - x.reward.target_height - heights
            world = np.stack([hits_w[:, 0], hits_w[:, 1], eff_z], axis=-1)
            world = np.nan_to_num(world, nan=root_pos[2], posinf=root_pos[2],
                                  neginf=root_pos[2])
            t = np.clip((heights + x.scandots.height_clip)
                        / (2 * x.scandots.height_clip), 0.0, 1.0)
            idx = np.round(t * (n_colors - 1)).astype(np.int32)
            try:
                markers.visualize(
                    translations=torch.as_tensor(world, dtype=torch.float32,
                                                 device=device),
                    marker_indices=torch.as_tensor(idx, dtype=torch.long,
                                                   device=device),
                )
            except Exception as e:  # noqa: BLE001
                print(f"[inspect][WARN] marker update failed ({e}); disabling")
                markers = None

        if args.save_dir:
            saved["heights"].append(heights)
            if x.camera.enabled:
                saved["depth"].append(
                    obs["depth"][0, 0].detach().cpu().numpy()
                )

        # ── Print the panes ────────────────────────────────────────────
        if step % args.every == 0 and order is not None:
            grid = to_heatmap(heights, x.scandots, order=order)
            print("\n" + "=" * 70)
            print(f"step {step}  base_h={float(env._base_height[0]):.3f} m  "
                  f"pitch={math.degrees(pitch):+.1f}deg "
                  f"roll={math.degrees(roll):+.1f}deg  "
                  f"terrain_lvl="
                  f"{int(env.scene.terrain.terrain_levels[0]) if hasattr(env.scene.terrain, 'terrain_levels') else -1}")
            print(f"PANE A  scandots  min={sat['min']:+.3f} max={sat['max']:+.3f} "
                  f"mean={sat['mean']:+.3f} std={sat['std']:.3f}  "
                  f"SATURATED={100 * sat['frac_saturated']:.1f}% "
                  f"(+clip {100 * sat['frac_at_pos_clip']:.1f}%, "
                  f"-clip {100 * sat['frac_at_neg_clip']:.1f}%)")
            print("        front of robot at TOP, robot's left at LEFT; "
                  "'.'=low/far below, '@'=high/at base")
            for line in ascii_heatmap(
                grid, -x.scandots.height_clip, x.scandots.height_clip
            ).splitlines():
                print(f"        {line}")
            if cov is not None:
                vis_grid = to_heatmap(m.astype(float), x.scandots, order=order)
                print(f"PANE C  depth camera sees {100 * cov:.1f}% of the "
                      f"scandots this step")
                print("        visibility mask over the same grid "
                      "('@'=seen by camera)")
                for line in ascii_heatmap(vis_grid, 0.0, 1.0).splitlines():
                    print(f"        {line}")
                d = obs["depth"][0, 0].detach().cpu().numpy()
                print(f"        depth image {d.shape[0]}x{d.shape[1]}  "
                      f"min={d.min():.2f} max={d.max():.2f} m  "
                      f"new_frame={bool(obs['depth_new_frame'][0])}")
                for line in ascii_depth(d, x.camera.max_depth).splitlines():
                    print(f"        {line}")

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print(f"  steps                    {args.steps}")
    print(f"  grid flatten order       {order}")
    print(f"  forward_offset           {x.scandots.forward_offset:+.3f} m")
    print(f"  mean scandot saturation  {100 * float(np.mean(sat_acc)):.1f}%  "
          f"(height_clip=+/-{x.scandots.height_clip})")
    if cov_acc:
        c = np.array(cov_acc)
        print(f"  depth/scandot overlap    mean {100 * c.mean():.1f}%  "
              f"min {100 * c.min():.1f}%  max {100 * c.max():.1f}%")
        print("  ^ scandots outside this overlap are teacher exteroception the "
              "student can never recover.")
    else:
        print("  depth/scandot overlap    not measured (pass --camera)")

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(args.save_dir, "scandot_capture.npz"),
            heights=np.array(saved["heights"]),
            depth=(np.array(saved["depth"]) if saved["depth"] else np.zeros(0)),
            grid_x=x.scandots.grid_x, grid_y=x.scandots.grid_y,
            order=str(order), forward_offset=x.scandots.forward_offset,
        )
        print(f"  wrote {args.save_dir}/scandot_capture.npz")
    print("=" * 70)

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
    raise SystemExit(code)
