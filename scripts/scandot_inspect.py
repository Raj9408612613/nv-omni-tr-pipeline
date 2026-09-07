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
_p.add_argument("--ckpt", default=None,
                help="checkpoint to drive with. Teacher OR student — the phase "
                     "is detected from the weights. A student checkpoint also "
                     "requires --camera. Without --ckpt the robot holds its "
                     "standing pose and never traverses terrain.")
_p.add_argument("--camera", action="store_true",
                help="enable the depth rig (Pane C + overlap measurement)")
_p.add_argument("--markers", action="store_true",
                help="draw 3D scandot markers from the composited tensor")
_p.add_argument("--forward_offset", type=float, default=None,
                help="override scandots.forward_offset for this run")
_p.add_argument("--watch", default="0",
                help="which env to display: an index, 'auto' (latch onto the "
                     "first env with real terrain relief and stay on it), or "
                     "'roam' (re-pick the most varied env every step — note "
                     "that consecutive prints are then DIFFERENT robots)")
_p.add_argument("--terrain_col", type=int, default=None,
                help="force every env onto this curriculum COLUMN (the column "
                     "picks the sub-terrain type). The startup banner lists "
                     "which column is which. Implies curriculum off.")
_p.add_argument("--terrain_row", type=int, default=None,
                help="force every env onto this difficulty ROW (0 = easiest). "
                     "Implies curriculum off.")
_p.add_argument("--fixed_scale", action="store_true",
                help="scale the heatmap to +/-height_clip instead of to the "
                     "data (use to compare absolute magnitudes across steps)")
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

def subterrain_columns(x) -> list[str]:
    """Which sub-terrain each curriculum COLUMN holds.

    Mirrors Isaac's TerrainGenerator: with curriculum=True each column is
    assigned one sub-terrain by cumulative proportion. This matters because a
    single env always lands in column 0 — so `--num_envs 1` does not sample a
    random terrain type, it deterministically gets whatever column 0 is.
    """
    t = x.terrain
    names = ["flat", "rough", "stairs_up", "stairs_down"]
    props = [t.flat_proportion, t.rough_proportion,
             t.stairs_up_proportion, t.stairs_down_proportion]
    for nm, key in (("discrete_obstacles", "discrete_obstacles_proportion"),
                    ("random_grid", "random_grid_proportion"),
                    ("rails", "rails_proportion"),
                    ("stepping_stones", "stepping_stones_proportion")):
        v = getattr(t, key, 0.0)
        if v > 0:
            names.append(nm)
            props.append(v)
    p = np.asarray(props, dtype=float)
    if p.sum() <= 0:
        return ["?"] * t.cols
    cum = np.cumsum(p / p.sum())
    out = []
    for c in range(t.cols):
        hit = np.where(c / t.cols + 0.001 < cum)[0]
        out.append(names[int(hit.min())] if hit.size else names[-1])
    return out


def terrain_verdict(grid: np.ndarray, spread_cm: float) -> str:
    """Describe the terrain from the scandots themselves, not from config.

    Config can say "difficulty row 3" while the robot stands on a flat patch;
    the heights are the ground truth about what it is actually looking at.
    """
    if spread_cm < 2.0:
        return f"FLAT (only {spread_cm:.1f} cm of relief across the whole grid)"
    # A step is a SUSTAINED front-to-rear offset, not one big neighbour gap —
    # random rough ground produces large gaps too, so requiring both keeps
    # noise from being reported as a staircase.
    n = grid.shape[0]
    third = max(1, n // 3)
    rear = float(np.median(grid[:third]))
    front = float(np.median(grid[-third:]))
    offset_cm = 100.0 * (front - rear)
    jump_cm = 100.0 * float(np.abs(np.diff(grid[:, grid.shape[1] // 2])).max())
    if abs(offset_cm) > 5.0 and jump_cm > 4.0:
        kind = "STEP UP" if offset_cm > 0 else "STEP DOWN / DROP"
        return (f"{kind} ahead ({abs(offset_cm):.1f} cm sustained offset "
                f"front-vs-rear, biggest single jump {jump_cm:.1f} cm)")
    if spread_cm < 8.0:
        return f"ROUGH ({spread_cm:.1f} cm of relief, no sustained step)"
    return (f"VERY UNEVEN ({spread_cm:.1f} cm of relief, "
            f"front-vs-rear offset only {offset_cm:+.1f} cm)")


def centreline(grid: np.ndarray, sc) -> str:
    """One-line front-to-back ELEVATION profile down the centre, in cm.

    This is the view that actually answers "is there a step ahead?" — the
    full grid is for spotting left/right asymmetry.
    """
    profile = grid[:, grid.shape[1] // 2]
    x_min = -sc.size[0] / 2.0 + sc.forward_offset
    ix_base = int(round((0.0 - x_min) / sc.spacing))
    cells, marks = [], []
    for ix in range(grid.shape[0] - 1, -1, -1):       # front -> rear
        cells.append(f"{100 * profile[ix]:+4.0f}")
        marks.append(" ^  " if ix == ix_base else "    ")
    x_max = x_min + (grid.shape[0] - 1) * sc.spacing
    return (f"centre line, front({x_max:+.2f}m) -> rear({x_min:+.2f}m), cm "
            f"(^ marks the point under the base):\n"
            f"        " + "".join(cells) + "\n"
            f"        " + "".join(marks).rstrip())


def detect_phase(state: dict, declared: str | None = None) -> str:
    """'teacher' | 'student' | 'unknown', from the state_dict's own keys.

    The two policies differ only in which encoder occupies
    `actor.extero_encoder` — ScandotEncoder exposes `.net.*`, DepthGRUEncoder
    exposes `.cnn.*`/`.gru.*`. Keys are the authority here; the checkpoint's
    `phase` string is only cross-checked, since it is metadata that can be
    stale while the weights cannot.
    """
    keys = list(state)
    student = any(k.startswith(("actor.extero_encoder.cnn.",
                                "actor.extero_encoder.gru.")) for k in keys)
    teacher = (any(k.startswith("actor.extero_encoder.net.") for k in keys)
               or any(k.startswith(("priv_encoder.", "critic.")) for k in keys))
    if student and not teacher:
        found = "student"
    elif teacher and not student:
        found = "teacher"
    else:
        return "unknown"
    if declared and declared != found:
        print(f"[inspect][WARN] checkpoint says phase='{declared}' but its "
              f"weights are a {found} policy; trusting the weights.")
    return found


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

    cols = subterrain_columns(x)
    print("[INIT] terrain columns (the column picks the sub-terrain TYPE, "
          "the row picks difficulty):")
    for c, nm in enumerate(cols):
        print(f"         col {c} -> {nm}")
    missing = sorted({"flat", "rough", "stairs_up", "stairs_down"} - set(cols))
    if missing:
        print(f"[INIT][WARN] these configured sub-terrains get NO column and "
              f"are never generated: {missing}")
    forced = args.terrain_col is not None or args.terrain_row is not None
    if forced:
        # Curriculum promotion would immediately move envs off the forced cell.
        x.curriculum.enabled = False
        print("[INIT] terrain forced -> curriculum disabled for this run")
    elif args.num_envs == 1:
        print(f"[INIT][NOTE] with --num_envs 1 Isaac puts the single env in "
              f"COLUMN 0 ('{cols[0]}') — it is not a random draw. Use "
              f"--terrain_col N to place it deliberately.")

    # ── Pre-flight: identify the checkpoint BEFORE the slow env build ──
    ckpt_state = ckpt_phase = None
    if args.ckpt:
        from omni_spot.checkpoint import load_checkpoint
        ck = load_checkpoint(args.ckpt, "cpu")
        ckpt_state = ck["model_state_dict"]
        ckpt_phase = detect_phase(ckpt_state, ck.get("phase"))
        print(f"[INIT] {args.ckpt} -> {ckpt_phase} checkpoint")
        if ckpt_phase == "unknown":
            print("[ERROR] cannot tell whether this is a teacher or student "
                  "checkpoint; its actor.extero_encoder.* keys match neither "
                  "ScandotEncoder (.net.*) nor DepthGRUEncoder (.cnn/.gru.*).")
            return 2
        if ckpt_phase == "student" and not x.camera.enabled:
            print(
                "[ERROR] this is a STUDENT checkpoint: it drives from the depth\n"
                "        camera, which is off in this run. Re-run with --camera:\n"
                f"          python scripts/scandot_inspect.py --robot {args.robot} "
                f"--camera --ckpt {args.ckpt}\n"
                "        (or pass a teacher checkpoint to inspect without "
                "rendering)."
            )
            return 2

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
    if ckpt_state is not None:
        from omni_spot.networks import StudentPolicy, TeacherPolicy
        cls = TeacherPolicy if ckpt_phase == "teacher" else StudentPolicy
        policy = cls(x).to(device)
        policy.load_state_dict(
            {k: v.to(device) for k, v in ckpt_state.items()}
        )
        policy.eval()
        print(f"[INIT] driving with the {ckpt_phase} policy")
    else:
        print("[INIT] driving with zero actions — the robot HOLDS ITS STANDING "
              "POSE and will not traverse terrain. Pass --ckpt to walk.")

    if forced:
        terr = env.scene.terrain
        try:
            origins = terr.terrain_origins            # (rows, cols, 3)
            n_rows, n_cols = origins.shape[0], origins.shape[1]
            if args.terrain_row is not None:
                terr.terrain_levels[:] = max(0, min(args.terrain_row, n_rows - 1))
            if args.terrain_col is not None:
                terr.terrain_types[:] = max(0, min(args.terrain_col, n_cols - 1))
            terr.env_origins[:] = origins[
                terr.terrain_levels, terr.terrain_types
            ]
            r0 = int(terr.terrain_levels[0])
            c0 = int(terr.terrain_types[0])
            print(f"[INIT] forced all envs onto row {r0}, col {c0} "
                  f"('{cols[c0] if c0 < len(cols) else '?'}')")
        except (AttributeError, TypeError, IndexError) as e:
            print(f"[INIT][WARN] could not force terrain placement ({e}); "
                  f"envs keep their default cells")

    obs, _ = env.reset()
    prev_done = torch.zeros(args.num_envs, dtype=torch.bool, device=device)

    def act() -> torch.Tensor:
        """One deterministic action from whichever policy was loaded."""
        if policy is None:
            return torch.zeros(args.num_envs, x.action_dim, device=device)
        with torch.no_grad():
            if ckpt_phase == "teacher":
                a = policy.act_mean(
                    obs["proprio"], obs["scandots"], obs["priv"]
                )
            else:
                a = policy.act_mean(
                    obs["proprio"], obs["depth"] / x.camera.max_depth,
                    obs["depth_new_frame"], obs["history"],
                    reset_mask=prev_done,
                )
        return a.clamp(-1.0, 1.0)
    markers, n_colors = build_markers(x.scandots.n_points) if args.markers else (None, 0)

    order = None
    sat_acc: list[float] = []
    max_relief_cm = 0.0
    watched: int | None = None
    cov_acc: list[float] = []
    saved: dict[str, list] = {"heights": [], "depth": []}

    for step in range(1, args.steps + 1):
        action = act()
        obs, _r, terminated, truncated, _info = env.step(action)
        prev_done = terminated | truncated

        all_heights = obs["scandots"].detach().cpu().numpy()
        # Per-env relief, used both to pick the watched env and to report
        # whether ANY robot is on interesting terrain this step.
        relief = all_heights.max(axis=1) - all_heights.min(axis=1)
        if args.watch == "roam":
            w = int(relief.argmax())          # re-pick every step
        elif args.watch == "auto":
            # Latch onto the first env with real relief and STAY on it. Without
            # the latch, consecutive prints show different robots and the
            # pitch/roll/base_h series cannot be read as one trajectory.
            if watched is None and float(relief.max()) > 0.05:
                watched = int(relief.argmax())
                print(f"[watch] locking onto env {watched} "
                      f"({100 * float(relief.max()):.0f} cm of relief). "
                      f"Use --watch roam to follow the most varied env instead.")
            w = watched if watched is not None else int(relief.argmax())
        else:
            w = min(int(args.watch), args.num_envs - 1)
        max_relief_cm = max(max_relief_cm, 100.0 * float(relief.max()))

        heights = all_heights[w]
        robot = env.scene["robot"]
        root_pos = robot.data.root_pos_w[w].detach().cpu().numpy()
        quat = robot.data.root_quat_w[w].detach().cpu().numpy()
        yaw, roll, pitch = yaw_roll_pitch(quat)

        hits_w = env.scene["height_scanner"].data.ray_hits_w[w].detach().cpu().numpy()
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
                    obs["depth"][w, 0].detach().cpu().numpy()
                )

        # ── Print the panes ────────────────────────────────────────────
        if step % args.every == 0 and order is not None:
            grid = to_heatmap(heights, x.scandots, order=order)
            spread_cm = 100.0 * (sat["max"] - sat["min"])
            lvl = (int(env.scene.terrain.terrain_levels[w])
                   if hasattr(env.scene.terrain, "terrain_levels") else -1)
            col = (int(env.scene.terrain.terrain_types[w])
                   if hasattr(env.scene.terrain, "terrain_types") else -1)
            print("\n" + "=" * 70)
            print(f"step {step}  base_h={float(env._base_height[w]):.3f} m  "
                  f"pitch={math.degrees(pitch):+.1f}deg "
                  f"roll={math.degrees(roll):+.1f}deg  "
                  f"terrain row={lvl} col={col}  [env {w}"
                  f"{' (roaming)' if args.watch == 'roam' else ''}]")
            # Display GROUND ELEVATION (= -heights). The raw tensor stores the
            # base-relative depth, where positive means the surface is FURTHER
            # BELOW (see obs.compose_scandots: "negative = surface higher").
            # Showing it unflipped makes a staircase read upside down.
            elev = -grid
            print(f"PANE A  ground elevation in cm, relative to nominal ground "
                  f"({x.reward.target_height:.2f} m under the base).")
            print(f"        + = ground HIGHER than nominal (step up / obstacle), "
                  f"- = LOWER (drop / step down).")
            print(f"        (the raw scandot tensor is the negative of this; "
                  f"shown flipped so up reads as up)")
            print(f"        range {-100 * sat['max']:+.1f} .. "
                  f"{-100 * sat['min']:+.1f} cm   spread {spread_cm:.1f} cm   "
                  f"-> {terrain_verdict(elev, spread_cm)}")
            if sat["frac_saturated"] > 0:
                bits = []
                if sat["frac_at_pos_clip"] > 0:
                    bits.append(f"{100 * sat['frac_at_pos_clip']:.1f}% at a DROP "
                                f"deeper than {x.scandots.height_clip} m")
                if sat["frac_at_neg_clip"] > 0:
                    bits.append(f"{100 * sat['frac_at_neg_clip']:.1f}% at ground "
                                f"higher than {x.scandots.height_clip} m")
                print(f"        SATURATED: {', '.join(bits)} — pinned at "
                      f"height_clip, so the encoder cannot tell how much "
                      f"further it goes")
            print("        " + centreline(elev, x.scandots))
            print(f"        full grid, {'auto' if not args.fixed_scale else 'fixed'}"
                  f"-scaled, front of robot at TOP, robot's LEFT at left:")
            if args.fixed_scale:
                lo, hi = -x.scandots.height_clip, x.scandots.height_clip
            else:
                # Auto-scale, floored at a 10 cm span. Without a floor, 1 cm of
                # raycast noise on flat ground stretches across the full ramp
                # and renders as speckle that reads like terrain. 10 cm is the
                # scale at which relief starts to matter to a walking Spot, so
                # anything below it correctly renders as near-uniform.
                lo, hi = sat["min"], sat["max"]
                if hi - lo < 0.10:
                    mid = 0.5 * (lo + hi)
                    lo, hi = mid - 0.05, mid + 0.05
            for line in ascii_heatmap(-grid, -hi, -lo).splitlines():
                print(f"        {line}")
            print(f"        scale: ' '={-100 * hi:+.1f} cm (lowest)  ...  "
                  f"'@'={-100 * lo:+.1f} cm (highest)")
            if cov is not None:
                vis_grid = to_heatmap(m.astype(float), x.scandots, order=order)
                print(f"PANE C  depth camera sees {100 * cov:.1f}% of the "
                      f"scandots this step")
                print("        visibility mask over the same grid "
                      "('@'=seen by camera)")
                for line in ascii_heatmap(vis_grid, 0.0, 1.0).splitlines():
                    print(f"        {line}")
                d = obs["depth"][w, 0].detach().cpu().numpy()
                print(f"        depth image {d.shape[0]}x{d.shape[1]}  "
                      f"min={d.min():.2f} max={d.max():.2f} m  "
                      f"new_frame={bool(obs['depth_new_frame'][w])}")
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
    print(f"  max terrain relief seen  {max_relief_cm:.1f} cm "
          f"(across all {args.num_envs} env(s), all steps)")
    if max_relief_cm < 3.0:
        print(
            "  ^ NO ROBOT LEFT FLAT GROUND during this run, so the scandots had\n"
            "    nothing to show. Each env sits in ONE terrain column, and the\n"
            "    column decides the sub-terrain type (flat / rough / stairs_up /\n"
            "    stairs_down) — with --num_envs 1 you get one type, usually flat.\n"
            "    To see stairs, run more envs and follow the interesting one:\n"
            f"      PYTHONPATH=. python scripts/scandot_inspect.py --robot "
            f"{args.robot} \\\n"
            f"          --num_envs 64 --watch auto --steps 600 --every 50 "
            f"--headless \\\n"
            f"          --ckpt <teacher>.pt\n"
            "    height_clip saturation cannot be judged from a flat-ground run."
        )
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
