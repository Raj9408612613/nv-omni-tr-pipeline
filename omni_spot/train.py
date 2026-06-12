"""
Training Entry Point — teacher-student pipeline (Isaac Lab)
============================================================
Single entrypoint for both phases:

    # Phase 1: privileged PPO teacher (no rendering, scandots via raycast)
    python -m omni_spot.train --phase teacher --robot spot --headless

    # Phase 2: DAgger depth distillation (cameras enabled automatically)
    python -m omni_spot.train --phase student --robot spot \
        --teacher_ckpt omni_logs/<run>/best.pt --headless

IMPORTANT: Isaac Lab sub-modules require the Omniverse app to be running.
AppLauncher MUST run before importing isaaclab/env modules — keep the
import order below.
"""

import argparse
import csv
import os
import subprocess
import sys
import threading
import time
from datetime import datetime


def _heartbeat(stop_event: threading.Event, label: str, interval: int = 30):
    """Print periodic progress during silent C++/Omniverse init phases."""
    start = time.time()
    while not stop_event.wait(interval):
        elapsed = time.time() - start
        print(f"  [WAIT] {label} still in progress... ({elapsed:.0f}s elapsed)",
              flush=True)


class _Phase:
    """Context manager that prints a heartbeat during a slow blocking call."""

    def __init__(self, label: str, interval: int = 30):
        self._label = label
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=_heartbeat, args=(self._stop, label, interval), daemon=True
        )

    def __enter__(self):
        self._start = time.time()
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join()
        elapsed = time.time() - self._start
        print(f"  [DONE] {self._label} completed in {elapsed:.1f}s", flush=True)


# ── Step 1: Launch Isaac Sim BEFORE importing Isaac Lab sub-modules ──
print("[INIT] Launching Isaac Sim (first run takes ~5 min for shader "
      "compilation)...", flush=True)
try:
    from isaaclab.app import AppLauncher
except ImportError:
    try:
        from omni.isaac.lab.app import AppLauncher
    except ImportError:
        print("[ERROR] Cannot import AppLauncher from isaaclab or "
              "omni.isaac.lab")
        print("        Is Isaac Lab installed? pip list | grep isaaclab")
        sys.exit(1)

_parser = argparse.ArgumentParser(
    description="Teacher-student navigation RL — Isaac Lab"
)
_parser.add_argument("--phase", choices=["teacher", "student"],
                     default="teacher")
_parser.add_argument("--robot", type=str, default="spot",
                     help="Config module name in omni_spot/configs/")
# Common overrides (None -> use the value from the robot config)
_parser.add_argument("--num_envs", type=int, default=None)
_parser.add_argument("--lr", type=float, default=None)
_parser.add_argument("--seed", type=int, default=42)
_parser.add_argument("--log_interval", type=int, default=None)
_parser.add_argument("--save_interval", type=int, default=None)
_parser.add_argument("--log_dir", type=str, default="omni_logs")
_parser.add_argument("--resume", type=str, default=None)
# Teacher-specific
_parser.add_argument("--n_steps", type=int, default=None)
_parser.add_argument("--total_updates", type=int, default=None)
_parser.add_argument("--profile", type=int, default=0, metavar="N",
                     help="Profile first N updates")
# Student-specific
_parser.add_argument("--total_iters", type=int, default=None)
_parser.add_argument("--teacher_ckpt", type=str, default=None,
                     help="Phase 1 checkpoint (required for --phase student)")
AppLauncher.add_app_launcher_args(_parser)
args = _parser.parse_args()

if args.phase == "student":
    if not args.teacher_ckpt:
        _parser.error("--teacher_ckpt is required for --phase student")
    # Cameras exist only in Phase 2; never enabled for the teacher.
    args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
print("[INIT] Isaac Sim launched successfully.", flush=True)

# ── Step 2: NOW safe to import Isaac Lab sub-modules & PyTorch ───────
import torch  # noqa: E402

from .configs import get_experiment_cfg  # noqa: E402
from .diagnostics import print_diagnostics  # noqa: E402

REWARD_COMPONENT_KEYS = (
    "r_progress", "r_goal", "r_collision", "r_near",
    "r_upright", "r_height", "r_energy", "r_smooth",
    "r_alive", "r_heading", "r_vel_track", "dist_goal", "terrain_level",
)

TEACHER_CSV_FIELDS = (
    "update", "timesteps", "wall_time", "rollout_sec", "update_sec", "sps",
    "rew_mean", "rew_min", "rew_max", "done_rate", "ep_count", "adapt_loss",
    "ret_raw_mean", "ret_raw_std", "ret_raw_min", "ret_raw_max",
    "ret_norm_mean", "ret_norm_std", "ret_norm_min", "ret_norm_max",
    "ret_scale_mean", "ret_scale_std",
    "adv_mean", "adv_std", "adv_min", "adv_max",
    "val_raw_mean", "val_raw_std", "val_raw_min", "val_raw_max",
    "explained_var",
    "action_mean_abs", "action_std_mean", "entropy",
    "approx_kl", "clip_frac", "ratio_mean", "ratio_max",
    "policy_loss", "value_loss", "total_loss",
    "grad_norm", "skipped_steps",
    "epochs_run", "early_stop_epoch", "running_kl", "lr",
    "proprio_mean", "proprio_std", "proprio_nan_frac",
    "scandot_mean", "scandot_std", "scandot_nan_frac",
    *REWARD_COMPONENT_KEYS,
)

STUDENT_CSV_FIELDS = (
    "iter", "timesteps", "wall_time", "sps",
    "dagger_loss", "dagger_loss_ema",
    "depth_new_frame_rate", "action_gap_p50", "action_gap_p95",
    "done_rate", "grad_norm", "lr", "vram_alloc_gb",
)


def _fmt(x, spec=".6g"):
    try:
        return format(float(x), spec)
    except (TypeError, ValueError):
        return ""


class SimpleLogger:
    """Progressive CSV + TensorBoard logger. Every row is flushed
    immediately so a crash mid-run still leaves usable logs."""

    def __init__(self, log_dir: str, run_id: str, fields: tuple):
        self.log_dir = os.path.join(log_dir, run_id)
        os.makedirs(self.log_dir, exist_ok=True)
        self.csv_path = os.path.join(self.log_dir, "train_log.csv")
        self.csv_file = open(self.csv_path, "w", newline="")
        self.fields = fields
        self.csv_writer = csv.DictWriter(
            self.csv_file, fieldnames=list(fields), extrasaction="ignore"
        )
        self.csv_writer.writeheader()
        self.csv_file.flush()
        self.tb = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tb = SummaryWriter(os.path.join(self.log_dir, "tb"))
        except Exception as e:  # noqa: BLE001 — TB is optional
            print(f"[logger] TensorBoard unavailable ({e}); CSV only")

    def log(self, step: int, row: dict):
        self.csv_writer.writerow(
            {k: (v if isinstance(v, str) else _fmt(v))
             for k, v in row.items() if k in self.fields}
        )
        self.csv_file.flush()
        if self.tb is not None:
            for k, v in row.items():
                try:
                    self.tb.add_scalar(k, float(v), step)
                except (TypeError, ValueError):
                    pass

    def close(self):
        self.csv_file.close()
        if self.tb is not None:
            self.tb.close()


def report_gpu_memory(tag: str):
    """Print torch allocator stats AND total device usage (nvidia-smi —
    captures PhysX/RTX memory that the torch allocator can't see)."""
    if torch.cuda.is_available():
        dev = torch.cuda.current_device()
        total = torch.cuda.get_device_properties(dev).total_memory / 2**30
        print(
            f"[VRAM][{tag}] torch: allocated="
            f"{torch.cuda.memory_allocated(dev) / 2**30:.2f} GiB  "
            f"max_allocated="
            f"{torch.cuda.max_memory_allocated(dev) / 2**30:.2f} GiB  "
            f"reserved={torch.cuda.memory_reserved(dev) / 2**30:.2f} GiB  "
            f"device_total={total:.1f} GiB",
            flush=True,
        )
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader"],
            text=True, timeout=10,
        ).strip()
        print(f"[VRAM][{tag}] nvidia-smi: {out}", flush=True)
    except Exception:  # noqa: BLE001 — informational only
        pass


def _apply_overrides(cfg, a):
    """CLI values override the robot-config training fields when provided."""
    t, s = cfg.teacher, cfg.student
    if a.num_envs is not None:
        t.num_envs = a.num_envs
        s.num_envs = a.num_envs
    if a.lr is not None:
        t.lr = a.lr
        s.lr = a.lr
    if a.n_steps is not None:
        t.n_steps = a.n_steps
    if a.total_updates is not None:
        t.total_updates = a.total_updates
    if a.total_iters is not None:
        s.total_iters = a.total_iters
    if a.log_interval is not None:
        t.log_interval = a.log_interval
        s.log_interval = a.log_interval
    if a.save_interval is not None:
        t.save_interval = a.save_interval
        s.save_interval = a.save_interval


# ════════════════════════════════════════════════════════════════════════════
# Phase 1 — privileged PPO teacher
# ════════════════════════════════════════════════════════════════════════════

def run_teacher(cfg, a) -> int:
    from .env_cfg import build_env_cfg
    from .nav_env import NavEnv
    from .ppo import PPOTrainer

    tc = cfg.teacher
    env_cfg = build_env_cfg(cfg, tc.num_envs)
    env_cfg.seed = a.seed

    with _Phase("Environment creation"):
        env = NavEnv(env_cfg, cfg)
    device = str(env.device)
    print(f"[INIT] {tc.num_envs} envs on {device}; scandots "
          f"{cfg.scandots.grid_x}x{cfg.scandots.grid_y}; cameras: OFF",
          flush=True)

    trainer = PPOTrainer(cfg, device=device)
    if a.resume:
        trainer.load(a.resume)
        print(f"[INIT] Resumed from {a.resume}")

    run_id = datetime.now().strftime("teacher_%Y%m%d_%H%M%S")
    logger = SimpleLogger(a.log_dir, run_id, TEACHER_CSV_FIELDS)
    print(f"[INIT] Logging to {logger.log_dir}")

    with _Phase("First reset"):
        obs, _ = env.reset()
    report_gpu_memory("after reset")

    best_rew = float("-inf")
    t_start = time.time()
    timesteps = 0

    for update in range(1, tc.total_updates + 1):
        trainer.anneal_lr(update)

        t0 = time.time()
        obs, batch, stats = trainer.collect_rollout(
            env, obs, profile=(update <= a.profile)
        )
        rollout_sec = time.time() - t0

        t0 = time.time()
        info = trainer.update(batch)
        update_sec = time.time() - t0

        timesteps += tc.num_envs * tc.n_steps
        sps = tc.num_envs * tc.n_steps / max(1e-6, rollout_sec + update_sec)

        row = {
            "update": update, "timesteps": timesteps,
            "wall_time": time.time() - t_start,
            "rollout_sec": rollout_sec, "update_sec": update_sec, "sps": sps,
            **{k: v for k, v in stats.items() if not k.startswith("_")},
            **stats.get("_diag", {}),
            **info,
        }
        row.update(stats.get("_diag", {}).get("reward_components", {}))
        logger.log(update, row)

        if update % tc.log_interval == 0:
            print(
                f"[{update:4d}/{tc.total_updates}] "
                f"rew={stats['rew_mean']:8.3f}  "
                f"done={stats['done_rate']:.3f}  "
                f"adapt={stats['adapt_loss']:.4f}  "
                f"kl={info.get('running_kl', 0):.4f}  "
                f"lr={info.get('lr', 0):.2e}  sps={sps:,.0f}",
                flush=True,
            )
            if "_timing" in stats:
                print(f"    profile: {stats['_timing']}")
        if update == 1 or update % 25 == 0:
            print_diagnostics(update, stats.get("_diag", {}), info,
                              max_grad=tc.max_grad)
        if update == 1:
            report_gpu_memory("after update 1")

        if update % tc.save_interval == 0:
            path = os.path.join(logger.log_dir, f"ckpt_{update:05d}.pt")
            trainer.save(path)
            print(f"  [SAVE] {path}")
        if stats["rew_mean"] > best_rew:
            best_rew = stats["rew_mean"]
            trainer.save(os.path.join(logger.log_dir, "best.pt"))

    trainer.save(os.path.join(logger.log_dir, "final.pt"))
    report_gpu_memory("end of training")

    # Success marker (smoke scripts check this to avoid false positives from
    # Isaac Sim's shutdown masking the Python exit code)
    with open(os.path.join(a.log_dir, "SUCCESS"), "w") as f:
        f.write(f"phase=teacher robot={cfg.robot.name} "
                f"total_timesteps={timesteps} run={run_id}\n")
    print(f"[DONE] Teacher training complete: {timesteps:,} timesteps. "
          f"Best reward {best_rew:.3f}. Checkpoints in {logger.log_dir}")
    logger.close()
    env.close()
    return 0


# ════════════════════════════════════════════════════════════════════════════
# Entry
# ════════════════════════════════════════════════════════════════════════════

def main() -> int:
    torch.manual_seed(args.seed)
    cfg = get_experiment_cfg(args.robot)
    _apply_overrides(cfg, args)

    if args.phase == "teacher":
        return run_teacher(cfg, args)

    cfg.camera.enabled = True  # Phase 2: construct the camera rig
    from .dagger import run_student
    return run_student(
        cfg, args,
        logger_cls=SimpleLogger,
        csv_fields=STUDENT_CSV_FIELDS,
        report_gpu_memory=report_gpu_memory,
        phase_ctx=_Phase,
    )


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    finally:
        simulation_app.close()
    sys.exit(code)
