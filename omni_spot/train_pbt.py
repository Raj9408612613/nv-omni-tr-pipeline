"""
Population-Based Training Entry Point — Phase 1 teacher, single sim
===================================================================
N members in ONE Isaac Sim process sharing a single env.step. Each member has
its own policy + PPO optimizer + 4 reward knobs + 3 PPO knobs; periodically the
population ranks by a weight-free fitness, copies the bottom quartile from the
top quartile (exploit) and perturbs the copies' knobs (explore).

    PYTHONPATH=. python -m omni_spot.train_pbt --robot spot \
        --pop_size 24 --envs_per_member 2048 --total_updates 1500 \
        --pbt_interval 50 --pbt_warmup 100 --headless

VRAM probe before committing to a larger population (plan Phase 5):

    python -m omni_spot.train_pbt --pop_size 32 --vram_probe 3 --headless

CPU smoke (no Isaac Lab) — exercises the full PBT loop on the mock env:

    python -m omni_spot.train_pbt --mock --device cpu --pop_size 4 \
        --envs_per_member 8 --total_updates 12 --pbt_warmup 4 --pbt_interval 4

IMPORTANT: AppLauncher MUST run before importing isaaclab/env modules — the
import order below mirrors train.py. In --mock mode Isaac Sim is never
launched.
"""

import argparse
import csv
import os
import subprocess
import sys
import threading
import time
from datetime import datetime


# ── Heartbeat for slow blocking init phases (mirrors train.py) ───────────────
def _heartbeat(stop_event: threading.Event, label: str, interval: int = 30):
    start = time.time()
    while not stop_event.wait(interval):
        print(f"  [WAIT] {label} still in progress... "
              f"({time.time() - start:.0f}s elapsed)", flush=True)


class _Phase:
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

    def __exit__(self, exc_type, *_):
        self._stop.set()
        self._thread.join()
        word = "FAILED" if exc_type else "DONE"
        print(f"  [{word}] {self._label} after "
              f"{time.time() - self._start:.1f}s", flush=True)


# ── Args ─────────────────────────────────────────────────────────────────────
_parser = argparse.ArgumentParser(description="Population-Based Training teacher")
_parser.add_argument("--robot", type=str, default="spot")
_parser.add_argument("--pop_size", type=int, default=24)
_parser.add_argument("--envs_per_member", type=int, default=2048)
_parser.add_argument("--total_updates", type=int, default=1500)
_parser.add_argument("--pbt_interval", type=int, default=50)
_parser.add_argument("--pbt_warmup", type=int, default=100)
_parser.add_argument("--fitness_dist_weight", type=float, default=0.1)
_parser.add_argument("--n_steps", type=int, default=None,
                     help="Rollout horizon (default: cfg.teacher.n_steps)")
_parser.add_argument("--seed", type=int, default=42)
_parser.add_argument("--log_dir", type=str, default="omni_logs")
_parser.add_argument("--log_interval", type=int, default=1)
_parser.add_argument("--save_interval", type=int, default=50)
_parser.add_argument("--resume", type=str, default=None,
                     help="Population checkpoint (population_latest.pt) to resume")
_parser.add_argument("--init_ckpt", type=str, default=None,
                     help="Warm-start every member from a teacher checkpoint")
_parser.add_argument("--vram_probe", type=int, default=0, metavar="N",
                     help="Run N updates, print peak VRAM, then exit (0 = off)")
# CPU smoke path (no Isaac Lab).
_parser.add_argument("--mock", action="store_true",
                     help="Use the pure-PyTorch MockEnv (no Isaac Sim)")
_parser.add_argument("--device", type=str, default="cuda")

# ── Launch Isaac Sim BEFORE importing Isaac Lab sub-modules (unless --mock) ──
_HAS_APP = False
try:
    from isaaclab.app import AppLauncher
    _HAS_APP = True
except ImportError:
    try:
        from omni.isaac.lab.app import AppLauncher
        _HAS_APP = True
    except ImportError:
        AppLauncher = None

if _HAS_APP:
    AppLauncher.add_app_launcher_args(_parser)
else:
    # Keep --headless accepted even when Isaac Lab is absent (mock runs).
    _parser.add_argument("--headless", action="store_true")

args = _parser.parse_args()

simulation_app = None
if not args.mock:
    if not _HAS_APP:
        print("[ERROR] Isaac Lab not found and --mock not set. Install Isaac "
              "Lab or pass --mock for the CPU smoke path.", flush=True)
        sys.exit(1)
    print("[INIT] Launching Isaac Sim (first run takes ~5 min for shader "
          "compilation)...", flush=True)
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    print("[INIT] Isaac Sim launched successfully.", flush=True)

# ── NOW safe to import torch + Isaac Lab sub-modules ─────────────────────────
import torch  # noqa: E402

from .configs import get_experiment_cfg  # noqa: E402
from .pbt import KNOB_NAMES, Population  # noqa: E402


def report_gpu_memory(tag: str):
    if torch.cuda.is_available():
        dev = torch.cuda.current_device()
        total = torch.cuda.get_device_properties(dev).total_memory / 2**30
        print(
            f"[VRAM][{tag}] torch: allocated="
            f"{torch.cuda.memory_allocated(dev) / 2**30:.2f} GiB  "
            f"max_allocated="
            f"{torch.cuda.max_memory_allocated(dev) / 2**30:.2f} GiB  "
            f"reserved={torch.cuda.memory_reserved(dev) / 2**30:.2f} GiB  "
            f"device_total={total:.1f} GiB", flush=True,
        )
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader"], text=True, timeout=10,
        ).strip()
        print(f"[VRAM][{tag}] nvidia-smi: {out}", flush=True)
    except Exception:  # noqa: BLE001 — informational only
        pass


class _Csv:
    """Minimal flush-every-row CSV writer (crash-safe partial logs)."""

    def __init__(self, path: str, fields: list[str]):
        self.fields = fields
        self.f = open(path, "w", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=fields, extrasaction="ignore")
        self.w.writeheader()
        self.f.flush()

    def row(self, d: dict):
        self.w.writerow(d)
        self.f.flush()

    def close(self):
        self.f.close()


def _build_env(cfg, total_envs: int, device: str, mock: bool):
    if mock:
        from .mock_env import MockEnv
        return MockEnv(cfg, num_envs=total_envs, device=device)
    from .env_cfg import build_env_cfg
    from .nav_env import NavEnv
    env_cfg = build_env_cfg(cfg, total_envs)
    env_cfg.seed = args.seed
    return NavEnv(env_cfg, cfg)


def _save_full(path: str, pop: Population, update: int):
    torch.save({"update": update, "population": pop.state_dict()}, path)


def main() -> int:
    torch.manual_seed(args.seed)
    cfg = get_experiment_cfg(args.robot)
    if args.n_steps is not None:
        cfg.teacher.n_steps = args.n_steps
    n_steps = cfg.teacher.n_steps

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA unavailable; falling back to CPU.", flush=True)
        device = "cpu"

    # ── Population (members + per-env reward-weight tiling) ──────────────
    with _Phase("Population construction"):
        pop = Population(
            cfg, n_members=args.pop_size, envs_per_member=args.envs_per_member,
            device=device, seed=args.seed,
            fitness_dist_weight=args.fitness_dist_weight,
            init_ckpt=args.init_ckpt,
        )
    start_update = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        pop.load_state_dict(ck["population"])
        start_update = int(ck.get("update", 0))
        print(f"[INIT] Resumed population from {args.resume} "
              f"@ update {start_update}", flush=True)

    total_envs = pop.total_envs
    with _Phase("Environment creation"):
        env = _build_env(cfg, total_envs, device, args.mock)
    # Hand the live per-env reward-weight tensors to the env (re-pointed after
    # any resume, since load reallocates them).
    env._reward_weights = pop.reward_weights

    print(f"[CONFIG] pop_size={args.pop_size} envs_per_member="
          f"{args.envs_per_member} total_envs={total_envs} n_steps={n_steps} "
          f"interval={args.pbt_interval} warmup={args.pbt_warmup} "
          f"device={device} mock={args.mock}", flush=True)

    run_id = datetime.now().strftime("pbt_%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.log_dir, run_id)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INIT] Logging to {out_dir}", flush=True)

    log = _Csv(os.path.join(out_dir, "train_log.csv"), [
        "update", "timesteps", "wall_time", "rollout_sec", "update_sec", "sps",
        "rew_mean_pop", "rew_min_pop", "rew_max_pop", "done_rate_pop",
        "best_fitness", "mean_fitness", "vram_alloc_gb",
    ])
    members_csv = _Csv(os.path.join(out_dir, "pbt_members.csv"), [
        "update", "member_id", "fitness", "success_rate", "mean_final_dist",
        *KNOB_NAMES,
    ])
    events_csv = _Csv(os.path.join(out_dir, "pbt_events.csv"), [
        "update", "target", "source", "donor_fitness",
        "target_fitness_before", "perturb_seed",
        *[f"old_{k}" for k in KNOB_NAMES], *[f"new_{k}" for k in KNOB_NAMES],
    ])
    corr_csv = _Csv(os.path.join(out_dir, "pbt_corr.csv"), [
        "update", *[f"corr_{k}" for k in KNOB_NAMES],
    ])

    with _Phase("First reset"):
        obs, _ = env.reset()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    report_gpu_memory("after reset")

    probe = args.vram_probe
    last_updates = (start_update + probe) if probe > 0 else args.total_updates

    t_start = time.time()
    timesteps = 0
    best_fitness = float("-inf")
    mean_fitness = float("nan")

    for update in range(start_update + 1, last_updates + 1):
        t0 = time.time()
        obs, batches, stats = pop.collect_rollouts(env, obs, n_steps)
        rollout_sec = time.time() - t0

        t0 = time.time()
        infos = pop.update_members(batches)
        update_sec = time.time() - t0

        timesteps += total_envs * n_steps
        sps = total_envs * n_steps / max(1e-6, rollout_sec + update_sec)

        rew_means = [s["rew_mean"] for s in stats]
        rew_mins = [s["rew_min"] for s in stats]
        rew_maxs = [s["rew_max"] for s in stats]
        done_rates = [s["done_rate"] for s in stats]
        vram = (torch.cuda.memory_allocated() / 2**30
                if torch.cuda.is_available() else 0.0)
        log.row({
            "update": update, "timesteps": timesteps,
            "wall_time": time.time() - t_start,
            "rollout_sec": rollout_sec, "update_sec": update_sec, "sps": sps,
            "rew_mean_pop": sum(rew_means) / len(rew_means),
            "rew_min_pop": min(rew_mins), "rew_max_pop": max(rew_maxs),
            "done_rate_pop": sum(done_rates) / len(done_rates),
            "best_fitness": best_fitness, "mean_fitness": mean_fitness,
            "vram_alloc_gb": vram,
        })

        if update % args.log_interval == 0:
            print(f"[{update:5d}/{last_updates}] "
                  f"rew(pop μ)={sum(rew_means)/len(rew_means):7.3f}  "
                  f"done={sum(done_rates)/len(done_rates):.3f}  "
                  f"best_fit={best_fitness:.3f}  sps={sps:,.0f}", flush=True)

        if update == start_update + 1:
            report_gpu_memory("after update 1")

        # ── Sanity (plan Phase 5): top member's reward-component mix ────
        if update % 200 == 0:
            top = pop.top_member()
            comps = stats[top.id]["_diag"].get("reward_components", {})
            mix = "  ".join(f"{k}={comps.get(k, float('nan')):+.3f}"
                            for k in ("r_alive", "r_goal", "r_vel_track"))
            print(f"    [sanity] top member {top.id} components: {mix}",
                  flush=True)

        # ── PBT: exploit / explore ──────────────────────────────────────
        do_pbt = (update >= args.pbt_warmup
                  and update % args.pbt_interval == 0
                  and probe == 0)
        if do_pbt:
            events = pop.evolve()
            fits = [m.fitness for m in pop.members if m.fitness == m.fitness]
            best_fitness = max(fits) if fits else float("-inf")
            mean_fitness = sum(fits) / len(fits) if fits else float("nan")

            for m in pop.members:
                members_csv.row({
                    "update": update, "member_id": m.id, "fitness": m.fitness,
                    "success_rate": m.success_rate,
                    "mean_final_dist": m.mean_final_dist,
                    **{k: m.knobs[k] for k in KNOB_NAMES},
                })
            for ev in events:
                events_csv.row({
                    "update": update, "target": ev["target"],
                    "source": ev["source"],
                    "donor_fitness": ev["donor_fitness"],
                    "target_fitness_before": ev["target_fitness_before"],
                    "perturb_seed": ev["perturb_seed"],
                    **{f"old_{k}": ev["old_knobs"][k] for k in KNOB_NAMES},
                    **{f"new_{k}": ev["new_knobs"][k] for k in KNOB_NAMES},
                })
            corr = pop.knob_fitness_correlations()
            corr_csv.row({"update": update,
                          **{f"corr_{k}": corr[k] for k in KNOB_NAMES}})
            print(f"  [PBT] update {update}: evolved {len(events)} members; "
                  f"best_fit={best_fitness:.3f} mean_fit={mean_fitness:.3f} "
                  f"corr(goal_bonus)={corr['goal_bonus']:.2f}", flush=True)

        # ── Checkpoints ─────────────────────────────────────────────────
        if update % args.save_interval == 0 and probe == 0:
            _save_full(os.path.join(out_dir, "population_latest.pt"), pop, update)
            # best.pt = top member in the standard teacher format (eval-loadable)
            top = pop.top_member()
            top.trainer.save(os.path.join(out_dir, "best.pt"))
            print(f"  [SAVE] population_latest.pt + best.pt @ update {update}",
                  flush=True)

    if probe > 0:
        report_gpu_memory(f"after {probe} probe updates")
        print(f"[PROBE] {probe} updates done for pop_size={args.pop_size}, "
              f"envs_per_member={args.envs_per_member}. See peak VRAM above.",
              flush=True)
    else:
        _save_full(os.path.join(out_dir, "population_final.pt"), pop, last_updates)
        pop.top_member().trainer.save(os.path.join(out_dir, "best.pt"))
        with open(os.path.join(args.log_dir, "SUCCESS"), "w") as f:
            f.write(f"phase=pbt robot={cfg.robot.name} "
                    f"pop_size={args.pop_size} timesteps={timesteps} "
                    f"run={run_id}\n")
        print(f"[DONE] PBT complete: {timesteps:,} timesteps. "
              f"Best fitness {best_fitness:.3f}. Artifacts in {out_dir}",
              flush=True)

    report_gpu_memory("end")
    log.close(); members_csv.close(); events_csv.close(); corr_csv.close()
    close = getattr(env, "close", None)
    if callable(close):
        close()
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
        if simulation_app is not None:
            simulation_app.close()
    sys.exit(code)
