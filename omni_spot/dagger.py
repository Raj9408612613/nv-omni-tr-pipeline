"""
Phase 2 — DAgger depth distillation
====================================
The STUDENT's actions drive the simulation; the frozen Phase 1 teacher
labels every visited state; loss = MSE(student_mean, teacher_mean). This is
DAgger, not behavior cloning: the state distribution is the student's own.

Updates are streaming/on-policy with no aggregation buffer: classic DAgger
aggregates because single-env trajectories are correlated and labels are
expensive, but here N parallel envs give a decorrelated batch every step and
the teacher is free to query. This also keeps GRU TBPTT trivially correct —
gradients never cross a stored hidden state (the depth encoder detaches its
hidden after each tick; encoder gradients flow on render ticks, trunk/head
gradients flow every step).

The adaptation module phi is reused from Phase 1 AS-IS (frozen); only
exteroception is distilled. The teacher consumes scandots+priv, which the
shared NavEnv always provides — cameras are merely added on top in Phase 2.
"""

from __future__ import annotations

import os
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

from .checkpoint import (
    load_checkpoint,
    load_teacher_into_student,
    save_checkpoint,
)
from .configs.base import ExperimentCfg
from .networks import StudentPolicy, TeacherPolicy


class DaggerTrainer:
    """Owns the frozen teacher, the student, and the streaming update."""

    def __init__(self, cfg: ExperimentCfg, teacher_ckpt: str,
                 device: str = "cuda"):
        self.cfg = cfg
        self.device = torch.device(device)
        sc = cfg.student

        ckpt = load_checkpoint(teacher_ckpt, self.device)
        teacher_state = ckpt["model_state_dict"]

        self.teacher = TeacherPolicy(cfg).to(self.device)
        self.teacher.load_state_dict(teacher_state)
        self.teacher.eval()
        self.teacher.requires_grad_(False)

        self.student = StudentPolicy(cfg).to(self.device)
        missing, _ = load_teacher_into_student(self.student, teacher_state)
        print(f"[dagger] warm-started student; fresh modules: "
              f"{sorted({m.split('.')[2] for m in missing})}")

        if sc.distill_extero_only:
            # Freeze everything except the depth encoder so hold-steps
            # (cached latent, no grad path) skip cleanly and no gradients
            # accumulate on parameters the optimizer never touches.
            self.student.actor.requires_grad_(False)
            self.student.actor.extero_encoder.requires_grad_(True)
            params = list(self.student.actor.extero_encoder.parameters())
        else:
            params = [p for p in self.student.actor.parameters()
                      if p.requires_grad]
        self.optimizer = torch.optim.Adam(params, lr=sc.lr)
        self._params = params

        B = sc.num_envs
        self._prev_done = torch.ones(B, dtype=torch.bool, device=self.device)
        self._step_i = 0
        self.loss_ema: float | None = None

    # ──────────────────────────────────────────────────────────────────
    def step(self, obs: dict) -> tuple[torch.Tensor, dict]:
        """One DAgger step. Returns (action to execute, metrics)."""
        cfg = self.cfg
        sc = cfg.student
        self.student.train()

        depth = obs["depth"] / cfg.camera.max_depth
        e_s = self.student.actor.extero_encoder.step(
            depth, obs["depth_new_frame"], reset_mask=self._prev_done
        )
        with torch.no_grad():
            z_hat = self.student.adaptation_module(obs["history"])
        a_student, _ = self.student.actor.forward_with_latent(
            obs["proprio"], e_s, z_hat
        )

        with torch.no_grad():
            a_teacher = self.teacher.act_mean(
                obs["proprio"], obs["scandots"], obs["priv"]
            )

        loss = F.mse_loss(a_student, a_teacher)
        grad_norm = 0.0
        if loss.requires_grad:
            # (On hold steps with distill_extero_only the cached latent is
            # detached, so there is nothing to train — skip cleanly.)
            loss.backward()
            self._step_i += 1
            if self._step_i % sc.optimizer_step_every == 0:
                gn = nn.utils.clip_grad_norm_(self._params, sc.max_grad)
                grad_norm = float(gn)
                self.optimizer.step()
                self.optimizer.zero_grad()

        with torch.no_grad():
            gap = (a_student - a_teacher).abs()
            p50 = float(gap.median())
            p95 = float(torch.quantile(gap.flatten(), 0.95))

        loss_val = float(loss.detach())
        self.loss_ema = (loss_val if self.loss_ema is None
                         else 0.99 * self.loss_ema + 0.01 * loss_val)

        action = a_student.detach()
        if sc.action_noise_std > 0:
            action = action + sc.action_noise_std * torch.randn_like(action)
        action = torch.clamp(action, -1.0, 1.0)

        metrics = {
            "dagger_loss": loss_val,
            "dagger_loss_ema": self.loss_ema,
            "action_gap_p50": p50,
            "action_gap_p95": p95,
            "grad_norm": grad_norm,
            "new_frame_frac": float(
                obs["depth_new_frame"].float().mean()
            ),
        }
        return action, metrics

    def update_dones(self, terminated: torch.Tensor,
                     truncated: torch.Tensor):
        """Call after env.step — resets the GRU state next step."""
        self._prev_done = terminated | truncated

    # ──────────────────────────────────────────────────────────────────
    def save(self, path: str):
        save_checkpoint(
            path,
            model_state=self.student.state_dict(),
            phase="student",
            robot=self.cfg.robot.name,
            optimizer_state=self.optimizer.state_dict(),
        )

    def load(self, path: str):
        ckpt = load_checkpoint(path, self.device)
        self.student.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        # phi stays frozen regardless of what was loaded
        self.student.adaptation_module.requires_grad_(False)
        self.student.adaptation_module.eval()


# ════════════════════════════════════════════════════════════════════════════
# Entry (called by train.py — Isaac app is already running)
# ════════════════════════════════════════════════════════════════════════════

def run_student(cfg: ExperimentCfg, a, *, logger_cls, csv_fields,
                report_gpu_memory, phase_ctx) -> int:
    from .env_cfg import build_env_cfg
    from .nav_env import NavEnv

    sc = cfg.student
    assert cfg.camera.enabled, "Phase 2 requires cfg.camera.enabled"
    env_cfg = build_env_cfg(cfg, sc.num_envs)
    env_cfg.seed = a.seed

    with phase_ctx("Environment creation (with cameras)"):
        env = NavEnv(env_cfg, cfg)
    device = str(env.device)
    print(f"[INIT] {sc.num_envs} envs on {device}; depth "
          f"{cfg.camera.n_cams}x{cfg.camera.height}x{cfg.camera.width} @ "
          f"{1.0 / cfg.camera.update_period_s:.0f} Hz, policy @ "
          f"{1.0 / cfg.sim.control_dt:.0f} Hz", flush=True)

    trainer = DaggerTrainer(cfg, a.teacher_ckpt, device)
    if a.resume:
        trainer.load(a.resume)
        print(f"[INIT] Resumed student from {a.resume}")

    run_id = datetime.now().strftime("student_%Y%m%d_%H%M%S")
    logger = logger_cls(a.log_dir, run_id, csv_fields)
    print(f"[INIT] Logging to {logger.log_dir}")

    with phase_ctx("First reset"):
        obs, _ = env.reset()
    report_gpu_memory("after reset")

    best_ema = float("inf")
    t_start = time.time()
    done_acc = 0.0
    frame_acc = 0.0
    window = 0

    for it in range(1, sc.total_iters + 1):
        action, m = trainer.step(obs)
        obs, _reward, terminated, truncated, _info = env.step(action)
        trainer.update_dones(terminated, truncated)

        done_acc += float((terminated | truncated).float().mean())
        frame_acc += m["new_frame_frac"]
        window += 1

        if it % sc.log_interval == 0:
            wall = time.time() - t_start
            sps = it * sc.num_envs / max(1e-6, wall)
            row = {
                "iter": it,
                "timesteps": it * sc.num_envs,
                "wall_time": wall,
                "sps": sps,
                "dagger_loss": m["dagger_loss"],
                "dagger_loss_ema": m["dagger_loss_ema"],
                "depth_new_frame_rate": frame_acc / window,
                "action_gap_p50": m["action_gap_p50"],
                "action_gap_p95": m["action_gap_p95"],
                "done_rate": done_acc / window,
                "grad_norm": m["grad_norm"],
                "lr": sc.lr,
                "vram_alloc_gb": (
                    torch.cuda.max_memory_allocated() / 2**30
                    if torch.cuda.is_available() else 0.0
                ),
            }
            logger.log(it, row)
            print(
                f"[{it:6d}/{sc.total_iters}] "
                f"loss={m['dagger_loss']:.5f} (ema {m['dagger_loss_ema']:.5f})"
                f"  gap p50={m['action_gap_p50']:.4f} p95={m['action_gap_p95']:.4f}"
                f"  frame_rate={frame_acc / window:.3f}  sps={sps:,.0f}",
                flush=True,
            )
            done_acc = frame_acc = 0.0
            window = 0
        if it == sc.log_interval:
            report_gpu_memory("after first log window")

        if it % sc.save_interval == 0:
            trainer.save(os.path.join(logger.log_dir, f"ckpt_{it:06d}.pt"))
        if trainer.loss_ema is not None and trainer.loss_ema < best_ema:
            best_ema = trainer.loss_ema
            if it % sc.log_interval == 0:
                trainer.save(os.path.join(logger.log_dir, "best.pt"))

    trainer.save(os.path.join(logger.log_dir, "final.pt"))
    report_gpu_memory("end of training")

    with open(os.path.join(a.log_dir, "SUCCESS"), "w") as f:
        f.write(f"phase=student robot={cfg.robot.name} "
                f"iters={sc.total_iters} run={run_id}\n")
    print(f"[DONE] Distillation complete: {sc.total_iters} iters, "
          f"best loss EMA {best_ema:.5f}. Checkpoints in {logger.log_dir}")
    logger.close()
    env.close()
    return 0
