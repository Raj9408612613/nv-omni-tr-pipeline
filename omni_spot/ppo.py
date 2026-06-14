"""
PyTorch PPO Trainer — Phase 1 privileged teacher
=================================================
Asymmetric actor-critic PPO over the obs dict
    {proprio, scandots, priv, history, critic_extras}
with the adaptation module phi trained CONCURRENTLY (ROA style): during the
rollout, phi regresses z_hat = phi(history) onto z = priv_encoder(priv).detach()
with its own optimizer, so the regression never perturbs the PPO objective
and the rollout buffer never has to store the (T, B, 50, 57) history.

Unlike the old depth pipeline (which cached CNN features computed under
no_grad, so the encoder never trained), minibatches here re-encode the raw
scandot/priv observations — gradients flow end-to-end through both encoders.

GAE, running return normalization, KL early stopping, LR annealing, and the
NaN/exploding-gradient guards are carried over from the original trainer.
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn

from .configs.base import ExperimentCfg
from .networks import TeacherPolicy, gaussian_entropy, gaussian_log_prob
from .obs import build_critic_obs


class RolloutBatch:
    """Flattened rollout experience (T*B, ...). Critic input is recomposed
    from the stored fields at minibatch time (critic_extras is only 3 floats
    per sample; storing the full 253-dim critic obs would double the buffer).
    """

    __slots__ = [
        "proprio", "scandots", "priv", "critic_extras",
        "action", "log_prob", "advantage", "ret", "old_value",
    ]

    def __init__(self, **kw: torch.Tensor):
        for k in self.__slots__:
            setattr(self, k, kw[k])

    def __getitem__(self, idx):
        return RolloutBatch(**{k: getattr(self, k)[idx] for k in self.__slots__})


# ── GAE Advantage Computation ────────────────────────────────────────────────

def compute_gae(
    rewards: torch.Tensor,   # (T, B)
    values: torch.Tensor,    # (T+1, B)
    dones: torch.Tensor,     # (T, B)
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns advantages (T,B) and returns (T,B)."""
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(rewards.shape[1], device=rewards.device)

    for t in reversed(range(T)):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * mask - values[t]
        gae = delta + gamma * gae_lambda * mask * gae
        advantages[t] = gae

    returns = advantages + values[:T]
    return advantages, returns


# ── PPO Trainer ──────────────────────────────────────────────────────────────

class PPOTrainer:
    """Owns the TeacherPolicy, both optimizers, and the rollout/update loop."""

    def __init__(self, cfg: ExperimentCfg, device: str = "cuda"):
        self.cfg = cfg
        tc = cfg.teacher
        self.n_envs = tc.num_envs
        self.n_steps = tc.n_steps
        self.device = torch.device(device)

        self.net = TeacherPolicy(cfg).to(self.device)

        # PPO optimizer covers actor + priv encoder + critic; phi has its own
        # optimizer so the regression cadence/LR is independent of PPO.
        ppo_params = (
            list(self.net.actor.parameters())
            + list(self.net.priv_encoder.parameters())
            + list(self.net.critic.parameters())
        )
        self.optimizer = torch.optim.Adam(ppo_params, lr=tc.lr)
        self.adapt_optimizer = torch.optim.Adam(
            self.net.adaptation_module.parameters(), lr=tc.adaptation_lr
        )

        # Running stats for return normalization (PopArt-style). The critic
        # is trained on normalized returns; GAE consumes denormalized values
        # so reward and V(s) stay on the same scale.
        self._ret_mean = 0.0
        self._ret_std = 1.0
        self._ret_ema_alpha = 0.05

        # Linear LR decay over the run (PPO optimizer only).
        self._base_lr = tc.lr
        self._total_updates = max(1, tc.total_updates)
        self._cur_lr = tc.lr
        # Entropy-coefficient anneal (base -> ent_coef_final over the run) so
        # late-stage exploration noise stops inflating the action std.
        self._base_ent_coef = tc.ent_coef
        self._cur_ent_coef = tc.ent_coef

    # ──────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def sample_action(self, obs: dict) -> tuple:
        """Returns (action, log_prob, value). Value is in NORMALIZED scale."""
        critic_obs = build_critic_obs(
            obs["proprio"], obs["scandots"], obs["priv"], obs["critic_extras"]
        )
        mean, log_std, value = self.net.evaluate(
            obs["proprio"], obs["scandots"], obs["priv"], critic_obs
        )
        std = torch.exp(log_std)
        action = torch.clamp(mean + std * torch.randn_like(mean), -1.0, 1.0)
        log_prob = gaussian_log_prob(mean, log_std, action)
        return action, log_prob, value

    # ──────────────────────────────────────────────────────────────────
    def adaptation_step(self, obs: dict) -> float:
        """One phi regression step: ||phi(history) - z.detach()||^2."""
        with torch.no_grad():
            z = self.net.priv_encoder(obs["priv"])
        z_hat = self.net.adaptation_module(obs["history"])
        loss = torch.mean(torch.sum((z_hat - z) ** 2, dim=-1))
        if not torch.isfinite(loss):
            return float("nan")
        self.adapt_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.net.adaptation_module.parameters(), self.cfg.teacher.max_grad
        )
        self.adapt_optimizer.step()
        return float(loss.detach())

    # ──────────────────────────────────────────────────────────────────
    def collect_rollout(self, env, obs: dict, profile: bool = False):
        """Collect n_steps of experience; train phi concurrently.

        Returns (obs, RolloutBatch, rollout_stats).
        """
        tc = self.cfg.teacher
        buf = {k: [] for k in (
            "proprio", "scandots", "priv", "critic_extras",
            "action", "log_prob", "value", "reward", "done",
        )}
        adapt_losses: list[float] = []
        reward_sums = None
        reward_count = 0
        t_inference = 0.0
        t_env_step = 0.0

        for step in range(self.n_steps):
            if profile:
                torch.cuda.synchronize()
                t0 = time.time()

            action, log_prob, value = self.sample_action(obs)

            if step % tc.adapt_every == 0:
                adapt_losses.append(self.adaptation_step(obs))

            if profile:
                torch.cuda.synchronize()
                t_inference += time.time() - t0
                t0 = time.time()

            buf["proprio"].append(obs["proprio"])
            buf["scandots"].append(obs["scandots"])
            buf["priv"].append(obs["priv"])
            buf["critic_extras"].append(obs["critic_extras"])
            buf["action"].append(action)
            buf["log_prob"].append(log_prob)
            buf["value"].append(value)

            # Environment step — Isaac Lab auto-resets internally
            obs, reward, terminated, truncated, step_info = env.step(action)

            if profile:
                torch.cuda.synchronize()
                t_env_step += time.time() - t0

            buf["reward"].append(reward)
            buf["done"].append((terminated | truncated).float())

            if reward_sums is None:
                reward_sums = {k: v.clone() for k, v in step_info.items()
                               if isinstance(v, torch.Tensor)}
            else:
                for k, v in step_info.items():
                    if k in reward_sums and isinstance(v, torch.Tensor):
                        reward_sums[k] = reward_sums[k] + v
            reward_count += 1

        # Bootstrap value for the last state (normalized scale)
        _, _, last_value = self.sample_action(obs)

        rewards = torch.stack(buf["reward"])
        dones = torch.stack(buf["done"])
        stacked_values = torch.stack(buf["value"])          # (T, B) normalized
        values_norm = torch.cat(
            [stacked_values, last_value.unsqueeze(0)], dim=0
        )
        # Denormalize critic outputs before GAE so deltas are unit-consistent
        values_raw = values_norm * self._ret_std + self._ret_mean

        advantages, returns = compute_gae(
            rewards, values_raw, dones, tc.gamma, tc.gae_lambda
        )

        # Update running return stats (bootstrap on the very first batch)
        batch_ret_mean = float(returns.mean())
        batch_ret_std = float(returns.std()) + 1e-8
        if self._ret_mean == 0.0 and self._ret_std == 1.0:
            self._ret_mean = batch_ret_mean
            self._ret_std = batch_ret_std
        else:
            a = self._ret_ema_alpha
            self._ret_mean = (1.0 - a) * self._ret_mean + a * batch_ret_mean
            self._ret_std = (1.0 - a) * self._ret_std + a * batch_ret_std

        returns_norm = (returns - self._ret_mean) / self._ret_std
        stacked_values_raw = values_raw[:-1]

        def flat(x: torch.Tensor) -> torch.Tensor:
            return x.reshape(-1, *x.shape[2:]) if x.ndim > 2 else x.reshape(-1)

        adv_flat = flat(advantages)
        adv_norm = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        batch = RolloutBatch(
            proprio=flat(torch.stack(buf["proprio"])),
            scandots=flat(torch.stack(buf["scandots"])),
            priv=flat(torch.stack(buf["priv"])),
            critic_extras=flat(torch.stack(buf["critic_extras"])),
            action=flat(torch.stack(buf["action"])),
            log_prob=flat(torch.stack(buf["log_prob"])),
            advantage=adv_norm,
            ret=flat(returns_norm),
            old_value=flat(stacked_values),
        )

        finite_adapt = [a for a in adapt_losses if a == a]  # drop NaN
        rollout_stats = {
            "rew_mean": float(rewards.mean()),
            "rew_min": float(rewards.min()),
            "rew_max": float(rewards.max()),
            "done_rate": float(dones.mean()),
            "ep_count": int(dones.sum()),
            "adapt_loss": (sum(finite_adapt) / len(finite_adapt)
                           if finite_adapt else float("nan")),
        }

        stacked_proprio = torch.stack(buf["proprio"])
        stacked_scandots = torch.stack(buf["scandots"])
        rollout_stats["_diag"] = {
            "adapt_loss": rollout_stats["adapt_loss"],
            "ret_raw_mean": float(returns.mean()),
            "ret_raw_std": float(returns.std()),
            "ret_raw_min": float(returns.min()),
            "ret_raw_max": float(returns.max()),
            "ret_norm_mean": float(returns_norm.mean()),
            "ret_norm_std": float(returns_norm.std()),
            "ret_norm_min": float(returns_norm.min()),
            "ret_norm_max": float(returns_norm.max()),
            "ret_scale_mean": float(self._ret_mean),
            "ret_scale_std": float(self._ret_std),
            "adv_mean": float(adv_norm.mean()),
            "adv_std": float(adv_norm.std()),
            "adv_min": float(adv_norm.min()),
            "adv_max": float(adv_norm.max()),
            "val_raw_mean": float(stacked_values_raw.mean()),
            "val_raw_std": float(stacked_values_raw.std()),
            "val_raw_min": float(stacked_values_raw.min()),
            "val_raw_max": float(stacked_values_raw.max()),
            "explained_var": float(
                1.0 - torch.var(returns - stacked_values_raw)
                / (torch.var(returns) + 1e-8)
            ),
            "proprio_mean": float(stacked_proprio.mean()),
            "proprio_std": float(stacked_proprio.std()),
            "proprio_nan_frac": float(
                (~torch.isfinite(stacked_proprio)).float().mean()
            ),
            "scandot_mean": float(stacked_scandots.mean()),
            "scandot_std": float(stacked_scandots.std()),
            "scandot_nan_frac": float(
                (~torch.isfinite(stacked_scandots)).float().mean()
            ),
        }

        if reward_sums is not None and reward_count > 0:
            rollout_stats["_diag"]["reward_components"] = {
                k: float(v.mean() / reward_count) for k, v in reward_sums.items()
            }

        if profile:
            rollout_stats["_timing"] = {
                "inference_sec": t_inference,
                "env_step_sec": t_env_step,
            }

        return obs, batch, rollout_stats

    # ──────────────────────────────────────────────────────────────────
    def ppo_update_step(self, mb: RolloutBatch) -> dict:
        """One gradient update on a minibatch with NaN/explosion safeguards.

        Re-encodes raw scandots/priv so encoder gradients flow.
        """
        tc = self.cfg.teacher
        self.net.train()

        critic_obs = build_critic_obs(
            mb.proprio, mb.scandots, mb.priv, mb.critic_extras
        )
        mean, log_std, value = self.net.evaluate(
            mb.proprio, mb.scandots, mb.priv, critic_obs
        )

        # ── Policy loss with ratio clamping ──────────────────────────
        log_prob_new = gaussian_log_prob(mean, log_std, mb.action)
        # Tight log-ratio clamp: outside [0.8, 1.2] the clipped gradient is
        # already saturated; a wide clamp only inflates ratio_max and Adam's
        # gradient estimates on tail samples.
        log_ratio = torch.clamp(log_prob_new - mb.log_prob, -2.0, 2.0)
        ratio = torch.exp(log_ratio)

        adv = torch.clamp(mb.advantage, -5.0, 5.0)
        pg_loss1 = ratio * adv
        pg_loss2 = torch.clamp(ratio, 1 - tc.clip_eps, 1 + tc.clip_eps) * adv
        policy_loss = -torch.mean(torch.minimum(pg_loss1, pg_loss2))

        # ── Value loss (simple MSE on normalized returns) ────────────
        value_loss = 0.5 * torch.mean((value - mb.ret) ** 2)
        value_loss = torch.clamp(value_loss, 0.0, 1_000_000.0)

        entropy = torch.mean(gaussian_entropy(log_std))

        # The actor mean is tanh-squashed into (-1, 1) (see networks.Actor),
        # so the old action-bounds penalty was identically zero — dropped.
        total = (policy_loss + tc.vf_coef * value_loss
                 - self._cur_ent_coef * entropy)
        total = torch.clamp(total, -1e6, 1e6)
        if not torch.isfinite(total):
            total = torch.tensor(0.0, device=total.device, requires_grad=True)

        self.optimizer.zero_grad()
        total.backward()

        params = [p for g in self.optimizer.param_groups for p in g["params"]]
        for p in params:
            if p.grad is not None:
                p.grad = torch.where(
                    torch.isfinite(p.grad), p.grad, torch.zeros_like(p.grad)
                )
        grad_norm = nn.utils.clip_grad_norm_(params, tc.max_grad)
        grad_norm_val = (grad_norm.item()
                         if isinstance(grad_norm, torch.Tensor) else grad_norm)
        # Skip catastrophic batches outright (pre-clip norm > 10x target)
        skipped_step = grad_norm_val > 10.0 * tc.max_grad
        if skipped_step:
            self.optimizer.zero_grad()
        else:
            self.optimizer.step()

        with torch.no_grad():
            approx_kl = torch.mean((ratio - 1.0) - log_ratio).item()
            clip_frac = torch.mean(
                ((ratio < 1.0 - tc.clip_eps) | (ratio > 1.0 + tc.clip_eps))
                .float()
            ).item()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "ent_coef": self._cur_ent_coef,
            "total_loss": total.item(),
            "ratio_mean": ratio.mean().item(),
            "ratio_max": ratio.max().item(),
            "approx_kl": approx_kl,
            "clip_frac": clip_frac,
            "action_mean_abs": mean.abs().mean().item(),
            "action_std_mean": torch.exp(log_std).mean().item(),
            "value_pred_mean": value.mean().item(),
            "value_pred_std": value.std().item(),
            "value_pred_min": value.min().item(),
            "value_pred_max": value.max().item(),
            "adv_mb_mean": mb.advantage.mean().item(),
            "adv_mb_std": mb.advantage.std().item(),
            "grad_norm": grad_norm_val,
            "skipped_step": int(skipped_step),
        }

    # ──────────────────────────────────────────────────────────────────
    def anneal_lr(self, update_idx: int) -> float:
        """Linear LR + entropy-coef decay over the run; 1-indexed; PPO only."""
        frac = max(0.0, 1.0 - (update_idx - 1) / float(self._total_updates))
        new_lr = self._base_lr * frac
        for pg in self.optimizer.param_groups:
            pg["lr"] = new_lr
        self._cur_lr = new_lr
        ef = self.cfg.teacher.ent_coef_final
        self._cur_ent_coef = ef + (self._base_ent_coef - ef) * frac
        return new_lr

    # ──────────────────────────────────────────────────────────────────
    def update(self, batch: RolloutBatch) -> dict:
        """n_epochs of PPO with running-mean KL early stopping:
        mid-epoch break at 1.5x target_kl, end-of-epoch break at target_kl."""
        tc = self.cfg.teacher
        total_samples = batch.proprio.shape[0]
        mb_size = max(1, total_samples // tc.num_minibatches)

        kl_sum = 0.0
        kl_n = 0
        skipped_total = 0
        last_info: dict = {}
        early_stop_epoch: int | str = ""
        epoch = 0
        stop = False

        for epoch in range(tc.n_epochs):
            perm = torch.randperm(total_samples, device=self.device)
            for start in range(0, total_samples, mb_size):
                idx = perm[start: start + mb_size]
                info = self.ppo_update_step(batch[idx])
                last_info = info
                kl_sum += float(info["approx_kl"])
                kl_n += 1
                skipped_total += int(info.get("skipped_step", 0))
                if kl_sum / max(1, kl_n) > 1.5 * tc.target_kl:
                    early_stop_epoch = epoch + 1
                    stop = True
                    break
            if stop:
                break
            if kl_sum / max(1, kl_n) > tc.target_kl:
                early_stop_epoch = epoch + 1
                break

        last_info["epochs_run"] = epoch + 1
        last_info["early_stop_epoch"] = early_stop_epoch
        last_info["running_kl"] = kl_sum / max(1, kl_n)
        last_info["skipped_steps"] = skipped_total
        last_info["lr"] = self._cur_lr
        return last_info

    # ──────────────────────────────────────────────────────────────────
    def save(self, path: str):
        from .checkpoint import save_checkpoint
        save_checkpoint(
            path,
            model_state=self.net.state_dict(),
            phase="teacher",
            robot=self.cfg.robot.name,
            optimizer_state=self.optimizer.state_dict(),
            adapt_optimizer_state=self.adapt_optimizer.state_dict(),
            ret_mean=self._ret_mean,
            ret_std=self._ret_std,
        )

    def load(self, path: str):
        from .checkpoint import load_checkpoint
        ckpt = load_checkpoint(path, self.device)
        self.net.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "adapt_optimizer_state_dict" in ckpt:
            self.adapt_optimizer.load_state_dict(
                ckpt["adapt_optimizer_state_dict"]
            )
        if "ret_mean" in ckpt:
            self._ret_mean = float(ckpt["ret_mean"])
        if "ret_std" in ckpt:
            self._ret_std = float(ckpt["ret_std"])
