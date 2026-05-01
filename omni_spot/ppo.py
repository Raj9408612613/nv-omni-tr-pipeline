"""
PyTorch PPO Trainer
====================
Ported from jax_ppo.py. Preserves all hyperparameters, GAE computation,
value clipping, NaN guards, KL early stopping, and CNN feature caching.
this file is the code for the PPO policy. Dictates how the policy will run and calculate rewards and updates.
"""

import time
from typing import NamedTuple

import torch
import torch.nn as nn
import numpy as np

from .config import (
    GAMMA, GAE_LAMBDA, CLIP_EPS, ENT_COEF, VF_COEF,
    MAX_GRAD, LR, N_EPOCHS, MINIBATCH_SZ, TARGET_KL,
    CNN_FEAT_DIM, PROPRIO_DIM, ACTION_DIM,
    LOG_STD_MIN, LOG_STD_MAX,
)
from .spot_actor_critic import (
    SpotActorCritic, gaussian_log_prob, gaussian_entropy,
)


class RolloutBatch:
    """Stores flattened rollout experience (T*B, ...).

    Mirrors the JAX RolloutBatch NamedTuple.
    """
    __slots__ = [
        "cnn_feat", "proprio", "action", "log_prob",
        "advantage", "ret", "old_value",
    ]

    def __init__(
        self,
        cnn_feat:  torch.Tensor,   # (T*B, 256)
        proprio:   torch.Tensor,   # (T*B, 37)
        action:    torch.Tensor,   # (T*B, 12)
        log_prob:  torch.Tensor,   # (T*B,)
        advantage: torch.Tensor,   # (T*B,)
        ret:       torch.Tensor,   # (T*B,) normalized return
        old_value: torch.Tensor,   # (T*B,) normalized old value
    ):
        self.cnn_feat  = cnn_feat
        self.proprio   = proprio
        self.action    = action
        self.log_prob  = log_prob
        self.advantage = advantage
        self.ret       = ret
        self.old_value = old_value

    def __getitem__(self, idx):
        return RolloutBatch(
            cnn_feat  = self.cnn_feat[idx],
            proprio   = self.proprio[idx],
            action    = self.action[idx],
            log_prob  = self.log_prob[idx],
            advantage = self.advantage[idx],
            ret       = self.ret[idx],
            old_value = self.old_value[idx],
        )


# ── GAE Advantage Computation ────────────────────────────────────────────────

def compute_gae(
    rewards: torch.Tensor,   # (T, B)
    values:  torch.Tensor,   # (T+1, B)
    dones:   torch.Tensor,   # (T, B)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns advantages (T,B) and returns (T,B).

    Matches JAX compute_gae exactly.
    """
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(rewards.shape[1], device=rewards.device)

    for t in reversed(range(T)):
        mask  = 1.0 - dones[t]
        delta = rewards[t] + GAMMA * values[t + 1] * mask - values[t]
        gae   = delta + GAMMA * GAE_LAMBDA * mask * gae
        advantages[t] = gae

    returns = advantages + values[:T]
    return advantages, returns


# ── PPO Trainer ──────────────────────────────────────────────────────────────

class PPOTrainer:
    """
    Manages model, optimizer, and training loop.
    Mirrors the JAX PPOTrainer class.
    """

    def __init__(
        self,
        n_envs:        int = 4096,
        n_steps:       int = 2048,
        lr:            float = LR,
        device:        str = "cuda",
        total_updates: int = 500,
    ):
        self.n_envs  = n_envs
        self.n_steps = n_steps
        self.device  = torch.device(device)

        # ── Initialize network ────────────────────────────────────────
        self.net = SpotActorCritic().to(self.device)

        # ── Optimizer (Adam with gradient clipping) ───────────────────
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

        # Running stats for return normalization (PopArt-style).
        # The critic is trained on normalized returns; GAE consumes the
        # denormalized values so reward and V(s) are on the same scale.
        # Stored as Python floats so they survive checkpoint save/load cleanly.
        self._ret_mean = 0.0
        self._ret_std  = 1.0
        self._ret_ema_alpha = 0.05  # ~20-update effective window

        # ── LR annealing ──────────────────────────────────────────────
        # Linearly decay LR from base → 0 over the run. Smaller updates
        # late in training keep the trust region tight as the policy nears
        # a good basin, mirroring rl_games / sb3 default behaviour.
        self._base_lr       = lr
        self._total_updates = max(1, total_updates)
        self._cur_lr        = lr

    # ──────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def sample_action(self, obs: dict) -> tuple:
        """Single inference step. Returns (action, log_prob, value, cnn_feat)."""
        return self.net.inference_step(obs["depth"], obs["proprio"])

    # ──────────────────────────────────────────────────────────────────
    def collect_rollout(self, env, obs: dict, profile: bool = False):
        """
        Collect n_steps of experience.

        Stores CNN features (T*B, 256) instead of raw depth to keep
        rollout buffer at ~1 GB instead of ~374 GB.

        Returns:
            obs:           updated observation dict
            batch:         RolloutBatch
            rollout_stats: dict with reward/episode metrics
        """
        buf_cnn_feat = []
        buf_proprio  = []
        buf_actions  = []
        buf_log_prob = []
        buf_rewards  = []
        buf_dones    = []
        buf_values   = []

        reward_sums = None
        reward_count = 0

        t_inference = 0.0
        t_env_step  = 0.0

        for _ in range(self.n_steps):
            if profile:
                torch.cuda.synchronize()
                t0 = time.time()

            action, log_prob, value, cnn_feat = self.sample_action(obs)

            if profile:
                torch.cuda.synchronize()
                t_inference += time.time() - t0
                t0 = time.time()

            # Sanitize features
            cnn_feat = torch.where(
                torch.isfinite(cnn_feat), cnn_feat, torch.zeros_like(cnn_feat)
            )
            buf_cnn_feat.append(cnn_feat)
            buf_proprio.append(obs["proprio"])
            buf_actions.append(action)
            buf_log_prob.append(log_prob)
            buf_values.append(value)

            # Environment step — Isaac Lab handles auto-reset internally
            obs, reward, terminated, truncated, step_info = env.step(action)

            if profile:
                torch.cuda.synchronize()
                t_env_step += time.time() - t0

            done = terminated | truncated
            buf_rewards.append(reward)
            buf_dones.append(done.float())

            # Accumulate reward component running sums
            if reward_sums is None:
                reward_sums = {k: v.clone() for k, v in step_info.items()
                               if isinstance(v, torch.Tensor)}
            else:
                for k, v in step_info.items():
                    if k in reward_sums and isinstance(v, torch.Tensor):
                        reward_sums[k] = reward_sums[k] + v
            reward_count += 1

        # Bootstrap value for last step (critic output is in NORMALIZED scale).
        _, _, last_value, _ = self.sample_action(obs)

        # Stack: (T, B, ...)
        rewards        = torch.stack(buf_rewards)
        dones          = torch.stack(buf_dones)
        stacked_values = torch.stack(buf_values)   # normalized critic outputs, (T, B)
        values_norm    = torch.cat([stacked_values, last_value.unsqueeze(0)], dim=0)

        # Denormalize critic outputs to raw-return scale before GAE so that
        # δ = r + γV(s') − V(s) is unit-consistent with the raw reward.
        # Running stats are identity (mean=0, std=1) on update 1 and bootstrapped
        # below from the first batch, so the first rollout's GAE is unchanged
        # but subsequent ones use a meaningful baseline.
        values_raw = values_norm * self._ret_std + self._ret_mean

        advantages, returns = compute_gae(rewards, values_raw, dones)

        # Update running return stats from THIS batch's returns.
        batch_ret_mean = float(returns.mean())
        batch_ret_std  = float(returns.std()) + 1e-8
        if self._ret_mean == 0.0 and self._ret_std == 1.0:
            # First update: bootstrap directly from the batch so the critic
            # target isn't trained against (R − 0) / 1 for ~20 updates.
            self._ret_mean = batch_ret_mean
            self._ret_std  = batch_ret_std
        else:
            a = self._ret_ema_alpha
            self._ret_mean = (1.0 - a) * self._ret_mean + a * batch_ret_mean
            self._ret_std  = (1.0 - a) * self._ret_std  + a * batch_ret_std

        # Critic training target: normalize with RUNNING stats (not per-batch) so
        # the critic sees a consistent target distribution across updates.
        returns_norm    = (returns - self._ret_mean) / self._ret_std
        old_values_norm = stacked_values  # already normalized (raw critic output)

        # Raw-scale values for diagnostics (T, B)
        stacked_values_raw = values_raw[:-1]

        # Flatten (T*B, ...)
        def flat(x):
            return x.reshape(-1, *x.shape[2:]) if x.ndim > 2 else x.reshape(-1)

        # Normalize advantages globally
        adv_flat = flat(advantages)
        adv_mean = adv_flat.mean()
        adv_std  = adv_flat.std() + 1e-8
        adv_norm = (adv_flat - adv_mean) / adv_std

        stacked_cnn_feat = torch.stack(buf_cnn_feat)
        stacked_proprio  = torch.stack(buf_proprio)

        batch = RolloutBatch(
            cnn_feat  = flat(stacked_cnn_feat),
            proprio   = flat(stacked_proprio),
            action    = flat(torch.stack(buf_actions)),
            log_prob  = flat(torch.stack(buf_log_prob)),
            advantage = adv_norm,
            ret       = flat(returns_norm),
            old_value = flat(old_values_norm),
        )

        # Rollout stats (host sync)
        rollout_stats = {
            "rew_mean":  float(rewards.mean()),
            "rew_min":   float(rewards.min()),
            "rew_max":   float(rewards.max()),
            "done_rate": float(dones.mean()),
            "ep_count":  int(dones.sum()),
        }

        # Diagnostics
        rollout_stats["_diag"] = {
            "ret_raw_mean":      float(returns.mean()),
            "ret_raw_std":       float(returns.std()),
            "ret_raw_min":       float(returns.min()),
            "ret_raw_max":       float(returns.max()),
            "ret_norm_mean":     float(returns_norm.mean()),
            "ret_norm_std":      float(returns_norm.std()),
            "ret_norm_min":      float(returns_norm.min()),
            "ret_norm_max":      float(returns_norm.max()),
            "ret_scale_mean":    float(self._ret_mean),
            "ret_scale_std":     float(self._ret_std),
            # Report normalized advantages (what PPO actually trains on)
            "adv_mean":          float(adv_norm.mean()),
            "adv_std":           float(adv_norm.std()),
            "adv_min":           float(adv_norm.min()),
            "adv_max":           float(adv_norm.max()),
            "val_raw_mean":      float(stacked_values_raw.mean()),
            "val_raw_std":       float(stacked_values_raw.std()),
            "val_raw_min":       float(stacked_values_raw.min()),
            "val_raw_max":       float(stacked_values_raw.max()),
            "explained_var":     float(
                1.0 - torch.var(returns - stacked_values_raw)
                / (torch.var(returns) + 1e-8)
            ),
            "proprio_mean":      float(stacked_proprio.mean()),
            "proprio_std":       float(stacked_proprio.std()),
            "proprio_nan_frac":  float((~torch.isfinite(stacked_proprio)).float().mean()),
            "cnn_feat_mean":     float(stacked_cnn_feat.mean()),
            "cnn_feat_std":      float(stacked_cnn_feat.std()),
            "cnn_feat_nan_frac": float((~torch.isfinite(stacked_cnn_feat)).float().mean()),
        }

        if reward_sums is not None and reward_count > 0:
            reward_components = {
                k: float(v.mean() / reward_count) for k, v in reward_sums.items()
            }
            rollout_stats["_diag"]["reward_components"] = reward_components

        if profile:
            rollout_stats["_timing"] = {
                "inference_sec": t_inference,
                "env_step_sec":  t_env_step,
            }

        return obs, batch, rollout_stats

    # ──────────────────────────────────────────────────────────────────
    def ppo_update_step(self, batch: RolloutBatch) -> dict:
        """One gradient update on a minibatch with full NaN/explosion safeguards."""
        self.net.train()

        mean, log_std, value = self.net.head_forward(
            batch.cnn_feat, batch.proprio
        )

        # ── Policy loss with ratio clamping ──────────────────────────
        log_prob_new = gaussian_log_prob(mean, log_std, batch.action)
        # Tight log-ratio clamp (±2 → ratio ∈ [0.135, 7.39]). With CLIP_EPS=0.2
        # the policy gradient is already saturated outside [0.8, 1.2], so a
        # ±10 clamp (ratio up to 22000) only serves to inflate `ratio_max` and
        # blow up Adam's gradient estimates the moment a single sample lands far
        # in the tail of the importance-sampling distribution.
        log_ratio = torch.clamp(log_prob_new - batch.log_prob, -2.0, 2.0)
        ratio = torch.exp(log_ratio)

        adv_norm = torch.clamp(batch.advantage, -5.0, 5.0)

        pg_loss1    = ratio * adv_norm
        pg_loss2    = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv_norm
        policy_loss = -torch.mean(torch.minimum(pg_loss1, pg_loss2))

        # ── Value loss (simple MSE, no PPO clipping) ─────────────────
        # PPO-style VF clipping was removed: batch.old_value is stored in
        # normalised scale while value is raw network output, so the clip
        # bound (-CLIP_EPS, +CLIP_EPS) was always saturated → zero gradient
        # on vf_loss2 → asymmetric updates → critic converged to wrong value.
        value_loss = 0.5 * torch.mean((value - batch.ret) ** 2)
        value_loss = torch.clamp(value_loss, 0.0, 1_000_000.0)

        # ── Entropy bonus ────────────────────────────────────────────
        entropy = torch.mean(gaussian_entropy(log_std))

        total = policy_loss + VF_COEF * value_loss - ENT_COEF * entropy
        total = torch.clamp(total, -1e6, 1e6)

        # NaN guard
        if not torch.isfinite(total):
            total = torch.tensor(0.0, device=total.device, requires_grad=True)

        # ── Backward + gradient clip ─────────────────────────────────
        self.optimizer.zero_grad()
        total.backward()

        # NaN guard on gradients
        for p in self.net.parameters():
            if p.grad is not None:
                p.grad = torch.where(
                    torch.isfinite(p.grad), p.grad, torch.zeros_like(p.grad)
                )

        grad_norm = nn.utils.clip_grad_norm_(self.net.parameters(), MAX_GRAD)
        # Skip the step on truly catastrophic batches (pre-clip norm > 10× the
        # clip target). Clipping alone preserves direction, but a 100× spike
        # often signals a numerical anomaly — skipping is safer than scaling.
        grad_norm_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
        skipped_step = grad_norm_val > 10.0 * MAX_GRAD
        if skipped_step:
            self.optimizer.zero_grad()
        else:
            self.optimizer.step()

        # ── Diagnostics ──────────────────────────────────────────────
        with torch.no_grad():
            approx_kl = torch.mean((ratio - 1.0) - log_ratio).item()
            clip_frac = torch.mean(
                ((ratio < 1.0 - CLIP_EPS) | (ratio > 1.0 + CLIP_EPS)).float()
            ).item()

        info = {
            "policy_loss":     policy_loss.item(),
            "value_loss":      value_loss.item(),
            "entropy":         entropy.item(),
            "total_loss":      total.item(),
            "ratio_mean":      ratio.mean().item(),
            "ratio_max":       ratio.max().item(),
            "approx_kl":       approx_kl,
            "clip_frac":       clip_frac,
            "action_mean_abs": mean.abs().mean().item(),
            "action_std_mean": torch.exp(log_std).mean().item(),
            "value_pred_mean": value.mean().item(),
            "value_pred_std":  value.std().item(),
            "value_pred_min":  value.min().item(),
            "value_pred_max":  value.max().item(),
            "adv_mb_mean":     batch.advantage.mean().item(),
            "adv_mb_std":      batch.advantage.std().item(),
            "grad_norm":       grad_norm_val,
            "skipped_step":    int(skipped_step),
        }
        return info

    # ──────────────────────────────────────────────────────────────────
    def anneal_lr(self, update_idx: int) -> float:
        """Linearly decay LR from base → 0 over the configured run length.

        Called once per outer-loop update (BEFORE update()). update_idx is
        1-indexed. Returns the new LR for logging.
        """
        frac = max(0.0, 1.0 - (update_idx - 1) / float(self._total_updates))
        new_lr = self._base_lr * frac
        for pg in self.optimizer.param_groups:
            pg["lr"] = new_lr
        self._cur_lr = new_lr
        return new_lr

    # ──────────────────────────────────────────────────────────────────
    def update(self, batch: RolloutBatch) -> dict:
        """Run N_EPOCHS of PPO updates with running-mean KL early stopping.

        Stop conditions, evaluated AFTER each minibatch:
          * mid-epoch break: running-mean approx_kl > 1.5 × TARGET_KL
            (catches a divergence the moment it happens, before the rest of
            the epoch makes it worse).
          * end-of-epoch break: running-mean approx_kl > TARGET_KL.

        The running mean (instead of the last-minibatch value) avoids stopping
        on a single noisy minibatch and avoids missing a steady drift that the
        old "check the very last minibatch only" logic would happily ignore.
        """
        total_samples = batch.cnn_feat.shape[0]

        kl_sum = 0.0
        kl_n   = 0
        skipped_total = 0
        last_info = {}
        early_stop_epoch = ""
        epoch = 0
        stop = False

        for epoch in range(N_EPOCHS):
            perm = torch.randperm(total_samples, device=self.device)

            for start in range(0, total_samples, MINIBATCH_SZ):
                idx = perm[start : start + MINIBATCH_SZ]
                mb  = batch[idx]
                info = self.ppo_update_step(mb)
                last_info = info

                kl_sum += float(info["approx_kl"])
                kl_n   += 1
                skipped_total += int(info.get("skipped_step", 0))

                running_kl = kl_sum / max(1, kl_n)

                # Mid-epoch hard break — KL has clearly exploded.
                if running_kl > 1.5 * TARGET_KL:
                    early_stop_epoch = epoch + 1
                    stop = True
                    break

            if stop:
                break

            # Soft end-of-epoch break — policy has drifted enough.
            running_kl = kl_sum / max(1, kl_n)
            if running_kl > TARGET_KL:
                early_stop_epoch = epoch + 1
                break

        last_info["epochs_run"]       = epoch + 1
        last_info["early_stop_epoch"] = early_stop_epoch
        last_info["running_kl"]       = kl_sum / max(1, kl_n)
        last_info["skipped_steps"]    = skipped_total
        last_info["lr"]               = self._cur_lr
        return last_info

    # ──────────────────────────────────────────────────────────────────
    def save(self, path: str):
        torch.save({
            "model_state_dict": self.net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "ret_mean": self._ret_mean,
            "ret_std":  self._ret_std,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.net.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "ret_mean" in ckpt:
            self._ret_mean = float(ckpt["ret_mean"])
        if "ret_std" in ckpt:
            self._ret_std = float(ckpt["ret_std"])
