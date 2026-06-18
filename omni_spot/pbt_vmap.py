"""
Vmapped batched PPO update for PBT (opt-in)
===========================================
The default PBT update (pbt.Population.update_members) is a Python loop over
members — the correctness reference. At high N the N sequential forward/backward
passes over small MLP minibatches underuse the GPU (kernel-launch bound). This
module batches the N members' forward/backward with
`torch.func.functional_call` + `vmap` over stacked per-member parameters, then
does the per-member gradient clip + optimizer step (cheap elementwise work) in a
short loop.

What this path reproduces vs. the loop:
  - SAME loss math as ppo.ppo_update_step (policy clip, value MSE, entropy),
  - SAME NaN-grad guard, per-member grad-norm clip, catastrophic-grad skip,
  - SAME torch.optim.Adam step (the real per-member optimizer is reused, so
    exploit/explore, save, and resume keep working unchanged).
What it intentionally drops (per-member, data-dependent control flow that does
not vectorize):
  - KL early stopping — every member runs all n_epochs,
  - independent per-member minibatch shuffles — one shared permutation per
    epoch is applied to each member's own data.

`reference_update_members` is the same algorithm written as an explicit
per-member loop using ordinary autograd; the unit test asserts the two produce
identical parameter updates, which is what validates the vmap machinery.

The teacher has no recurrence on the PPO path (ScandotEncoder is an MLP,
PrivEncoder an MLP+LayerNorm, Critic an MLP; the adaptation module's Conv1d is
NOT on the evaluate() path), so vmap is clean.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from torch.func import functional_call, grad, stack_module_state, vmap

from .networks import gaussian_entropy, gaussian_log_prob
from .obs import build_critic_obs

# Rollout fields consumed by the PPO loss (order is fixed for vmap in_dims).
_FIELDS = (
    "proprio", "scandots", "priv", "critic_extras",
    "action", "log_prob", "advantage", "ret",
)


class _EvalWrap(nn.Module):
    """Wrap a TeacherPolicy so functional_call's forward == net.evaluate."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, proprio, scandots, priv, critic_obs):
        return self.net.evaluate(proprio, scandots, priv, critic_obs)


def _ppo_loss(mean, log_std, value, action, old_log_prob, advantage, ret,
              clip_eps, ent_coef, vf_coef):
    """PPO surrogate loss for ONE member, matching ppo.ppo_update_step.
    Returns (total, aux) where aux holds detached scalars for logging.
    clip_eps / ent_coef may be Python floats (loop) or 0-d tensors (vmap)."""
    log_prob_new = gaussian_log_prob(mean, log_std, action)
    log_ratio = torch.clamp(log_prob_new - old_log_prob, -2.0, 2.0)
    ratio = torch.exp(log_ratio)

    adv = torch.clamp(advantage, -5.0, 5.0)
    pg_loss1 = ratio * adv
    pg_loss2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
    policy_loss = -torch.mean(torch.minimum(pg_loss1, pg_loss2))

    value_loss = 0.5 * torch.mean((value - ret) ** 2)
    value_loss = torch.clamp(value_loss, 0.0, 1_000_000.0)
    entropy = torch.mean(gaussian_entropy(log_std))

    total = policy_loss + vf_coef * value_loss - ent_coef * entropy
    total = torch.clamp(total, -1e6, 1e6)
    total = torch.where(torch.isfinite(total), total, torch.zeros_like(total))

    approx_kl = torch.mean((ratio - 1.0) - log_ratio)
    clip_frac = torch.mean(
        ((ratio < 1.0 - clip_eps) | (ratio > 1.0 + clip_eps)).float()
    )
    aux = {
        "policy_loss": policy_loss.detach(),
        "value_loss": value_loss.detach(),
        "entropy": entropy.detach(),
        "total_loss": total.detach(),
        "approx_kl": approx_kl.detach(),
        "clip_frac": clip_frac.detach(),
    }
    return total, aux


def _ppo_param_names(member) -> list[str]:
    """Stacked-state keys (with the _EvalWrap "net." prefix) for the params the
    PPO optimizer owns — actor + priv_encoder + critic, i.e. everything except
    the adaptation module (which has its own optimizer and is off the evaluate
    path)."""
    params, _ = stack_module_state([_EvalWrap(member.trainer.net)])
    return [k for k in params if "adaptation_module" not in k]


def _guard_clip_step(member, params_list, max_grad) -> tuple[float, int]:
    """NaN-guard grads, clip to max_grad, skip catastrophic batches; else step.
    Mirrors ppo.ppo_update_step's tail exactly."""
    for p in params_list:
        if p.grad is not None:
            p.grad = torch.where(
                torch.isfinite(p.grad), p.grad, torch.zeros_like(p.grad)
            )
    grad_norm = nn.utils.clip_grad_norm_(params_list, max_grad)
    gnv = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
    skipped = gnv > 10.0 * max_grad
    if skipped:
        member.trainer.optimizer.zero_grad()
    else:
        member.trainer.optimizer.step()
    return gnv, int(skipped)


def _mb_size(total_samples: int, num_minibatches: int) -> int:
    return max(1, total_samples // num_minibatches)


def _make_perms(total_samples, n_epochs, device):
    return [torch.randperm(total_samples, device=device) for _ in range(n_epochs)]


def _empty_info(member, n_epochs):
    return {
        "policy_loss": float("nan"), "value_loss": float("nan"),
        "entropy": float("nan"), "total_loss": float("nan"),
        "approx_kl": float("nan"), "clip_frac": float("nan"),
        "grad_norm": float("nan"), "running_kl": float("nan"),
        "epochs_run": n_epochs, "skipped_steps": 0,
        "lr": float(member.knobs["lr"]), "ent_coef": float(member.knobs["ent_coef"]),
    }


# ════════════════════════════════════════════════════════════════════════════
# Vmapped update
# ════════════════════════════════════════════════════════════════════════════

def vmap_update_members(pop, batches, perms=None) -> list[dict]:
    """Batched per-member PPO update. Mutates each member's net + optimizer in
    place (the real torch.optim.Adam), so it is a drop-in for the loop path."""
    tc = pop.cfg.teacher
    members = pop.members
    N = len(members)
    device = pop.device

    for m in members:                      # lr onto each optimizer; knobs ready
        m.apply_knobs_to_trainer()
    clip_eps = torch.tensor([float(m.knobs["clip_eps"]) for m in members],
                            device=device)
    ent_coef = torch.tensor([float(m.knobs["ent_coef"]) for m in members],
                            device=device)

    total_samples = batches[0].proprio.shape[0]
    mb_size = _mb_size(total_samples, tc.num_minibatches)
    if perms is None:
        perms = _make_perms(total_samples, tc.n_epochs, device)

    # Stacked per-member rollout data (constant during the update).
    stacked = {f: torch.stack([getattr(b, f) for b in batches], dim=0)
               for f in _FIELDS}

    base = copy.deepcopy(_EvalWrap(members[0].trainer.net)).to("meta")
    ppo_names = _ppo_param_names(members[0])
    wraps = [_EvalWrap(m.trainer.net) for m in members]   # live refs to nets
    member_param_maps = [dict(m.trainer.net.named_parameters()) for m in members]

    vf_coef = tc.vf_coef

    def compute_loss(ppo_params, proprio, scandots, priv, critic_extras,
                     action, old_log_prob, advantage, ret, c_eps, e_coef):
        critic_obs = build_critic_obs(proprio, scandots, priv, critic_extras)
        mean, log_std, value = functional_call(
            base, ppo_params, (proprio, scandots, priv, critic_obs)
        )
        return _ppo_loss(mean, log_std, value, action, old_log_prob, advantage,
                         ret, c_eps, e_coef, vf_coef)

    grad_fn = grad(compute_loss, argnums=0, has_aux=True)
    vmapped = vmap(grad_fn, in_dims=(0,) + (0,) * 8 + (0, 0))

    kl_running = [0.0] * N
    skipped = [0] * N
    last_gn = [float("nan")] * N
    last_aux = None
    kl_n = 0

    for ep in range(tc.n_epochs):
        perm = perms[ep]
        for start in range(0, total_samples, mb_size):
            idx = perm[start:start + mb_size]
            # Re-stack CURRENT params (they changed after the previous step).
            params, _buffers = stack_module_state(wraps)
            ppo_params = {k: params[k] for k in ppo_names}
            mb = tuple(stacked[f][:, idx] for f in _FIELDS)
            grads, aux = vmapped(ppo_params, *mb, clip_eps, ent_coef)
            last_aux = aux
            kl_n += 1
            for i, m in enumerate(members):
                pmap = member_param_maps[i]
                plist = []
                for name in ppo_names:
                    p = pmap[name[len("net."):]]
                    p.grad = grads[name][i]
                    plist.append(p)
                gnv, sk = _guard_clip_step(m, plist, tc.max_grad)
                last_gn[i] = gnv
                skipped[i] += sk
                kl_running[i] += float(aux["approx_kl"][i])

    infos = []
    for i, m in enumerate(members):
        info = _empty_info(m, tc.n_epochs)
        if last_aux is not None:
            info.update({
                "policy_loss": float(last_aux["policy_loss"][i]),
                "value_loss": float(last_aux["value_loss"][i]),
                "entropy": float(last_aux["entropy"][i]),
                "total_loss": float(last_aux["total_loss"][i]),
                "approx_kl": float(last_aux["approx_kl"][i]),
                "clip_frac": float(last_aux["clip_frac"][i]),
                "grad_norm": last_gn[i],
                "running_kl": kl_running[i] / max(1, kl_n),
                "skipped_steps": skipped[i],
            })
        infos.append(info)
    return infos


# ════════════════════════════════════════════════════════════════════════════
# Explicit-loop reference (test oracle; same algorithm, ordinary autograd)
# ════════════════════════════════════════════════════════════════════════════

def reference_update_members(pop, batches, perms=None) -> list[dict]:
    """Same update as vmap_update_members but as a plain per-member loop using
    autograd. Used to validate the vmap path produces identical updates."""
    tc = pop.cfg.teacher
    members = pop.members
    device = pop.device

    for m in members:
        m.apply_knobs_to_trainer()

    total_samples = batches[0].proprio.shape[0]
    mb_size = _mb_size(total_samples, tc.num_minibatches)
    if perms is None:
        perms = _make_perms(total_samples, tc.n_epochs, device)

    N = len(members)
    kl_running = [0.0] * N
    skipped = [0] * N
    last_gn = [float("nan")] * N
    last_aux = [None] * N
    kl_n = 0

    for ep in range(tc.n_epochs):
        perm = perms[ep]
        for start in range(0, total_samples, mb_size):
            idx = perm[start:start + mb_size]
            kl_n += 1
            for i, m in enumerate(members):
                b = batches[i]
                critic_obs = build_critic_obs(
                    b.proprio[idx], b.scandots[idx], b.priv[idx],
                    b.critic_extras[idx]
                )
                mean, log_std, value = m.trainer.net.evaluate(
                    b.proprio[idx], b.scandots[idx], b.priv[idx], critic_obs
                )
                total, aux = _ppo_loss(
                    mean, log_std, value, b.action[idx], b.log_prob[idx],
                    b.advantage[idx], b.ret[idx],
                    float(m.knobs["clip_eps"]), float(m.knobs["ent_coef"]),
                    tc.vf_coef,
                )
                m.trainer.optimizer.zero_grad()
                total.backward()
                plist = [p for g in m.trainer.optimizer.param_groups
                         for p in g["params"]]
                gnv, sk = _guard_clip_step(m, plist, tc.max_grad)
                last_gn[i] = gnv
                skipped[i] += sk
                last_aux[i] = aux
                kl_running[i] += float(aux["approx_kl"])

    infos = []
    for i, m in enumerate(members):
        info = _empty_info(m, tc.n_epochs)
        if last_aux[i] is not None:
            info.update({
                "policy_loss": float(last_aux[i]["policy_loss"]),
                "value_loss": float(last_aux[i]["value_loss"]),
                "entropy": float(last_aux[i]["entropy"]),
                "total_loss": float(last_aux[i]["total_loss"]),
                "approx_kl": float(last_aux[i]["approx_kl"]),
                "clip_frac": float(last_aux[i]["clip_frac"]),
                "grad_norm": last_gn[i],
                "running_kl": kl_running[i] / max(1, kl_n),
                "skipped_steps": skipped[i],
            })
        infos.append(info)
    return infos
