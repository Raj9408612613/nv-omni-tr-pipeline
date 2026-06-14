"""
Teacher / Student Networks
===========================
Module attribute names are a CONTRACT: Phase 2 loads Phase 1 actor weights
by state_dict prefix (see checkpoint.py). TeacherPolicy and StudentPolicy
share the same `Actor` class with the exteroception encoder injected under
the same attribute name, so all shared keys match byte-for-byte:

    actor.proprio_mlp.*   copied teacher -> student
    actor.trunk.*         copied
    actor.head.*          copied
    actor.log_std         copied
    adaptation_module.*   copied, then frozen in the student
    actor.extero_encoder.*  NOT copied (ScandotEncoder vs DepthGRUEncoder)
    priv_encoder.* / critic.*  teacher-only

All dimensions derive from ExperimentCfg — nothing robot-specific here.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .configs.base import ExperimentCfg


def _mlp(in_dim: int, hidden: tuple[int, ...], out_dim: int,
         final_act: bool = False) -> nn.Sequential:
    """ELU MLP. final_act=True puts an ELU after the output layer too."""
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ELU()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    if final_act:
        layers.append(nn.ELU())
    return nn.Sequential(*layers)


# ════════════════════════════════════════════════════════════════════════════
# Encoders
# ════════════════════════════════════════════════════════════════════════════

class ScandotEncoder(nn.Module):
    """Heightfield scandots -> exteroception latent e_t. (Teacher)"""

    def __init__(self, cfg: ExperimentCfg):
        super().__init__()
        pc = cfg.policy
        self.net = _mlp(cfg.scandots.n_points, pc.scandot_hidden,
                        pc.extero_latent_dim, final_act=True)

    def forward(self, scandots: torch.Tensor) -> torch.Tensor:
        return self.net(scandots)


class PrivEncoder(nn.Module):
    """Privileged obs -> latent z_t. Teacher-only; never deployed.

    The output is LayerNorm'd so z stays unit-scale as the encoder trains.
    Without it, z drifts upward in magnitude and the adaptation module phi
    (which regresses onto z.detach()) chases a moving target -> adapt_loss
    diverges and the Phase 2 student inherits a useless extrinsics estimator.
    """

    def __init__(self, cfg: ExperimentCfg):
        super().__init__()
        pc = cfg.policy
        self.net = _mlp(cfg.priv_dim, pc.priv_hidden, pc.z_dim)
        self.norm = nn.LayerNorm(pc.z_dim)

    def forward(self, priv: torch.Tensor) -> torch.Tensor:
        return self.norm(self.net(priv))


class AdaptationModule(nn.Module):
    """phi (ROA): 1D conv over the (proprio, action) history -> z_hat.

    Trained alongside PPO with ||z_hat - z.detach()||^2; deployed in Phase 2
    and on hardware in place of the privileged encoder.
    """

    def __init__(self, cfg: ExperimentCfg):
        super().__init__()
        pc = cfg.policy
        d = pc.adapt_embed_dim
        self.embed = nn.Linear(cfg.history_feat_dim, d)
        self.conv = nn.Sequential(
            nn.Conv1d(d, d, kernel_size=8, stride=4), nn.ELU(),
            nn.Conv1d(d, d, kernel_size=5, stride=1), nn.ELU(),
            nn.Conv1d(d, d, kernel_size=5, stride=1), nn.ELU(),
        )
        # Conv output length depends on history_len; measure it once.
        with torch.no_grad():
            try:
                t = self.conv(torch.zeros(1, d, pc.history_len))
            except RuntimeError as e:
                raise ValueError(
                    f"history_len={pc.history_len} too short for the phi conv "
                    f"stack (needs >= 16 steps): {e}"
                ) from e
        self.fc = nn.Linear(d * t.shape[-1], pc.z_dim)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """history: (B, H, F) oldest -> newest. Returns (B, z_dim)."""
        x = F.elu(self.embed(history))      # (B, H, D)
        x = x.transpose(1, 2)               # (B, D, H)
        x = self.conv(x)                    # (B, D, T)
        return self.fc(x.flatten(1))


class DepthGRUEncoder(nn.Module):
    """Depth images -> e_t via CNN + GRU. (Student)

    Depth renders at 10 Hz while the policy runs at 50 Hz: the GRU ticks only
    on new frames; between renders the cached latent is reused. Hidden state
    is detached when stored (TBPTT = 1 frame tick), so gradients flow through
    the CNN+GRU only on tick steps — hold steps return the detached cache.
    """

    def __init__(self, cfg: ExperimentCfg):
        super().__init__()
        pc, cam = cfg.policy, cfg.camera
        self.cnn = nn.Sequential(
            nn.Conv2d(cam.n_cams, 32, kernel_size=5, stride=2), nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2), nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2), nn.ELU(),
        )
        # Flatten size measured by a dummy forward so any rig (e.g. the old
        # 3-cam 120x160 setup) works from config alone.
        with torch.no_grad():
            flat = self.cnn(
                torch.zeros(1, cam.n_cams, cam.height, cam.width)
            ).flatten(1).shape[1]
        self.fc = nn.Sequential(nn.Linear(flat, pc.depth_fc_dim), nn.ELU())
        self.gru = nn.GRUCell(pc.depth_fc_dim, pc.gru_hidden_dim)
        self.out = nn.Sequential(
            nn.Linear(pc.gru_hidden_dim, pc.extero_latent_dim), nn.ELU()
        )
        self._gru_dim = pc.gru_hidden_dim
        self._e_dim = pc.extero_latent_dim
        self._hidden: torch.Tensor | None = None
        self._e_cache: torch.Tensor | None = None

    @property
    def hidden(self) -> torch.Tensor | None:
        return self._hidden

    def _ensure_state(self, batch: int, device, dtype):
        if self._hidden is None or self._hidden.shape[0] != batch:
            self._hidden = torch.zeros(batch, self._gru_dim,
                                       device=device, dtype=dtype)
            self._e_cache = torch.zeros(batch, self._e_dim,
                                        device=device, dtype=dtype)

    def reset_state(self):
        self._hidden = None
        self._e_cache = None

    def step(
        self,
        depth: torch.Tensor,           # (B, C, H, W), pre-normalized
        new_frame_mask: torch.Tensor,  # (B,) bool — a fresh render arrived
        reset_mask: torch.Tensor | None = None,  # (B,) bool — env was reset
    ) -> torch.Tensor:
        """Returns e_t (B, extero_latent_dim). Gradient flows only through
        rows that ticked this step; held rows return the detached cache."""
        self._ensure_state(depth.shape[0], depth.device, depth.dtype)
        if reset_mask is not None and reset_mask.any():
            keep = (~reset_mask).unsqueeze(-1).to(self._hidden.dtype)
            self._hidden = self._hidden * keep
            self._e_cache = self._e_cache * keep

        if not bool(new_frame_mask.any()):
            return self._e_cache

        feat = self.fc(self.cnn(depth).flatten(1))
        h_new = self.gru(feat, self._hidden)
        e_new = self.out(h_new)
        m = new_frame_mask.unsqueeze(-1)
        e = torch.where(m, e_new, self._e_cache)
        self._hidden = torch.where(m, h_new.detach(), self._hidden)
        self._e_cache = torch.where(m, e_new.detach(), self._e_cache)
        return e


# ════════════════════════════════════════════════════════════════════════════
# Actor / Critic
# ════════════════════════════════════════════════════════════════════════════

class Actor(nn.Module):
    """MLP(proprio_feat, e_t, z_t) -> action mean. Shared by teacher/student;
    the exteroception encoder is injected (ScandotEncoder or DepthGRUEncoder)
    under the same attribute name so checkpoints cross-load by prefix."""

    def __init__(self, cfg: ExperimentCfg, extero_encoder: nn.Module):
        super().__init__()
        pc = cfg.policy
        self.proprio_mlp = _mlp(cfg.proprio_dim, pc.proprio_mlp_hidden[:-1],
                                pc.proprio_mlp_hidden[-1], final_act=True)
        self.extero_encoder = extero_encoder
        trunk_in = pc.proprio_mlp_hidden[-1] + pc.extero_latent_dim + pc.z_dim
        self.trunk = _mlp(trunk_in, pc.trunk_hidden[:-1], pc.trunk_hidden[-1],
                          final_act=True)
        self.head = nn.Linear(pc.trunk_hidden[-1], cfg.action_dim)
        self.log_std = nn.Parameter(
            torch.full((cfg.action_dim,), pc.log_std_init)
        )
        self._mean_clip = pc.action_mean_clip
        self._log_std_min = pc.log_std_min
        self._log_std_max = pc.log_std_max

    def forward_with_latent(
        self, proprio: torch.Tensor, e: torch.Tensor, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (action_mean clamped, log_std clamped)."""
        p = self.proprio_mlp(proprio)
        x = self.trunk(torch.cat([p, e, z], dim=-1))
        # tanh keeps the mean strictly inside (-1, 1) with restoring
        # gradients; a hard clamp has zero gradient outside the range,
        # so nothing opposed the mean drifting ever further out.
        mean = torch.tanh(self.head(x))
        log_std = torch.clamp(self.log_std, self._log_std_min, self._log_std_max)
        return mean, log_std


class Critic(nn.Module):
    """Asymmetric critic: sees proprio + scandots + priv + true base lin vel."""

    def __init__(self, cfg: ExperimentCfg):
        super().__init__()
        self.net = _mlp(cfg.critic_obs_dim, cfg.policy.critic_hidden, 1)
        # Near-zero output init (kept from the original SpotActorCritic)
        final = self.net[-1]
        nn.init.orthogonal_(final.weight, gain=0.01)
        nn.init.zeros_(final.bias)

    def forward(self, critic_obs: torch.Tensor) -> torch.Tensor:
        return self.net(critic_obs).squeeze(-1)


# ════════════════════════════════════════════════════════════════════════════
# Policies
# ════════════════════════════════════════════════════════════════════════════

class TeacherPolicy(nn.Module):
    """Phase 1: privileged asymmetric actor-critic + concurrent phi."""

    def __init__(self, cfg: ExperimentCfg):
        super().__init__()
        self.actor = Actor(cfg, ScandotEncoder(cfg))
        self.priv_encoder = PrivEncoder(cfg)
        self.adaptation_module = AdaptationModule(cfg)
        self.critic = Critic(cfg)

    def act_mean(
        self, proprio: torch.Tensor, scandots: torch.Tensor, priv: torch.Tensor
    ) -> torch.Tensor:
        """Deterministic teacher action (used for DAgger labels)."""
        e = self.actor.extero_encoder(scandots)
        z = self.priv_encoder(priv)
        return self.actor.forward_with_latent(proprio, e, z)[0]

    def evaluate(
        self,
        proprio: torch.Tensor,
        scandots: torch.Tensor,
        priv: torch.Tensor,
        critic_obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full grad path: (mean, log_std, value). Used in rollout (no_grad)
        and PPO minibatches (with grad — encoders train end-to-end)."""
        e = self.actor.extero_encoder(scandots)
        z = self.priv_encoder(priv)
        mean, log_std = self.actor.forward_with_latent(proprio, e, z)
        value = self.critic(critic_obs)
        return mean, log_std, value


class StudentPolicy(nn.Module):
    """Phase 2: deployable policy — depth CNN+GRU exteroception + frozen phi."""

    def __init__(self, cfg: ExperimentCfg):
        super().__init__()
        self.actor = Actor(cfg, DepthGRUEncoder(cfg))
        self.adaptation_module = AdaptationModule(cfg)

    def act_mean(
        self,
        proprio: torch.Tensor,
        depth: torch.Tensor,
        new_frame_mask: torch.Tensor,
        history: torch.Tensor,
        reset_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Student action mean. phi runs without grad (frozen in Phase 2)."""
        e = self.actor.extero_encoder.step(depth, new_frame_mask, reset_mask)
        with torch.no_grad():
            z_hat = self.adaptation_module(history)
        return self.actor.forward_with_latent(proprio, e, z_hat)[0]


# ════════════════════════════════════════════════════════════════════════════
# Diagonal Gaussian helpers (kept verbatim from spot_actor_critic.py)
# ════════════════════════════════════════════════════════════════════════════

def gaussian_log_prob(
    mean: torch.Tensor, log_std: torch.Tensor, action: torch.Tensor
) -> torch.Tensor:
    """Log probability of action under diagonal Gaussian."""
    std = torch.exp(log_std)
    log_p = -0.5 * (((action - mean) / std) ** 2 + 2 * log_std
                    + math.log(2 * math.pi))
    return log_p.sum(-1)


def gaussian_entropy(log_std: torch.Tensor) -> torch.Tensor:
    """Entropy of diagonal Gaussian."""
    return (0.5 + 0.5 * math.log(2 * math.pi) + log_std).sum(-1)
