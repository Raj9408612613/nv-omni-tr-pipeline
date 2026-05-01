"""
Spot Actor-Critic Network — PyTorch
=====================================
Ported from jax_ppo.py (Flax/JAX).

Architecture (identical to JAX version):
    Depth (3x120x160) -> CNN encoder -> 256-dim
    Proprio (37)       -> MLP encoder -> 64-dim
                              | concat (320-dim)
                          MLP (256 -> 128)
                              |
                    Policy head -> 12 actions (mean, log_std)
                    Value  head -> 1 scalar

all the above is now changed to run on the isaac sim and rl-games(isaac gym environment). 
instead of importing FLAX/JAX we now import torch.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    CNN_FEAT_DIM, PROPRIO_DIM, ACTION_DIM,
    LOG_STD_MIN, LOG_STD_MAX,
)


class DepthCNNEncoder(nn.Module):
    """Nx120x160 depth images -> CNN_FEAT_DIM features (N = N_CAMS = 3).

    Matches Flax version exactly:
        Conv(32, 8x8, stride=4) -> ELU
        Conv(64, 4x4, stride=2) -> ELU
        Conv(64, 3x3, stride=1) -> ELU
        Flatten -> Dense(256) -> ELU
    """

    def __init__(self, features: int = CNN_FEAT_DIM):
        super().__init__()
        from .config import N_CAMS
        self.conv1 = nn.Conv2d(N_CAMS, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # Compute flattened size: input (5, 120, 160)
        # After conv1: (32, 29, 39)  [(120-8)/4+1=29, (160-8)/4+1=39]
        # After conv2: (64, 13, 18)  [(29-4)/2+1=13, (39-4)/2+1=18]
        # After conv3: (64, 11, 16)  [(13-3)/1+1=11, (18-3)/1+1=16]
        self.fc = nn.Linear(64 * 11 * 16, features)

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        """
        Args:
            depth: (B, N_CAMS, 120, 160) — already channels-first for PyTorch
        Returns:
            (B, CNN_FEAT_DIM) features
        """
        x = F.elu(self.conv1(depth))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = x.reshape(x.shape[0], -1)  # flatten spatial dims
        x = F.elu(self.fc(x))
        return x


class PropriEncoder(nn.Module):
    """37-dim proprio -> 64-dim features.

    Matches Flax version:
        Dense(128) -> ELU -> Dense(64) -> ELU
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(PROPRIO_DIM, 128)
        self.fc2 = nn.Linear(128, 64)

    def forward(self, proprio: torch.Tensor) -> torch.Tensor:
        x = F.elu(self.fc1(proprio))
        x = F.elu(self.fc2(x))
        return x


class SpotActorCritic(nn.Module):
    """Combined actor-critic for Spot navigation.

    Exactly matches the Flax SpotActorCritic architecture.
    Supports CNN feature caching: call encode_depth() during rollout,
    then head_forward() with cached features during PPO update.
    """

    def __init__(self):
        super().__init__()
        self.cnn = DepthCNNEncoder()
        self.proprio_enc = PropriEncoder()

        # Shared trunk: concat(256, 64) = 320 -> 256 -> 128
        self.trunk0 = nn.Linear(CNN_FEAT_DIM + 64, 256)
        self.trunk1 = nn.Linear(256, 128)

        # Actor head
        self.actor = nn.Linear(128, ACTION_DIM)

        # Learnable log_std (not per-observation, shared across batch)
        # Init std = e^-1 ≈ 0.37. Balances exploration vs stability:
        # -2.0 (std=0.14) was too conservative — near-zero entropy (-6.9)
        # killed exploration. -1.0 gives entropy ≈ +5, still 3× less
        # chaotic than the original std=1.0 that threw the robot over.
        self.log_std = nn.Parameter(torch.full((ACTION_DIM,), -1.0))

        # Critic head
        self.critic0 = nn.Linear(128, 64)
        # critic1: near-zero init (orthogonal @ 0.01) like Flax version
        self.critic1 = nn.Linear(64, 1)
        nn.init.orthogonal_(self.critic1.weight, gain=0.01)
        nn.init.zeros_(self.critic1.bias)

    def forward(
        self, depth: torch.Tensor, proprio: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass.

        Args:
            depth:  (B, N_CAMS, 120, 160)
            proprio: (B, 37)

        Returns:
            action_mean: (B, 12) clipped to [-2, 2]
            log_std:     (12,) clamped to [LOG_STD_MIN, LOG_STD_MAX]
            value:       (B,)
        """
        cnn_feat = self.cnn(depth)
        return self.head_forward(cnn_feat, proprio)

    def encode_depth(self, depth: torch.Tensor) -> torch.Tensor:
        """Run only the CNN encoder (for feature caching during rollout).

        Args:
            depth: (B, N_CAMS, 120, 160)
        Returns:
            (B, CNN_FEAT_DIM) features
        """
        return self.cnn(depth / 10.0)  # matches _inference_step normalization

    @torch.no_grad()
    def inference_step(
        self, depth: torch.Tensor, proprio: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full rollout inference: encode -> forward -> sample.

        Returns (action, log_prob, value, cnn_feat) — matches JAX _inference_step.
        """
        cnn_feat = self.cnn(depth / 10.0)
        action_mean, log_std, value = self.head_forward(cnn_feat, proprio)

        std = torch.exp(log_std)
        noise = torch.randn_like(action_mean)
        action = torch.clamp(action_mean + std * noise, -1.0, 1.0)
        log_prob = gaussian_log_prob(action_mean, log_std, action)
        return action, log_prob, value, cnn_feat

    def head_forward(
        self, cnn_feat: torch.Tensor, proprio: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward from pre-encoded CNN features (skips DepthCNNEncoder).

        Used during PPO update with cached CNN features.
        Gradients flow through this (not @torch.no_grad).
        """
        pro_feat = self.proprio_enc(proprio)
        x = torch.cat([cnn_feat, pro_feat], dim=-1)  # (B, 320)

        x = F.elu(self.trunk0(x))
        x = F.elu(self.trunk1(x))

        # Actor
        action_mean = torch.clamp(self.actor(x), -2.0, 2.0)
        log_std = torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX)

        # Critic
        v = F.elu(self.critic0(x))
        value = self.critic1(v).squeeze(-1)

        return action_mean, log_std, value


# ── Distribution helpers ─────────────────────────────────────────────────────

def gaussian_log_prob(
    mean: torch.Tensor, log_std: torch.Tensor, action: torch.Tensor
) -> torch.Tensor:
    """Log probability of action under diagonal Gaussian.

    Matches JAX _gaussian_log_prob exactly.
    """
    std = torch.exp(log_std)
    log_p = -0.5 * (((action - mean) / std) ** 2 + 2 * log_std
                     + math.log(2 * math.pi))
    return log_p.sum(-1)  # sum over action dims


def gaussian_entropy(log_std: torch.Tensor) -> torch.Tensor:
    """Entropy of diagonal Gaussian. Matches JAX _gaussian_entropy."""
    return (0.5 + 0.5 * math.log(2 * math.pi) + log_std).sum(-1)
