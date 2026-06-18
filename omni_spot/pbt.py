"""
Population-Based Training — single-process, single-sim
======================================================
A `Population` of N members trained in ONE Isaac Sim process that share a
single `env.step`. Each member owns:

    - its own TeacherPolicy + PPO optimizer + adaptation optimizer
      (a full `PPOTrainer`, reused so the GAE / return-normalization /
      PPO-update math is the validated single-policy path),
    - 4 reward knobs   : alive_bonus, progress_w, vel_track_w, goal_bonus
    - 3 PPO knobs      : clip_eps, ent_coef, lr
    - per-member return stats (ret_mean / ret_std, kept inside its PPOTrainer)
    - a fitness history.

Env layout: total = N x envs_per_member envs in N contiguous slices; member k
owns envs [k*epm : (k+1)*epm]. The per-env reward-weight tensors are tiled
across each slice and handed to the env via `env._reward_weights`; the env's
curriculum state is per-env, so slices never cross-contaminate.

Rollout ordering (correctness reference, the naive loop):
    1. each member runs its policy on its obs slice -> action slice
    2. concat all N action slices -> ONE shared env.step
    3. store per-member transitions
    4. per-member PPO update runs AFTER the T-step rollout.

Exploit / explore (every interval, after a warm-up): rank by a WEIGHT-FREE
fitness (success_rate + a small -dist_goal term), copy the bottom quartile
from the top quartile (weights + optimizer state + ret stats + all 7 knobs),
then perturb each copied knob x0.8 or x1.2 within its clamp range, and re-tile
that member's slice of the reward-weight tensor.

NOTE on vmap: the plan calls for an optional `torch.func.functional_call` +
`vmap` batched update at high N. This module implements the loop path (the
correctness reference). The structure (per-member `PPOTrainer.update`) is the
slot where a vmapped update would replace the Python loop; it is intentionally
left as a follow-up so the loop stays the verified path.
"""

from __future__ import annotations

import dataclasses
import math
import random

import torch

from .configs.base import ExperimentCfg
from .ppo import PPOTrainer

# ── Knob ranges (plan Phase 3) ───────────────────────────────────────────────
# The 4 reward knobs become per-env tensors; the 3 PPO knobs stay per-member
# scalars applied to each member's PPOTrainer before its update.
REWARD_KNOB_RANGES: dict[str, tuple[float, float]] = {
    "alive_bonus": (0.0, 0.2),
    "progress_w": (10.0, 100.0),
    "vel_track_w": (0.5, 3.0),
    "goal_bonus": (10.0, 50.0),
}
PPO_KNOB_RANGES: dict[str, tuple[float, float]] = {
    "clip_eps": (0.1, 0.3),
    "ent_coef": (0.0, 0.02),
    "lr": (1e-5, 1e-3),
}
ALL_KNOB_RANGES: dict[str, tuple[float, float]] = {
    **REWARD_KNOB_RANGES, **PPO_KNOB_RANGES
}
REWARD_KNOBS: tuple[str, ...] = tuple(REWARD_KNOB_RANGES)
PPO_KNOBS: tuple[str, ...] = tuple(PPO_KNOB_RANGES)
KNOB_NAMES: tuple[str, ...] = tuple(ALL_KNOB_RANGES)

PERTURB_FACTORS = (0.8, 1.2)

# Per-step buffer keys, matching PPOTrainer.collect_rollout / _finalize_rollout.
_BUF_KEYS = (
    "proprio", "scandots", "priv", "critic_extras",
    "action", "log_prob", "value", "reward", "done",
)


def _clamp(value: float, lo: float, hi: float) -> float:
    return float(min(max(value, lo), hi))


class Member:
    """One PBT member: a PPOTrainer plus its 7 searched knobs and fitness."""

    def __init__(self, member_id: int, trainer: PPOTrainer, knobs: dict):
        self.id = member_id
        self.trainer = trainer
        self.knobs: dict[str, float] = {k: float(knobs[k]) for k in KNOB_NAMES}
        self.fitness: float = float("-inf")
        self.fitness_history: list[float] = []
        # Fitness accumulators over the current interval (reset at each evolve).
        self.ep_count: int = 0
        self.goal_count: int = 0
        self.final_dist_sum: float = 0.0
        # Last computed components (for logging / contamination guard).
        self.success_rate: float = float("nan")
        self.mean_final_dist: float = float("nan")

    # ── knobs -> trainer ──────────────────────────────────────────────
    def apply_knobs_to_trainer(self) -> None:
        """Push this member's 3 PPO knobs onto its PPOTrainer before update().
        Reward knobs are applied through the env's per-env weight tensor, not
        here."""
        t = self.trainer
        t._clip_eps = float(self.knobs["clip_eps"])
        t._cur_ent_coef = float(self.knobs["ent_coef"])
        lr = float(self.knobs["lr"])
        for pg in t.optimizer.param_groups:
            pg["lr"] = lr
        t._cur_lr = lr

    def reset_accumulators(self) -> None:
        self.ep_count = 0
        self.goal_count = 0
        self.final_dist_sum = 0.0


class Population:
    """N members sharing one env. Owns the per-env reward-weight tensors."""

    def __init__(
        self,
        cfg: ExperimentCfg,
        n_members: int,
        envs_per_member: int,
        device: str = "cuda",
        seed: int = 0,
        fitness_dist_weight: float = 0.1,
        init_ckpt: str | None = None,
    ):
        self.cfg = cfg
        self.envs_per_member = envs_per_member
        self.device = torch.device(device)
        self.fitness_dist_weight = fitness_dist_weight
        self.rng = random.Random(seed)

        self.members: list[Member] = []
        for i in range(n_members):
            trainer = PPOTrainer(cfg, device=str(self.device))
            if init_ckpt is not None:
                # Warm-start every member from a validated teacher; knobs still
                # diverge, so the population searches around a good policy.
                trainer.load(init_ckpt)
            knobs = self._sample_initial_knobs()
            self.members.append(Member(i, trainer, knobs))

        # Per-env reward-weight tensors (allocated once, mutated in place).
        self._weight_tensors: dict[str, torch.Tensor] = {}
        self._reward_weights = None
        self.build_reward_weights()

    # ── layout helpers ────────────────────────────────────────────────
    @property
    def n_members(self) -> int:
        return len(self.members)

    @property
    def total_envs(self) -> int:
        return self.n_members * self.envs_per_member

    def member_slice(self, member_id: int) -> slice:
        epm = self.envs_per_member
        return slice(member_id * epm, (member_id + 1) * epm)

    # ── reward-weight tiling (FIX B target) ───────────────────────────
    def build_reward_weights(self):
        """(Re)allocate the per-env reward-weight tensors and tile every
        member's knobs across its slice. Returns the RewardWeightsCfg to assign
        to `env._reward_weights` (it holds references to the live tensors, so
        in-place re-tiling after PBT is visible to the env automatically)."""
        total = self.total_envs
        self._weight_tensors = {
            knob: torch.empty(total, device=self.device) for knob in REWARD_KNOBS
        }
        self._reward_weights = dataclasses.replace(
            self.cfg.reward, **self._weight_tensors
        )
        for m in self.members:
            self.retile_member(m.id)
        return self._reward_weights

    @property
    def reward_weights(self):
        return self._reward_weights

    def retile_member(self, member_id: int) -> None:
        sl = self.member_slice(member_id)
        knobs = self.members[member_id].knobs
        for knob in REWARD_KNOBS:
            self._weight_tensors[knob][sl] = float(knobs[knob])

    # ── initial knob sampling ─────────────────────────────────────────
    def _sample_initial_knobs(self) -> dict[str, float]:
        knobs: dict[str, float] = {}
        for knob, (lo, hi) in ALL_KNOB_RANGES.items():
            if knob == "lr":
                knobs[knob] = math.exp(
                    self.rng.uniform(math.log(lo), math.log(hi))
                )
            elif knob == "ent_coef":
                # Start strictly positive: multiplicative perturbation can never
                # revive a knob that has hit exactly 0.
                knobs[knob] = self.rng.uniform(max(lo, 1e-3), hi)
            else:
                knobs[knob] = self.rng.uniform(lo, hi)
        return knobs

    # ── rollout (shared step, per-member buffers) ─────────────────────
    def collect_rollouts(self, env, obs: dict, n_steps: int):
        """Run n_steps with ONE shared env.step per step. Returns
        (next_obs, batches, stats) — one RolloutBatch + stats dict per member,
        each built via that member's PPOTrainer._finalize_rollout."""
        tc = self.cfg.teacher
        M = self.n_members
        bufs = [{k: [] for k in _BUF_KEYS} for _ in range(M)]
        adapt_losses: list[list[float]] = [[] for _ in range(M)]
        reward_sums: list[dict | None] = [None for _ in range(M)]
        reward_count = 0

        for step in range(n_steps):
            actions = []
            obs_slices = []
            for m in self.members:
                sl = self.member_slice(m.id)
                obs_m = {k: v[sl] for k, v in obs.items()}
                obs_slices.append(obs_m)
                action, log_prob, value = m.trainer.sample_action(obs_m)
                if step % tc.adapt_every == 0:
                    adapt_losses[m.id].append(m.trainer.adaptation_step(obs_m))
                b = bufs[m.id]
                b["proprio"].append(obs_m["proprio"])
                b["scandots"].append(obs_m["scandots"])
                b["priv"].append(obs_m["priv"])
                b["critic_extras"].append(obs_m["critic_extras"])
                b["action"].append(action)
                b["log_prob"].append(log_prob)
                b["value"].append(value)
                actions.append(action)

            full_action = torch.cat(actions, dim=0)
            obs, reward, terminated, truncated, info = env.step(full_action)
            done = terminated | truncated
            done_f = done.float()
            at_goal = getattr(env, "_at_goal", None)
            dist_goal = info.get("dist_goal", None)

            for m in self.members:
                sl = self.member_slice(m.id)
                b = bufs[m.id]
                b["reward"].append(reward[sl])
                b["done"].append(done_f[sl])

                # Per-member reward-component sums (diagnostics), sliced from
                # the env extras exactly as collect_rollout accumulates them.
                step_info_m = {
                    k: v[sl] for k, v in info.items()
                    if isinstance(v, torch.Tensor)
                }
                if reward_sums[m.id] is None:
                    reward_sums[m.id] = {
                        k: v.clone() for k, v in step_info_m.items()
                    }
                else:
                    rs = reward_sums[m.id]
                    for k, v in step_info_m.items():
                        if k in rs:
                            rs[k] = rs[k] + v

                # Weight-free fitness accumulators.
                done_m = done[sl]
                n_done = int(done_m.sum())
                if n_done > 0:
                    m.ep_count += n_done
                    if at_goal is not None:
                        m.goal_count += int((at_goal[sl] & done_m).sum())
                    if dist_goal is not None:
                        m.final_dist_sum += float(dist_goal[sl][done_m].sum())
            reward_count += 1

        batches, stats = [], []
        for m in self.members:
            sl = self.member_slice(m.id)
            obs_m = {k: v[sl] for k, v in obs.items()}
            _, _, last_value = m.trainer.sample_action(obs_m)
            batch, st = m.trainer._finalize_rollout(
                bufs[m.id], last_value, adapt_losses[m.id],
                reward_sums[m.id], reward_count, None,
            )
            batches.append(batch)
            stats.append(st)
        return obs, batches, stats

    # ── per-member update (loop = correctness reference) ──────────────
    def update_members(self, batches: list) -> list[dict]:
        infos = []
        for m, batch in zip(self.members, batches):
            m.apply_knobs_to_trainer()
            infos.append(m.trainer.update(batch))
        return infos

    # ── fitness ───────────────────────────────────────────────────────
    def compute_fitness(self, m: Member) -> float:
        """WEIGHT-FREE fitness: success_rate minus a small mean-final-distance
        term. Both come from geometry (goal_tol, dist_goal), NOT from any
        knob-scaled reward term, so selection ranks behavior not knobs."""
        if m.ep_count == 0:
            m.success_rate = float("nan")
            m.mean_final_dist = float("nan")
            return float("-inf")
        m.success_rate = m.goal_count / m.ep_count
        m.mean_final_dist = m.final_dist_sum / m.ep_count
        return m.success_rate - self.fitness_dist_weight * m.mean_final_dist

    # ── exploit / explore ─────────────────────────────────────────────
    def evolve(self) -> list[dict]:
        """Rank by fitness, copy bottom quartile from a random top-quartile
        member, perturb its knobs, and re-tile its env slice. Returns a list of
        copy/perturb events for logging. Resets fitness accumulators after."""
        for m in self.members:
            m.fitness = self.compute_fitness(m)
            m.fitness_history.append(m.fitness)

        ranked = sorted(self.members, key=lambda m: m.fitness, reverse=True)
        cut = max(1, self.n_members // 4)
        top = ranked[:cut]
        bottom = ranked[-cut:]

        events: list[dict] = []
        for weak in bottom:
            donor = self.rng.choice(top)
            seed = self.rng.randrange(2 ** 31)
            old_knobs = dict(weak.knobs)
            if donor.id != weak.id:
                self._copy_member(donor, weak)
            self._perturb_knobs(weak)
            self.retile_member(weak.id)
            events.append({
                "target": weak.id,
                "source": donor.id,
                "donor_fitness": donor.fitness,
                "target_fitness_before": weak.fitness,
                "perturb_seed": seed,
                "old_knobs": old_knobs,
                "new_knobs": dict(weak.knobs),
            })

        for m in self.members:
            m.reset_accumulators()
        return events

    def _copy_member(self, src: Member, dst: Member) -> None:
        """Exploit: clone weights + optimizer state + ret stats + all knobs.
        load_state_dict deep-copies, so src and dst keep independent storage."""
        dst.trainer.net.load_state_dict(src.trainer.net.state_dict())
        dst.trainer.optimizer.load_state_dict(
            src.trainer.optimizer.state_dict()
        )
        dst.trainer.adapt_optimizer.load_state_dict(
            src.trainer.adapt_optimizer.state_dict()
        )
        # ret stats matter: the critic trains on normalized returns and GAE
        # denormalizes with the same stats, so donor weights expect donor norm.
        dst.trainer._ret_mean = src.trainer._ret_mean
        dst.trainer._ret_std = src.trainer._ret_std
        dst.knobs = dict(src.knobs)

    def _perturb_knobs(self, m: Member) -> None:
        for knob, (lo, hi) in ALL_KNOB_RANGES.items():
            factor = self.rng.choice(PERTURB_FACTORS)
            m.knobs[knob] = _clamp(m.knobs[knob] * factor, lo, hi)

    # ── contamination guard (Phase 5) ─────────────────────────────────
    def knob_fitness_correlations(self) -> dict[str, float]:
        """Pearson correlation between member fitness and each mutated knob,
        across the population. A strong correlation with goal_bonus would mean
        fitness is still weight-contaminated."""
        fits = [m.fitness for m in self.members]
        finite = [f for f in fits if math.isfinite(f)]
        out: dict[str, float] = {}
        if len(finite) < 2:
            return {k: float("nan") for k in KNOB_NAMES}
        idx = [i for i, f in enumerate(fits) if math.isfinite(f)]
        fy = [fits[i] for i in idx]
        for knob in KNOB_NAMES:
            fx = [self.members[i].knobs[knob] for i in idx]
            out[knob] = _pearson(fx, fy)
        return out

    # ── selection helpers ─────────────────────────────────────────────
    def top_member(self) -> Member:
        return max(self.members, key=lambda m: m.fitness)

    # ── checkpoint I/O ─────────────────────────────────────────────────
    def state_dict(self) -> dict:
        return {
            "n_members": self.n_members,
            "envs_per_member": self.envs_per_member,
            "fitness_dist_weight": self.fitness_dist_weight,
            "members": [
                {
                    "id": m.id,
                    "knobs": dict(m.knobs),
                    "model_state": m.trainer.net.state_dict(),
                    "optimizer_state": m.trainer.optimizer.state_dict(),
                    "adapt_optimizer_state":
                        m.trainer.adapt_optimizer.state_dict(),
                    "ret_mean": m.trainer._ret_mean,
                    "ret_std": m.trainer._ret_std,
                    "fitness": m.fitness,
                    "fitness_history": list(m.fitness_history),
                }
                for m in self.members
            ],
        }

    def load_state_dict(self, sd: dict) -> None:
        if sd["n_members"] != self.n_members:
            raise ValueError(
                f"population size mismatch: checkpoint has {sd['n_members']} "
                f"members, this run has {self.n_members}"
            )
        if sd["envs_per_member"] != self.envs_per_member:
            raise ValueError(
                "envs_per_member mismatch: checkpoint "
                f"{sd['envs_per_member']} vs run {self.envs_per_member}"
            )
        for m, ms in zip(self.members, sd["members"]):
            m.trainer.net.load_state_dict(ms["model_state"])
            m.trainer.optimizer.load_state_dict(ms["optimizer_state"])
            m.trainer.adapt_optimizer.load_state_dict(
                ms["adapt_optimizer_state"]
            )
            m.trainer._ret_mean = float(ms["ret_mean"])
            m.trainer._ret_std = float(ms["ret_std"])
            m.knobs = {k: float(ms["knobs"][k]) for k in KNOB_NAMES}
            m.fitness = float(ms.get("fitness", float("-inf")))
            m.fitness_history = list(ms.get("fitness_history", []))
        # Restore slice ownership AND per-env weight tiling, not just weights.
        self.build_reward_weights()

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        sd = torch.load(path, map_location=self.device, weights_only=False)
        self.load_state_dict(sd)


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    denom = math.sqrt(sxx * syy)
    if denom < 1e-12:
        return float("nan")
    return sxy / denom
