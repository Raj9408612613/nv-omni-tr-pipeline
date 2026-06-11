"""
Checkpoint I/O + named teacher -> student cross-loading
========================================================
Checkpoint format (extends the original trainer's):
    {
      "model_state_dict":            policy state dict,
      "optimizer_state_dict":        main optimizer (PPO or DAgger),
      "adapt_optimizer_state_dict":  phi optimizer (teacher ckpts only),
      "ret_mean", "ret_std":         return-normalization stats (teacher),
      "phase":                       "teacher" | "student",
      "robot":                       config module name,
    }

Cross-loading is BY MODULE NAME: every parameter whose key starts with a
shared prefix is copied teacher -> student; the exteroception encoder is
intentionally excluded (ScandotEncoder vs DepthGRUEncoder).
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Keys copied from the Phase 1 teacher into the Phase 2 student.
SHARED_PREFIXES = (
    "actor.proprio_mlp.",
    "actor.trunk.",
    "actor.head.",
    "actor.log_std",
    "adaptation_module.",
)
# Teacher-only namespaces (must never appear as student keys).
TEACHER_ONLY_PREFIXES = ("priv_encoder.", "critic.")
# Student keys allowed to be missing after a cross-load.
STUDENT_FRESH_PREFIX = "actor.extero_encoder."


def save_checkpoint(path: str, *, model_state: dict, phase: str, robot: str,
                    optimizer_state: dict | None = None,
                    adapt_optimizer_state: dict | None = None,
                    ret_mean: float | None = None,
                    ret_std: float | None = None,
                    extra: dict | None = None):
    ckpt = {
        "model_state_dict": model_state,
        "phase": phase,
        "robot": robot,
    }
    if optimizer_state is not None:
        ckpt["optimizer_state_dict"] = optimizer_state
    if adapt_optimizer_state is not None:
        ckpt["adapt_optimizer_state_dict"] = adapt_optimizer_state
    if ret_mean is not None:
        ckpt["ret_mean"] = float(ret_mean)
    if ret_std is not None:
        ckpt["ret_std"] = float(ret_std)
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


def load_checkpoint(path: str, device: str | torch.device = "cpu") -> dict:
    return torch.load(path, map_location=device, weights_only=True)


def load_teacher_into_student(
    student: nn.Module, teacher_state: dict, freeze_adaptation: bool = True
) -> tuple[list[str], list[str]]:
    """Copy shared modules from a teacher state_dict into a StudentPolicy.

    Asserts that the ONLY keys the student did not receive are its fresh
    depth encoder, and that nothing unexpected was offered — so any drift
    in module naming fails loudly instead of silently skipping weights.

    Returns (missing_keys, unexpected_keys) for logging.
    """
    filtered = {
        k: v for k, v in teacher_state.items()
        if k.startswith(SHARED_PREFIXES)
    }
    if not filtered:
        raise ValueError(
            "No shared keys found in the teacher checkpoint — expected "
            f"prefixes {SHARED_PREFIXES}. Got keys like: "
            f"{list(teacher_state)[:5]}"
        )
    result = student.load_state_dict(filtered, strict=False)

    bad_missing = [
        k for k in result.missing_keys
        if not k.startswith(STUDENT_FRESH_PREFIX)
    ]
    if bad_missing:
        raise RuntimeError(
            "Teacher checkpoint did not cover these student modules "
            f"(naming drift?): {bad_missing}"
        )
    if result.unexpected_keys:
        raise RuntimeError(
            "Teacher checkpoint offered keys the student does not have: "
            f"{result.unexpected_keys}"
        )

    if freeze_adaptation:
        student.adaptation_module.requires_grad_(False)
        student.adaptation_module.eval()
    return list(result.missing_keys), list(result.unexpected_keys)
