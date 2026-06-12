"""
PPO Training Diagnostics — PyTorch
====================================
Ported from ppo_diagnostics.py. Framework-agnostic (uses plain floats/dicts).
"""

import numpy as np


def compute_explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Explained variance: 1 - Var(y_true - y_pred) / Var(y_true)."""
    var_true = np.var(y_true)
    if var_true < 1e-10:
        return 0.0
    return float(1.0 - np.var(y_true - y_pred) / var_true)


def print_diagnostics(update: int, rollout_diag: dict, update_diag: dict,
                      max_grad: float = 1.5):
    """Print a diagnostic block to stdout."""
    lines = []
    lines.append("")
    lines.append(f"{'='*72}")
    lines.append(f"  DIAGNOSTICS -- Update {update}")
    lines.append(f"{'='*72}")

    # ── 1. Return / Advantage Stats ──────────────────────────────────
    lines.append("")
    lines.append("  [RETURNS & ADVANTAGES]")
    rd = rollout_diag
    lines.append(f"    Raw returns    : mean={rd['ret_raw_mean']:>10.4f}  std={rd['ret_raw_std']:>10.4f}  "
                 f"min={rd['ret_raw_min']:>10.4f}  max={rd['ret_raw_max']:>10.4f}")
    lines.append(f"    Norm returns   : mean={rd['ret_norm_mean']:>10.4f}  std={rd['ret_norm_std']:>10.4f}  "
                 f"min={rd['ret_norm_min']:>10.4f}  max={rd['ret_norm_max']:>10.4f}")
    lines.append(f"    Advantages     : mean={rd['adv_mean']:>10.4f}  std={rd['adv_std']:>10.4f}  "
                 f"min={rd['adv_min']:>10.4f}  max={rd['adv_max']:>10.4f}")
    lines.append(f"    Ret norm scale : ret_mean={rd['ret_scale_mean']:>10.4f}  ret_std={rd['ret_scale_std']:>10.4f}")

    # ── 2. Value Predictions ─────────────────────────────────────────
    lines.append("")
    lines.append("  [VALUE PREDICTIONS]")
    lines.append(f"    Raw values     : mean={rd['val_raw_mean']:>10.4f}  std={rd['val_raw_std']:>10.4f}  "
                 f"min={rd['val_raw_min']:>10.4f}  max={rd['val_raw_max']:>10.4f}")
    ev = rd.get("explained_var", 0.0)
    lines.append(f"    Explained var  : {ev:>10.4f}  {'<-- BAD (< 0)' if ev < 0 else '<-- OK' if ev > 0.1 else '<-- LOW'}")

    # ── 3. Policy Health ─────────────────────────────────────────────
    lines.append("")
    lines.append("  [POLICY HEALTH]")
    ud = update_diag
    lines.append(f"    action_mean mag: mean={ud.get('action_mean_abs', 0.0):>10.4f}")
    lines.append(f"    action_std     : mean={ud.get('action_std_mean', 0.0):>10.4f}")
    lines.append(f"    entropy        : {ud.get('entropy', 0.0):>10.4f}")
    lines.append(f"    approx_kl      : {ud.get('approx_kl', 0.0):>10.6f}  "
                 f"{'<-- HIGH' if ud.get('approx_kl', 0) > 0.05 else ''}")
    lines.append(f"    clip_frac      : {ud.get('clip_frac', 0.0):>10.4f}  "
                 f"(ratio outside [{1-0.2:.1f}, {1+0.2:.1f}])")
    lines.append(f"    ratio          : mean={ud.get('ratio_mean', 0.0):>10.4f}  "
                 f"max={ud.get('ratio_max', 0.0):>10.4f}")

    # ── 4. Loss Components ───────────────────────────────────────────
    lines.append("")
    lines.append("  [LOSS COMPONENTS]")
    lines.append(f"    policy_loss    : {ud.get('policy_loss', 0.0):>12.6f}")
    lines.append(f"    value_loss     : {ud.get('value_loss', 0.0):>12.6f}  (xVF_COEF=0.5 -> {ud.get('value_loss', 0.0)*0.5:>12.6f})")
    lines.append(f"    entropy_bonus  : {ud.get('entropy', 0.0):>12.6f}  (xENT_COEF=0.01 -> {ud.get('entropy', 0.0)*0.01:>12.6f})")
    lines.append(f"    total_loss     : {ud.get('total_loss', 0.0):>12.6f}")

    # ── 5. Gradient Health ───────────────────────────────────────────
    lines.append("")
    lines.append("  [GRADIENT HEALTH]")
    lines.append(f"    grad_global_norm: {ud.get('grad_norm', 0.0):>10.6f}  "
                 f"(clipped to {max_grad})")
    if ud.get('skipped_steps', 0):
        lines.append(f"    skipped_steps   : {ud.get('skipped_steps', 0):>10d}  "
                     f"(grad_norm > 10x max_grad)")

    # ── 6. Reward Components ─────────────────────────────────────────
    if 'reward_components' in rd:
        lines.append("")
        lines.append("  [REWARD COMPONENTS (per-step mean)]")
        for name, val in rd['reward_components'].items():
            lines.append(f"    {name:>15s} : {val:>10.4f}")

    # ── 7. Observation Health ────────────────────────────────────────
    lines.append("")
    lines.append("  [OBSERVATION HEALTH]")
    lines.append(f"    proprio        : mean={rd.get('proprio_mean', 0.0):>8.4f}  "
                 f"std={rd.get('proprio_std', 0.0):>8.4f}  "
                 f"nan_frac={rd.get('proprio_nan_frac', 0.0):>8.6f}")
    lines.append(f"    scandots       : mean={rd.get('scandot_mean', 0.0):>8.4f}  "
                 f"std={rd.get('scandot_std', 0.0):>8.4f}  "
                 f"nan_frac={rd.get('scandot_nan_frac', 0.0):>8.6f}")
    if 'adapt_loss' in rd:
        lines.append(f"    adapt_loss     : {rd['adapt_loss']:>8.4f}  "
                     f"(phi regression ||z_hat - z||^2)")

    # ── 8. Flags / Warnings ──────────────────────────────────────────
    warnings = []
    if ev < 0:
        warnings.append("VALUE FUNCTION worse than mean prediction (explained_var < 0)")
    if ud.get('approx_kl', 0) > 0.05:
        warnings.append(f"KL divergence too high ({ud.get('approx_kl', 0):.4f}) -- policy changing too fast")
    if ud.get('action_std_mean', 0) > 5.0:
        warnings.append(f"Action std very high ({ud.get('action_std_mean', 0):.2f}) -- entropy explosion")
    if rd.get('ret_raw_std', 0) < 1e-6:
        warnings.append("Return std near zero -- no learning signal")
    if rd.get('ret_raw_std', 0) > 1000:
        warnings.append(f"Return std very high ({rd.get('ret_raw_std', 0):.1f}) -- reward scale issue")
    if abs(rd.get('adv_mean', 0)) > 1.0:
        warnings.append(f"Advantage mean not centered ({rd.get('adv_mean', 0):.4f})")
    if ud.get('clip_frac', 0) > 0.5:
        warnings.append(f"High clip fraction ({ud.get('clip_frac', 0):.2f}) -- updates too aggressive")
    if ud.get('value_loss', 0) > 100:
        warnings.append(f"Value loss very high ({ud.get('value_loss', 0):.1f}) -- value predictions far off")

    if warnings:
        lines.append("")
        lines.append("  [WARNINGS]")
        for w in warnings:
            lines.append(f"    !! {w}")

    lines.append(f"{'='*72}")
    lines.append("")

    print("\n".join(lines))
