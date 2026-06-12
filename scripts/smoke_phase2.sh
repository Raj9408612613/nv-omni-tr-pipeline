#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 (student/DAgger) smoke test — run on the Isaac Lab machine.
#
#   bash scripts/smoke_phase2.sh                          # random teacher
#   TEACHER_CKPT=omni_logs/<run>/best.pt bash scripts/smoke_phase2.sh
#
# 16 envs, 200 iterations. Checks:
#   * depth renders at the configured rate (~0.2 of policy steps at
#     10 Hz render / 50 Hz policy; resets add a little)
#   * GRU latent persists between renders (asserted in tests/, and the
#     frame-rate column here confirms cadence in-sim)
#   * DAgger loss DECREASES over the run
#   * VRAM printed
#
# Without TEACHER_CKPT a randomly initialized teacher is saved and used —
# that is a fixed labeling function, sufficient to verify the distillation
# machinery end-to-end (loss must still decrease).
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")/.."

NUM_ENVS="${NUM_ENVS:-16}"
ITERS="${ITERS:-200}"
LOG_DIR="omni_logs/smoke_phase2"
mkdir -p "${LOG_DIR}"
rm -f "${LOG_DIR}/SUCCESS"

TEACHER_CKPT="${TEACHER_CKPT:-}"
if [[ -z "${TEACHER_CKPT}" ]]; then
    TEACHER_CKPT="${LOG_DIR}/random_teacher.pt"
    echo "[SETUP] No TEACHER_CKPT given — creating random teacher at ${TEACHER_CKPT}"
    PYTHONPATH=. CKPT_PATH="${TEACHER_CKPT}" ${PYTHON:-python} - <<'EOF'
import os
from omni_spot.checkpoint import save_checkpoint
from omni_spot.configs import get_experiment_cfg
from omni_spot.networks import TeacherPolicy

cfg = get_experiment_cfg("spot")
teacher = TeacherPolicy(cfg)
save_checkpoint(os.environ["CKPT_PATH"], model_state=teacher.state_dict(),
                phase="teacher", robot=cfg.robot.name)
print("[SETUP] random teacher saved")
EOF
fi

PYTHONPATH=. ${PYTHON:-python} -m omni_spot.train \
    --phase student --robot spot \
    --teacher_ckpt "${TEACHER_CKPT}" \
    --num_envs "${NUM_ENVS}" --total_iters "${ITERS}" \
    --log_interval 10 --save_interval "${ITERS}" \
    --log_dir "${LOG_DIR}" --headless

test -f "${LOG_DIR}/SUCCESS" || { echo "[FAIL] no SUCCESS marker"; exit 1; }
echo "[CHECK] SUCCESS marker present"

LOG_DIR="${LOG_DIR}" ${PYTHON:-python} - <<'EOF'
import csv, glob, math, os

log_dir = os.environ["LOG_DIR"]
runs = sorted(glob.glob(f"{log_dir}/student_*/train_log.csv"))
assert runs, f"no train_log.csv under {log_dir}"
with open(runs[-1]) as f:
    rows = list(csv.DictReader(f))
loss = [float(r["dagger_loss"]) for r in rows if r.get("dagger_loss")]
rate = [float(r["depth_new_frame_rate"]) for r in rows
        if r.get("depth_new_frame_rate")]
assert loss and all(math.isfinite(x) for x in loss), "bad dagger_loss column"

k = max(1, len(loss) // 5)
first, last = sum(loss[:k]) / k, sum(loss[-k:]) / k
print(f"[CHECK] dagger_loss first={first:.5f} last={last:.5f} "
      f"(ratio {last / max(first, 1e-12):.3f})")
assert last < 0.8 * first, "DAgger loss did not decrease over the run"

mean_rate = sum(rate) / len(rate)
print(f"[CHECK] depth_new_frame_rate = {mean_rate:.3f} (expected ~0.2)")
assert 0.12 <= mean_rate <= 0.5, "depth render cadence off (expected ~0.2)"
EOF

echo "[PASS] Phase 2 smoke (${NUM_ENVS} envs, ${ITERS} iters) — see [VRAM] lines above for memory usage"
