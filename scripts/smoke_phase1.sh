#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 (teacher) smoke test — run on the Isaac Lab machine.
#
#   bash scripts/smoke_phase1.sh              # 256 envs x 50 updates
#   bash scripts/smoke_phase1.sh --probe32k   # 32768 envs x 3 updates (VRAM probe)
#
# Checks:
#   * training completes (SUCCESS marker)
#   * rew_mean is finite and NON-FLAT across updates (skipped for the probe)
#   * VRAM is printed by train.py after reset / update 1 / end
# Expected output ends with "[PASS] ..."
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")/.."

NUM_ENVS=256
UPDATES=50
TAG=smoke_phase1
CHECK_FLAT=1
if [[ "${1:-}" == "--probe32k" ]]; then
    NUM_ENVS=32768
    UPDATES=3
    TAG=probe32k_phase1
    CHECK_FLAT=0
fi
LOG_DIR="omni_logs/${TAG}"
rm -f "${LOG_DIR}/SUCCESS"

PYTHONPATH=. ${PYTHON:-python} -m omni_spot.train \
    --phase teacher --robot spot \
    --num_envs "${NUM_ENVS}" --total_updates "${UPDATES}" \
    --save_interval "${UPDATES}" \
    --log_dir "${LOG_DIR}" --headless

test -f "${LOG_DIR}/SUCCESS" || { echo "[FAIL] no SUCCESS marker"; exit 1; }
echo "[CHECK] SUCCESS marker present"

CHECK_FLAT="${CHECK_FLAT}" LOG_DIR="${LOG_DIR}" ${PYTHON:-python} - <<'EOF'
import csv, glob, math, os, sys

log_dir = os.environ["LOG_DIR"]
check_flat = os.environ["CHECK_FLAT"] == "1"
runs = sorted(glob.glob(f"{log_dir}/teacher_*/train_log.csv"))
assert runs, f"no train_log.csv under {log_dir}"
with open(runs[-1]) as f:
    rows = list(csv.DictReader(f))
rews = [float(r["rew_mean"]) for r in rows if r.get("rew_mean")]
assert rews, "empty training log"
assert all(math.isfinite(x) for x in rews), f"non-finite rew_mean in {runs[-1]}"
lo, hi = min(rews), max(rews)
print(f"[CHECK] {len(rews)} updates, rew_mean in [{lo:.3f}, {hi:.3f}] — finite")
if check_flat:
    mean = sum(rews) / len(rews)
    var = sum((x - mean) ** 2 for x in rews) / len(rews)
    assert var > 1e-12, f"reward is FLAT across {len(rews)} updates"
    print("[CHECK] reward is non-flat")
adapt = [float(r["adapt_loss"]) for r in rows if r.get("adapt_loss")]
if adapt:
    print(f"[CHECK] adapt_loss first={adapt[0]:.4f} last={adapt[-1]:.4f}")
EOF

echo "[PASS] Phase 1 smoke (${NUM_ENVS} envs, ${UPDATES} updates) — see [VRAM] lines above for memory usage"
