#!/usr/bin/env bash
# =============================================================================
# Smoke Test — Validates training pipeline end-to-end
# =============================================================================
# Level 1 (default): MockSpotEnv + PPO — no Omniverse needed
# Level 2 (--full):  Isaac Lab SpotNavEnv + PPO only — skips Level 1
#
# Usage:
#   bash scripts/smoke_test.sh                          # Level 1 (mock env)
#   bash scripts/smoke_test.sh --full                   # Level 2 only (Isaac Lab)
#   NUM_ENVS=8 bash scripts/smoke_test.sh --full        # Level 2 with 8 envs
# =============================================================================
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="$REPO_DIR"

LEVEL="${1:-mock}"
NUM_ENVS="${NUM_ENVS:-64}"
N_STEPS="${N_STEPS:-128}"
UPDATES="${UPDATES:-5}"

echo "=============================================="
echo "  Smoke Test — $(date)"
echo "=============================================="
echo ""

# ── Level 1: Mock Environment (default, skipped with --full) ────────────────
if [[ "$LEVEL" != "--full" ]]; then
    # ── Ensure conda env is active ───────────────────────────────────────────
    if [[ "${CONDA_DEFAULT_ENV:-}" != "isaaclab" ]]; then
        eval "$($HOME/miniconda3/bin/conda shell.bash hook)" 2>/dev/null || true
        conda activate isaaclab 2>/dev/null || true
    fi

    echo ">>> Level 1: MockSpotEnv + PPO ($NUM_ENVS envs, $N_STEPS steps, $UPDATES updates)"
    echo ""

    python -c "
import time, torch
from omni_spot.mock_env import MockSpotEnv
from omni_spot.ppo import PPOTrainer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
env = MockSpotEnv(num_envs=$NUM_ENVS, device=device)
trainer = PPOTrainer(n_envs=$NUM_ENVS, n_steps=$N_STEPS, lr=3e-4, device=device)
obs, _ = env.reset()

total_steps = 0
t_start = time.time()

for i in range($UPDATES):
    t0 = time.time()
    obs, batch, stats = trainer.collect_rollout(env, obs)
    update_info = trainer.update(batch)
    dt = time.time() - t0
    total_steps += $NUM_ENVS * $N_STEPS
    sps = ($NUM_ENVS * $N_STEPS) / dt
    print(f'  Update {i+1}/$UPDATES: rew={stats[\"rew_mean\"]:+.3f}  loss={update_info[\"policy_loss\"]:+.4f}  {sps:.0f} SPS  {dt:.1f}s')

wall = time.time() - t_start
print()
print(f'  Total: {total_steps:,} steps in {wall:.1f}s ({total_steps/wall:.0f} SPS)')
print()
print('  [PASS] Level 1: Mock environment smoke test passed')
"

    if [ $? -ne 0 ]; then
        echo ""
        echo "  [FAIL] Level 1 failed. Fix errors above before proceeding."
        exit 1
    fi

    echo ""
    echo "  Skipping Level 2 (Isaac Lab). Use --full to run it."

# ── Level 2: Full Isaac Lab (--full only) ────────────────────────────────────
else
    echo ">>> Level 2: Isaac Lab SpotNavEnv + PPO ($NUM_ENVS envs, $N_STEPS steps, $UPDATES updates)"
    echo ""

    CUSTOM_IMAGE="isaac-lab-spot:latest"

    if ! sudo docker image inspect "$CUSTOM_IMAGE" &>/dev/null; then
        echo "  [FAIL] Docker image '$CUSTOM_IMAGE' not found. Run setup_ec2_isaac.sh first."
        exit 1
    fi

    if [ ! -f "$REPO_DIR/models/spot_scene.usd" ]; then
        echo "  [FAIL] models/spot_scene.usd not found. Run MJCF→USD conversion first."
        exit 1
    fi

    mkdir -p "$HOME/.isaac_cache/kit" \
             "$HOME/.isaac_cache/ov" \
             "$HOME/.isaac_cache/glcache" \
             "$HOME/.isaac_cache/computecache"

    sudo docker run --rm --gpus all \
        --entrypoint="" \
        --stop-timeout 30 \
        -e "ACCEPT_EULA=Y" \
        -e "PYTHONUNBUFFERED=1" \
        -v "$REPO_DIR":/workspace \
        -v "$HOME/omni_logs":/workspace/omni_logs \
        -v "$HOME/.isaac_cache/kit:/root/.cache/kit" \
        -v "$HOME/.isaac_cache/ov:/root/.nvidia-omniverse" \
        -v "$HOME/.isaac_cache/glcache:/root/.cache/nvidia/GLCache" \
        -v "$HOME/.isaac_cache/computecache:/root/.nv/ComputeCache" \
        "$CUSTOM_IMAGE" \
        /isaac-sim/python.sh -m omni_spot.train \
            --headless \
            --enable_cameras \
            --num_envs $NUM_ENVS \
            --n_steps $N_STEPS \
            --total_updates $UPDATES \
            --log_dir /workspace/smoke_test_output

    DOCKER_EXIT=$?

    if [ -f "$REPO_DIR/smoke_test_output/SUCCESS" ]; then
        echo ""
        echo "  [PASS] Level 2: Isaac Lab smoke test passed (via Docker container)"
        sudo rm -f "$REPO_DIR/smoke_test_output/SUCCESS"
    else
        echo ""
        echo "  [FAIL] Level 2: Isaac Lab smoke test failed (docker exit=$DOCKER_EXIT, no SUCCESS marker)"
        exit 1
    fi
fi

echo ""
echo "=============================================="
echo "  Smoke test complete!"
echo "=============================================="
