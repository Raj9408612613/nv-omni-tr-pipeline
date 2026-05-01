#!/usr/bin/env bash
# =============================================================================
# Isaac Sim + IsaacLab Setup Verification
# =============================================================================
# Checks all installed components are working.
# Run after setup_ec2_isaac.sh completes.
#
# Usage: bash scripts/verify_setup.sh
# =============================================================================
set -u

PASS=0
FAIL=0
WARN=0

pass()  { echo "  [PASS] $1"; ((PASS++)); }
fail()  { echo "  [FAIL] $1"; ((FAIL++)); }
warn()  { echo "  [WARN] $1"; ((WARN++)); }

echo "=============================================="
echo "  Isaac Sim + IsaacLab Verification"
echo "  $(date)"
echo "=============================================="
echo ""

# ── Ensure conda env is active ──────────────────────────────────────────────
if [[ "${CONDA_DEFAULT_ENV:-}" != "isaaclab" ]]; then
    echo ">>> Activating isaaclab conda env..."
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)" 2>/dev/null || true
    conda activate isaaclab 2>/dev/null || true
fi

# ── Detect repo root ────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

echo "--- Hardware & Drivers ---"

# Test 1: NVIDIA GPU
echo -n "  1. NVIDIA GPU present... "
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
    DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    pass "$GPU_NAME, ${GPU_MEM}, driver $DRIVER"
else
    fail "nvidia-smi not found or failed"
fi

# Test 2: Vulkan
echo -n "  2. Vulkan support... "
if command -v vulkaninfo &>/dev/null && vulkaninfo --summary 2>/dev/null | grep -q "GPU"; then
    pass "Vulkan available"
else
    warn "Vulkan not detected (may affect rendering, not required for headless training)"
fi

echo ""
echo "--- Python & PyTorch ---"

# Test 3: Python version
echo -n "  3. Python version... "
PY_VER=$(python --version 2>&1)
if echo "$PY_VER" | grep -q "3.11"; then
    pass "$PY_VER"
elif echo "$PY_VER" | grep -q "3.10\|3.12"; then
    warn "$PY_VER (expected 3.11 for Isaac Sim 5.x)"
else
    fail "$PY_VER (incompatible — need 3.10-3.12)"
fi

# Test 4: PyTorch + CUDA
echo -n "  4. PyTorch CUDA... "
CUDA_OK=$(python -c "
import torch
if torch.cuda.is_available():
    print(f'torch {torch.__version__}, CUDA {torch.version.cuda}, {torch.cuda.get_device_name(0)}')
else:
    print('NO_CUDA')
" 2>&1)
if echo "$CUDA_OK" | grep -q "NO_CUDA"; then
    fail "PyTorch CUDA not available"
else
    pass "$CUDA_OK"
fi

echo ""
echo "--- Isaac Sim & IsaacLab ---"

# Test 5: Isaac Sim importable
echo -n "  5. Isaac Sim import... "
ISAACSIM_OK=$(python -c "import isaacsim; print('OK')" 2>&1)
if [[ "$ISAACSIM_OK" == "OK" ]]; then
    pass "isaacsim importable"
else
    fail "import isaacsim failed: $ISAACSIM_OK"
fi

# Test 6: IsaacLab importable
echo -n "  6. IsaacLab import... "
ISAACLAB_OK=$(python -c "import isaaclab; print('OK')" 2>&1)
if [[ "$ISAACLAB_OK" == "OK" ]]; then
    pass "isaaclab importable"
else
    fail "import isaaclab failed: $ISAACLAB_OK"
fi

echo ""
echo "--- ML Libraries ---"

# Test 7: ray
echo -n "  7. Ray... "
RAY_OK=$(python -c "import ray; print(f'ray {ray.__version__}')" 2>&1)
if echo "$RAY_OK" | grep -q "ray "; then
    pass "$RAY_OK"
else
    warn "ray not installed (needed for rl_games, not for basic training)"
fi

# Test 8: gymnasium
echo -n "  8. Gymnasium... "
GYM_OK=$(python -c "import gymnasium; print(f'gymnasium {gymnasium.__version__}')" 2>&1)
if echo "$GYM_OK" | grep -q "gymnasium "; then
    pass "$GYM_OK"
else
    fail "gymnasium not installed: $GYM_OK"
fi

# Test 9: tensorboard
echo -n "  9. TensorBoard... "
TB_OK=$(python -c "import tensorboard; print(f'tensorboard {tensorboard.__version__}')" 2>&1)
if echo "$TB_OK" | grep -q "tensorboard "; then
    pass "$TB_OK"
else
    warn "tensorboard not installed (optional for logging)"
fi

echo ""
echo "--- Project Pipeline (No Omniverse Required) ---"

# Test 10: omni_spot imports
echo -n "  10. omni_spot module... "
OMNI_OK=$(PYTHONPATH="$REPO_DIR" python -c "
from omni_spot.config import ACTION_DIM, PROPRIO_DIM, CNN_FEAT_DIM
print(f'OK (action={ACTION_DIM}, proprio={PROPRIO_DIM}, cnn={CNN_FEAT_DIM})')
" 2>&1)
if echo "$OMNI_OK" | grep -q "OK"; then
    pass "$OMNI_OK"
else
    fail "omni_spot import failed: $OMNI_OK"
fi

# Test 11: MockSpotEnv instantiation
echo -n "  11. MockSpotEnv... "
MOCK_OK=$(PYTHONPATH="$REPO_DIR" python -c "
import torch
from omni_spot.mock_env import MockSpotEnv
env = MockSpotEnv(num_envs=4, device='cuda' if torch.cuda.is_available() else 'cpu')
obs, info = env.reset()
print(f'OK depth={list(obs[\"depth\"].shape)} proprio={list(obs[\"proprio\"].shape)}')
" 2>&1)
if echo "$MOCK_OK" | grep -q "OK"; then
    pass "$MOCK_OK"
else
    fail "MockSpotEnv failed: $MOCK_OK"
fi

# Test 12: PPOTrainer + 2-step rollout
echo -n "  12. PPO 2-step rollout... "
PPO_OK=$(PYTHONPATH="$REPO_DIR" python -c "
import torch
from omni_spot.mock_env import MockSpotEnv
from omni_spot.ppo import PPOTrainer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
env = MockSpotEnv(num_envs=4, device=device)
trainer = PPOTrainer(n_envs=4, n_steps=2, lr=3e-4, device=device)
obs, info = env.reset()
obs, batch, stats = trainer.collect_rollout(env, obs)
update_info = trainer.update(batch)
print(f'OK rew_mean={stats[\"rew_mean\"]:.4f} policy_loss={update_info[\"policy_loss\"]:.4f}')
" 2>&1)
if echo "$PPO_OK" | grep -q "OK"; then
    pass "$PPO_OK"
else
    fail "PPO rollout failed: $PPO_OK"
fi

# Test 13: SpotActorCritic network
echo -n "  13. SpotActorCritic network... "
NET_OK=$(PYTHONPATH="$REPO_DIR" python -c "
import torch
from omni_spot.spot_actor_critic import SpotActorCritic
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = SpotActorCritic().to(device)
depth = torch.randn(2, 5, 120, 160, device=device)
proprio = torch.randn(2, 37, device=device)
mean, log_std, value = net(depth, proprio)
params = sum(p.numel() for p in net.parameters())
print(f'OK params={params:,} action={list(mean.shape)} value={list(value.shape)}')
" 2>&1)
if echo "$NET_OK" | grep -q "OK"; then
    pass "$NET_OK"
else
    fail "Network test failed: $NET_OK"
fi

echo ""
echo "--- Asset Files ---"

# Test 14: MJCF model
echo -n "  14. MJCF model... "
if [ -f "$REPO_DIR/models/spot_scene.xml" ]; then
    MESH_COUNT=$(ls "$REPO_DIR/models/assets/"*.obj 2>/dev/null | wc -l)
    pass "spot_scene.xml found ($MESH_COUNT mesh files)"
else
    fail "models/spot_scene.xml not found"
fi

# Test 15: USD model (generated from MJCF)
echo -n "  15. USD model... "
if [ -f "$REPO_DIR/models/spot_scene.usd" ]; then
    pass "spot_scene.usd found"
else
    warn "spot_scene.usd not found (MJCF→USD conversion still needed for full Isaac Lab training)"
fi

echo ""
echo "--- Docker & Containers ---"

# Test 16: Docker
echo -n "  16. Docker... "
if command -v docker &>/dev/null; then
    DOCKER_VER=$(docker --version 2>/dev/null | awk '{print $3}' | tr -d ',')
    pass "Docker $DOCKER_VER"
else
    fail "Docker not installed"
fi

# Test 17: NVIDIA Container Toolkit
echo -n "  17. NVIDIA Container Toolkit... "
if dpkg -l nvidia-container-toolkit &>/dev/null 2>&1; then
    pass "installed"
else
    fail "nvidia-container-toolkit not installed"
fi

# Test 18: GPU visible in Docker
echo -n "  18. GPU in Docker... "
DOCKER_GPU=$(sudo docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu22.04 nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)
if [ -n "$DOCKER_GPU" ]; then
    pass "$DOCKER_GPU"
else
    fail "GPU not accessible in Docker containers"
fi

# Test 19: Isaac Sim container
echo -n "  19. Isaac Sim container... "
if sudo docker image inspect nvcr.io/nvidia/isaac-sim:4.5.0 &>/dev/null; then
    pass "nvcr.io/nvidia/isaac-sim:4.5.0 pulled"
else
    fail "Isaac Sim container not pulled"
fi

# Test 20: Custom isaac-lab-spot image
echo -n "  20. isaac-lab-spot image... "
if sudo docker image inspect isaac-lab-spot:latest &>/dev/null; then
    pass "isaac-lab-spot:latest built"
else
    fail "isaac-lab-spot:latest not built (run setup_ec2_isaac.sh)"
fi

echo ""
echo "--- Remote Desktop ---"

# Test 21: DCV server
echo -n "  21. NICE DCV... "
if command -v dcv &>/dev/null && sudo systemctl is-active dcvserver &>/dev/null 2>&1; then
    pass "DCV server running"
elif command -v dcv &>/dev/null; then
    warn "DCV installed but not running"
else
    warn "DCV not installed (optional — only needed for GUI)"
fi

# ── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo "=============================================="
TOTAL=$((PASS + FAIL + WARN))
echo "  Results: $PASS passed, $FAIL failed, $WARN warnings (of $TOTAL checks)"
echo "=============================================="

if [ "$FAIL" -eq 0 ]; then
    echo ""
    echo "  All critical checks passed!"
    echo "  Next: bash scripts/smoke_test.sh"
else
    echo ""
    echo "  Some checks failed. Review output above."
    exit 1
fi
