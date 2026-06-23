#!/usr/bin/env bash
# =============================================================================
# isaac_run.sh — native (non-Docker) Isaac Sim + Isaac Lab setup & training
# =============================================================================
# One-stop, IDEMPOTENT setup for the omni_spot teacher-student pipeline on a
# fresh Ubuntu + NVIDIA box (tuned for the RTX PRO 6000 Blackwell, 96 GB).
# Re-running is safe — each stage skips work already done.
#
# Stages:
#   0. System packages + NVIDIA driver  (driver install REQUIRES a reboot:
#                                         the script installs it, prints a
#                                         reboot instruction, and exits; you
#                                         reboot and re-run to continue)
#   1. Miniconda + conda env 'isaac' (py3.11)
#   2. PyTorch (CUDA) + sanity check
#   3. 32 GB swap (Isaac Sim shader compile spikes host RAM)
#   4. Isaac Sim + Isaac Lab (pip: --no-deps core + manual deps)
#   5. Smoke tests + (optional) full teacher training
#
# Usage:
#   bash scripts/isaac_run.sh            # setup + smoke; then print train cmd
#   bash scripts/isaac_run.sh --train    # also launch the full teacher run
#
# ASSUMPTIONS (edit the tunables below if any are wrong):
#   * GPU is Blackwell  -> PyTorch wheels = cu128. Change TORCH_CUDA otherwise
#     (cu124 / cu121 / ... ) to match your driver's CUDA version.
#   * conda env is 'isaac' (your current shell). The old notes said 'isaaclab'
#     in one place — standardised to 'isaac' here.
#   * The training repo is the parent directory of this script.
# =============================================================================
set -euo pipefail

# ── Tunables ─────────────────────────────────────────────────────────────────
ENV_NAME="isaac"
PY_VERSION="3.11"
TORCH_CUDA="cu128"                 # Blackwell / driver 580. Adjust if needed.
TORCH_INDEX="https://download.pytorch.org/whl/${TORCH_CUDA}"
TORCH_NIGHTLY_INDEX="https://download.pytorch.org/whl/nightly/${TORCH_CUDA}"
# Compute capability the GPU needs in torch.cuda.get_arch_list(). Blackwell
# (RTX PRO 6000) is sm_120 — stable cu128 wheels have lacked it, so we verify
# and fall back to nightly. Set to "" to skip the arch check on other GPUs.
TORCH_REQUIRED_ARCH="sm_120"
NVIDIA_DRIVER="nvidia-driver-580-open"
SWAP_SIZE="32G"
CONDA_DIR="$HOME/miniconda3"
ISAACLAB_DIR="$HOME/IsaacLab"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$HOME/isaac_run_setup.log"

RUN_TRAINING=false
[[ "${1:-}" == "--train" ]] && RUN_TRAINING=true

exec > >(tee -a "$LOG_FILE") 2>&1
echo "=== isaac_run.sh started $(date) | repo=$REPO_DIR | env=$ENV_NAME ==="

# ── Torch helpers (Blackwell sm_120 needs a recent cu128 / nightly build) ────
torch_arch_ok() {
    # exit 0 if torch imports AND (no required arch OR it's in the arch list)
    python - "${TORCH_REQUIRED_ARCH:-}" <<'PY' 2>/dev/null
import sys
req = sys.argv[1] if len(sys.argv) > 1 else ""
try:
    import torch
except Exception:
    sys.exit(1)
sys.exit(0 if (not req or req in torch.cuda.get_arch_list()) else 2)
PY
}

ensure_torch() {
    # Guarantee an installed torch whose kernels cover TORCH_REQUIRED_ARCH.
    if torch_arch_ok; then
        echo "    torch OK: $(python -c 'import torch;print(torch.__version__)') | arch=$(python -c 'import torch;print(torch.cuda.get_arch_list())' 2>/dev/null)"
        return 0
    fi
    # torch ONLY — the pipeline never imports torchvision, and on nightly
    # torch+torchvision pin each other (ResolutionImpossible). Uninstall any
    # stray torchvision so it can't drag torch to a non-Blackwell build.
    echo "    [torch] need ${TORCH_REQUIRED_ARCH:-any}; (re)installing stable ${TORCH_CUDA}..."
    pip uninstall -y torch torchvision >/dev/null 2>&1 || true
    pip install torch --index-url "$TORCH_INDEX" || true
    if ! torch_arch_ok; then
        echo "    [torch] stable ${TORCH_CUDA} lacks ${TORCH_REQUIRED_ARCH}; trying nightly ${TORCH_CUDA}..."
        pip uninstall -y torch >/dev/null 2>&1 || true
        pip install --pre torch --index-url "$TORCH_NIGHTLY_INDEX"
    fi
    torch_arch_ok || echo "    [torch][WARN] ${TORCH_REQUIRED_ARCH} still missing — check pytorch.org for a Blackwell build"
}

# =============================================================================
# Stage 0 — system packages + NVIDIA driver (reboot gate)
# =============================================================================
echo ">>> Stage 0: system packages + NVIDIA driver"
sudo apt-get update -qq
sudo apt-get install -y wget curl git build-essential libglu1-mesa

if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    echo "    NVIDIA driver active:"
    nvidia-smi --query-gpu=name,driver_version,memory.total \
        --format=csv,noheader | sed 's/^/      /'
else
    echo "    NVIDIA driver not active — installing $NVIDIA_DRIVER ..."
    sudo apt-get install -y "$NVIDIA_DRIVER" nvidia-utils-580
    echo "==================================================================="
    echo "  Driver installed. A REBOOT is required before the GPU is usable:"
    echo "      sudo reboot"
    echo "  After reboot, re-run this script to continue:"
    echo "      bash $SCRIPT_DIR/isaac_run.sh"
    echo "==================================================================="
    exit 0
fi

# =============================================================================
# Stage 1 — Miniconda + conda env
# =============================================================================
echo ">>> Stage 1: Miniconda + conda env '$ENV_NAME'"
if [ ! -d "$CONDA_DIR" ]; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$CONDA_DIR"
    rm -f /tmp/miniconda.sh
else
    echo "    Miniconda already at $CONDA_DIR"
fi

eval "$("$CONDA_DIR/bin/conda" shell.bash hook)"
"$CONDA_DIR/bin/conda" init bash >/dev/null 2>&1 || true
conda tos accept --override-channels \
    --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels \
    --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

if conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
    echo "    conda env '$ENV_NAME' already exists"
else
    conda create -n "$ENV_NAME" python="$PY_VERSION" -y
fi
conda activate "$ENV_NAME"
echo "    active python: $(python --version)"

# =============================================================================
# Stage 2 — PyTorch (CUDA) + sanity check
# =============================================================================
echo ">>> Stage 2: PyTorch ($TORCH_CUDA, need arch ${TORCH_REQUIRED_ARCH:-any}) + build tooling"
pip install "setuptools<75.0.0"
ensure_torch
python - <<'PY'
import torch
ok = torch.cuda.is_available()
name = torch.cuda.get_device_name(0) if ok else "NO GPU VISIBLE"
print(f"    torch {torch.__version__} | cuda_available={ok} | {name}")
print(f"    arch_list: {torch.cuda.get_arch_list()}")
PY

# =============================================================================
# Stage 3 — swap (Isaac Sim shader compile is RAM-hungry)
# =============================================================================
echo ">>> Stage 3: ${SWAP_SIZE} swap"
if swapon --show 2>/dev/null | grep -q '/swapfile'; then
    echo "    /swapfile already active"
else
    sudo fallocate -l "$SWAP_SIZE" /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    grep -q '/swapfile' /etc/fstab \
        || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
fi

# =============================================================================
# Stage 4 — Isaac Sim + Isaac Lab
# =============================================================================
echo ">>> Stage 4: Isaac Sim + Isaac Lab"
if python -c "import isaacsim" 2>/dev/null; then
    echo "    isaacsim already importable"
else
    pip install isaacsim-rl isaacsim-replicator \
        isaacsim-extscache-physics isaacsim-extscache-kit-sdk
fi
pip install "ray[default]==2.45.0"
pip install "setuptools<75.0.0"        # re-pin: some installs bump it back up

if [ ! -d "$ISAACLAB_DIR" ]; then
    git clone https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_DIR"
else
    echo "    IsaacLab already cloned at $ISAACLAB_DIR"
fi

if python -c "import isaaclab" 2>/dev/null; then
    echo "    isaaclab already importable"
else
    pushd "$ISAACLAB_DIR" >/dev/null
    pip install --no-deps -e source/isaaclab
    pip install toml gymnasium==1.2.1 trimesh einops warp-lang \
        prettytable==3.3.0 flatdict
    pip install --use-deprecated=legacy-resolver -e source/isaaclab_assets
    pip install --use-deprecated=legacy-resolver -e source/isaaclab_tasks
    popd >/dev/null
fi
pip install tensorboard "imageio[ffmpeg]" h5py

echo "    import check:"
python -c "import isaacsim; print('      isaacsim OK')" || echo "      isaacsim FAILED"
python -c "import isaaclab; print('      isaaclab OK')" || echo "      isaaclab FAILED"

# Isaac Sim's deps can swap torch for a build without sm_120 — repair if so.
echo ">>> Re-verifying torch arch after Isaac install"
ensure_torch

# =============================================================================
# Stage 5 — smoke tests + (optional) full teacher training
# =============================================================================
echo ">>> Stage 5: smoke tests"
cd "$REPO_DIR"

bash scripts/smoke_phase1.sh 2>&1 | tee /tmp/smoke1.log
grep -q "\[PASS\]" /tmp/smoke1.log \
    && echo "    smoke_phase1: PASS" \
    || echo "    smoke_phase1: NO [PASS] — inspect /tmp/smoke1.log"

bash scripts/smoke_phase1.sh --probe32k 2>&1 | tee /tmp/probe.log
grep -q "\[PASS\]" /tmp/probe.log \
    && echo "    probe32k: PASS" \
    || echo "    probe32k: NO [PASS] — inspect /tmp/probe.log"
PHYSX_ERRS=$(grep -c "PhysX error" /tmp/probe.log || true)
echo "    PhysX errors in probe: ${PHYSX_ERRS} (must be 0)"

TRAIN_CMD="PYTHONPATH=. python -m omni_spot.train --phase teacher --robot spot --headless --lr 1e-4"
if $RUN_TRAINING; then
    echo ">>> Launching full teacher training (this runs for hours)..."
    eval "$TRAIN_CMD 2>&1 | tee /tmp/teacher_full2.log"
else
    echo "==================================================================="
    echo "  Setup + smoke complete. To launch the full teacher run:"
    echo "      conda activate $ENV_NAME && cd $REPO_DIR"
    echo "      $TRAIN_CMD 2>&1 | tee /tmp/teacher_full2.log"
    echo "  (or re-run this script with --train to do it automatically)"
    echo "==================================================================="
fi
echo "=== isaac_run.sh finished $(date) | full log: $LOG_FILE ==="
