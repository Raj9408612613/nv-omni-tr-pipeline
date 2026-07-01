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
TORCH_VERSION="2.7.0"              # MUST match Isaac Lab's pin. The cu128 build
                                   # of this version is what carries sm_120
                                   # (Blackwell) kernels; the default-index
                                   # cu126 build of the SAME version does not.
TORCHVISION_VERSION="0.22.0"       # pairs with torch 2.7.0
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
echo ">>> Stage 2: PyTorch ($TORCH_CUDA) + build tooling"
pip install "setuptools<75.0.0"

# (Re)install torch if it is missing OR if the installed build lacks compiled
# kernels for THIS GPU's compute arch. The latter is the classic Blackwell
# trap: `pip install torch` (default index) gives a CUDA 12.6 build whose
# kernels stop at sm_90, so on an sm_120 card it imports fine but dies at
# runtime with "no kernel image is available for execution on the device".
# We compare the device capability against torch's compiled arch list — both
# are static/property queries that do NOT launch a kernel, so the check is
# safe even on a mismatched build. A plain `import torch` check is NOT enough
# (pip matches on version, not CUDA build, so a wrong-arch torch slips by).
torch_arch_ok() {
    python - <<'PY' 2>/dev/null
import sys
try:
    import torch
    cap = torch.cuda.get_device_capability()       # e.g. (12, 0) on Blackwell
except Exception:
    sys.exit(1)
arch = f"sm_{cap[0]}{cap[1]}"
sys.exit(0 if arch in torch.cuda.get_arch_list() else 1)
PY
}

# Install the pinned, GPU-arch-correct torch ONLY when the current install is
# missing or built for the wrong arch. Idempotent: a no-op once the right wheel
# is in place, so re-running the script does NOT redownload/reinstall torch
# every time. No --no-cache-dir, so the rare reinstall reuses the pip cache
# instead of pulling the ~3 GB wheel again.
ensure_torch() {
    if torch_arch_ok; then
        echo "    torch OK — present and has kernels for this GPU's compute arch"
    else
        echo "    torch missing or wrong CUDA build — installing torch==$TORCH_VERSION ($TORCH_CUDA) ..."
        pip install --force-reinstall \
            "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
            --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"
        # If $TORCH_CUDA stable still lacks your arch (very new GPU), switch to
        # the matching nightly index, e.g. .../whl/nightly/cu128, and re-run.
    fi
}
ensure_torch
python - <<'PY'
import torch
ok = torch.cuda.is_available()
name = torch.cuda.get_device_name(0) if ok else "NO GPU VISIBLE"
print(f"    torch {torch.__version__} | cuda_available={ok} | {name}")
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

# Isaac Sim / Isaac Lab declare their own torch dependency and, when they (re)
# install, can silently swap our cu128 wheel for a default-index cu126 build
# that has NO sm_120 kernels — the exact "no kernel image" trap. Re-assert the
# correct build now, AFTER those installs, so torch has the last word. This is
# why earlier runs seemed to "reinstall every time": Stage 2 fixed torch, then
# Stage 4 clobbered it. Idempotent once stable.
echo ">>> Re-verifying torch CUDA arch after Isaac install"
ensure_torch

echo "    import check:"
python -c "import isaacsim; print('      isaacsim OK')" || echo "      isaacsim FAILED"
python -c "import isaaclab; print('      isaaclab OK')" || echo "      isaaclab FAILED"

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
