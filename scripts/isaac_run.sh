#!/usr/bin/env bash
# Native (non-Docker) Isaac Sim + Isaac Lab setup & training. Idempotent.
#   bash scripts/isaac_run.sh          # setup + smoke
#   bash scripts/isaac_run.sh --train  # also launch full teacher run
set -euo pipefail

ENV_NAME="isaac"
PY_VERSION="3.11"
TORCH_CUDA="cu128"                 # Blackwell/driver 580; change if CUDA differs
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

echo ">>> Stage 0: system packages + NVIDIA driver"
sudo apt-get update -qq
sudo apt-get install -y wget curl git build-essential libglu1-mesa
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    echo "    NVIDIA driver active:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | sed 's/^/      /'
else
    echo "    NVIDIA driver not active — installing $NVIDIA_DRIVER ..."
    sudo apt-get install -y "$NVIDIA_DRIVER" nvidia-utils-580
    echo "  Driver installed. REBOOT then re-run:  sudo reboot ; bash $SCRIPT_DIR/isaac_run.sh"
    exit 0
fi

echo ">>> Stage 1: Miniconda + conda env '$ENV_NAME'"
if [ ! -d "$CONDA_DIR" ]; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$CONDA_DIR"; rm -f /tmp/miniconda.sh
else echo "    Miniconda already at $CONDA_DIR"; fi
eval "$("$CONDA_DIR/bin/conda" shell.bash hook)"
"$CONDA_DIR/bin/conda" init bash >/dev/null 2>&1 || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true
if conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then echo "    env '$ENV_NAME' exists"
else conda create -n "$ENV_NAME" python="$PY_VERSION" -y; fi
conda activate "$ENV_NAME"
echo "    active python: $(python --version)"

echo ">>> Stage 2: PyTorch ($TORCH_CUDA)"
pip install "setuptools<75.0.0"
python -c "import torch" 2>/dev/null && echo "    torch present" || \
    pip install torch torchvision --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"
python - <<'PY'
import torch
ok = torch.cuda.is_available()
print(f"    torch {torch.__version__} | cuda={ok} | {torch.cuda.get_device_name(0) if ok else 'NO GPU'}")
PY

echo ">>> Stage 3: ${SWAP_SIZE} swap"
if swapon --show 2>/dev/null | grep -q '/swapfile'; then echo "    swap active"
else
    sudo fallocate -l "$SWAP_SIZE" /swapfile && sudo chmod 600 /swapfile
    sudo mkswap /swapfile && sudo swapon /swapfile
    grep -q '/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
fi

echo ">>> Stage 4: Isaac Sim + Isaac Lab"
python -c "import isaacsim" 2>/dev/null && echo "    isaacsim present" || \
    pip install isaacsim-rl isaacsim-replicator isaacsim-extscache-physics isaacsim-extscache-kit-sdk
pip install "ray[default]==2.45.0"
pip install "setuptools<75.0.0"
[ -d "$ISAACLAB_DIR" ] || git clone https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_DIR"
if python -c "import isaaclab" 2>/dev/null; then echo "    isaaclab present"
else
    pushd "$ISAACLAB_DIR" >/dev/null
    pip install --no-deps -e source/isaaclab
    pip install toml gymnasium==1.2.1 trimesh einops warp-lang prettytable==3.3.0 flatdict
    pip install --use-deprecated=legacy-resolver -e source/isaaclab_assets
    pip install --use-deprecated=legacy-resolver -e source/isaaclab_tasks
    popd >/dev/null
fi
pip install tensorboard "imageio[ffmpeg]" h5py
python -c "import isaacsim; print('    isaacsim OK')" || echo "    isaacsim FAILED"
python -c "import isaaclab; print('    isaaclab OK')" || echo "    isaaclab FAILED"

echo ">>> Stage 5: smoke tests"
cd "$REPO_DIR"
bash scripts/smoke_phase1.sh 2>&1 | tee /tmp/smoke1.log
TRAIN_CMD="PYTHONPATH=. python -m omni_spot.train --phase teacher --robot spot --headless --lr 1e-4"
if $RUN_TRAINING; then eval "$TRAIN_CMD 2>&1 | tee /tmp/teacher_full2.log"
else echo "  Setup done. Train with: $TRAIN_CMD"; fi
echo "=== isaac_run.sh finished $(date) ==="
