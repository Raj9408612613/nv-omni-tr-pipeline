#!/usr/bin/env bash
# Stage-4-only installer: Isaac Sim + Isaac Lab into the ACTIVE conda env.
# Use when stages 0-3 (driver/conda/torch) are already done.
#   bash scripts/install_isaaclab.sh [path-to-IsaacLab-clone]
# No fragile line continuations; every pip command is a single line.
set -euo pipefail

ISAACLAB_DIR="${1:-}"
if [ -z "$ISAACLAB_DIR" ]; then
    for d in "$HOME/IsaacLab" "$(pwd)/IsaacLab"; do
        if [ -d "$d" ]; then ISAACLAB_DIR="$d"; break; fi
    done
fi
if [ -z "$ISAACLAB_DIR" ] || [ ! -d "$ISAACLAB_DIR" ]; then
    ISAACLAB_DIR="$HOME/IsaacLab"
    git clone https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_DIR"
fi
echo ">>> Using IsaacLab at: $ISAACLAB_DIR"

if python -c "import isaacsim" 2>/dev/null; then
    echo ">>> isaacsim already importable"
else
    pip install isaacsim-rl isaacsim-replicator isaacsim-extscache-physics isaacsim-extscache-kit-sdk
fi

pip install --no-deps -e "$ISAACLAB_DIR/source/isaaclab"
pip install toml gymnasium==1.2.1 trimesh einops warp-lang prettytable==3.3.0 flatdict
pip install --use-deprecated=legacy-resolver -e "$ISAACLAB_DIR/source/isaaclab_assets"
pip install --use-deprecated=legacy-resolver -e "$ISAACLAB_DIR/source/isaaclab_tasks"
pip install tensorboard "imageio[ffmpeg]" h5py

python -c "import isaacsim; print('isaacsim OK')"
python -c "import isaaclab;  print('isaaclab OK')"
echo ">>> Isaac Lab install complete."
