#!/usr/bin/env bash
# =============================================================================
# Run commands inside the Isaac Sim + IsaacLab Docker container
# =============================================================================
# Usage:
#   bash scripts/isaac_run.sh                          # Interactive shell
#   bash scripts/isaac_run.sh python -m omni_spot.train --num_envs 64
#   bash scripts/isaac_run.sh bash scripts/smoke_test_full.sh
#
# The project repo is mounted at /workspace inside the container.
# All Isaac Sim / IsaacLab / Omniverse APIs are available via /isaac-sim/python.sh
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
IMAGE_NAME="isaac-lab-spot:latest"

# Check Docker is available
if ! command -v docker &>/dev/null; then
    echo "ERROR: Docker not installed. Run setup_ec2_isaac.sh first."
    exit 1
fi

# Check image exists
if ! sudo docker image inspect "$IMAGE_NAME" &>/dev/null; then
    echo "ERROR: Docker image '$IMAGE_NAME' not found."
    echo "Run: cd ~/claude_code && sudo docker build -t $IMAGE_NAME -f Dockerfile.isaac ."
    exit 1
fi

# Persist Isaac Sim shader/kit/compute caches across container runs
mkdir -p "$HOME/.isaac_cache/kit" \
         "$HOME/.isaac_cache/ov" \
         "$HOME/.isaac_cache/glcache" \
         "$HOME/.isaac_cache/computecache"

CACHE_VOLUMES=(
    -v "$HOME/.isaac_cache/kit:/root/.cache/kit"
    -v "$HOME/.isaac_cache/ov:/root/.nvidia-omniverse"
    -v "$HOME/.isaac_cache/glcache:/root/.cache/nvidia/GLCache"
    -v "$HOME/.isaac_cache/computecache:/root/.nv/ComputeCache"
)

# If no command given, start interactive shell
if [ $# -eq 0 ]; then
    echo "Starting interactive Isaac Sim shell..."
    echo "  Use /isaac-sim/python.sh for Isaac Sim Python"
    echo "  Project mounted at /workspace"
    sudo docker run --rm -it --gpus all \
        --entrypoint="" \
        --shm-size=32g \
        -e "ACCEPT_EULA=Y" \
        -v "$REPO_DIR":/workspace \
        -v "$HOME/omni_logs":/workspace/omni_logs \
        "${CACHE_VOLUMES[@]}" \
        "$IMAGE_NAME" \
        /bin/bash
else
    # Run the given command
    # Replace 'python' with '/isaac-sim/python.sh' for Isaac Sim compatibility
    CMD="$@"
    if [[ "$1" == "python" ]]; then
        shift
        sudo docker run --rm --gpus all \
            --entrypoint="" \
            --shm-size=32g \
            -e "ACCEPT_EULA=Y" \
            -e "PYTHONUNBUFFERED=1" \
            -v "$REPO_DIR":/workspace \
            -v "$HOME/omni_logs":/workspace/omni_logs \
            "${CACHE_VOLUMES[@]}" \
            "$IMAGE_NAME" \
            /isaac-sim/python.sh "$@"
    else
        sudo docker run --rm --gpus all \
            --entrypoint="" \
            --shm-size=32g \
            -e "ACCEPT_EULA=Y" \
            -e "PYTHONUNBUFFERED=1" \
            -v "$REPO_DIR":/workspace \
            -v "$HOME/omni_logs":/workspace/omni_logs \
            "${CACHE_VOLUMES[@]}" \
            "$IMAGE_NAME" \
            "$@"
    fi
fi
