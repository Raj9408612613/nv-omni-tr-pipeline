#!/usr/bin/env bash
# =============================================================================
# Download required asset files for Spot Isaac Sim training
# =============================================================================
# Usage:
#   bash scripts/download_assets.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$REPO_DIR/models"

echo "=============================================="
echo "  Downloading Spot Assets"
echo "  $(date)"
echo "=============================================="
echo ""

mkdir -p "$MODELS_DIR"

# -----------------------------------------------------------------------------
# Spot USD (Omniverse official — Boston Dynamics Spot for Isaac Sim 4.5)
# -----------------------------------------------------------------------------
SPOT_USD="$MODELS_DIR/spot_omniverse.usd"
if [ -f "$SPOT_USD" ]; then
    echo "  [SKIP] spot_omniverse.usd already exists"
else
    echo "  [DOWNLOAD] spot_omniverse.usd ..."
    wget -q --show-progress \
        -O "$SPOT_USD" \
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Robots/BostonDynamics/spot/spot.usd"
    echo "  [DONE] spot_omniverse.usd -> $SPOT_USD"
fi

echo ""
echo "=============================================="
echo "  Asset download complete!"
echo "  Files saved to: $MODELS_DIR"
echo "=============================================="
