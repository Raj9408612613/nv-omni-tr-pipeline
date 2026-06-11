#!/usr/bin/env bash
# =============================================================================
# Smoke Test — Validates the teacher-student pipeline end-to-end
# =============================================================================
# Level 1 (default): CPU unit tests + mock-env PPO — no Omniverse/GPU needed
# Level 2 (--full):  Phase 1 Isaac Lab smoke (delegates to smoke_phase1.sh)
#
# Usage:
#   bash scripts/smoke_test.sh           # Level 1 (pytest/standalone tests)
#   bash scripts/smoke_test.sh --full    # Level 2 (Isaac Lab, 256 envs)
# =============================================================================
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="$REPO_DIR"
cd "$REPO_DIR"

LEVEL="${1:-mock}"

echo "=============================================="
echo "  Smoke Test — $(date)"
echo "=============================================="

if [[ "$LEVEL" != "--full" ]]; then
    if [[ "${CONDA_DEFAULT_ENV:-}" != "isaaclab" ]]; then
        eval "$($HOME/miniconda3/bin/conda shell.bash hook)" 2>/dev/null || true
        conda activate isaaclab 2>/dev/null || true
    fi

    echo ">>> Level 1: CPU unit + mock-env trainer tests (no Isaac needed)"
    if command -v pytest >/dev/null 2>&1; then
        pytest tests/ -v || { echo "  [FAIL] Level 1"; exit 1; }
    else
        ${PYTHON:-python} tests/test_shapes.py || { echo "  [FAIL] Level 1"; exit 1; }
        ${PYTHON:-python} tests/test_ppo_mock.py || { echo "  [FAIL] Level 1"; exit 1; }
        if [ -f tests/test_dagger_mock.py ]; then
            ${PYTHON:-python} tests/test_dagger_mock.py || { echo "  [FAIL] Level 1"; exit 1; }
        fi
    fi
    echo "  [PASS] Level 1"
    echo ""
    echo "  Run 'bash scripts/smoke_test.sh --full' on the Isaac Lab machine"
    echo "  for the Phase 1 sim smoke (or scripts/smoke_phase1.sh directly)."
else
    echo ">>> Level 2: Phase 1 Isaac Lab smoke (256 envs x 50 updates)"
    bash scripts/smoke_phase1.sh || { echo "  [FAIL] Level 2"; exit 1; }
fi

echo ""
echo "=============================================="
echo "  Smoke test complete!"
echo "=============================================="
