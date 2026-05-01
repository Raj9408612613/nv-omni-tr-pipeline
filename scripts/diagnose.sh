#!/usr/bin/env bash
# =============================================================================
# Diagnostics — shows what Isaac Sim / training is doing right now
# =============================================================================
# Usage: bash scripts/diagnose.sh
# =============================================================================

echo "=============================================="
echo "  Isaac Sim Diagnostics — $(date)"
echo "=============================================="

# --- Docker containers ---
echo ""
echo "--- Running Docker Containers ---"
sudo docker ps --format "table {{.ID}}\t{{.Image}}\t{{.Status}}\t{{.Names}}" 2>/dev/null
if [ $? -ne 0 ] || [ -z "$(sudo docker ps -q 2>/dev/null)" ]; then
    echo "  No containers running"
fi

# --- GPU processes ---
echo ""
echo "--- GPU Processes (what is using VRAM) ---"
nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv,noheader 2>/dev/null || echo "  No GPU processes found"

# --- Full nvidia-smi ---
echo ""
echo "--- GPU State ---"
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw \
    --format=csv,noheader 2>/dev/null

# --- What's writing to disk ---
echo ""
echo "--- Top Disk Writers (lsof) ---"
sudo lsof 2>/dev/null | awk '$4 ~ /[0-9]+w/ {print $1, $2, $9}' | sort -u | head -20

# --- Smoke test log tail ---
echo ""
echo "--- Last 10 lines of smoke_test.log ---"
if [ -f ~/smoke_test.log ]; then
    tail -10 ~/smoke_test.log
else
    echo "  ~/smoke_test.log not found"
fi

# --- omni_logs ---
echo ""
echo "--- Last 20 lines of omni_logs (if any) ---"
LATEST_LOG=$(ls -t ~/omni_logs/*.log 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
    echo "  File: $LATEST_LOG"
    tail -20 "$LATEST_LOG"
else
    echo "  No omni_logs found"
fi

# --- Python/training processes ---
echo ""
echo "--- Running Python/Training Processes ---"
ps aux | grep -E "python|train|isaac|omni_spot" | grep -v grep

# --- Disk I/O source ---
echo ""
echo "--- Disk I/O Summary ---"
iostat -x 1 1 2>/dev/null | grep -E "nvme|Device" | head -5

echo ""
echo "=============================================="
echo "  Done"
echo "=============================================="
