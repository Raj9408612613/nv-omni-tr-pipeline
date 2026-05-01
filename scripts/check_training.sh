#!/usr/bin/env bash
# =============================================================================
# Check if Isaac Lab training is progressing or hanging
# Usage: bash scripts/check_training.sh
# =============================================================================

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_DIR="$HOME/.isaac_cache"

echo "=============================================="
echo "  Training Health Check — $(date)"
echo "=============================================="

# ── 1. Is the Docker container still running? ──────────────────────────────
echo ""
echo ">>> [1] Docker container"
CONTAINER=$(sudo docker ps --format "table {{.ID}}\t{{.Status}}\t{{.CreatedAt}}\t{{.Names}}" \
    | grep "isaac-lab-spot" | head -1)
if [ -z "$CONTAINER" ]; then
    echo "  [DEAD] No isaac-lab-spot container is running — training has stopped."
else
    echo "  [ALIVE] $CONTAINER"
fi

# ── 2. GPU utilization ─────────────────────────────────────────────────────
echo ""
echo ">>> [2] GPU state"
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,temperature.gpu \
    --format=csv,noheader | awk -F',' '{
    gpu=$1; mem=$2; used=$3; temp=$4
    printf "  GPU util: %s   Mem util: %s   VRAM used: %s   Temp: %s\n", gpu, mem, used, temp
    # Interpret state
    gsub(/ %/, "", gpu)
    if (gpu+0 == 0) print "  [STATUS] 0% GPU → shader compilation or hang"
    else if (gpu+0 < 30) print "  [STATUS] Low GPU → env reset / data transfer"
    else print "  [STATUS] Active GPU → training running"
}'

# ── 3. Disk I/O — is shader cache being written? ──────────────────────────
echo ""
echo ">>> [3] Disk I/O (2-second sample)"
WRITE_BEFORE=$(cat /sys/block/nvme0n1/stat 2>/dev/null | awk '{print $7}')
sleep 2
WRITE_AFTER=$(cat /sys/block/nvme0n1/stat 2>/dev/null | awk '{print $7}')
if [ -n "$WRITE_BEFORE" ] && [ -n "$WRITE_AFTER" ]; then
    SECTORS=$(( WRITE_AFTER - WRITE_BEFORE ))
    KB=$(( SECTORS / 2 ))
    echo "  Write activity: ${KB} KB/s (over 2s sample)"
    if [ "$KB" -gt 500 ]; then
        echo "  [STATUS] High disk writes → shader cache being compiled (normal)"
    elif [ "$KB" -gt 50 ]; then
        echo "  [STATUS] Moderate disk writes → logging / checkpoint saving"
    else
        echo "  [STATUS] Low disk writes → idle or training in memory"
    fi
fi

# ── 4. Shader cache size — is it growing? ─────────────────────────────────
echo ""
echo ">>> [4] Isaac shader cache"
if [ -d "$CACHE_DIR" ]; then
    du -sh "$CACHE_DIR" 2>/dev/null | awk '{print "  Cache size: "$1" at '$CACHE_DIR'"}'
    # Count shader files
    SHADER_COUNT=$(find "$CACHE_DIR/glcache" "$CACHE_DIR/computecache" \
        -type f 2>/dev/null | wc -l)
    echo "  Shader files: $SHADER_COUNT"
    if [ "$SHADER_COUNT" -gt 0 ]; then
        NEWEST=$(find "$CACHE_DIR" -type f -printf '%T@ %p\n' 2>/dev/null \
            | sort -n | tail -1 | awk '{print $2}')
        NEWEST_AGE=$(( $(date +%s) - $(stat -c %Y "$NEWEST" 2>/dev/null || echo 0) ))
        echo "  Newest cache file: ${NEWEST_AGE}s ago"
        if [ "$NEWEST_AGE" -lt 30 ]; then
            echo "  [STATUS] Cache written <30s ago → actively compiling shaders"
        elif [ "$NEWEST_AGE" -lt 300 ]; then
            echo "  [STATUS] Cache written <5min ago → recently compiled or training"
        else
            echo "  [STATUS] Cache not updated recently → past compilation phase"
        fi
    fi
else
    echo "  Cache dir not found: $CACHE_DIR"
fi

# ── 5. Log file progress ───────────────────────────────────────────────────
echo ""
echo ">>> [5] Training log (last 5 lines of ~/train.log)"
if [ -f ~/train.log ]; then
    LOG_SIZE=$(wc -l < ~/train.log)
    LOG_AGE=$(( $(date +%s) - $(stat -c %Y ~/train.log) ))
    echo "  Log: $LOG_SIZE lines, last updated ${LOG_AGE}s ago"
    echo "  ---"
    tail -5 ~/train.log | sed 's/^/  /'
    echo "  ---"
    if [ "$LOG_AGE" -gt 120 ]; then
        echo "  [WARN] Log not updated in ${LOG_AGE}s — may be hanging"
    else
        echo "  [OK] Log updated ${LOG_AGE}s ago"
    fi
else
    echo "  ~/train.log not found"
fi

# ── 6. Checkpoint files ────────────────────────────────────────────────────
echo ""
echo ">>> [6] Saved checkpoints"
CKPT_COUNT=$(find "$REPO_DIR/omni_logs" -name "*.pt" 2>/dev/null | wc -l)
if [ "$CKPT_COUNT" -gt 0 ]; then
    echo "  $CKPT_COUNT checkpoint(s) saved:"
    find "$REPO_DIR/omni_logs" -name "*.pt" -printf "  %TY-%Tm-%Td %TH:%TM  %f  (%s bytes)\n" \
        2>/dev/null | sort
else
    echo "  No checkpoints yet (saved every 50 updates)"
fi

echo ""
echo "=============================================="
echo "  VERDICT"
echo "=============================================="

# Final verdict
CONTAINER_ALIVE=$(sudo docker ps -q --filter ancestor=isaac-lab-spot:latest | wc -l)
LOG_AGE_FINAL=9999
[ -f ~/train.log ] && LOG_AGE_FINAL=$(( $(date +%s) - $(stat -c %Y ~/train.log) ))

if [ "$CONTAINER_ALIVE" -eq 0 ]; then
    echo "  [DEAD] Container not running. Training stopped or crashed."
    echo "  Check: sudo docker ps -a | head -5"
elif [ "$LOG_AGE_FINAL" -gt 600 ]; then
    echo "  [STUCK?] Container alive but log hasn't updated in ${LOG_AGE_FINAL}s"
    echo "  If disk writes also low → likely hung. Consider restarting."
elif [ "$LOG_AGE_FINAL" -lt 120 ] && [ "$CKPT_COUNT" -gt 0 ]; then
    echo "  [TRAINING] Active — checkpoints exist, log is recent."
else
    echo "  [INITIALIZING] Container alive, no checkpoints yet."
    echo "  If shader cache is growing → wait 5-15 min for compilation to finish."
fi
echo ""
