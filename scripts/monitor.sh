#!/bin/bash
LOG=~/hw_monitor.log
INTERVAL=5  # seconds between samples

echo "=== Monitoring started at $(date) ===" | tee -a $LOG

while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "\n[$TIMESTAMP]" | tee -a $LOG

    echo "--- GPU ---" | tee -a $LOG
    nvidia-smi --query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu \
        --format=csv,noheader 2>/dev/null | tee -a $LOG || echo "No GPU found" | tee -a $LOG

    echo "--- CPU & RAM ---" | tee -a $LOG
    top -bn1 | grep -E "^(%Cpu|MiB Mem)" | tee -a $LOG

    echo "--- Disk ---" | tee -a $LOG
    df -h / | tail -1 | awk '{print "Disk used: "$3" / "$2" ("$5")"}' | tee -a $LOG
    iostat -d 1 1 2>/dev/null | grep -v "^$\|Device\|Linux" | tee -a $LOG

    sleep $INTERVAL
done
