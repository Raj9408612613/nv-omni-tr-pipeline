#!/usr/bin/env python3
"""
Combined CPU + GPU Monitor
Runs both monitors in parallel with synchronized display
"""

import threading
import time
import subprocess
import sys

def run_gpu_monitor():
    """Run GPU monitor in separate thread"""
    subprocess.run([sys.executable, "gpu_monitor.py"])

def run_cpu_monitor():
    """Run CPU monitor in separate thread"""
    subprocess.run([sys.executable, "cpu_monitor.py"])

def main():
    print("🚀 Starting Combined CPU + GPU Monitor...")
    print("This will open two monitoring windows.")
    print("Press Ctrl+C in each window to stop.")
     
    # Start GPU monitor in new terminal
    gpu_thread = threading.Thread(target=lambda: subprocess.run([
        "gnome-terminal", "--", "python3", "gpu_monitor.py"
    ]))
     
    # Start CPU monitor in new terminal  
    cpu_thread = threading.Thread(target=lambda: subprocess.run([
        "gnome-terminal", "--", "python3", "cpu_monitor.py"
    ]))
     
    gpu_thread.start()
    time.sleep(1)
    cpu_thread.start()
     
    print("✅ Monitors started in separate terminals")
    print("Close the terminal windows to stop monitoring")

if __name__ == "__main__":
    main()
