#!/usr/bin/env python3
"""
Advanced GPU Monitoring Script for NVIDIA RTX 6000 Pro
Provides detailed real-time GPU metrics with logging and alerts
"""

import subprocess
import json
import time
import datetime
import psutil
import threading
from collections import deque
import signal
import sys

class GPUMonitor:
    def __init__(self, log_file="gpu_monitor.log", alert_threshold=90):
        self.log_file = log_file
        self.alert_threshold = alert_threshold
        self.running = True
        self.gpu_history = deque(maxlen=100)  # Keep last 100 readings
        self.start_time = time.time()
        
    def get_gpu_info(self):
        """Get comprehensive GPU information"""
        try:
            # Query multiple GPU metrics at once
            cmd = [
                'nvidia-smi', 
                '--query-gpu=timestamp,name,driver_version,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,memory.free,power.draw,power.limit,clocks.current.graphics,clocks.current.memory,fan.speed,pcie.link.gen.current,pcie.link.width.current',
                '--format=csv,noheader,nounits'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            if result.stdout.strip():
                values = result.stdout.strip().split(', ')
                
                gpu_data = {
                    'timestamp': values[0],
                    'name': values[1],
                    'driver_version': values[2],
                    'temperature': float(values[3]) if values[3] != '[Not Supported]' else 0,
                    'gpu_utilization': float(values[4]) if values[4] != '[Not Supported]' else 0,
                    'memory_utilization': float(values[5]) if values[5] != '[Not Supported]' else 0,
                    'memory_used': float(values[6]) if values[6] != '[Not Supported]' else 0,
                    'memory_total': float(values[7]) if values[7] != '[Not Supported]' else 0,
                    'memory_free': float(values[8]) if values[8] != '[Not Supported]' else 0,
                    'power_draw': float(values[9]) if values[9] != '[Not Supported]' else 0,
                    'power_limit': float(values[10]) if values[10] != '[Not Supported]' else 0,
                    'clock_graphics': float(values[11]) if values[11] != '[Not Supported]' else 0,
                    'clock_memory': float(values[12]) if values[12] != '[Not Supported]' else 0,
                    'fan_speed': float(values[13]) if values[13] != '[Not Supported]' else 0,
                    'pcie_gen': values[14] if values[14] != '[Not Supported]' else 'N/A',
                    'pcie_width': values[15] if values[15] != '[Not Supported]' else 'N/A'
                }
                
                return gpu_data
                
        except subprocess.CalledProcessError as e:
            print(f"Error getting GPU info: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None
    
    def get_gpu_processes(self):
        """Get detailed information about GPU processes"""
        try:
            cmd = ['nvidia-smi', '--query-compute-apps=pid,process_name,gpu_uuid,used_memory', '--format=csv,noheader,nounits']
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            processes = []
            if result.stdout.strip():
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = line.split(', ')
                        if len(parts) >= 4:
                            processes.append({
                                'pid': int(parts[0]),
                                'name': parts[1],
                                'gpu_uuid': parts[2],
                                'memory_used': float(parts[3])
                            })
            return processes
        except:
            return []
    
    def calculate_averages(self):
        """Calculate running averages"""
        if not self.gpu_history:
            return {}
            
        gpu_util_avg = sum(d['gpu_utilization'] for d in self.gpu_history) / len(self.gpu_history)
        mem_util_avg = sum(d['memory_utilization'] for d in self.gpu_history) / len(self.gpu_history)
        temp_avg = sum(d['temperature'] for d in self.gpu_history) / len(self.gpu_history)
        power_avg = sum(d['power_draw'] for d in self.gpu_history) / len(self.gpu_history)
        
        return {
            'gpu_utilization_avg': gpu_util_avg,
            'memory_utilization_avg': mem_util_avg,
            'temperature_avg': temp_avg,
            'power_avg': power_avg
        }
    
    def display_gpu_info(self, gpu_data, processes, averages):
        """Display formatted GPU information"""
        if not gpu_data:
            return
            
        # Clear screen
        print("\033[2J\033[H")
        
        # Header
        print("=" * 80)
        print(f"🚀 ADVANCED GPU MONITOR - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️  Uptime: {time.time() - self.start_time:.1f}s | Samples: {len(self.gpu_history)}")
        print("=" * 80)
        
        # GPU Information
        print(f"🎮 GPU: {gpu_data['name']}")
        print(f"🔧 Driver: {gpu_data['driver_version']}")
        print()
        
        # Performance Metrics
        print("📊 PERFORMANCE METRICS:")
        print(f"   GPU Utilization:    {gpu_data['gpu_utilization']:6.1f}% {'🔥' if gpu_data['gpu_utilization'] > 80 else '✅'}")
        print(f"   Memory Utilization: {gpu_data['memory_utilization']:6.1f}% {'🔥' if gpu_data['memory_utilization'] > 80 else '✅'}")
        print(f"   Temperature:        {gpu_data['temperature']:6.1f}°C {'🌡️' if gpu_data['temperature'] > 80 else '❄️'}")
        print()
        
        # Memory Information
        memory_used_gb = gpu_data['memory_used'] / 1024
        memory_total_gb = gpu_data['memory_total'] / 1024
        memory_free_gb = gpu_data['memory_free'] / 1024
        memory_percent = (memory_used_gb / memory_total_gb) * 100 if memory_total_gb > 0 else 0
        
        print("💾 MEMORY USAGE:")
        print(f"   Used:  {memory_used_gb:6.2f} GB ({memory_percent:5.1f}%)")
        print(f"   Free:  {memory_free_gb:6.2f} GB")
        print(f"   Total: {memory_total_gb:6.2f} GB")
        print(f"   Bar:   {'█' * int(memory_percent/5)}{'░' * (20-int(memory_percent/5))} {memory_percent:.1f}%")
        print()
        
        # Power and Clocks
        power_percent = (gpu_data['power_draw'] / gpu_data['power_limit']) * 100 if gpu_data['power_limit'] > 0 else 0
        print("⚡ POWER & CLOCKS:")
        print(f"   Power Draw:    {gpu_data['power_draw']:6.1f}W / {gpu_data['power_limit']:.1f}W ({power_percent:.1f}%)")
        print(f"   Graphics Clock: {gpu_data['clock_graphics']:6.0f} MHz")
        print(f"   Memory Clock:   {gpu_data['clock_memory']:6.0f} MHz")
        print(f"   Fan Speed:      {gpu_data['fan_speed']:6.1f}%")
        print(f"   PCIe:          Gen{gpu_data['pcie_gen']} x{gpu_data['pcie_width']}")
        print()
        
        # Running Averages
        if averages:
            print("📈 RUNNING AVERAGES:")
            print(f"   GPU Util:  {averages['gpu_utilization_avg']:6.1f}%")
            print(f"   Mem Util:  {averages['memory_utilization_avg']:6.1f}%")
            print(f"   Temp:      {averages['temperature_avg']:6.1f}°C")
            print(f"   Power:     {averages['power_avg']:6.1f}W")
            print()
        
        # Active Processes
        if processes:
            print("🔄 GPU PROCESSES:")
            for proc in processes:
                print(f"   PID {proc['pid']:>6}: {proc['name']:<20} | {proc['memory_used']:>8.1f} MB")
        else:
            print("🔄 GPU PROCESSES: None")
        
        print("=" * 80)
        print("Press Ctrl+C to stop monitoring")
    
    def log_data(self, gpu_data):
        """Log data to file"""
        if gpu_data:
            timestamp = datetime.datetime.now().isoformat()
            log_entry = f"{timestamp},{gpu_data['gpu_utilization']},{gpu_data['memory_utilization']},{gpu_data['temperature']},{gpu_data['power_draw']}\n"
            
            try:
                with open(self.log_file, 'a') as f:
                    f.write(log_entry)
            except Exception as e:
                print(f"Error writing to log: {e}")
    
    def check_alerts(self, gpu_data):
        """Check for alert conditions"""
        alerts = []
        
        if gpu_data['temperature'] > 85:
            alerts.append(f"🚨 HIGH TEMPERATURE: {gpu_data['temperature']:.1f}°C")
        
        if gpu_data['gpu_utilization'] > self.alert_threshold:
            alerts.append(f"⚠️  HIGH GPU USAGE: {gpu_data['gpu_utilization']:.1f}%")
        
        if gpu_data['memory_utilization'] > self.alert_threshold:
            alerts.append(f"⚠️  HIGH MEMORY USAGE: {gpu_data['memory_utilization']:.1f}%")
        
        power_percent = (gpu_data['power_draw'] / gpu_data['power_limit']) * 100 if gpu_data['power_limit'] > 0 else 0
        if power_percent > 95:
            alerts.append(f"⚡ POWER LIMIT REACHED: {power_percent:.1f}%")
        
        if alerts:
            print("\n" + "!" * 50)
            for alert in alerts:
                print(alert)
            print("!" * 50)
    
    def signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully"""
        print("\n\n🛑 Stopping GPU monitor...")
        self.running = False
        sys.exit(0)
    
    def run(self, interval=1):
        """Main monitoring loop"""
        signal.signal(signal.SIGINT, self.signal_handler)
        
        # Create log header
        try:
            with open(self.log_file, 'w') as f:
                f.write("timestamp,gpu_utilization,memory_utilization,temperature,power_draw\n")
        except Exception as e:
            print(f"Error creating log file: {e}")
        
        print("🚀 Starting Advanced GPU Monitor...")
        print(f"📝 Logging to: {self.log_file}")
        print(f"⏰ Update interval: {interval}s")
        time.sleep(2)
        
        while self.running:
            try:
                gpu_data = self.get_gpu_info()
                processes = self.get_gpu_processes()
                
                if gpu_data:
                    self.gpu_history.append(gpu_data)
                    averages = self.calculate_averages()
                    
                    self.display_gpu_info(gpu_data, processes, averages)
                    self.log_data(gpu_data)
                    self.check_alerts(gpu_data)
                
                time.sleep(interval)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(interval)

if __name__ == "__main__":
    monitor = GPUMonitor(log_file="gpu_monitor.log", alert_threshold=85)
    monitor.run(interval=1)

