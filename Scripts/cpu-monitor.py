#!/usr/bin/env python3
"""
Advanced CPU Monitoring Script
Provides detailed real-time CPU metrics, process analysis, and performance insights
"""

import psutil
import time
import datetime
import threading
from collections import deque, defaultdict
import signal
import sys
import os

class CPUMonitor:
    def __init__(self, log_file="cpu_monitor.log"):
        self.log_file = log_file
        self.running = True
        self.cpu_history = deque(maxlen=100)
        self.process_history = defaultdict(lambda: deque(maxlen=20))
        self.start_time = time.time()
        self.cpu_count = psutil.cpu_count()
        self.cpu_count_logical = psutil.cpu_count(logical=True)
         
    def get_cpu_info(self):
        """Get comprehensive CPU information"""
        try:
            # CPU percentages per core
            cpu_percent_per_core = psutil.cpu_percent(percpu=True, interval=0.1)
            cpu_percent_total = psutil.cpu_percent(interval=0.1)
             
            # CPU frequencies
            cpu_freq = psutil.cpu_freq()
            cpu_freq_per_core = psutil.cpu_freq(percpu=True)
             
            # Load averages (Linux/macOS)
            try:
                load_avg = os.getloadavg()
            except:
                load_avg = (0, 0, 0)
             
            # CPU times
            cpu_times = psutil.cpu_times()
             
            # CPU stats
            cpu_stats = psutil.cpu_stats()
             
            cpu_data = {
                'timestamp': datetime.datetime.now(),
                'cpu_percent_total': cpu_percent_total,
                'cpu_percent_per_core': cpu_percent_per_core,
                'cpu_freq_current': cpu_freq.current if cpu_freq else 0,
                'cpu_freq_min': cpu_freq.min if cpu_freq else 0,
                'cpu_freq_max': cpu_freq.max if cpu_freq else 0,
                'cpu_freq_per_core': [(f.current if f else 0) for f in cpu_freq_per_core] if cpu_freq_per_core else [],
                'load_avg_1min': load_avg[0],
                'load_avg_5min': load_avg[1],
                'load_avg_15min': load_avg[2],
                'cpu_times': cpu_times,
                'cpu_stats': cpu_stats,
                'cpu_count_physical': self.cpu_count,
                'cpu_count_logical': self.cpu_count_logical
            }
             
            return cpu_data
             
        except Exception as e:
            print(f"Error getting CPU info: {e}")
            return None
     
    def get_memory_info(self):
        """Get detailed memory information"""
        try:
            virtual_mem = psutil.virtual_memory()
            swap_mem = psutil.swap_memory()
             
            return {
                'virtual_total': virtual_mem.total,
                'virtual_available': virtual_mem.available,
                'virtual_used': virtual_mem.used,
                'virtual_percent': virtual_mem.percent,
                'virtual_free': virtual_mem.free,
                'virtual_buffers': getattr(virtual_mem, 'buffers', 0),
                'virtual_cached': getattr(virtual_mem, 'cached', 0),
                'swap_total': swap_mem.total,
                'swap_used': swap_mem.used,
                'swap_free': swap_mem.free,
                'swap_percent': swap_mem.percent
            }
        except Exception as e:
            print(f"Error getting memory info: {e}")
            return None
     
    def get_top_processes(self, limit=10):
        """Get top CPU consuming processes"""
        try:
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'memory_info', 'status', 'create_time']):
                try:
                    proc_info = proc.info
                    if proc_info['cpu_percent'] is not None and proc_info['cpu_percent'] > 0:
                        processes.append(proc_info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
             
            # Sort by CPU usage
            processes.sort(key=lambda x: x['cpu_percent'], reverse=True)
            return processes[:limit]
             
        except Exception as e:
            print(f"Error getting processes: {e}")
            return []
     
    def get_disk_io(self):
        """Get disk I/O statistics"""
        try:
            disk_io = psutil.disk_io_counters()
            if disk_io:
                return {
                    'read_count': disk_io.read_count,
                    'write_count': disk_io.write_count,
                    'read_bytes': disk_io.read_bytes,
                    'write_bytes': disk_io.write_bytes,
                    'read_time': disk_io.read_time,
                    'write_time': disk_io.write_time
                }
        except:
            return None
     
    def get_network_io(self):
        """Get network I/O statistics"""
        try:
            net_io = psutil.net_io_counters()
            if net_io:
                return {
                    'bytes_sent': net_io.bytes_sent,
                    'bytes_recv': net_io.bytes_recv,
                    'packets_sent': net_io.packets_sent,
                    'packets_recv': net_io.packets_recv,
                    'errin': net_io.errin,
                    'errout': net_io.errout,
                    'dropin': net_io.dropin,
                    'dropout': net_io.dropout
                }
        except:
            return None
     
    def calculate_averages(self):
        """Calculate running averages"""
        if not self.cpu_history:
            return {}
         
        cpu_avg = sum(d['cpu_percent_total'] for d in self.cpu_history) / len(self.cpu_history)
        load_1min_avg = sum(d['load_avg_1min'] for d in self.cpu_history) / len(self.cpu_history)
         
        return {
            'cpu_avg': cpu_avg,
            'load_1min_avg': load_1min_avg
        }
     
    def format_bytes(self, bytes_value):
        """Format bytes to human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_value < 1024.0:
                return f"{bytes_value:.1f} {unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.1f} PB"
     
    def display_cpu_info(self, cpu_data, memory_info, processes, disk_io, net_io, averages):
        """Display formatted CPU information"""
        if not cpu_data:
            return
         
        # Clear screen
        print("\033[2J\033[H")
         
        # Header
        print("=" * 100)
        print(f"🖥️  ADVANCED CPU MONITOR - {cpu_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️  Uptime: {time.time() - self.start_time:.1f}s | Samples: {len(self.cpu_history)}")
        print("=" * 100)
         
        # CPU Overview
        print(f"🔧 CPU INFO: {cpu_data['cpu_count_physical']} cores / {cpu_data['cpu_count_logical']} threads")
        print(f"⚡ Frequency: {cpu_data['cpu_freq_current']:.0f} MHz (Min: {cpu_data['cpu_freq_min']:.0f}, Max: {cpu_data['cpu_freq_max']:.0f})")
        print()
         
        # CPU Usage
        print("📊 CPU USAGE:")
        print(f"   Overall: {cpu_data['cpu_percent_total']:6.1f}% {'🔥' if cpu_data['cpu_percent_total'] > 80 else '✅'}")
         
        # Per-core usage
        cores_per_row = 8
        for i in range(0, len(cpu_data['cpu_percent_per_core']), cores_per_row):
            core_group = cpu_data['cpu_percent_per_core'][i:i+cores_per_row]
            core_str = " ".join([f"C{i+j:2d}:{usage:5.1f}%" for j, usage in enumerate(core_group)])
            print(f"   {core_str}")
         
        # CPU Usage Bar
        usage_bar = '█' * int(cpu_data['cpu_percent_total']/5) + '░' * (20-int(cpu_data['cpu_percent_total']/5))
        print(f"   Bar: {usage_bar} {cpu_data['cpu_percent_total']:.1f}%")
        print()
         
        # Load Average
        print("📈 LOAD AVERAGE:")
        print(f"   1min: {cpu_data['load_avg_1min']:6.2f} | 5min: {cpu_data['load_avg_5min']:6.2f} | 15min: {cpu_data['load_avg_15min']:6.2f}")
        load_percent = (cpu_data['load_avg_1min'] / cpu_data['cpu_count_logical']) * 100
        load_status = '🔥' if load_percent > 100 else '⚠️' if load_percent > 80 else '✅'
        print(f"   Load%: {load_percent:6.1f}% {load_status}")
        print()
         
        # Memory Information
        if memory_info:
            mem_used_gb = memory_info['virtual_used'] / (1024**3)
            mem_total_gb = memory_info['virtual_total'] / (1024**3)
            mem_available_gb = memory_info['virtual_available'] / (1024**3)
             
            print("💾 MEMORY USAGE:")
            print(f"   Used:      {mem_used_gb:6.2f} GB ({memory_info['virtual_percent']:5.1f}%)")
            print(f"   Available: {mem_available_gb:6.2f} GB")
            print(f"   Total:     {mem_total_gb:6.2f} GB")
             
            if memory_info['virtual_buffers'] > 0:
                buffers_gb = memory_info['virtual_buffers'] / (1024**3)
                cached_gb = memory_info['virtual_cached'] / (1024**3)
                print(f"   Buffers:   {buffers_gb:6.2f} GB | Cached: {cached_gb:6.2f} GB")
             
            mem_bar = '█' * int(memory_info['virtual_percent']/5) + '░' * (20-int(memory_info['virtual_percent']/5))
            print(f"   Bar:       {mem_bar} {memory_info['virtual_percent']:.1f}%")
             
            if memory_info['swap_total'] > 0:
                swap_gb = memory_info['swap_used'] / (1024**3)
                swap_total_gb = memory_info['swap_total'] / (1024**3)
                print(f"   Swap:      {swap_gb:6.2f} GB / {swap_total_gb:.2f} GB ({memory_info['swap_percent']:.1f}%)")
            print()
         
        # Running Averages
        if averages:
            print("📈 RUNNING AVERAGES:")
            print(f"   CPU Usage: {averages['cpu_avg']:6.1f}%")
            print(f"   Load 1min: {averages['load_1min_avg']:6.2f}")
            print()
         
        # Top Processes
        if processes:
            print("🔄 TOP CPU PROCESSES:")
            print("   PID      NAME                    CPU%    MEM%    MEMORY      STATUS")
            print("   " + "-" * 70)
            for proc in processes[:8]:
                mem_mb = proc['memory_info'].rss / (1024*1024) if proc['memory_info'] else 0
                print(f"   {proc['pid']:>6}   {proc['name'][:20]:<20} {proc['cpu_percent']:6.1f}% {proc['memory_percent']:6.1f}% {mem_mb:8.1f}MB {proc['status'][:10]}")
            print()
         
        # I/O Statistics
        if disk_io or net_io:
            print("💿 I/O STATISTICS:")
            if disk_io:
                print(f"   Disk Read:  {self.format_bytes(disk_io['read_bytes']):>10} ({disk_io['read_count']:>8} ops)")
                print(f"   Disk Write: {self.format_bytes(disk_io['write_bytes']):>10} ({disk_io['write_count']:>8} ops)")
             
            if net_io:
                print(f"   Net Recv:   {self.format_bytes(net_io['bytes_recv']):>10} ({net_io['packets_recv']:>8} pkts)")
                print(f"   Net Sent:   {self.format_bytes(net_io['bytes_sent']):>10} ({net_io['packets_sent']:>8} pkts)")
            print()
         
        print("=" * 100)
        print("Press Ctrl+C to stop monitoring")
     
    def log_data(self, cpu_data, memory_info):
        """Log data to file"""
        if cpu_data and memory_info:
            timestamp = cpu_data['timestamp'].isoformat()
            log_entry = f"{timestamp},{cpu_data['cpu_percent_total']},{cpu_data['load_avg_1min']},{memory_info['virtual_percent']}\n"
             
            try:
                with open(self.log_file, 'a') as f:
                    f.write(log_entry)
            except Exception as e:
                print(f"Error writing to log: {e}")
     
    def signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully"""
        print("\n\n🛑 Stopping CPU monitor...")
        self.running = False
        sys.exit(0)
     
    def run(self, interval=2):
        """Main monitoring loop"""
        signal.signal(signal.SIGINT, self.signal_handler)
         
        # Create log header
        try:
            with open(self.log_file, 'w') as f:
                f.write("timestamp,cpu_percent,load_avg_1min,memory_percent\n")
        except Exception as e:
            print(f"Error creating log file: {e}")
         
        print("🚀 Starting Advanced CPU Monitor...")
        print(f"📝 Logging to: {self.log_file}")
        print(f"⏰ Update interval: {interval}s")
        time.sleep(2)
         
        while self.running:
            try:
                cpu_data = self.get_cpu_info()
                memory_info = self.get_memory_info()
                processes = self.get_top_processes(10)
                disk_io = self.get_disk_io()
                net_io = self.get_network_io()
                 
                if cpu_data:
                    self.cpu_history.append(cpu_data)
                    averages = self.calculate_averages()
                     
                    self.display_cpu_info(cpu_data, memory_info, processes, disk_io, net_io, averages)
                    self.log_data(cpu_data, memory_info)
                 
                time.sleep(interval)
                 
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(interval)

if __name__ == "__main__":
    monitor = CPUMonitor(log_file="cpu_monitor.log")
    monitor.run(interval=2)
