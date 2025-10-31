"""
GPU monitoring utility using NVML for real-time telemetry.
"""

import time
import threading
from typing import Optional, Callable
from dataclasses import dataclass


@dataclass
class GPUStats:
    """GPU statistics snapshot."""
    
    timestamp_ms: int
    sm_utilization: float  # 0-100%
    memory_used_mb: float
    memory_total_mb: float
    temperature_c: float
    power_watts: float


class GPUMonitor:
    """Real-time GPU monitoring using NVML."""
    
    def __init__(self, device_id: int = 0, sample_interval_ms: int = 1000):
        """Initialize GPU monitor.
        
        Args:
            device_id: GPU device index
            sample_interval_ms: Sampling interval in milliseconds
        """
        self.device_id = device_id
        self.sample_interval_ms = sample_interval_ms
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._callback: Optional[Callable[[GPUStats], None]] = None
        self._nvml_initialized = False
        
        try:
            import pynvml
            self.pynvml = pynvml
            self._nvml_initialized = True
        except ImportError:
            print("Warning: pynvml not installed. GPU monitoring disabled.")
            print("Install with: pip install nvidia-ml-py3")
            self.pynvml = None
    
    def start(self, callback: Callable[[GPUStats], None]):
        """Start monitoring in background thread.
        
        Args:
            callback: Function to call with each GPU stats snapshot
        """
        if not self._nvml_initialized:
            print("Warning: NVML not available, GPU monitoring skipped")
            return
        
        self._callback = callback
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        """Stop monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        try:
            self.pynvml.nvmlInit()
            handle = self.pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
            
            while self._running:
                start_time = time.time()
                
                try:
                    # Get utilization
                    util = self.pynvml.nvmlDeviceGetUtilizationRates(handle)
                    sm_util = util.gpu
                    
                    # Get memory
                    mem_info = self.pynvml.nvmlDeviceGetMemoryInfo(handle)
                    mem_used_mb = mem_info.used / (1024 * 1024)
                    mem_total_mb = mem_info.total / (1024 * 1024)
                    
                    # Get temperature
                    temp_c = self.pynvml.nvmlDeviceGetTemperature(
                        handle, self.pynvml.NVML_TEMPERATURE_GPU
                    )
                    
                    # Get power
                    power_mw = self.pynvml.nvmlDeviceGetPowerUsage(handle)
                    power_w = power_mw / 1000.0
                    
                    # Create stats snapshot
                    stats = GPUStats(
                        timestamp_ms=int(time.time() * 1000),
                        sm_utilization=float(sm_util),
                        memory_used_mb=mem_used_mb,
                        memory_total_mb=mem_total_mb,
                        temperature_c=float(temp_c),
                        power_watts=power_w
                    )
                    
                    # Callback with stats
                    if self._callback:
                        self._callback(stats)
                
                except Exception as e:
                    print(f"Error reading GPU stats: {e}")
                
                # Sleep until next sample
                elapsed = time.time() - start_time
                sleep_time = max(0, (self.sample_interval_ms / 1000.0) - elapsed)
                time.sleep(sleep_time)
        
        except Exception as e:
            print(f"GPU monitoring error: {e}")
        finally:
            try:
                self.pynvml.nvmlShutdown()
            except:
                pass
    
    def get_current_stats(self) -> Optional[GPUStats]:
        """Get current GPU stats (blocking call).
        
        Returns:
            GPUStats or None if NVML not available
        """
        if not self._nvml_initialized:
            return None
        
        try:
            self.pynvml.nvmlInit()
            handle = self.pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
            
            util = self.pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem_info = self.pynvml.nvmlDeviceGetMemoryInfo(handle)
            temp_c = self.pynvml.nvmlDeviceGetTemperature(
                handle, self.pynvml.NVML_TEMPERATURE_GPU
            )
            power_mw = self.pynvml.nvmlDeviceGetPowerUsage(handle)
            
            stats = GPUStats(
                timestamp_ms=int(time.time() * 1000),
                sm_utilization=float(util.gpu),
                memory_used_mb=mem_info.used / (1024 * 1024),
                memory_total_mb=mem_info.total / (1024 * 1024),
                temperature_c=float(temp_c),
                power_watts=power_mw / 1000.0
            )
            
            self.pynvml.nvmlShutdown()
            return stats
        
        except Exception as e:
            print(f"Error getting GPU stats: {e}")
            return None


if __name__ == "__main__":
    # Test GPU monitoring
    monitor = GPUMonitor(device_id=0, sample_interval_ms=1000)
    
    def print_stats(stats: GPUStats):
        print(f"GPU: {stats.sm_utilization:.1f}% | "
              f"Mem: {stats.memory_used_mb:.0f}/{stats.memory_total_mb:.0f} MB | "
              f"Temp: {stats.temperature_c:.1f}°C | "
              f"Power: {stats.power_watts:.1f}W")
    
    print("Starting GPU monitoring for 10 seconds...")
    monitor.start(print_stats)
    time.sleep(10)
    monitor.stop()
    print("Monitoring stopped.")

