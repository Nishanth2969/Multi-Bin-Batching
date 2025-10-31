"""
Metrics collection and analysis for MBB benchmarking.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import time
import statistics


@dataclass
class RequestMetrics:
    """Metrics for a single request with comprehensive telemetry."""
    
    request_id: int
    prompt_tokens: int
    arrival_time_ms: int
    start_time_ms: int = 0
    completion_time_ms: int = 0
    output_tokens: int = 0
    predicted_bin: int = -1
    admit_bin: int = -1
    wait_time_ms: int = 0
    
    @property
    def ttft_ms(self) -> int:
        """Time to first token."""
        if self.start_time_ms == 0:
            return 0
        return self.start_time_ms - self.arrival_time_ms
    
    @property
    def total_latency_ms(self) -> int:
        """Total request latency."""
        if self.completion_time_ms == 0:
            return 0
        return self.completion_time_ms - self.arrival_time_ms
    
    @property
    def tpot_ms(self) -> float:
        """Time per output token."""
        if self.output_tokens == 0 or self.completion_time_ms == 0:
            return 0.0
        duration = self.completion_time_ms - self.start_time_ms
        return duration / self.output_tokens
    
    @property
    def prompt_tokens_per_second(self) -> float:
        """Prompt throughput."""
        if self.start_time_ms == 0:
            return 0.0
        return (self.prompt_tokens / (self.start_time_ms - self.arrival_time_ms)) * 1000


@dataclass
class GPUSnapshot:
    """Per-second GPU utilization snapshot."""
    
    timestamp_ms: int
    sm_utilization: float  # 0-100%
    memory_used_mb: float
    memory_total_mb: float
    temperature_c: float = 0.0
    power_watts: float = 0.0
    
    @property
    def memory_utilization(self) -> float:
        """Memory utilization percentage."""
        if self.memory_total_mb == 0:
            return 0.0
        return (self.memory_used_mb / self.memory_total_mb) * 100


@dataclass
class BatchMetrics:
    """Metrics for a single batch."""
    
    batch_id: int
    num_requests: int
    total_tokens: int
    batch_size_tokens: int
    batch_time_ms: int
    bin_id: int = -1
    
    @property
    def tokens_per_second(self) -> float:
        """Batch throughput."""
        if self.batch_time_ms == 0:
            return 0.0
        return (self.total_tokens / self.batch_time_ms) * 1000


class MetricsCollector:
    """Collects and aggregates metrics during benchmarking with full telemetry."""
    
    def __init__(self):
        """Initialize metrics collector."""
        self.request_metrics: List[RequestMetrics] = []
        self.batch_metrics: List[BatchMetrics] = []
        self.gpu_snapshots: List[GPUSnapshot] = []
        self.bin_occupancy_history: List[Dict[int, int]] = []
        self.start_time_ms = time.time() * 1000
        self.current_time_ms = 0
        self.total_output_tokens = 0  # For token parity checks
    
    def record_request_start(self, req_id: int, prompt_tokens: int, 
                            arrival_time_ms: int, predicted_bin: int = -1, 
                            admit_bin: int = -1):
        """Record request start with bin information."""
        metrics = RequestMetrics(
            request_id=req_id,
            prompt_tokens=prompt_tokens,
            arrival_time_ms=arrival_time_ms,
            start_time_ms=self.current_time_ms,
            predicted_bin=predicted_bin,
            admit_bin=admit_bin
        )
        self.request_metrics.append(metrics)
    
    def record_request_completion(self, req_id: int, output_tokens: int):
        """Record request completion with output token count."""
        for m in self.request_metrics:
            if m.request_id == req_id:
                m.completion_time_ms = self.current_time_ms
                m.output_tokens = output_tokens
                m.wait_time_ms = m.ttft_ms
                self.total_output_tokens += output_tokens
                break
    
    def record_batch(self, batch_id: int, num_reqs: int, total_toks: int,
                    batch_size_toks: int, batch_time_ms: int, bin_id: int = -1):
        """Record batch completion."""
        metrics = BatchMetrics(
            batch_id=batch_id,
            num_requests=num_reqs,
            total_tokens=total_toks,
            batch_size_tokens=batch_size_toks,
            batch_time_ms=batch_time_ms,
            bin_id=bin_id
        )
        self.batch_metrics.append(metrics)
    
    def record_gpu_snapshot(self, sm_util: float, mem_used_mb: float, 
                           mem_total_mb: float, temp_c: float = 0.0, 
                           power_w: float = 0.0):
        """Record per-second GPU stats."""
        snapshot = GPUSnapshot(
            timestamp_ms=self.current_time_ms,
            sm_utilization=sm_util,
            memory_used_mb=mem_used_mb,
            memory_total_mb=mem_total_mb,
            temperature_c=temp_c,
            power_watts=power_w
        )
        self.gpu_snapshots.append(snapshot)
    
    def record_bin_occupancy(self, bin_occupancy: Dict[int, int]):
        """Record current bin queue sizes."""
        self.bin_occupancy_history.append({
            'timestamp_ms': self.current_time_ms,
            **bin_occupancy
        })
    
    def update_time(self, time_ms: int):
        """Update current time."""
        self.current_time_ms = time_ms
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary statistics."""
        if not self.request_metrics:
            return {}
        
        ttfts = [m.ttft_ms for m in self.request_metrics if m.ttft_ms > 0]
        latencies = [m.total_latency_ms for m in self.request_metrics if m.total_latency_ms > 0]
        tpots = [m.tpot_ms for m in self.request_metrics if m.tpot_ms > 0]
        wait_times = [m.wait_time_ms for m in self.request_metrics if m.wait_time_ms > 0]
        
        duration_s = (self.current_time_ms - self.start_time_ms) / 1000.0
        total_tokens = sum([m.prompt_tokens + m.output_tokens for m in self.request_metrics])
        
        summary = {
            'num_requests': len(self.request_metrics),
            'duration_s': duration_s,
            'total_tokens': total_tokens,
            'total_output_tokens': self.total_output_tokens,
            'tps': total_tokens / duration_s if duration_s > 0 else 0,
            'ttft_p50_ms': statistics.median(ttfts) if ttfts else 0,
            'ttft_p95_ms': self._percentile(ttfts, 0.95) if ttfts else 0,
            'latency_p50_ms': statistics.median(latencies) if latencies else 0,
            'latency_p95_ms': self._percentile(latencies, 0.95) if latencies else 0,
            'tpot_p50_ms': statistics.median(tpots) if tpots else 0,
            'tpot_p95_ms': self._percentile(tpots, 0.95) if tpots else 0,
            'wait_p50_ms': statistics.median(wait_times) if wait_times else 0,
            'wait_p95_ms': self._percentile(wait_times, 0.95) if wait_times else 0,
            'avg_prompt_tokens': statistics.mean([m.prompt_tokens for m in self.request_metrics]),
            'avg_output_tokens': statistics.mean([m.output_tokens for m in self.request_metrics if m.output_tokens > 0]) if self.total_output_tokens > 0 else 0,
        }
        
        # Add GPU stats if available
        if self.gpu_snapshots:
            summary['avg_sm_util'] = statistics.mean([s.sm_utilization for s in self.gpu_snapshots])
            summary['avg_mem_util'] = statistics.mean([s.memory_utilization for s in self.gpu_snapshots])
            summary['avg_mem_used_mb'] = statistics.mean([s.memory_used_mb for s in self.gpu_snapshots])
        
        return summary
    
    def _percentile(self, data: List[float], p: float) -> float:
        """Compute percentile."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        idx = int(len(sorted_data) * p)
        return sorted_data[min(idx, len(sorted_data) - 1)]
    
    def export_detailed_csv(self, filepath: str):
        """Export per-request telemetry to CSV."""
        import csv
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'request_id', 'prompt_tokens', 'output_tokens', 
                'predicted_bin', 'admit_bin', 
                'arrival_ms', 'start_ms', 'completion_ms',
                'ttft_ms', 'tpot_ms', 'wait_ms', 'total_latency_ms'
            ])
            for m in self.request_metrics:
                writer.writerow([
                    m.request_id, m.prompt_tokens, m.output_tokens,
                    m.predicted_bin, m.admit_bin,
                    m.arrival_time_ms, m.start_time_ms, m.completion_time_ms,
                    m.ttft_ms, m.tpot_ms, m.wait_time_ms, m.total_latency_ms
                ])
    
    def export_gpu_stats_csv(self, filepath: str):
        """Export per-second GPU stats to CSV."""
        import csv
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp_ms', 'sm_utilization', 'memory_used_mb', 
                'memory_total_mb', 'memory_utilization', 'temperature_c', 'power_watts'
            ])
            for s in self.gpu_snapshots:
                writer.writerow([
                    s.timestamp_ms, s.sm_utilization, s.memory_used_mb,
                    s.memory_total_mb, s.memory_utilization, s.temperature_c, s.power_watts
                ])


def compare_metrics(baseline: Dict[str, Any], mbb: Dict[str, Any]) -> Dict[str, Any]:
    """Compare baseline vs MBB metrics.
    
    Args:
        baseline: Baseline metrics summary
        mbb: MBB metrics summary
        
    Returns:
        Comparison dictionary with percent differences
    """
    comparison = {}
    
    for key in baseline:
        if key in mbb and isinstance(baseline[key], (int, float)):
            baseline_val = baseline[key]
            mbb_val = mbb[key]
            if baseline_val != 0:
                pct_diff = ((mbb_val - baseline_val) / baseline_val) * 100
                comparison[key] = f"{pct_diff:+.2f}%"
            else:
                comparison[key] = "N/A"
    
    return comparison

