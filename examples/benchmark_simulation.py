"""
Benchmark simulation example.

This example shows how to run a simple benchmark to compare
MBB against a baseline scheduler.
"""

from mbb_core import MBScheduler
from bench.loadgen import RequestGenerator, short_heavy_mix, mixed_mix, long_tail_mix
from bench.metrics import MetricsCollector
import time


def simulate_benchmark(scheduler, mix_name, num_iterations=100):
    """Run a benchmark simulation."""
    
    print(f"\n{'='*60}")
    print(f"Benchmark: {mix_name}")
    print(f"{'='*60}")
    
    # Generate requests
    if mix_name == "short_heavy":
        mix = short_heavy_mix()
    elif mix_name == "mixed":
        mix = mixed_mix()
    else:
        mix = long_tail_mix()
    
    generator = RequestGenerator(mix, seed=42)
    requests = generator.generate_requests()
    
    # Setup metrics
    collector = MetricsCollector()
    
    # Simulate
    start_time = time.time()
    
    for i, (arrival_time_s, prompt_tokens, output_tokens) in enumerate(requests[:num_iterations]):
        arrival_ms = int(arrival_time_s * 1000)
        
        # Enqueue
        rid = scheduler.enqueue_request(
            prompt_tokens=prompt_tokens,
            arrival_time_ms=arrival_ms
        )
        
        collector.record_request_start(rid, prompt_tokens, arrival_ms)
        collector.update_time(arrival_ms)
        
        # Periodically build batches
        if i % 5 == 0:
            batch = scheduler.build_batch(current_time_ms=arrival_ms)
            
            for req in batch:
                collector.record_request_completion(req.rid)
        
        collector.update_time(arrival_ms)
    
    elapsed = time.time() - start_time
    
    # Print results
    print(f"\nSimulated {num_iterations} requests in {elapsed:.2f}s")
    stats = collector.get_summary()
    
    print("\nMetrics:")
    print(f"  Requests: {stats.get('num_requests', 0)}")
    print(f"  TTFT P50: {stats.get('ttft_p50', 0):.2f}ms")
    print(f"  TTFT P95: {stats.get('ttft_p95', 0):.2f}ms")
    print(f"  Latency P50: {stats.get('latency_p50', 0):.2f}ms")
    print(f"  Latency P95: {stats.get('latency_p95', 0):.2f}ms")
    
    return stats


def main():
    """Run benchmark simulation."""
    
    print("Multi-Bin Batching Benchmark Simulation")
    print("=" * 60)
    
    # Create scheduler
    scheduler = MBScheduler(
        num_bins=3,
        bin_edges=[128, 512],
        kv_budget_tokens=8192,
        max_batched_tokens=2048,
        starvation_ms=500
    )
    
    # Run benchmarks
    results = {}
    
    for mix in ["short_heavy", "mixed", "long_tail"]:
        results[mix] = simulate_benchmark(scheduler, mix, num_iterations=200)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    for mix, stats in results.items():
        print(f"\n{mix}:")
        print(f"  Throughput: {stats.get('total_throughput_tokens_per_sec', 0):.2f} tokens/sec")
        print(f"  TTFT P50: {stats.get('ttft_p50', 0):.2f}ms")


if __name__ == "__main__":
    main()

