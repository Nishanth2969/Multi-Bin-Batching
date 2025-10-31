"""
Basic usage example for Multi-Bin Batching.

This example demonstrates:
1. Creating a scheduler
2. Enqueuing requests
3. Building batches
4. Collecting statistics
"""

import time
from mbb_core import MBScheduler


def main():
    """Run basic MBB example."""
    
    # Create scheduler
    scheduler = MBScheduler(
        num_bins=3,
        bin_edges=[128, 512],
        kv_budget_tokens=8192,
        max_batched_tokens=2048,
        starvation_ms=500,
        predictor_type="heuristic",
        selection_policy="round_robin"
    )
    
    print("Multi-Bin Batching Example")
    print("=" * 50)
    
    # Simulate request arrivals
    arrival_times = [
        (0, 100, "short"),
        (10, 300, "medium"),
        (20, 50, "short"),
        (30, 800, "long"),
        (40, 150, "medium"),
    ]
    
    print("\nEnqueuing requests...")
    for arrival_ms, prompt_tokens, label in arrival_times:
        rid = scheduler.enqueue_request(
            prompt_tokens=prompt_tokens,
            arrival_time_ms=arrival_ms
        )
        print(f"  Request {rid}: {prompt_tokens} tokens ({label})")
    
    print(f"\nTotal pending: {scheduler.total_pending()}")
    print("\nBin occupancy:")
    for bin_id, size in scheduler.get_bin_occupancy():
        if size > 0:
            print(f"  Bin {bin_id}: {size} requests")
    
    # Build batches
    print("\nBuilding batches...")
    current_time = 100
    batch_num = 1
    
    while not scheduler.is_empty():
        batch = scheduler.build_batch(current_time_ms=current_time)
        
        if batch:
            print(f"\nBatch {batch_num}:")
            print(f"  Size: {len(batch)} requests")
            total_tokens = sum(r.prompt_tokens + r.pred_out_tokens for r in batch)
            print(f"  Total tokens: {total_tokens}")
            
            for req in batch:
                print(f"    Request {req.rid}: prompt={req.prompt_tokens}, "
                      f"pred={req.pred_out_tokens}, wait={req.wait_time_ms}ms")
            
            batch_num += 1
        
        current_time += 50
        scheduler.update_time(50)
    
    # Get final statistics
    print("\n" + "=" * 50)
    print("Final Statistics:")
    stats = scheduler.get_statistics()
    print(f"  Total requests: {stats['request_counter']}")
    print(f"  Total pending: {stats['bin_statistics']['total_pending']}")
    print(f"  Batches formed: {stats.get('batches_formed', 'N/A')}")


if __name__ == "__main__":
    main()

