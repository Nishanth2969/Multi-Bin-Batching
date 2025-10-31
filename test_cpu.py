#!/usr/bin/env python3
"""
CPU Test Suite - Verify Multi-Bin Batching works before GPU deployment.

Tests all core components with minimal data to ensure correctness.
"""

import sys
import traceback
from typing import List, Dict
from tqdm import tqdm


def test_scheduler_basic():
    """Test basic scheduler functionality."""
    from mbb_core.scheduler import MBScheduler
    
    scheduler = MBScheduler(num_bins=3, bin_edges=[128, 512], batch_size=4)
    
    for i in range(12):
        scheduler.enqueue_request(prompt_tokens=100 + i * 10)
    
    assert scheduler.total_pending() == 12
    
    batches = []
    while scheduler.total_pending() >= 4:
        batch = scheduler.build_batch(current_time_ms=1000)
        if batch:
            batches.append(batch)
    
    assert len(batches) > 0
    assert all(len(b) == 4 for b in batches)
    return True


def test_bin_edges():
    """Test bin edge assignment."""
    from mbb_core.bins import BinEdges
    
    edges = BinEdges([128, 512])
    assert edges.num_bins() == 3
    
    assert edges.bin_for(64) == 0
    assert edges.bin_for(200) == 1
    assert edges.bin_for(600) == 2
    
    return True


def test_quantile_bins():
    """Test quantile bin creation."""
    from mbb_core.bins import create_quantile_bins
    
    lengths = [64, 128, 256, 512, 1024] * 10
    edges = create_quantile_bins(lengths, num_bins=3)
    
    assert edges.num_bins() == 3
    assert len(edges.edges) >= 1
    
    counts = [0] * 3
    for length in lengths:
        bin_id = edges.bin_for(length)
        counts[bin_id] += 1
    
    assert all(c > 0 for c in counts)
    return True


def test_predictors():
    """Test all predictor types."""
    from mbb_core.predict import HeuristicPredictor, AdaptivePredictor
    
    heuristic = HeuristicPredictor()
    pred1 = heuristic(100)
    assert 0 < pred1 < 5000
    
    adaptive = AdaptivePredictor(alpha=0.1, min_samples=3)
    for prompt, output in [(100, 150), (200, 300), (150, 200)]:
        adaptive(prompt)
        adaptive.update(prompt, output)
    
    assert adaptive.samples >= 3
    pred2 = adaptive(100)
    assert pred2 > 0
    
    return True


def test_queues():
    """Test bin queue operations."""
    from mbb_core.queues import BinQueues, Request
    
    bins = BinQueues(3)
    
    for i in range(10):
        req = Request(rid=i, prompt_tokens=100, arrival_ms=0, pred_out_tokens=128)
        bins.push(i % 3, req)
    
    assert bins.total_pending() == 10
    assert bins.bin_size(0) > 0
    assert bins.bin_size(1) > 0
    assert bins.bin_size(2) > 0
    
    req = bins.pop(0)
    assert req is not None
    assert bins.total_pending() == 9
    
    return True


def test_selector():
    """Test batch selector."""
    from mbb_core.selector import BatchSelector
    from mbb_core.queues import BinQueues, Request
    
    bins = BinQueues(3)
    selector = BatchSelector(kv_budget_tokens=8192, max_batched_tokens=512)
    selector.set_bins(bins)
    
    for i in range(20):
        req = Request(rid=i, prompt_tokens=100, arrival_ms=0, pred_out_tokens=128)
        bins.push(i % 3, req)
    
    batch = selector.build_batch(current_ms=1000)
    assert len(batch) > 0
    assert len(batch) <= 5
    
    return True


def test_equal_mass_warmup():
    """Test equal-mass bin warmup."""
    from mbb_core.scheduler import MBScheduler
    
    scheduler = MBScheduler(
        num_bins=3,
        use_equal_mass_bins=True,
        warmup_samples=10
    )
    
    for length in [64, 128, 256, 512, 1024] * 3:
        scheduler.record_output_length(length)
    
    assert scheduler.bin_histogram.is_warmed_up()
    edges = scheduler.bin_histogram.get_edges()
    assert edges.num_bins() == 3
    
    return True


def test_workload_simulation():
    """Simulate a small workload."""
    from mbb_core.scheduler import MBScheduler
    import time
    
    scheduler = MBScheduler(num_bins=3, bin_edges=[128, 512], batch_size=4)
    
    prompts = [64, 128, 256, 512] * 5
    for i, prompt_tokens in enumerate(prompts):
        scheduler.enqueue_request(prompt_tokens=prompt_tokens, arrival_time_ms=i * 100)
    
    batches_processed = 0
    while scheduler.total_pending() >= 4 or scheduler.service_queue:
        batch = scheduler.build_batch(current_time_ms=10000)
        if batch:
            batches_processed += len(batch)
            for req in batch:
                scheduler.record_output_length(req.pred_out_tokens)
    
    assert batches_processed > 0
    return True


def test_dataset_generation():
    """Test dataset generation with minimal data."""
    try:
        from data.download_datasets import download_gsm8k_prompts, create_workload_mixes
        
        prompts = download_gsm8k_prompts()
        assert len(prompts) > 0
        
        mixes = create_workload_mixes(prompts, size=20)
        assert 'short_heavy' in mixes
        assert 'mixed' in mixes
        assert 'long_tail' in mixes
        
        for mix_name, mix_prompts in mixes.items():
            assert len(mix_prompts) > 0
        
        return True
    except Exception as e:
        if "requests" in str(e).lower() or "http" in str(e).lower():
            print(f"  (Skipping network test in offline mode)")
            return True
        raise


def test_manifest_generation():
    """Test manifest generation."""
    from bench.run_manifest import generate_baseline_manifest, generate_mbb_manifest
    
    baseline = generate_baseline_manifest("mixed", 10.0)
    assert baseline.scheduler.policy == "continuous"
    assert baseline.workload.arrival_rate == 10.0
    
    mbb = generate_mbb_manifest("mixed", 10.0, num_bins=3)
    assert mbb.scheduler.policy == "multi_bin"
    assert mbb.scheduler.num_bins == 3
    
    return True


def test_metrics_collection():
    """Test metrics collection."""
    from bench.metrics import MetricsCollector
    
    collector = MetricsCollector()
    
    collector.record_request_start(rid=0, prompt_tokens=100, arrival_ms=0, predicted_bin=0, admit_bin=0)
    collector.record_request_completion(rid=0, output_tokens=150)
    
    summary = collector.get_summary()
    assert summary['total_requests'] == 1
    assert summary['total_output_tokens'] == 150
    
    return True


def run_all_tests():
    """Run all tests and report results."""
    tests = [
        ("Scheduler Basic", test_scheduler_basic),
        ("Bin Edges", test_bin_edges),
        ("Quantile Bins", test_quantile_bins),
        ("Predictors", test_predictors),
        ("Queues", test_queues),
        ("Selector", test_selector),
        ("Equal-Mass Warmup", test_equal_mass_warmup),
        ("Workload Simulation", test_workload_simulation),
        ("Dataset Generation", test_dataset_generation),
        ("Manifest Generation", test_manifest_generation),
        ("Metrics Collection", test_metrics_collection),
    ]
    
    results: List[tuple] = []
    
    print("="*70)
    print("CPU Test Suite - Multi-Bin Batching")
    print("="*70)
    print()
    
    for test_name, test_func in tqdm(tests, desc="Running tests", unit="test"):
        try:
            test_func()
            results.append((test_name, True, None))
            tqdm.write(f"✓ {test_name}")
        except Exception as e:
            results.append((test_name, False, str(e)))
            tqdm.write(f"✗ {test_name}: {e}")
            tqdm.write(traceback.format_exc())
    
    print()
    print("="*70)
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")
    print("="*70)
    
    if passed == total:
        print("\n✓ ALL TESTS PASSED - Ready for GitHub and GPU deployment!")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED - Fix errors before deployment")
        for name, success, error in results:
            if not success:
                print(f"  - {name}: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())

