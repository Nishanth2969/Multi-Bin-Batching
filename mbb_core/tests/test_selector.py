"""Tests for batch selection."""

import pytest
from mbb_core.queues import Request, BinQueues
from mbb_core.selector import RoundRobinSelector, StarvationGuard, BatchSelector


def test_round_robin_selector():
    """Test round-robin selection."""
    bins = BinQueues(n_bins=3)
    bins.push(0, Request(rid=1, prompt_tokens=100, arrival_ms=0, pred_out_tokens=50))
    bins.push(1, Request(rid=2, prompt_tokens=100, arrival_ms=0, pred_out_tokens=50))
    bins.push(2, Request(rid=3, prompt_tokens=100, arrival_ms=0, pred_out_tokens=50))
    
    selector = RoundRobinSelector(bins)
    
    # Should cycle through bins
    bin1 = selector.get_next_bin()
    bin2 = selector.get_next_bin()
    bin3 = selector.get_next_bin()
    
    assert bin1 in [0, 1, 2]
    assert bin2 in [0, 1, 2]
    assert bin3 in [0, 1, 2]
    assert len({bin1, bin2, bin3}) >= 2  # Should get different bins


def test_starvation_guard():
    """Test starvation detection."""
    guard = StarvationGuard(max_wait_ms=500)
    
    req = Request(rid=1, prompt_tokens=100, arrival_ms=0, pred_out_tokens=50)
    
    # Not starved yet
    assert not guard.check_starvation(req, current_ms=400)
    
    # Starved now
    assert guard.check_starvation(req, current_ms=600)


def test_batch_selector_basic():
    """Test basic batch building."""
    bins = BinQueues(n_bins=3)
    bins.push(0, Request(rid=1, prompt_tokens=100, arrival_ms=0, pred_out_tokens=50))
    bins.push(0, Request(rid=2, prompt_tokens=100, arrival_ms=0, pred_out_tokens=50))
    
    selector = BatchSelector(
        kv_budget_tokens=8192,
        max_batched_tokens=2048,
        starvation_ms=500
    )
    selector.set_bins(bins)
    
    batch = selector.build_batch(current_ms=100)
    assert len(batch) > 0


def test_batch_selector_respects_limits():
    """Test batch selector respects token limits."""
    bins = BinQueues(n_bins=1)
    
    # Create many requests
    for i in range(100):
        bins.push(0, Request(
            rid=i,
            prompt_tokens=100,
            arrival_ms=0,
            pred_out_tokens=50
        ))
    
    selector = BatchSelector(
        kv_budget_tokens=8192,
        max_batched_tokens=200,  # Small limit
        starvation_ms=500
    )
    selector.set_bins(bins)
    
    batch = selector.build_batch(current_ms=100)
    # Should not exceed limits
    total_tokens = sum(r.prompt_tokens + 1 for r in batch)  # +1 for decode
    assert total_tokens <= 200


if __name__ == "__main__":
    pytest.main([__file__])

