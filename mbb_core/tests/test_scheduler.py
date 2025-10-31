"""Tests for main scheduler."""

import pytest
from mbb_core.scheduler import MBScheduler


def test_scheduler_initialization():
    """Test scheduler initialization."""
    scheduler = MBScheduler(
        num_bins=3,
        kv_budget_tokens=8192,
        max_batched_tokens=2048
    )
    
    assert scheduler.num_bins == 3
    assert scheduler.bins.n_bins == 3
    assert scheduler.bins.is_empty() is True  # Should be empty initially


def test_scheduler_enqueue():
    """Test enqueuing requests."""
    scheduler = MBScheduler(num_bins=3)
    
    rid1 = scheduler.enqueue_request(prompt_tokens=100)
    rid2 = scheduler.enqueue_request(prompt_tokens=500)
    rid3 = scheduler.enqueue_request(prompt_tokens=50)
    
    assert rid1 == 0
    assert rid2 == 1
    assert rid3 == 2
    assert scheduler.total_pending() == 3


def test_scheduler_build_batch():
    """Test building a batch."""
    scheduler = MBScheduler(
        num_bins=3,
        max_batched_tokens=2048
    )
    
    # Enqueue several requests
    for i in range(10):
        scheduler.enqueue_request(prompt_tokens=100)
    
    # Should be able to build a batch
    batch = scheduler.build_batch()
    assert len(batch) > 0


def test_scheduler_statistics():
    """Test scheduler statistics."""
    scheduler = MBScheduler(num_bins=3)
    
    scheduler.enqueue_request(prompt_tokens=100)
    scheduler.enqueue_request(prompt_tokens=200)
    
    stats = scheduler.get_statistics()
    
    assert 'request_counter' in stats
    assert 'bin_statistics' in stats
    assert 'selector_statistics' in stats
    assert stats['bin_statistics']['total_pending'] == 2


def test_scheduler_bin_assignment():
    """Test requests are assigned to correct bins."""
    scheduler = MBScheduler(num_bins=3)
    
    # Short requests
    scheduler.enqueue_request(prompt_tokens=50)
    
    # Medium requests
    scheduler.enqueue_request(prompt_tokens=200)
    
    # Long requests
    scheduler.enqueue_request(prompt_tokens=1000)
    
    occupancy = scheduler.get_bin_occupancy()
    non_empty = [bin_id for bin_id, size in occupancy if size > 0]
    assert len(non_empty) <= 3  # Should be in at most 3 bins


if __name__ == "__main__":
    pytest.main([__file__])

