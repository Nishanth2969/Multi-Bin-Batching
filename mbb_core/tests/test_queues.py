"""Tests for request queues."""

import pytest
from mbb_core.queues import Request, BinQueues


def test_request_creation():
    """Test request creation."""
    req = Request(
        rid=1,
        prompt_tokens=100,
        arrival_ms=0,
        pred_out_tokens=50
    )
    
    assert req.rid == 1
    assert req.prompt_tokens == 100
    assert req.bin_id == -1  # Unassigned initially


def test_request_wait_time():
    """Test request wait time update."""
    req = Request(
        rid=1,
        prompt_tokens=100,
        arrival_ms=100,
        pred_out_tokens=50
    )
    
    req.update_wait_time(current_ms=250)
    assert req.wait_time_ms == 150


def test_bin_queues_initialization():
    """Test bin queues initialization."""
    bins = BinQueues(n_bins=3)
    
    assert bins.n_bins == 3
    assert len(bins.queues) == 3
    assert bins.is_empty()


def test_bin_queues_push_pop():
    """Test pushing and popping requests."""
    bins = BinQueues(n_bins=3)
    
    req = Request(rid=1, prompt_tokens=100, arrival_ms=0, pred_out_tokens=50)
    bins.push(0, req)
    
    assert not bins.is_empty()
    assert bins.bin_size(0) == 1
    assert bins.total_pending() == 1
    
    req2 = bins.pop(0)
    assert req2 == req
    assert bins.is_empty()


def test_bin_queues_non_empty():
    """Test non-empty bin detection."""
    bins = BinQueues(n_bins=3)
    
    bins.push(0, Request(rid=1, prompt_tokens=100, arrival_ms=0, pred_out_tokens=50))
    bins.push(2, Request(rid=2, prompt_tokens=200, arrival_ms=0, pred_out_tokens=100))
    
    non_empty = bins.non_empty_bins()
    assert 0 in non_empty
    assert 2 in non_empty
    assert 1 not in non_empty


def test_bin_queues_statistics():
    """Test queue statistics."""
    bins = BinQueues(n_bins=3)
    
    bins.push(0, Request(rid=1, prompt_tokens=100, arrival_ms=0, pred_out_tokens=50))
    bins.push(0, Request(rid=2, prompt_tokens=100, arrival_ms=0, pred_out_tokens=50))
    
    stats = bins.get_stats()
    assert stats['total_enqueued'] == 2
    assert stats['bin_sizes'] == [2, 0, 0]
    assert stats['total_pending'] == 2


def test_bin_queues_out_of_range():
    """Test error handling for out-of-range bin IDs."""
    bins = BinQueues(n_bins=3)
    req = Request(rid=1, prompt_tokens=100, arrival_ms=0, pred_out_tokens=50)
    
    with pytest.raises(ValueError):
        bins.push(3, req)


if __name__ == "__main__":
    pytest.main([__file__])

