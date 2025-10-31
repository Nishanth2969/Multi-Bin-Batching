"""Tests for bin management."""

import pytest
from mbb_core.bins import BinEdges, create_equal_width_bins, create_quantile_bins


def test_bin_edges_initialization():
    """Test bin edges initialization."""
    edges = BinEdges([128, 512])
    
    assert edges.num_bins() == 3
    assert edges.edges == [128, 512]


def test_bin_edges_auto_sorted():
    """Test edges are auto-sorted."""
    edges = BinEdges([512, 128, 256])
    assert edges.edges == [128, 256, 512]


def test_bin_assignment():
    """Test bin assignment logic."""
    edges = BinEdges([128, 512])
    
    assert edges.bin_for(50) == 0
    assert edges.bin_for(128) == 1  # >= 128, goes to bin 1
    assert edges.bin_for(129) == 1  # >= 128 and < 512
    assert edges.bin_for(512) == 2  # >= 512, goes to bin 2
    assert edges.bin_for(513) == 2


def test_bin_ranges():
    """Test bin range retrieval."""
    edges = BinEdges([128, 512])
    
    assert edges.get_bin_range(0) == (0, 128)
    assert edges.get_bin_range(1) == (128, 512)
    assert edges.get_bin_range(2) == (512, float('inf'))


def test_single_bin():
    """Test single bin case."""
    edges = BinEdges([])
    assert edges.num_bins() == 1
    assert edges.bin_for(0) == 0
    assert edges.bin_for(10000) == 0


def test_create_equal_width_bins():
    """Test equal width bin creation."""
    edges = create_equal_width_bins(max_length=1000, num_bins=4)
    
    assert edges.num_bins() == 4
    assert len(edges.edges) == 3
    
    # Should have edges at 250, 500, 750
    assert edges.edges[0] == 250
    assert edges.edges[1] == 500
    assert edges.edges[2] == 750


def test_create_quantile_bins():
    """Test quantile bin creation."""
    observed = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    edges = create_quantile_bins(observed, num_bins=3)
    
    assert edges.num_bins() == 3
    assert len(edges.edges) == 2


if __name__ == "__main__":
    pytest.main([__file__])

