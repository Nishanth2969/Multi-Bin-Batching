"""
Bin management for grouping requests by predicted length.
"""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class BinEdges:
    """Defines bin boundaries for request grouping.
    
    Example: edges=[128, 512] creates bins:
    - Bin 0: predicted_length < 128
    - Bin 1: 128 <= predicted_length < 512
    - Bin 2: predicted_length >= 512
    """
    
    edges: List[int]
    
    def __post_init__(self):
        """Validate and sort edges."""
        if self.edges:
            self.edges = sorted(set(self.edges))
    
    def num_bins(self) -> int:
        """Return number of bins (num_edges + 1)."""
        return len(self.edges) + 1
    
    def bin_for(self, predicted_len: int) -> int:
        """Assign a predicted length to a bin.
        
        Args:
            predicted_len: Predicted output length in tokens
            
        Returns:
            Bin index (0 to num_bins-1)
        """
        for i, edge in enumerate(self.edges):
            if predicted_len < edge:
                return i
        return len(self.edges)
    
    def get_bin_range(self, bin_id: int) -> Tuple[int, int]:
        """Get the min and max predicted lengths for a bin.
        
        Args:
            bin_id: Bin index
            
        Returns:
            Tuple of (min, max) predicted length for this bin
        """
        n_bins = self.num_bins()
        if bin_id < 0 or bin_id >= n_bins:
            raise ValueError(f"Bin ID {bin_id} out of range [0, {n_bins})")
        
        min_len = self.edges[bin_id - 1] if bin_id > 0 else 0
        max_len = self.edges[bin_id] if bin_id < len(self.edges) else float('inf')
        
        return (min_len, max_len)
    
    def __repr__(self) -> str:
        ranges = [f"bin{i}: {self.get_bin_range(i)}" for i in range(self.num_bins())]
        return f"BinEdges({', '.join(ranges)})"


def create_equal_width_bins(max_length: int, num_bins: int) -> BinEdges:
    """Create bins with equal width.
    
    Args:
        max_length: Maximum expected output length
        num_bins: Number of bins to create
        
    Returns:
        BinEdges with equal-width bins
    """
    if num_bins <= 1:
        return BinEdges([])
    
    step = max_length / num_bins
    edges = [int(i * step) for i in range(1, num_bins)]
    return BinEdges(edges)


def create_quantile_bins(observed_lengths: List[int], num_bins: int) -> BinEdges:
    """Create bins based on quantiles of observed lengths (equal-mass bins).
    
    Args:
        observed_lengths: List of observed output lengths
        num_bins: Number of bins to create
        
    Returns:
        BinEdges with quantile-based bins
    """
    if not observed_lengths or num_bins <= 1:
        return BinEdges([])
    
    sorted_lengths = sorted(observed_lengths)
    n = len(sorted_lengths)
    quantiles = [i / num_bins for i in range(1, num_bins)]
    edges = [sorted_lengths[int(q * n)] for q in quantiles]
    edges = sorted(set(edges))
    
    return BinEdges(edges)


class BinHistogram:
    """Tracks output length distribution for dynamic bin edge computation."""
    
    def __init__(self, num_bins: int = 3, warmup_samples: int = 100):
        """Initialize histogram tracker.
        
        Args:
            num_bins: Target number of bins
            warmup_samples: Number of samples before computing edges
        """
        self.num_bins = num_bins
        self.warmup_samples = warmup_samples
        self.observed_lengths: List[int] = []
        self.current_edges: BinEdges = BinEdges([128, 512])  # Default
        self.warmup_complete = False
    
    def add_observation(self, output_length: int):
        """Add observed output length.
        
        Args:
            output_length: Actual output length in tokens
        """
        if not self.warmup_complete:
            self.observed_lengths.append(output_length)
            
            if len(self.observed_lengths) >= self.warmup_samples:
                self._update_edges()
                self.warmup_complete = True
    
    def _update_edges(self):
        """Update bin edges based on collected observations."""
        if len(self.observed_lengths) >= self.num_bins:
            self.current_edges = create_quantile_bins(
                self.observed_lengths, self.num_bins
            )
    
    def get_edges(self) -> BinEdges:
        """Get current bin edges.
        
        Returns:
            Current BinEdges (quantile-based after warmup)
        """
        return self.current_edges
    
    def is_warmed_up(self) -> bool:
        """Check if warmup phase is complete."""
        return self.warmup_complete
    
    def reset(self):
        """Reset histogram for new warmup."""
        self.observed_lengths.clear()
        self.warmup_complete = False

