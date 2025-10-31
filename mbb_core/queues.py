"""
Request queues and bin management for multi-bin batching.
"""

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple
from datetime import datetime


@dataclass
class Request:
    """Represents a single inference request."""
    
    rid: int
    prompt_tokens: int
    arrival_ms: int
    pred_out_tokens: int
    bin_id: int = -1
    wait_time_ms: int = 0
    prompt_text: str = ""  # Optional, for model-based predictors
    
    def update_wait_time(self, current_ms: int):
        """Update wait time based on current timestamp."""
        self.wait_time_ms = current_ms - self.arrival_ms


class BinQueues:
    """Manages per-bin queues of requests."""
    
    def __init__(self, n_bins: int):
        """Initialize bin queues.
        
        Args:
            n_bins: Number of bins
        """
        self.n_bins = n_bins
        self.queues: List[Deque[Request]] = [deque() for _ in range(n_bins)]
        self._total_enqueued = 0
        self._total_dequeued = 0
        
    def push(self, bin_id: int, req: Request):
        """Add request to specified bin queue.
        
        Args:
            bin_id: Target bin index
            req: Request to enqueue
        """
        if not (0 <= bin_id < self.n_bins):
            raise ValueError(f"Bin ID {bin_id} out of range [0, {self.n_bins})")
        
        req.bin_id = bin_id
        self.queues[bin_id].append(req)
        self._total_enqueued += 1
    
    def pop(self, bin_id: int) -> Optional[Request]:
        """Remove and return next request from bin queue.
        
        Args:
            bin_id: Bin index
            
        Returns:
            Request or None if queue is empty
        """
        if not self.queues[bin_id]:
            return None
        self._total_dequeued += 1
        return self.queues[bin_id].popleft()
    
    def peek(self, bin_id: int) -> Optional[Request]:
        """Return next request without removing it.
        
        Args:
            bin_id: Bin index
            
        Returns:
            Request or None if queue is empty
        """
        if not self.queues[bin_id]:
            return None
        return self.queues[bin_id][0]
    
    def non_empty_bins(self) -> List[int]:
        """Return list of bin indices with non-empty queues.
        
        Returns:
            List of bin indices with pending requests
        """
        return [i for i, q in enumerate(self.queues) if q]
    
    def is_empty(self) -> bool:
        """Check if all queues are empty."""
        return all(not q for q in self.queues)
    
    def total_pending(self) -> int:
        """Total number of pending requests across all bins."""
        return sum(len(q) for q in self.queues)
    
    def bin_size(self, bin_id: int) -> int:
        """Get queue size for a specific bin."""
        return len(self.queues[bin_id])
    
    def get_bin_occupancy(self) -> List[Tuple[int, int]]:
        """Get (bin_id, queue_size) for all bins.
        
        Returns:
            List of tuples (bin_id, queue_size)
        """
        return [(i, len(q)) for i, q in enumerate(self.queues)]
    
    def get_stats(self) -> dict:
        """Get statistics about queue operations.
        
        Returns:
            Dictionary with queue statistics
        """
        return {
            'total_enqueued': self._total_enqueued,
            'total_dequeued': self._total_dequeued,
            'total_pending': self.total_pending(),
            'bin_sizes': [len(q) for q in self.queues],
            'non_empty_bins': len(self.non_empty_bins())
        }


class RequestReassigner:
    """Handles reassignment of requests when predictions are inaccurate."""
    
    def __init__(self, bins: BinQueues):
        """Initialize reassigner.
        
        Args:
            bins: Bin queues to manage
        """
        self.bins = bins
        self._reassignment_count = 0
    
    def check_and_reassign(self, req: Request, true_output_tokens: int, 
                          bin_edges: 'BinEdges'):
        """Check if request should be moved to a different bin.
        
        Args:
            req: Request to check
            true_output_tokens: Actual output length
            bin_edges: BinEdges to determine target bin
        """
        pred_bin = bin_edges.bin_for(req.pred_out_tokens)
        true_bin = bin_edges.bin_for(true_output_tokens)
        
        if pred_bin != true_bin:
            self._reassignment_count += 1
    
    def get_reassignment_count(self) -> int:
        return self._reassignment_count

