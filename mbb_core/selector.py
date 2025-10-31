"""
Batch selection logic for multi-bin batching.

Implements policies for selecting requests from bins to form batches.
"""

from typing import List, Optional, Callable
from .queues import BinQueues, Request


class RoundRobinSelector:
    """Round-robin selection across non-empty bins."""
    
    def __init__(self, bins: BinQueues):
        """Initialize round-robin selector.
        
        Args:
            bins: Bin queues to select from
        """
        self.bins = bins
        self._cursor = 0
        self._cached_non_empty = []
    
    def get_next_bin(self, non_empty: Optional[List[int]] = None) -> Optional[int]:
        """Get next bin index in round-robin order.
        
        Args:
            non_empty: Optional pre-computed non-empty bins list (performance)
        
        Returns:
            Next bin index or None if all queues empty
        """
        if non_empty is None:
            non_empty = self.bins.non_empty_bins()
        
        if not non_empty:
            return None
        
        if self._cursor >= len(non_empty):
            self._cursor = 0
        
        bin_idx = non_empty[self._cursor]
        self._cursor = (self._cursor + 1) % len(non_empty)
        return bin_idx
    
    def reset(self):
        """Reset cursor."""
        self._cursor = 0


class WeightedSelector:
    """Weighted selection based on bin queue sizes."""
    
    def __init__(self, bins: BinQueues, weights: Optional[List[float]] = None):
        """Initialize weighted selector.
        
        Args:
            bins: Bin queues to select from
            weights: Optional weights per bin (uniform if None)
        """
        self.bins = bins
        self.n_bins = bins.n_bins
        self.weights = weights if weights else [1.0] * self.n_bins
    
    def get_next_bin(self) -> Optional[int]:
        """Get next bin index weighted by queue sizes.
        
        Returns:
            Next bin index or None if all queues empty
        """
        non_empty = self.bins.non_empty_bins()
        if not non_empty:
            return None
        
        # Weight by queue size * bin weight
        weighted_scores = [
            len(self.bins.queues[i]) * self.weights[i] for i in non_empty
        ]
        
        # Simple argmax (could use random weighted selection)
        max_idx = weighted_scores.index(max(weighted_scores))
        return non_empty[max_idx]


class StarvationGuard:
    """Prevents starvation by promoting old requests."""
    
    def __init__(self, max_wait_ms: int = 500):
        """Initialize starvation guard.
        
        Args:
            max_wait_ms: Maximum wait time before promoting (milliseconds)
        """
        self.max_wait_ms = max_wait_ms
        self._promotion_count = 0
    
    def check_starvation(self, req: Request, current_ms: int) -> bool:
        """Check if a request is starved.
        
        Args:
            req: Request to check
            current_ms: Current timestamp in milliseconds
            
        Returns:
            True if request should be promoted
        """
        wait_time = current_ms - req.arrival_ms
        if wait_time > self.max_wait_ms:
            self._promotion_count += 1
            return True
        return False
    
    def get_promotion_count(self) -> int:
        """Get number of promotions made."""
        return self._promotion_count


class BatchSelector:
    """Main batch builder using multi-bin selection."""
    
    def __init__(
        self,
        kv_budget_tokens: int,
        max_batched_tokens: int,
        starvation_ms: int = 500,
        selection_policy: str = "round_robin"
    ):
        """Initialize batch selector.
        
        Args:
            kv_budget_tokens: KV cache budget in tokens
            max_batched_tokens: Maximum tokens per batch
            starvation_ms: Maximum wait before promoting
            selection_policy: 'round_robin' or 'weighted'
        """
        self.kv_budget_tokens = kv_budget_tokens
        self.max_batched_tokens = max_batched_tokens
        self.starvation_guard = StarvationGuard(starvation_ms)
        self.bins = None
        self.selector = None
        self.selection_policy = selection_policy
        self._cached_non_empty = []
        self._cache_valid = False
    
    def set_bins(self, bins: BinQueues):
        """Set bin queues to select from.
        
        Args:
            bins: BinQueues instance
        """
        self.bins = bins
        if self.selection_policy == "round_robin":
            self.selector = RoundRobinSelector(bins)
        else:
            self.selector = WeightedSelector(bins)
        self._cache_valid = False
    
    def build_batch(self, current_ms: int) -> List[Request]:
        """Build a batch of requests from bins."""
        if not self.bins or self.bins.is_empty():
            return []
        
        if not self._cache_valid:
            self._cached_non_empty = self.bins.non_empty_bins()
            self._cache_valid = True
        
        if not self._cached_non_empty:
            return []
        
        batch = []
        used_tokens = 0
        visited_bins = set()
        starved_bins = []
        
        for bin_id in self._cached_non_empty:
            req = self.bins.peek(bin_id)
            if req and self.starvation_guard.check_starvation(req, current_ms):
                starved_bins.append(bin_id)
        
        bins_to_process = starved_bins + [b for b in self._cached_non_empty if b not in starved_bins]
        
        for bin_id in bins_to_process:
            if bin_id in visited_bins:
                continue
            visited_bins.add(bin_id)
            
            if self.bins.bin_size(bin_id) == 0:
                continue
            
            while self.bins.bin_size(bin_id) > 0:
                req = self.bins.peek(bin_id)
                if req is None:
                    break
                
                need_tokens = req.prompt_tokens + 1
                
                if need_tokens > self.kv_budget_tokens:
                    break
                
                if used_tokens + need_tokens > self.max_batched_tokens:
                    break
                
                req = self.bins.pop(bin_id)
                req.update_wait_time(current_ms)
                batch.append(req)
                used_tokens += need_tokens
                
                if used_tokens >= self.max_batched_tokens:
                    self._cache_valid = False
                    return batch
        
        if batch:
            self._cache_valid = False
        
        return batch
    
    def _estimate_token_cost(self, req: Request) -> int:
        """Estimate token cost for a request."""
        return req.prompt_tokens + 1
    
    def get_stats(self) -> dict:
        """Get selector statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            'starvation_promotions': self.starvation_guard.get_promotion_count(),
        }

