"""
SGLang Multi-Bin Batching Policy Integration.

Implements MBB as a custom scheduling policy for SGLang's batch scheduler.
"""

import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
from collections import deque

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mbb_core.scheduler import MBScheduler
from mbb_core.bins import BinEdges
from mbb_core.predict import HeuristicPredictor, AdaptivePredictor


class MultiBinBatchingPolicy:
    """Multi-Bin Batching policy for SGLang scheduler."""
    
    def __init__(
        self,
        num_bins: int = 8,
        bin_edges: Optional[List[int]] = None,
        batch_size: int = 256,
        predictor_type: str = "adaptive",
        starvation_ms: int = 1000,
        use_equal_mass_bins: bool = True,
        warmup_samples: int = 100
    ):
        """Initialize MBB policy.
        
        Args:
            num_bins: Number of bins for grouping requests
            bin_edges: Custom bin edges (if None, uses equal-mass)
            batch_size: Maximum batch size
            predictor_type: 'heuristic' or 'adaptive'
            starvation_ms: Timeout before promoting starved requests
            use_equal_mass_bins: Use quantile-based equal-mass bins
            warmup_samples: Number of samples for warm-up phase
        """
        self.num_bins = num_bins
        self.batch_size = batch_size
        self.starvation_ms = starvation_ms
        
        self.mbb_scheduler = MBScheduler(
            num_bins=num_bins,
            bin_edges=bin_edges or [],
            batch_size=batch_size,
            predictor_type=predictor_type,
            starvation_ms=starvation_ms,
            use_equal_mass_bins=use_equal_mass_bins,
            warmup_samples=warmup_samples
        )
        
        self.request_map: Dict[str, int] = {}
        self.mbb_id_map: Dict[int, str] = {}
        self.next_id = 0
        
        print(f"✓ MBB Policy initialized: {num_bins} bins, {predictor_type} predictor")
    
    def on_request_arrival(self, request_id: str, prompt_tokens: int) -> int:
        """Called when new request arrives.
        
        Args:
            request_id: Request identifier
            prompt_tokens: Number of prompt tokens
            
        Returns:
            Assigned bin ID
        """
        mbb_id = self.next_id
        self.next_id += 1
        
        self.request_map[request_id] = mbb_id
        self.mbb_id_map[mbb_id] = request_id
        
        self.mbb_scheduler.enqueue_request(prompt_tokens=prompt_tokens)
        
        predicted_output = self.mbb_scheduler.predictor.predict(prompt_tokens)
        bin_id = self.mbb_scheduler.bin_edges.bin_for(predicted_output)
        
        return bin_id
    
    def on_request_completion(self, request_id: str, output_tokens: int):
        """Called when request completes.
        
        Args:
            request_id: Request identifier
            output_tokens: Number of generated tokens
        """
        self.mbb_scheduler.record_output_length(output_tokens)
        
        if request_id in self.request_map:
            mbb_id = self.request_map.pop(request_id)
            if mbb_id in self.mbb_id_map:
                del self.mbb_id_map[mbb_id]
    
    def select_batch(
        self,
        waiting_requests: List[Any],
        current_time_ms: int,
        max_batch_size: Optional[int] = None
    ) -> List[Any]:
        """Select next batch using MBB logic.
        
        Args:
            waiting_requests: List of waiting request objects
            current_time_ms: Current timestamp in milliseconds
            max_batch_size: Maximum batch size (overrides default)
            
        Returns:
            List of selected request objects for next batch
        """
        effective_batch_size = max_batch_size or self.batch_size
        
        batch = self.mbb_scheduler.build_batch(
            current_time_ms=current_time_ms,
            max_batch_size=effective_batch_size
        )
        
        if not batch:
            return []
        
        selected_requests = []
        batch_mbb_ids = {id(req) for req in batch}
        
        for req in waiting_requests:
            req_id = getattr(req, 'request_id', str(req))
            if req_id in self.request_map:
                mbb_id = self.request_map[req_id]
                if mbb_id in batch_mbb_ids:
                    selected_requests.append(req)
        
        return selected_requests[:effective_batch_size]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get MBB statistics.
        
        Returns:
            Dictionary with current MBB stats
        """
        bin_sizes = [len(self.mbb_scheduler.bins.get_bin(i)) 
                     for i in range(self.num_bins)]
        
        return {
            'num_bins': self.num_bins,
            'bin_occupancy': bin_sizes,
            'service_queue_size': len(self.mbb_scheduler.service_queue),
            'batches_formed': self.mbb_scheduler._batches_formed,
            'requests_served': self.mbb_scheduler._requests_served,
            'warmup_complete': (
                hasattr(self.mbb_scheduler, 'bin_histogram') and
                self.mbb_scheduler.bin_histogram.is_ready()
                if self.mbb_scheduler.bin_histogram else True
            )
        }


def create_mbb_policy(config: Dict[str, Any]) -> MultiBinBatchingPolicy:
    """Factory function to create MBB policy from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized MultiBinBatchingPolicy instance
    """
    return MultiBinBatchingPolicy(
        num_bins=config.get('num_bins', 8),
        bin_edges=config.get('bin_edges'),
        batch_size=config.get('batch_size', 256),
        predictor_type=config.get('predictor_type', 'adaptive'),
        starvation_ms=config.get('starvation_ms', 1000),
        use_equal_mass_bins=config.get('use_equal_mass_bins', True),
        warmup_samples=config.get('warmup_samples', 100)
    )

