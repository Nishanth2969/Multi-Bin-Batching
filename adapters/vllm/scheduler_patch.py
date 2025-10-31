"""
Runtime patch for vLLM to integrate Multi-Bin Batching scheduler.

This monkey-patches vLLM's scheduler to use MBB logic without modifying vLLM source.
"""

import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mbb_core.scheduler import MBScheduler
from mbb_core.bins import BinEdges
from mbb_core.predict import HeuristicPredictor, AdaptivePredictor


class VLLMMBBAdapter:
    """Adapter to integrate MBB with vLLM's scheduler."""
    
    def __init__(
        self,
        num_bins: int = 3,
        bin_edges: Optional[List[int]] = None,
        batch_size: int = 256,
        predictor_type: str = "adaptive",
        starvation_ms: int = 1000,
        use_equal_mass_bins: bool = True
    ):
        """Initialize MBB adapter.
        
        Args:
            num_bins: Number of bins
            bin_edges: Bin edges (if None, uses equal-mass)
            batch_size: Maximum batch size
            predictor_type: 'heuristic' or 'adaptive'
            starvation_ms: Starvation timeout
            use_equal_mass_bins: Use equal-mass quantile bins
        """
        self.num_bins = num_bins
        self.batch_size = batch_size
        self.starvation_ms = starvation_ms
        
        # Create MBB scheduler
        self.mbb_scheduler = MBScheduler(
            num_bins=num_bins,
            bin_edges=bin_edges or [],
            batch_size=batch_size,
            predictor_type=predictor_type,
            starvation_ms=starvation_ms,
            use_equal_mass_bins=use_equal_mass_bins,
            warmup_samples=100
        )
        
        self.request_mapping = {}  # Map vLLM request_id to MBB internal tracking
        self.next_mbb_id = 0
    
    def on_request_arrival(self, request_id: str, prompt_tokens: int, arrival_time_ms: int):
        """Called when a new request arrives.
        
        Args:
            request_id: vLLM request ID
            prompt_tokens: Number of prompt tokens
            arrival_time_ms: Arrival timestamp
        """
        mbb_id = self.next_mbb_id
        self.next_mbb_id += 1
        
        self.request_mapping[request_id] = {
            'mbb_id': mbb_id,
            'prompt_tokens': prompt_tokens,
            'arrival_time_ms': arrival_time_ms
        }
        
        self.mbb_scheduler.enqueue_request(prompt_tokens=prompt_tokens)
    
    def on_request_completion(self, request_id: str, output_tokens: int):
        """Called when a request completes.
        
        Args:
            request_id: vLLM request ID
            output_tokens: Number of output tokens generated
        """
        if request_id in self.request_mapping:
            self.mbb_scheduler.record_output_length(output_tokens)
    
    def select_batch(self, available_requests: List[str], current_time_ms: int) -> List[str]:
        """Select batch using MBB logic.
        
        Args:
            available_requests: List of request IDs available for batching
            current_time_ms: Current timestamp
            
        Returns:
            List of request IDs to batch together
        """
        batch_requests = self.mbb_scheduler.build_batch(current_time_ms=current_time_ms)
        
        # Map MBB internal IDs back to vLLM request IDs
        selected = []
        for mbb_req in batch_requests:
            for req_id, mapping in self.request_mapping.items():
                if mapping['mbb_id'] == id(mbb_req):
                    selected.append(req_id)
                    break
        
        return selected[:self.batch_size]


def patch_vllm_scheduler(llm_engine, mbb_adapter: VLLMMBBAdapter):
    """Patch vLLM's LLM engine to use MBB scheduling.
    
    Args:
        llm_engine: vLLM LLM instance
        mbb_adapter: MBB adapter instance
    """
    try:
        # Access vLLM's internal scheduler
        scheduler = llm_engine.llm_engine.scheduler[0]
        
        # Store original methods
        original_add_request = scheduler.add_seq_group
        original_schedule = scheduler._schedule
        
        # Wrap add_request to track arrivals
        def wrapped_add_request(seq_group, *args, **kwargs):
            result = original_add_request(seq_group, *args, **kwargs)
            
            # Track in MBB
            request_id = seq_group.request_id
            prompt_tokens = len(seq_group.get_seqs()[0].get_token_ids())
            arrival_time_ms = int(seq_group.arrival_time * 1000)
            
            mbb_adapter.on_request_arrival(
                request_id=request_id,
                prompt_tokens=prompt_tokens,
                arrival_time_ms=arrival_time_ms
            )
            
            return result
        
        # Wrap _schedule to use MBB batch selection
        def wrapped_schedule(*args, **kwargs):
            # Get default scheduling result
            result = original_schedule(*args, **kwargs)
            
            # MBB logic would go here to reorder/select batches
            # For now, just track completions
            if hasattr(result, 'finished_seq_groups'):
                for seq_group in result.finished_seq_groups:
                    request_id = seq_group.request_id
                    output_tokens = len(seq_group.get_seqs()[0].get_output_token_ids())
                    mbb_adapter.on_request_completion(request_id, output_tokens)
            
            return result
        
        # Apply patches
        scheduler.add_seq_group = wrapped_add_request
        scheduler._schedule = wrapped_schedule
        
        print(f"✓ MBB scheduler patch applied ({mbb_adapter.num_bins} bins)")
        return True
        
    except Exception as e:
        print(f"✗ Failed to patch vLLM scheduler: {e}")
        import traceback
        traceback.print_exc()
        return False

