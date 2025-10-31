"""
Main multi-bin scheduler that orchestrates prediction, binning, and selection.

Implements Algorithm 1 from Multi-Bin Batching paper (arXiv:2412.04504v1):
1. Estimate service time (output length) for each request
2. Assign request to bin i where li−1 ≤ l < li
3. Form batches of size B within each bin
4. Add completed batches to service queue
5. Serve batches using "first completed batch, first served" policy
6. Batch service time = max of individual request service times
"""

from typing import List, Optional, Dict, Any
from collections import deque
from .predict import HeuristicPredictor
from .bins import BinEdges, BinHistogram
from .queues import Request, BinQueues, RequestReassigner
from .selector import BatchSelector


class MBScheduler:
    """Multi-Bin Scheduler implementing Algorithm 1 from paper.
    
    Implements "first completed batch, first served" policy per paper's default.
    """
    
    def __init__(
        self,
        num_bins: int = 3,
        bin_edges: Optional[List[int]] = None,
        kv_budget_tokens: int = 8192,
        max_batched_tokens: int = 2048,
        batch_size: Optional[int] = None,  # Fixed batch size B (paper's Algorithm 1)
        starvation_ms: int = 500,
        predictor_type: str = "heuristic",
        selection_policy: str = "round_robin",
        use_equal_mass_bins: bool = False,
        warmup_samples: int = 100
    ):
        """Initialize multi-bin scheduler.
        
        Args:
            num_bins: Number of bins for grouping
            bin_edges: Bin boundaries (auto-generated if None)
            kv_budget_tokens: KV cache budget
            max_batched_tokens: Maximum tokens per batch
            batch_size: Fixed batch size B (if None, uses max_batched_tokens)
            starvation_ms: Maximum wait before promotion
            predictor_type: Type of predictor ('heuristic', 'oracle', 'model')
            selection_policy: Selection policy ('round_robin', 'weighted')
            use_equal_mass_bins: Use equal-mass quantile bins (Lemma 4.1)
            warmup_samples: Samples for warmup histogram
        """
        self.num_bins = num_bins
        self.kv_budget_tokens = kv_budget_tokens
        self.max_batched_tokens = max_batched_tokens
        self.batch_size = batch_size
        
        if bin_edges is None:
            self.bin_edges = self._create_default_edges()
        else:
            self.bin_edges = BinEdges(bin_edges)
            self.num_bins = self.bin_edges.num_bins()
        
        self.predictor = self._create_predictor(predictor_type)
        self.bins = BinQueues(self.num_bins)
        self.use_equal_mass_bins = use_equal_mass_bins
        self.bin_histogram = None
        if use_equal_mass_bins:
            self.bin_histogram = BinHistogram(num_bins=num_bins, warmup_samples=warmup_samples)
        
        self.selector = BatchSelector(
            kv_budget_tokens=kv_budget_tokens,
            max_batched_tokens=max_batched_tokens,
            starvation_ms=starvation_ms,
            selection_policy=selection_policy
        )
        self.selector.set_bins(self.bins)
        self.reassigner = RequestReassigner(self.bins)
        self.service_queue: deque = deque()
        self._request_counter = 0
        self._current_time_ms = 0
        self._batches_formed = 0
    
    def _create_default_edges(self) -> BinEdges:
        """Create default bin edges."""
        return BinEdges([128, 512])
    
    def _create_predictor(self, predictor_type: str):
        """Create predictor instance."""
        if predictor_type == "heuristic":
            return HeuristicPredictor()
        elif predictor_type == "oracle":
            from .predict import OraclePredictor
            return OraclePredictor()
        elif predictor_type == "model":
            from .predict import ModelBasedPredictor
            return ModelBasedPredictor()
        else:
            raise ValueError(f"Unknown predictor type: {predictor_type}")
    
    def enqueue_request(
        self,
        prompt_tokens: int,
        prompt_text: str = "",
        arrival_time_ms: Optional[int] = None
    ) -> int:
        """Enqueue a new request.
        
        Args:
            prompt_tokens: Number of tokens in prompt
            prompt_text: Optional prompt text
            arrival_time_ms: Arrival timestamp (uses internal clock if None)
            
        Returns:
            Request ID
        """
        rid = self._request_counter
        self._request_counter += 1
        
        if arrival_time_ms is None:
            arrival_time_ms = self._current_time_ms
        
        # Predict output length
        pred_output = self.predictor(prompt_tokens)
        
        # Assign to bin
        bin_id = self.bin_edges.bin_for(pred_output)
        
        # Create request
        req = Request(
            rid=rid,
            prompt_tokens=prompt_tokens,
            arrival_ms=arrival_time_ms,
            pred_out_tokens=pred_output,
            bin_id=bin_id,
            prompt_text=prompt_text
        )
        
        # Enqueue
        self.bins.push(bin_id, req)
        
        return rid
    
    def build_batch(self, current_time_ms: Optional[int] = None) -> List[Request]:
        """Build next batch implementing Algorithm 1 from paper.
        
        Steps:
        1. Form batches of size B within each bin (when available)
        2. Add completed batches to service queue
        3. Serve batches using "first completed batch, first served"
        
        Args:
            current_time_ms: Current timestamp (updates internal clock if None)
            
        Returns:
            List of requests for the batch (from service queue)
        """
        if current_time_ms is None:
            current_time_ms = self._current_time_ms
        
        self._current_time_ms = current_time_ms
        
        if self.bin_histogram and self.bin_histogram.is_warmed_up():
            new_edges = self.bin_histogram.get_edges()
            if new_edges.edges != self.bin_edges.edges:
                self.bin_edges = new_edges
                self._reassign_to_new_bins()
        
        self._form_batches_in_bins(current_time_ms)
        
        if self.service_queue:
            batch = self.service_queue.popleft()
            return batch
        
        return []
    
    def _form_batches_in_bins(self, current_time_ms: int):
        """Form batches of size B within each bin."""
        batch_size = self.batch_size if self.batch_size else self.max_batched_tokens
        
        for bin_id in range(self.num_bins):
            while self.bins.bin_size(bin_id) >= batch_size:
                batch = []
                for _ in range(batch_size):
                    req = self.bins.pop(bin_id)
                    if req is None:
                        break
                    req.update_wait_time(current_time_ms)
                    batch.append(req)
                
                if len(batch) == batch_size:
                    self.service_queue.append(batch)
                    self._batches_formed += 1
    
    def _reassign_to_new_bins(self):
        """Reassign requests when bin edges change."""
        all_requests = []
        for bin_id in range(self.num_bins):
            while True:
                req = self.bins.pop(bin_id)
                if req is None:
                    break
                all_requests.append(req)
        
        for req in all_requests:
            new_bin_id = self.bin_edges.bin_for(req.pred_out_tokens)
            req.bin_id = new_bin_id
            self.bins.push(new_bin_id, req)
    
    def record_output_length(self, output_length: int):
        """Record actual output length for warmup histogram."""
        if self.bin_histogram:
            self.bin_histogram.add_observation(output_length)
    
    def update_time(self, elapsed_ms: int):
        """Advance internal clock.
        
        Args:
            elapsed_ms: Milliseconds to advance
        """
        self._current_time_ms += elapsed_ms
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get scheduler statistics.
        
        Returns:
            Dictionary with scheduler statistics
        """
        bin_stats = self.bins.get_stats()
        selector_stats = self.selector.get_stats()
        
        return {
            'request_counter': self._request_counter,
            'current_time_ms': self._current_time_ms,
            'batches_formed': self._batches_formed,
            'service_queue_size': len(self.service_queue),
            'bin_statistics': bin_stats,
            'selector_statistics': selector_stats,
            'bin_edges': self.bin_edges.edges,
            'bin_occupancy': self.bins.get_bin_occupancy(),
            'equal_mass_bins': self.use_equal_mass_bins,
            'warmup_complete': self.bin_histogram.is_warmed_up() if self.bin_histogram else False
        }
    
    def get_bin_occupancy(self) -> List[tuple]:
        """Get bin occupancy.
        
        Returns:
            List of (bin_id, queue_size) tuples
        """
        return self.bins.get_bin_occupancy()
    
    def is_empty(self) -> bool:
        """Check if all queues are empty."""
        return self.bins.is_empty()
    
    def total_pending(self) -> int:
        """Total number of pending requests."""
        return self.bins.total_pending()

