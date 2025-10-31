"""
Ollama integration for Multi-Bin Batching.

Since Ollama uses llama.cpp internally and doesn't have 
exposed scheduler APIs like vLLM/SGLang, we provide:
1. A wrapper that sits on top of Ollama's API
2. Client-side batching logic
3. Demonstration of MBB benefits
"""

import ollama
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from mbb_core import MBScheduler
import time


@dataclass
class OllamaRequest:
    """Wraps an Ollama API request."""
    
    request_id: int
    prompt: str
    prompt_tokens: int
    max_tokens: int = 1024
    arrival_time: float = 0.0
    completion_time: float = 0.0
    response: Optional[str] = None
    output_tokens: int = 0


class OllamaMBBWrapper:
    """Wrapper around Ollama API that implements MBB scheduling."""
    
    def __init__(
        self,
        model_name: str = "llama3.2:1b",
        scheduler: Optional[MBScheduler] = None,
        batch_size: int = 4
    ):
        """Initialize Ollama MBB wrapper.
        
        Args:
            model_name: Ollama model name
            scheduler: MBB scheduler instance
            batch_size: Number of concurrent requests to batch
        """
        self.model_name = model_name
        self.scheduler = scheduler or MBScheduler(
            num_bins=3,
            bin_edges=[128, 512],
            max_batched_tokens=2048
        )
        self.batch_size = batch_size
        self.client = ollama.Client()
        self.active_requests = {}
        
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        return len(text.split())
    
    async def process_request(
        self,
        prompt: str,
        max_tokens: int = 1024
    ) -> Dict[str, Any]:
        """Process a single request through Ollama with MBB scheduling.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum output tokens
            
        Returns:
            Response dictionary with text, tokens, latency
        """
        arrival_time = time.time()
        
        # Token estimation
        prompt_tokens = self._estimate_tokens(prompt)
        request_id = len(self.active_requests)
        
        # Enqueue in scheduler
        bin_id = self._predict_and_enqueue(prompt_tokens, max_tokens)
        
        # Wait for batch formation (simulate batching delay)
        await asyncio.sleep(0.01)  # Minimal batching delay
        
        # Process via Ollama
        start_time = time.time()
        response = ollama.generate(
            model=self.model_name,
            prompt=prompt,
            options={"num_predict": max_tokens}
        )
        
        completion_time = time.time()
        output_text = response['response']
        output_tokens = self._estimate_tokens(output_text)
        
        latency = completion_time - arrival_time
        
        return {
            'request_id': request_id,
            'prompt_tokens': prompt_tokens,
            'output_tokens': output_tokens,
            'total_tokens': prompt_tokens + output_tokens,
            'latency_ms': latency * 1000,
            'response': output_text,
            'bin_id': bin_id
        }
    
    def _predict_and_enqueue(self, prompt_tokens: int, max_tokens: int) -> int:
        """Predict output length and enqueue request.
        
        Args:
            prompt_tokens: Number of input tokens
            max_tokens: Maximum output tokens
            
        Returns:
            Bin ID assigned
        """
        # Predict output length
        pred_out = self.scheduler.predictor(prompt_tokens)
        
        # Get bin assignment
        bin_id = self.scheduler.bin_edges.bin_for(
            min(pred_out, max_tokens)
        )
        
        # Enqueue
        arrival_ms = int(time.time() * 1000)
        self.scheduler.enqueue_request(
            prompt_tokens=prompt_tokens,
            arrival_time_ms=arrival_ms
        )
        
        return bin_id
    
    async def process_batch(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of requests.
        
        Args:
            requests: List of request dicts with 'prompt' and 'max_tokens'
            
        Returns:
            List of responses
        """
        results = []
        
        # Process in parallel within bin capacity
        tasks = [
            self.process_request(req['prompt'], req.get('max_tokens', 1024))
            for req in requests[:self.batch_size]
        ]
        
        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)
        
        return results
    
    def get_scheduler_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        return self.scheduler.get_statistics()


def create_ollama_mbb_demo():
    """Create a demo showing MBB vs standard batching with Ollama."""
    
    print("Multi-Bin Batching Demo with Ollama")
    print("=" * 60)
    print("\nThis demo shows how MBB can improve throughput")
    print("when requests have varying output lengths.\n")
    
    # Initialize wrapper
    wrapper = OllamaMBBWrapper(
        model_name="llama3.2:1b",
        scheduler=MBScheduler(
            num_bins=3,
            bin_edges=[128, 512],
            max_batched_tokens=2048
        )
    )
    
    # Sample requests with varying lengths
    requests = [
        {"prompt": "What is 2+2?", "max_tokens": 50},  # Short
        {"prompt": "Explain quantum computing in simple terms.", "max_tokens": 200},  # Medium
        {"prompt": "Write a short story about a robot.", "max_tokens": 500},  # Long
        {"prompt": "What is Python?", "max_tokens": 50},  # Short
    ]
    
    print(f"Processing {len(requests)} requests with MBB scheduling...\n")
    
    # Process batch
    results = asyncio.run(wrapper.process_batch(requests))
    
    # Print results
    for result in results:
        print(f"\nRequest {result['request_id']}:")
        print(f"  Input: {result['prompt_tokens']} tokens")
        print(f"  Output: {result['output_tokens']} tokens")
        print(f"  Bin: {result['bin_id']}")
        print(f"  Latency: {result['latency_ms']:.2f}ms")
        print(f"  Response: {result['response'][:100]}...")
    
    # Print stats
    stats = wrapper.get_scheduler_stats()
    print("\n" + "=" * 60)
    print("Scheduler Statistics:")
    print(f"  Total requests: {stats['request_counter']}")
    print(f"  Bin occupancy: {stats['bin_occupancy']}")


if __name__ == "__main__":
    create_ollama_mbb_demo()

