"""
Load generation for benchmarking multi-bin batching.

Simulates realistic workload patterns for LLM inference benchmarking.
"""

import random
import numpy as np
from typing import List, Tuple, Dict, Callable
from dataclasses import dataclass
import time


@dataclass
class WorkloadMix:
    """Defines a workload mix with probabilities for different patterns."""
    
    name: str
    arrival_rate: float  # requests per second
    prompt_distribution: List[Tuple[int, float]]  # (num_tokens, probability)
    output_distribution: List[Tuple[int, float]]  # (num_tokens, probability)
    duration_s: int = 60  # benchmark duration in seconds
    
    def generate_arrival_times(self) -> List[float]:
        """Generate Poisson process arrival times."""
        arrival_times = []
        current_time = 0.0
        rate = self.arrival_rate
        
        while current_time < self.duration_s:
            # Poisson inter-arrival time
            interarrival = -np.log(random.random()) / rate
            current_time += interarrival
            if current_time < self.duration_s:
                arrival_times.append(current_time)
        
        return arrival_times
    
    def sample_prompt_length(self) -> int:
        """Sample prompt length from distribution."""
        rand = random.random()
        cumprob = 0.0
        
        for tokens, prob in self.prompt_distribution:
            cumprob += prob
            if rand <= cumprob:
                return tokens
        
        return self.prompt_distribution[0][0]
    
    def sample_output_length(self) -> int:
        """Sample output length from distribution."""
        rand = random.random()
        cumprob = 0.0
        
        for tokens, prob in self.output_distribution:
            cumprob += prob
            if rand <= cumprob:
                return tokens
        
        return self.output_distribution[0][0]


def short_heavy_mix() -> WorkloadMix:
    """Short-heavy workload: small prompts and short outputs."""
    return WorkloadMix(
        name="short_heavy",
        arrival_rate=150.0,
        prompt_distribution=[
            (64, 0.5),
            (128, 0.3),
            (256, 0.2),
        ],
        output_distribution=[
            (64, 0.4),
            (128, 0.4),
            (256, 0.2),
        ],
        duration_s=60
    )


def mixed_mix() -> WorkloadMix:
    """Mixed workload: varied prompt and output lengths."""
    return WorkloadMix(
        name="mixed",
        arrival_rate=100.0,
        prompt_distribution=[
            (64, 0.5),
            (512, 0.3),
            (2048, 0.2),
        ],
        output_distribution=[
            (64, 0.5),
            (512, 0.3),
            (1024, 0.2),
        ],
        duration_s=60
    )


def long_tail_mix() -> WorkloadMix:
    """Long-tail workload: mostly short, some very long."""
    return WorkloadMix(
        name="long_tail",
        arrival_rate=120.0,
        prompt_distribution=[
            (64, 0.7),
            (256, 0.2),
            (1024, 0.1),
        ],
        output_distribution=[
            (64, 0.7),
            (256, 0.2),
            (2048, 0.1),
        ],
        duration_s=60
    )


class RequestGenerator:
    """Generates synthetic requests based on workload mix."""
    
    def __init__(self, mix: WorkloadMix, seed: int = 42):
        """Initialize request generator.
        
        Args:
            mix: Workload mix configuration
            seed: Random seed for reproducibility
        """
        self.mix = mix
        random.seed(seed)
        np.random.seed(seed)
    
    def generate_requests(self) -> List[Tuple[float, int, int]]:
        """Generate list of requests.
        
        Returns:
            List of (arrival_time_s, prompt_tokens, output_tokens)
        """
        arrivals = self.mix.generate_arrival_times()
        requests = []
        
        for arrival_time in arrivals:
            prompt_tokens = self.mix.sample_prompt_length()
            output_tokens = self.mix.sample_output_length()
            requests.append((arrival_time, prompt_tokens, output_tokens))
        
        return requests


class LoadSimulator:
    """Simulates MBB vs continuous batching."""
    
    def __init__(self, scheduler):
        """Initialize load simulator.
        
        Args:
            scheduler: MBScheduler instance
        """
        self.scheduler = scheduler
        self.metrics = {
            'requests_admitted': 0,
            'requests_completed': 0,
            'batches_formed': 0,
            'total_tokens': 0,
            'idle_time_ms': 0,
            'bin_utilization': [],
        }
    
    def simulate(self, requests: List[Tuple[float, int, int]]):
        """Run simulation on request stream.
        
        Args:
            requests: List of (arrival_time_s, prompt_tokens, output_tokens)
        """
        next_arrival_idx = 0
        current_time_ms = 0
        pending_requests = []
        
        while True:
            # Process arrivals
            while (next_arrival_idx < len(requests) and
                   requests[next_arrival_idx][0] * 1000 <= current_time_ms):
                arr_time, prompt_toks, output_toks = requests[next_arrival_idx]
                self.scheduler.enqueue_request(
                    prompt_tokens=prompt_toks,
                    arrival_time_ms=int(arr_time * 1000)
                )
                self.metrics['requests_admitted'] += 1
                next_arrival_idx += 1
            
            # Build and process batch
            if not self.scheduler.bins.is_empty():
                batch = self.scheduler.build_batch(current_time_ms)
                if batch:
                    self.metrics['batches_formed'] += 1
                    for req in batch:
                        self.metrics['requests_completed'] += 1
                        self.metrics['total_tokens'] += req.prompt_tokens + req.pred_out_tokens
            
            # Advance time
            if next_arrival_idx >= len(requests) and self.scheduler.bins.is_empty():
                break
            
            current_time_ms += 50  # 50ms scheduling quantum
            scheduler.update_time(50)
    
    def get_metrics(self) -> Dict:
        """Get collected metrics."""
        return self.metrics.copy()


if __name__ == "__main__":
    # Example usage
    from mbb_core import MBScheduler
    
    scheduler = MBScheduler(num_bins=3)
    mix = short_heavy_mix()
    generator = RequestGenerator(mix, seed=42)
    requests = generator.generate_requests()
    
    simulator = LoadSimulator(scheduler)
    simulator.simulate(requests)
    
    print(f"Metrics: {simulator.get_metrics()}")

