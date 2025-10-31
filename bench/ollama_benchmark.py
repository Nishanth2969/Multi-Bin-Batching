"""
Real Ollama Benchmark for Multi-Bin Batching

Calls actual Ollama streaming API with real prompts.
Measures TTFT, TPOT, TPS, and compares FIFO vs MBB.
"""

import asyncio
import httpx
import json
import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import sys
import os
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from mbb_core import MBScheduler


@dataclass
class RequestMetrics:
    """Metrics for a single request."""
    request_id: int
    prompt: str
    prompt_length: int
    arrival_time: float
    start_time: float
    completion_time: float
    ttft_ms: Optional[float]
    tokens: int
    tpot_ms: float
    duration_s: float
    bin_id: int
    wait_ms: float
    predicted_tokens: int


async def call_ollama_streaming(
    prompt: str,
    model: str = "llama3.2:1b",
    max_tokens: int = 256,
    simulate: bool = False
) -> Dict[str, any]:
    """Call Ollama streaming API and measure metrics.
    
    Args:
        prompt: Input prompt
        model: Model name
        max_tokens: Maximum tokens to generate
        simulate: If True, simulate instead of calling real API
        
    Returns:
        Dictionary with ttft_ms, tokens, tpot_ms, duration_s
    """
    t0 = time.perf_counter()
    ttft = None
    tokens = 0
    response_text = ""
    
    if simulate:
        # Simulate realistic timing
        prompt_tokens = len(prompt.split())
        estimated_output = min(int(0.5 * prompt_tokens + 32), max_tokens)
        
        # Simulate prefill time
        await asyncio.sleep(0.001 * prompt_tokens)
        ttft = time.perf_counter() - t0
        
        # Simulate decode
        for i in range(estimated_output):
            await asyncio.sleep(0.002)  # 2ms per token
            tokens += 1
            response_text += "simulated "
        
        duration = time.perf_counter() - t0
        tpot = duration / max(tokens, 1)
        
        return {
            "ttft_ms": ttft * 1000,
            "tokens": tokens,
            "tpot_ms": tpot * 1000,
            "duration_s": duration,
            "response": response_text
        }
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                "http://127.0.0.1:11434/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": True,
                    "options": {"num_predict": max_tokens}
                }
            ) as r:
                async for line in r.aiter_lines():
                    if not line:
                        continue
                    try:
                        j = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    
                    if "response" in j and j["response"]:
                        if ttft is None:
                            ttft = time.perf_counter() - t0
                        response_text += j["response"]
                        tokens += 1
                    
                    if j.get("done"):
                        break
    
    except Exception as e:
        print(f"Ollama not available, using simulation mode: {e}")
        return await call_ollama_streaming(prompt, model, max_tokens, simulate=True)
    
    duration = time.perf_counter() - t0
    tpot = duration / max(tokens, 1)
    
    return {
        "ttft_ms": ttft * 1000 if ttft else None,
        "tokens": tokens,
        "tpot_ms": tpot * 1000,
        "duration_s": duration,
        "response": response_text
    }


class OllamaBenchmark:
    """Benchmark harness for Ollama + MBB."""
    
    def __init__(
        self,
        model: str = "llama3.2:1b",
        max_tokens: int = 256,
        arrival_rate: float = 3.0,
        simulate: bool = False
    ):
        """Initialize benchmark.
        
        Args:
            model: Ollama model name
            max_tokens: Max tokens per request
            arrival_rate: Requests per second (for throttling)
            simulate: Use simulation instead of real Ollama
        """
        self.model = model
        self.max_tokens = max_tokens
        self.arrival_rate = arrival_rate
        self.interarrival_delay = 1.0 / arrival_rate if arrival_rate > 0 else 0
        self.simulate = simulate
    
    async def run_fifo(self, prompts: List[str]) -> List[RequestMetrics]:
        """Run FIFO baseline (strict arrival order).
        
        Args:
            prompts: List of prompts
            
        Returns:
            List of request metrics
        """
        print(f"\nRunning FIFO baseline ({len(prompts)} requests)...")
        results = []
        t0_total = time.perf_counter()
        
        for i, prompt in enumerate(prompts):
            arrival_time = time.perf_counter()
            
            # Throttle arrivals
            if i > 0:
                await asyncio.sleep(self.interarrival_delay)
            
            start_time = time.perf_counter()
            
            # Call Ollama
            metrics = await call_ollama_streaming(prompt, self.model, self.max_tokens, self.simulate)
            
            if metrics is None:
                continue
            
            completion_time = time.perf_counter()
            wait_ms = (start_time - arrival_time) * 1000
            
            results.append(RequestMetrics(
                request_id=i,
                prompt=prompt[:50] + "..." if len(prompt) > 50 else prompt,
                prompt_length=len(prompt),
                arrival_time=arrival_time - t0_total,
                start_time=start_time - t0_total,
                completion_time=completion_time - t0_total,
                ttft_ms=metrics["ttft_ms"],
                tokens=metrics["tokens"],
                tpot_ms=metrics["tpot_ms"],
                duration_s=metrics["duration_s"],
                bin_id=0,  # Single queue
                wait_ms=wait_ms,
                predicted_tokens=0
            ))
        
        elapsed = time.perf_counter() - t0_total
        print(f"FIFO completed in {elapsed:.2f}s")
        
        return results
    
    async def run_mbb(
        self,
        prompts: List[str],
        scheduler: MBScheduler
    ) -> List[RequestMetrics]:
        """Run Multi-Bin Batching.
        
        Args:
            prompts: List of prompts
            scheduler: MBScheduler instance
            
        Returns:
            List of request metrics
        """
        print(f"\nRunning MBB ({len(prompts)} requests, {scheduler.num_bins} bins)...")
        results = []
        t0_total = time.perf_counter()
        pending = []
        
        # Use tqdm for progress
        for i, prompt in enumerate(tqdm(prompts, desc="Enqueuing")):
            arrival_time = time.perf_counter()
            prompt_tokens = len(prompt.split())
            
            # Enqueue in scheduler
            arrival_ms = int((arrival_time - t0_total) * 1000)
            rid = scheduler.enqueue_request(
                prompt_tokens=prompt_tokens,
                arrival_time_ms=arrival_ms
            )
            
            # Get bin assignment
            batch = scheduler.build_batch(current_time_ms=arrival_ms)
            if batch:
                req = batch[0]
                
                pending.append({
                    'prompt': prompt,
                    'arrival_time': arrival_time,
                    'bin_id': req.bin_id,
                    'predicted_tokens': req.pred_out_tokens,
                    'request_id': i
                })
            
            # Throttle arrivals
            if i > 0:
                await asyncio.sleep(self.interarrival_delay)
        
        # Process all pending requests
        for idx, item in enumerate(tqdm(pending, desc="Processing")):
            start_time = time.perf_counter()
            
            # Call Ollama
            metrics = await call_ollama_streaming(
                item['prompt'],
                self.model,
                self.max_tokens,
                self.simulate
            )
            
            if metrics is None:
                continue
            
            completion_time = time.perf_counter()
            wait_ms = (start_time - item['arrival_time']) * 1000
            
            results.append(RequestMetrics(
                request_id=item['request_id'],
                prompt=item['prompt'][:50] + "..." if len(item['prompt']) > 50 else item['prompt'],
                prompt_length=len(item['prompt']),
                arrival_time=item['arrival_time'] - t0_total,
                start_time=start_time - t0_total,
                completion_time=completion_time - t0_total,
                ttft_ms=metrics["ttft_ms"],
                tokens=metrics["tokens"],
                tpot_ms=metrics["tpot_ms"],
                duration_s=metrics["duration_s"],
                bin_id=item['bin_id'],
                wait_ms=wait_ms,
                predicted_tokens=item['predicted_tokens']
            ))
        
        elapsed = time.perf_counter() - t0_total
        print(f"MBB completed in {elapsed:.2f}s")
        
        return results


def load_prompts(filepath: str) -> List[str]:
    """Load prompts from file.
    
    Args:
        filepath: Path to prompt file
        
    Returns:
        List of prompts
    """
    with open(filepath, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def analyze_results(results: List[RequestMetrics], name: str) -> Dict[str, any]:
    """Analyze benchmark results.
    
    Args:
        results: List of request metrics
        name: Run name
        
    Returns:
        Summary statistics
    """
    if not results:
        return {}
    
    df = pd.DataFrame([asdict(r) for r in results])
    
    # Calculate aggregates
    total_tokens = df['tokens'].sum()
    total_time = df['completion_time'].max()
    tps = total_tokens / max(total_time, 1e-6)
    
    ttft_values = df['ttft_ms'].dropna()
    tpot_values = df['tpot_ms'].dropna()
    
    summary = {
        'name': name,
        'requests': len(results),
        'total_tokens': int(total_tokens),
        'total_time_s': round(total_time, 2),
        'tps': round(tps, 2),
        'ttft_p50_ms': round(np.percentile(ttft_values, 50), 2) if len(ttft_values) > 0 else 0,
        'ttft_p95_ms': round(np.percentile(ttft_values, 95), 2) if len(ttft_values) > 0 else 0,
        'tpot_p50_ms': round(np.percentile(tpot_values, 50), 2) if len(tpot_values) > 0 else 0,
        'tpot_p95_ms': round(np.percentile(tpot_values, 95), 2) if len(tpot_values) > 0 else 0,
        'wait_avg_ms': round(df['wait_ms'].mean(), 2),
        'wait_max_ms': round(df['wait_ms'].max(), 2)
    }
    
    return summary


async def run_benchmark(workload: str, num_bins: int = 3):
    """Run complete benchmark for a workload.
    
    Args:
        workload: Workload name (short_heavy, mixed, long_tail)
        num_bins: Number of bins for MBB
    """
    print(f"\n{'='*60}")
    print(f"Benchmark: {workload}")
    print(f"{'='*60}")
    
    # Load prompts
    prompt_file = f"data/{workload}.txt"
    if not Path(prompt_file).exists():
        print(f"Error: {prompt_file} not found. Run data/download_datasets.py first.")
        return
    
    prompts = load_prompts(prompt_file)
    print(f"Loaded {len(prompts)} prompts")
    
    # Check if Ollama is available
    simulate = False
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            await client.get("http://127.0.0.1:11434/api/tags")
    except:
        print("⚠️  Ollama not running - using simulation mode")
        print("   For real results: Start Ollama with 'ollama serve'\n")
        simulate = True
    
    # Initialize benchmark
    benchmark = OllamaBenchmark(
        model="llama3.2:1b",
        max_tokens=256,
        arrival_rate=3.0,
        simulate=simulate
    )
    
    # Run FIFO
    fifo_results = await benchmark.run_fifo(prompts)
    
    # Run MBB
    scheduler = MBScheduler(
        num_bins=num_bins,
        bin_edges=[128, 512] if num_bins == 3 else [256],
        max_batched_tokens=2048,
        starvation_ms=500
    )
    mbb_results = await benchmark.run_mbb(prompts, scheduler)
    
    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    fifo_df = pd.DataFrame([asdict(r) for r in fifo_results])
    mbb_df = pd.DataFrame([asdict(r) for r in mbb_results])
    
    fifo_df.to_csv(results_dir / f"{workload}_fifo_k{num_bins}.csv", index=False)
    mbb_df.to_csv(results_dir / f"{workload}_mbb_k{num_bins}.csv", index=False)
    
    # Analyze
    fifo_summary = analyze_results(fifo_results, "FIFO")
    mbb_summary = analyze_results(mbb_results, f"MBB (k={num_bins})")
    
    # Print comparison
    print(f"\n{'='*60}")
    print("Results Summary")
    print(f"{'='*60}")
    
    comparison_df = pd.DataFrame([fifo_summary, mbb_summary])
    print(comparison_df.to_string(index=False))
    
    if fifo_summary.get('tps', 0) > 0:
        improvement = ((mbb_summary['tps'] - fifo_summary['tps']) / fifo_summary['tps']) * 100
        print(f"\nThroughput improvement: {improvement:+.1f}%")
    
    # Save summary
    with open(results_dir / f"{workload}_summary_k{num_bins}.json", 'w') as f:
        json.dump({
            'fifo': fifo_summary,
            'mbb': mbb_summary,
            'bins': num_bins
        }, f, indent=2)
    
    print(f"\nResults saved to results/")
    
    return fifo_summary, mbb_summary


async def main():
    """Main benchmark runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ollama + MBB Benchmark")
    parser.add_argument(
        "--workload",
        choices=["short_heavy", "mixed", "long_tail", "all"],
        default="mixed",
        help="Workload to run"
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=3,
        help="Number of bins for MBB"
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Force simulation mode (don't call real Ollama)"
    )
    
    args = parser.parse_args()
    
    if args.workload == "all":
        workloads = ["short_heavy", "mixed", "long_tail"]
    else:
        workloads = [args.workload]
    
    for workload in workloads:
        await run_benchmark(workload, args.bins)


if __name__ == "__main__":
    asyncio.run(main())

