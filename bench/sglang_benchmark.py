"""
SGLang-based benchmarking harness for MBB evaluation.

Uses SGLang's OpenAI-compatible API for inference with optional MBB scheduling.
"""

import argparse
import os
import json
import time
import sys
import httpx
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from bench.metrics import MetricsCollector
from bench.gpu_monitor import GPUMonitor
from bench.run_manifest import RunManifest
from adapters.sglang.mbb_policy import MultiBinBatchingPolicy


class SGLangBenchmark:
    """Benchmark harness for SGLang with MBB integration."""
    
    def __init__(self, manifest: RunManifest, base_url: str = "http://localhost:30000"):
        """Initialize benchmark.
        
        Args:
            manifest: RunManifest configuration
            base_url: SGLang server base URL
        """
        self.manifest = manifest
        self.base_url = base_url.rstrip('/')
        self.metrics = MetricsCollector()
        self.gpu_monitor = GPUMonitor(
            device_id=manifest.system.gpu_device_id,
            sample_interval_ms=1000
        )
        self.mbb_policy = None
        self.client = httpx.Client(timeout=300.0)
        
        if manifest.scheduler.policy == "multi_bin":
            self.mbb_policy = MultiBinBatchingPolicy(
                num_bins=manifest.scheduler.num_bins,
                bin_edges=manifest.scheduler.bin_edges,
                batch_size=256,
                predictor_type=manifest.scheduler.predictor_type,
                starvation_ms=manifest.scheduler.starvation_ms,
                use_equal_mass_bins=True,
                warmup_samples=100
            )
    
    def load_prompts(self) -> List[str]:
        """Load prompts from dataset."""
        dataset_path = self.manifest.workload.dataset_path
        if not os.path.exists(dataset_path):
            print(f"ERROR: Dataset not found: {dataset_path}")
            sys.exit(1)
        
        with open(dataset_path, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
        
        print(f"Loaded {len(prompts)} prompts from {dataset_path}")
        return prompts
    
    def run_benchmark(self) -> Dict[str, Any]:
        """Execute benchmark run."""
        print("\n" + "="*80)
        print(f"Starting benchmark: {self.manifest.run_id}")
        print("="*80)
        
        prompts = self.load_prompts()
        
        if self.manifest.workload.num_requests:
            prompts = prompts[:self.manifest.workload.num_requests]
        
        def record_gpu(gpu_stats):
            self.metrics.record_gpu_snapshot(
                sm_util=gpu_stats.sm_utilization,
                mem_used_mb=gpu_stats.memory_used_mb,
                mem_total_mb=gpu_stats.memory_total_mb,
                temp_c=gpu_stats.temperature_c,
                power_w=gpu_stats.power_watts
            )
        
        self.gpu_monitor.start(record_gpu)
        
        print(f"\nRunning inference on {len(prompts)} prompts...")
        start_time = time.time()
        start_time_ms = int(start_time * 1000)
        
        try:
            for i, prompt in enumerate(tqdm(prompts, desc="Processing")):
                arrival_time_ms = int(time.time() * 1000)
                
                prompt_tokens = len(prompt.split())
                
                if self.mbb_policy:
                    bin_id = self.mbb_policy.on_request_arrival(str(i), prompt_tokens)
                else:
                    bin_id = -1
                
                response = self.client.post(
                    f"{self.base_url}/v1/completions",
                    json={
                        "model": self.manifest.system.model_name,
                        "prompt": prompt,
                        "max_tokens": self.manifest.decode.max_new_tokens,
                        "temperature": self.manifest.decode.temperature,
                        "top_p": self.manifest.decode.top_p,
                        "seed": self.manifest.decode.seed,
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    generated_text = result['choices'][0]['text']
                    output_tokens = len(generated_text.split())
                    
                    self.metrics.record_request_start(
                        req_id=i,
                        prompt_tokens=prompt_tokens,
                        arrival_time_ms=arrival_time_ms,
                        predicted_bin=bin_id,
                        admit_bin=bin_id
                    )
                    
                    self.metrics.record_request_completion(
                        req_id=i,
                        output_tokens=output_tokens
                    )
                    
                    if self.mbb_policy:
                        self.mbb_policy.on_request_completion(str(i), output_tokens)
                else:
                    print(f"\nERROR: Request {i} failed with status {response.status_code}")
        
        except KeyboardInterrupt:
            print("\nBenchmark interrupted by user")
        except Exception as e:
            print(f"\nERROR during benchmark: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.gpu_monitor.stop()
            self.client.close()
        
        end_time = time.time()
        end_time_ms = int(end_time * 1000)
        duration_s = end_time - start_time
        
        self.metrics.update_time(end_time_ms)
        
        print(f"\nBenchmark completed in {duration_s:.1f}s")
        
        summary = self.metrics.get_summary()
        summary['run_id'] = self.manifest.run_id
        summary['duration_seconds'] = duration_s
        summary['manifest'] = self.manifest.to_dict()
        
        if summary.get('total_output_tokens', 0) > 0 and duration_s > 0:
            summary['output_tps'] = summary['total_output_tokens'] / duration_s
        
        if self.mbb_policy:
            summary['mbb_stats'] = self.mbb_policy.get_stats()
        
        return summary
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save benchmark results."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        json_path = output_path.replace('.csv', '_summary.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved summary: {json_path}")
        
        csv_path = output_path
        self.metrics.export_detailed_csv(csv_path)
        print(f"Saved detailed metrics: {csv_path}")
        
        gpu_csv_path = output_path.replace('.csv', '_gpu.csv')
        self.metrics.export_gpu_stats_csv(gpu_csv_path)
        print(f"Saved GPU stats: {gpu_csv_path}")


def main():
    """Main benchmark entry point."""
    parser = argparse.ArgumentParser(description="SGLang Multi-Bin Batching Benchmark")
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to run manifest YAML/JSON file"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:30000",
        help="SGLang server base URL"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path"
    )
    
    args = parser.parse_args()
    
    manifest_path = args.manifest
    if manifest_path.endswith('.yaml') or manifest_path.endswith('.yml'):
        manifest = RunManifest.from_yaml(manifest_path)
    elif manifest_path.endswith('.json'):
        manifest = RunManifest.from_json(manifest_path)
    else:
        print(f"ERROR: Unsupported manifest format: {manifest_path}")
        sys.exit(1)
    
    if args.output:
        output_path = args.output
    else:
        output_dir = manifest.output_dir
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{manifest.run_id}.csv")
    
    benchmark = SGLangBenchmark(manifest, base_url=args.base_url)
    results = benchmark.run_benchmark()
    benchmark.save_results(results, output_path)
    
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print(f"Run ID: {results['run_id']}")
    print(f"Duration: {results.get('duration_seconds', 0):.1f}s")
    print(f"Requests: {results['num_requests']}")
    print(f"Total Output Tokens: {results.get('total_output_tokens', 0):,}")
    print(f"Output Throughput: {results.get('output_tps', 0):.1f} tokens/s")
    
    if 'avg_sm_util' in results:
        print(f"Avg GPU SM Util: {results['avg_sm_util']:.1f}%")
        print(f"Avg VRAM Used: {results['avg_mem_used_mb']:.0f}MB")
    
    if 'mbb_stats' in results:
        stats = results['mbb_stats']
        print(f"\nMBB Statistics:")
        print(f"  Bins: {stats['num_bins']}")
        print(f"  Batches Formed: {stats['batches_formed']}")
        print(f"  Requests Served: {stats['requests_served']}")
        print(f"  Bin Occupancy: {stats['bin_occupancy']}")
    
    print("="*80)


if __name__ == "__main__":
    main()

