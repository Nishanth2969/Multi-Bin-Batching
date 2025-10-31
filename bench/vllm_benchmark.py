"""
vLLM-based benchmarking harness for MBB evaluation.

Runs LLM inference with baseline or MBB scheduling and collects metrics.
"""

import argparse
import os
import json
import time
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bench.metrics import MetricsCollector
from bench.gpu_monitor import GPUMonitor
from bench.run_manifest import RunManifest


class VLLMBenchmark:
    """Benchmark harness for vLLM with MBB integration."""
    
    def __init__(self, manifest: RunManifest):
        """Initialize benchmark with run manifest.
        
        Args:
            manifest: RunManifest configuration
        """
        self.manifest = manifest
        self.metrics = MetricsCollector()
        self.gpu_monitor = GPUMonitor(
            device_id=manifest.system.gpu_device_id,
            sample_interval_ms=1000
        )
        self.vllm_engine = None
        self.sampling_params = None
    
    def setup_vllm(self):
        """Initialize vLLM engine with configuration."""
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            print("ERROR: vLLM not installed. Install with:")
            print("pip install vllm")
            sys.exit(1)
        
        # Create sampling params from decode config
        self.sampling_params = SamplingParams(
            temperature=self.manifest.decode.temperature,
            max_tokens=self.manifest.decode.max_new_tokens,
            top_p=self.manifest.decode.top_p,
            top_k=self.manifest.decode.top_k,
            seed=self.manifest.decode.seed,
            stop=self.manifest.decode.stop_tokens if self.manifest.decode.stop_tokens else None
        )
        
        # Initialize LLM engine
        print(f"Loading model: {self.manifest.system.model_name}")
        
        # Build vLLM kwargs (no custom scheduler args - we patch at runtime)
        vllm_kwargs = {
            "model": self.manifest.system.model_name,
            "dtype": self.manifest.system.dtype,
            "tensor_parallel_size": self.manifest.system.tensor_parallel_size,
            "max_model_len": self.manifest.system.max_model_len,
            "gpu_memory_utilization": 0.9,
        }
        
        self.vllm_engine = LLM(**vllm_kwargs)
        print(f"Model loaded successfully")
        
        # Apply MBB scheduler patch if needed
        if self.manifest.scheduler.policy == "multi_bin":
            from adapters.vllm.scheduler_patch import VLLMMBBAdapter, patch_vllm_scheduler
            
            self.mbb_adapter = VLLMMBBAdapter(
                num_bins=self.manifest.scheduler.num_bins,
                bin_edges=self.manifest.scheduler.bin_edges,
                batch_size=256,
                predictor_type=self.manifest.scheduler.predictor_type,
                starvation_ms=self.manifest.scheduler.starvation_ms,
                use_equal_mass_bins=True
            )
            
            patch_vllm_scheduler(self.vllm_engine, self.mbb_adapter)
        else:
            self.mbb_adapter = None
    
    def load_prompts(self) -> List[str]:
        """Load prompts from dataset.
        
        Returns:
            List of prompt strings
        """
        dataset_path = self.manifest.workload.dataset_path
        if not os.path.exists(dataset_path):
            print(f"ERROR: Dataset not found: {dataset_path}")
            sys.exit(1)
        
        with open(dataset_path, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
        
        print(f"Loaded {len(prompts)} prompts from {dataset_path}")
        return prompts
    
    def run_benchmark(self) -> Dict[str, Any]:
        """Execute benchmark run.
        
        Returns:
            Dictionary with benchmark results
        """
        print("\n" + "="*80)
        print(f"Starting benchmark: {self.manifest.run_id}")
        print("="*80)
        
        # Load prompts
        prompts = self.load_prompts()
        
        # Limit to num_requests if specified
        if self.manifest.workload.num_requests:
            prompts = prompts[:self.manifest.workload.num_requests]
        
        # Start GPU monitoring
        def record_gpu(gpu_stats):
            self.metrics.record_gpu_snapshot(
                sm_util=gpu_stats.sm_utilization,
                mem_used_mb=gpu_stats.memory_used_mb,
                mem_total_mb=gpu_stats.memory_total_mb,
                temp_c=gpu_stats.temperature_c,
                power_w=gpu_stats.power_watts
            )
        
        self.gpu_monitor.start(record_gpu)
        
        # Run inference
        print(f"\nRunning inference on {len(prompts)} prompts...")
        start_time = time.time()
        start_time_ms = int(start_time * 1000)
        
        try:
            # Generate outputs
            outputs = self.vllm_engine.generate(prompts, self.sampling_params)
            
            # Record metrics
            for i, output in enumerate(outputs):
                prompt = output.prompt
                generated = output.outputs[0].text
                prompt_tokens = len(output.prompt_token_ids)
                output_tokens = len(output.outputs[0].token_ids)
                
                # Record completion (simplified - in real impl would track TTFT separately)
                self.metrics.record_request_start(
                    req_id=i,
                    prompt_tokens=prompt_tokens,
                    arrival_time_ms=start_time_ms,
                    predicted_bin=-1,
                    admit_bin=-1
                )
                self.metrics.record_request_completion(
                    req_id=i,
                    output_tokens=output_tokens
                )
        
        except KeyboardInterrupt:
            print("\nBenchmark interrupted by user")
        except Exception as e:
            print(f"\nERROR during benchmark: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Stop monitoring
            self.gpu_monitor.stop()
        
        end_time = time.time()
        end_time_ms = int(end_time * 1000)
        duration_s = end_time - start_time
        
        # Update metrics collector time
        self.metrics.update_time(end_time_ms)
        
        print(f"\nBenchmark completed in {duration_s:.1f}s")
        
        # Get summary
        summary = self.metrics.get_summary()
        summary['run_id'] = self.manifest.run_id
        summary['duration_seconds'] = duration_s
        summary['manifest'] = self.manifest.to_dict()
        
        # Fix throughput calculation
        if summary.get('total_output_tokens', 0) > 0 and duration_s > 0:
            summary['output_tps'] = summary['total_output_tokens'] / duration_s
        
        return summary
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save benchmark results.
        
        Args:
            results: Results dictionary
            output_path: Output file path
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save summary JSON
        json_path = output_path.replace('.csv', '_summary.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved summary: {json_path}")
        
        # Save detailed CSV
        csv_path = output_path
        self.metrics.export_detailed_csv(csv_path)
        print(f"Saved detailed metrics: {csv_path}")
        
        # Save GPU stats
        gpu_csv_path = output_path.replace('.csv', '_gpu.csv')
        self.metrics.export_gpu_stats_csv(gpu_csv_path)
        print(f"Saved GPU stats: {gpu_csv_path}")


def main():
    """Main benchmark entry point."""
    parser = argparse.ArgumentParser(description="vLLM Multi-Bin Batching Benchmark")
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to run manifest YAML/JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: results/<run_id>.csv)"
    )
    
    args = parser.parse_args()
    
    # Load manifest
    manifest_path = args.manifest
    if manifest_path.endswith('.yaml') or manifest_path.endswith('.yml'):
        manifest = RunManifest.from_yaml(manifest_path)
    elif manifest_path.endswith('.json'):
        manifest = RunManifest.from_json(manifest_path)
    else:
        print(f"ERROR: Unsupported manifest format: {manifest_path}")
        print("Use .yaml, .yml, or .json")
        sys.exit(1)
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_dir = manifest.output_dir
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{manifest.run_id}.csv")
    
    # Run benchmark
    benchmark = VLLMBenchmark(manifest)
    benchmark.setup_vllm()
    results = benchmark.run_benchmark()
    benchmark.save_results(results, output_path)
    
    # Print summary
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print(f"Run ID: {results['run_id']}")
    print(f"Duration: {results.get('duration_seconds', results.get('duration_s', 0)):.1f}s")
    print(f"Requests: {results['num_requests']}")
    print(f"Total Output Tokens: {results.get('total_output_tokens', 0):,}")
    print(f"Output Throughput: {results.get('output_tps', results.get('tps', 0)):.1f} tokens/s")
    print(f"TTFT P50: {results.get('ttft_p50_ms', 0):.1f}ms")
    print(f"TTFT P95: {results.get('ttft_p95_ms', 0):.1f}ms")
    print(f"TPOT P50: {results.get('tpot_p50_ms', 0):.2f}ms")
    print(f"TPOT P95: {results.get('tpot_p95_ms', 0):.2f}ms")
    
    if 'avg_sm_util' in results:
        print(f"Avg GPU SM Util: {results['avg_sm_util']:.1f}%")
        print(f"Avg VRAM Used: {results['avg_mem_used_mb']:.0f}MB")
    
    print("="*80)


if __name__ == "__main__":
    main()

