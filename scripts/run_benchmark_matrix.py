#!/usr/bin/env python3
"""
Benchmark Matrix Runner for Multi-Bin Batching

Runs comprehensive benchmark matrix:
- Workloads: short_heavy, mixed, long_tail
- Policies: baseline (k=1), MBB (k=3)
- Ablations: k ∈ {1,2,3,4,8}, edges (static vs equal-mass), predictor (heuristic vs oracle)
- Loads: 75% and 90% SM utilization

Automates all runs and collects results.
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path
from typing import List, Dict


class BenchmarkMatrix:
    """Run complete benchmark matrix."""
    
    def __init__(self, output_dir: str = "results", gpu_id: int = 0):
        """Initialize benchmark matrix runner.
        
        Args:
            output_dir: Output directory for results
            gpu_id: GPU device ID
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.gpu_id = gpu_id
        self.results = []
    
    def run_baseline(self, workload: str, arrival_rate: float) -> Dict:
        """Run baseline (k=1) benchmark.
        
        Args:
            workload: Workload name
            arrival_rate: Requests per second
            
        Returns:
            Results dictionary
        """
        manifest = {
            'run_id': f'baseline_{workload}_r{int(arrival_rate)}',
            'workload': {'name': workload, 'dataset_path': f'data/{workload}.txt', 'arrival_rate': arrival_rate},
            'scheduler': {'policy': 'continuous', 'num_bins': 1},
            'decode': {'temperature': 0.0, 'max_new_tokens': 512, 'seed': 42},
            'system': {'gpu_device_id': self.gpu_id}
        }
        
        output_file = self.output_dir / f"{manifest['run_id']}.csv"
        
        print(f"\n[Baseline] {workload} @ {arrival_rate} req/s")
        cmd = [
            'python', 'bench/vllm_benchmark.py',
            '--manifest', json.dumps(manifest),
            '--output', str(output_file)
        ]
        
        result = self._run_command(cmd, manifest['run_id'])
        return result
    
    def run_mbb(self, workload: str, arrival_rate: float, num_bins: int, 
                use_equal_mass: bool = False, predictor: str = 'heuristic') -> Dict:
        """Run MBB benchmark.
        
        Args:
            workload: Workload name
            arrival_rate: Requests per second
            num_bins: Number of bins
            use_equal_mass: Use equal-mass quantile bins
            predictor: Predictor type
            
        Returns:
            Results dictionary
        """
        run_id = f'mbb_k{num_bins}_{workload}_r{int(arrival_rate)}'
        if use_equal_mass:
            run_id += '_equalmass'
        if predictor != 'heuristic':
            run_id += f'_{predictor}'
        
        manifest = {
            'run_id': run_id,
            'workload': {'name': workload, 'dataset_path': f'data/{workload}.txt', 'arrival_rate': arrival_rate},
            'scheduler': {
                'policy': 'multi_bin',
                'num_bins': num_bins,
                'bin_edges': [128, 512] if num_bins == 3 else None,
                'use_equal_mass_bins': use_equal_mass,
                'warmup_samples': 100,
                'predictor_type': predictor
            },
            'decode': {'temperature': 0.0, 'max_new_tokens': 512, 'seed': 42},
            'system': {'gpu_device_id': self.gpu_id}
        }
        
        output_file = self.output_dir / f"{run_id}.csv"
        
        print(f"\n[MBB k={num_bins}] {workload} @ {arrival_rate} req/s")
        cmd = [
            'python', 'bench/vllm_benchmark.py',
            '--manifest', json.dumps(manifest),
            '--output', str(output_file)
        ]
        
        result = self._run_command(cmd, run_id)
        return result
    
    def _run_command(self, cmd: List[str], run_id: str) -> Dict:
        """Run benchmark command.
        
        Args:
            cmd: Command to run
            run_id: Run identifier
            
        Returns:
            Results dictionary
        """
        print(f"Running: {' '.join(cmd)}")
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour max
            )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                print(f"✓ Completed in {duration:.1f}s")
                return {'run_id': run_id, 'status': 'success', 'duration_s': duration}
            else:
                print(f"✗ Failed: {result.stderr}")
                return {'run_id': run_id, 'status': 'failed', 'error': result.stderr}
        
        except subprocess.TimeoutExpired:
            print(f"✗ Timeout after 1 hour")
            return {'run_id': run_id, 'status': 'timeout'}
    
    def run_full_matrix(self):
        """Run complete benchmark matrix."""
        print("="*80)
        print("Multi-Bin Batching Benchmark Matrix")
        print("="*80)
        
        workloads = ['short_heavy', 'mixed', 'long_tail']
        arrival_rates = [100.0, 150.0]  # Adjust based on GPU capacity
        
        # Phase 1: Baseline runs
        print("\n" + "="*80)
        print("PHASE 1: Baseline Runs")
        print("="*80)
        
        for workload in workloads:
            for rate in arrival_rates:
                result = self.run_baseline(workload, rate)
                self.results.append(result)
                time.sleep(5)  # Cool down
        
        # Phase 2: MBB k=3 (main comparison)
        print("\n" + "="*80)
        print("PHASE 2: MBB k=3 (Main Comparison)")
        print("="*80)
        
        for workload in workloads:
            for rate in arrival_rates:
                result = self.run_mbb(workload, rate, num_bins=3)
                self.results.append(result)
                time.sleep(5)
        
        # Phase 3: Ablation - Number of bins
        print("\n" + "="*80)
        print("PHASE 3: Ablation - Number of Bins (k ∈ {1,2,3,4,8})")
        print("="*80)
        
        for k in [1, 2, 3, 4, 8]:
            result = self.run_mbb('mixed', 100.0, num_bins=k)
            self.results.append(result)
            time.sleep(5)
        
        # Phase 4: Ablation - Equal-mass bins
        print("\n" + "="*80)
        print("PHASE 4: Ablation - Equal-Mass Bins")
        print("="*80)
        
        result = self.run_mbb('mixed', 100.0, num_bins=3, use_equal_mass=True)
        self.results.append(result)
        time.sleep(5)
        
        # Phase 5: Oracle predictor
        print("\n" + "="*80)
        print("PHASE 5: Oracle Predictor (Upper Bound)")
        print("="*80)
        
        result = self.run_mbb('mixed', 100.0, num_bins=3, predictor='oracle')
        self.results.append(result)
        
        # Save summary
        summary_file = self.output_dir / 'benchmark_matrix_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print("\n" + "="*80)
        print("Benchmark Matrix Complete!")
        print("="*80)
        print(f"Results saved to: {summary_file}")
        print(f"Total runs: {len(self.results)}")
        print(f"Successful: {sum(1 for r in self.results if r.get('status') == 'success')}")
        print(f"Failed: {sum(1 for r in self.results if r.get('status') == 'failed')}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run benchmark matrix')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU device ID')
    parser.add_argument('--workload', type=str, help='Single workload to run (optional)')
    parser.add_argument('--ablation-only', action='store_true', help='Run only ablation studies')
    
    args = parser.parse_args()
    
    runner = BenchmarkMatrix(output_dir=args.output_dir, gpu_id=args.gpu_id)
    
    if args.ablation_only:
        # Run only ablations
        print("Running ablation studies only...")
        for k in [1, 2, 3, 4, 8]:
            runner.run_mbb('mixed', 100.0, num_bins=k)
        runner.run_mbb('mixed', 100.0, num_bins=3, use_equal_mass=True)
        runner.run_mbb('mixed', 100.0, num_bins=3, predictor='oracle')
    elif args.workload:
        # Run single workload
        runner.run_baseline(args.workload, 100.0)
        runner.run_mbb(args.workload, 100.0, num_bins=3)
    else:
        # Run full matrix
        runner.run_full_matrix()


if __name__ == "__main__":
    main()

