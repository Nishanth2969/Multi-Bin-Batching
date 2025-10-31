#!/usr/bin/env python3
"""
Adversarial Stress Test Generator for Multi-Bin Batching

Creates worst-case workload for baseline continuous batching to demonstrate
maximum MBB improvement (~2× throughput).

Design:
- Bursty arrivals mixing very short and very long outputs
- Small fixed batch size to maximize "max-within-batch" effect
- FIFO mixing to force stragglers
- MBB with k=16 and equal-mass bins should achieve ~2× improvement
"""

import random
import os
from pathlib import Path


def generate_stress_test_dataset(output_path: str, num_prompts: int = 500):
    """Generate adversarial stress test dataset.
    
    Args:
        output_path: Path to save dataset
        num_prompts: Number of prompts to generate
    """
    print(f"Generating adversarial stress test dataset ({num_prompts} prompts)...")
    
    # Very short prompts (target: ~64 tokens output)
    short_prompts = [
        "What is 2+2?",
        "Hello, how are you?",
        "Define AI.",
        "List colors.",
        "What is Python?",
        "Say yes or no.",
        "Count to 5.",
        "Name a fruit.",
        "What is 10*10?",
        "Brief answer only."
    ]
    
    # Very long prompts (target: ~2048 tokens output)
    long_prompts = [
        "Write a comprehensive 2000-word essay explaining the history, theory, and applications of quantum computing, including detailed discussions of qubits, superposition, entanglement, quantum algorithms like Shor's and Grover's, quantum error correction, quantum supremacy, current challenges, and future prospects in quantum information science.",
        "Provide a detailed technical explanation of transformer neural network architectures, covering attention mechanisms, multi-head attention, positional encoding, feed-forward networks, layer normalization, encoder-decoder structures, self-attention vs cross-attention, the mathematics behind attention scores, BERT and GPT architectures, training procedures, optimization techniques, and recent advances in large language models.",
        "Explain in depth the principles of distributed systems, including CAP theorem, consensus algorithms like Raft and Paxos, distributed transactions, two-phase commit, eventual consistency, distributed databases, microservices architecture, service mesh, load balancing, fault tolerance, replication strategies, network partitions, Byzantine fault tolerance, and real-world case studies.",
    ]
    
    prompts = []
    random.seed(42)
    
    # Create adversarial pattern: alternate short and long
    for i in range(num_prompts):
        if i % 2 == 0:
            # Short prompt
            prompt = random.choice(short_prompts)
            # Add some variation
            prompt += " " * random.randint(0, 20)
        else:
            # Long prompt
            prompt = random.choice(long_prompts)
            # Add variation
            prompt += " " * random.randint(0, 100)
        
        prompts.append(prompt)
    
    # Save to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for prompt in prompts:
            f.write(prompt + '\n')
    
    print(f"✓ Saved to: {output_path}")
    print(f"  Pattern: alternating short/long prompts")
    print(f"  Short prompts: {len([p for i, p in enumerate(prompts) if i % 2 == 0])}")
    print(f"  Long prompts: {len([p for i, p in enumerate(prompts) if i % 2 == 1])}")
    
    return prompts


def create_stress_test_manifest(output_dir: str = "bench/manifests"):
    """Create stress test run manifests.
    
    Args:
        output_dir: Directory to save manifests
    """
    import yaml
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Baseline stress test
    baseline_manifest = {
        'run_id': 'baseline_stress',
        'description': 'Baseline continuous batching on adversarial stress test',
        'workload': {
            'name': 'stress_test',
            'dataset_path': 'data/stress_test.txt',
            'arrival_rate': 50.0,  # Lower rate to ensure full utilization
            'duration_s': 600  # 10 minutes
        },
        'scheduler': {
            'policy': 'continuous',
            'num_bins': 1
        },
        'decode': {
            'temperature': 0.0,
            'max_new_tokens': 2048,  # Allow long outputs
            'seed': 42
        },
        'system': {
            'model_name': 'microsoft/Phi-3.5-mini-instruct',
            'dtype': 'float16'
        }
    }
    
    # MBB stress test (k=16 for maximum improvement)
    mbb_manifest = {
        'run_id': 'mbb_k16_stress',
        'description': 'MBB k=16 with equal-mass bins on adversarial stress test',
        'workload': {
            'name': 'stress_test',
            'dataset_path': 'data/stress_test.txt',
            'arrival_rate': 50.0,
            'duration_s': 600
        },
        'scheduler': {
            'policy': 'multi_bin',
            'num_bins': 16,  # High k for maximum throughput
            'use_equal_mass_bins': True,
            'warmup_samples': 200,  # More samples for better quantiles
            'predictor_type': 'heuristic'
        },
        'decode': {
            'temperature': 0.0,
            'max_new_tokens': 2048,
            'seed': 42
        },
        'system': {
            'model_name': 'microsoft/Phi-3.5-mini-instruct',
            'dtype': 'float16'
        }
    }
    
    # Save manifests
    baseline_path = f"{output_dir}/baseline_stress.yaml"
    mbb_path = f"{output_dir}/mbb_k16_stress.yaml"
    
    with open(baseline_path, 'w') as f:
        yaml.dump(baseline_manifest, f, default_flow_style=False)
    print(f"✓ Created: {baseline_path}")
    
    with open(mbb_path, 'w') as f:
        yaml.dump(mbb_manifest, f, default_flow_style=False)
    print(f"✓ Created: {mbb_path}")
    
    return baseline_path, mbb_path


def run_stress_test():
    """Run stress test comparison."""
    import subprocess
    
    print("="*80)
    print("ADVERSARIAL STRESS TEST")
    print("="*80)
    print("\nThis test demonstrates maximum MBB improvement (~2×)")
    print("by creating worst-case workload for baseline batching.\n")
    
    # Generate dataset
    dataset_path = "data/stress_test.txt"
    generate_stress_test_dataset(dataset_path, num_prompts=500)
    
    # Create manifests
    baseline_path, mbb_path = create_stress_test_manifest()
    
    print("\n" + "="*80)
    print("Running Baseline (Expected: Low throughput due to stragglers)")
    print("="*80)
    
    baseline_result = subprocess.run([
        'python', 'bench/vllm_benchmark.py',
        '--manifest', baseline_path,
        '--output', 'results/stress_baseline.csv'
    ])
    
    print("\n" + "="*80)
    print("Running MBB k=16 (Expected: ~2× improvement)")
    print("="*80)
    
    mbb_result = subprocess.run([
        'python', 'bench/vllm_benchmark.py',
        '--manifest', mbb_path,
        '--output', 'results/stress_mbb_k16.csv'
    ])
    
    print("\n" + "="*80)
    print("STRESS TEST COMPLETE")
    print("="*80)
    print("\nCompare results:")
    print("  python scripts/compare_results.py results/stress_baseline_summary.json results/stress_mbb_k16_summary.json")
    
    if baseline_result.returncode == 0 and mbb_result.returncode == 0:
        print("\n✓ Both runs completed successfully!")
        print("  Expected improvement: ~100% (2× throughput)")
    else:
        print("\n✗ Some runs failed - check logs above")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'generate':
        # Just generate dataset/manifests
        generate_stress_test_dataset("data/stress_test.txt")
        create_stress_test_manifest()
    else:
        # Run full stress test
        run_stress_test()

