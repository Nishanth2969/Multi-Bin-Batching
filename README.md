# Multi-Bin Batching (MBB) for LLM Inference

This repository implements **Multi-Bin Batching (MBB)** for improving LLM inference throughput by grouping requests into multiple bins based on predicted output length.

## Overview

Multi-Bin Batching addresses the head-of-line blocking problem in continuous batching by:

1. **Predicting output length** for each incoming request
2. **Assigning requests to bins** based on predicted length
3. **Batching within bins** to reduce straggler effects
4. **Enforcing fairness** via starvation guards

### Architecture

```
Request Arrival → Predictor → Bin Assignment → Bin Queues → Batch Selector → GPU Batch
```

- **Predictor**: Estimates output length from prompt characteristics
- **BinEdges**: Defines bin boundaries (e.g., short/medium/long)
- **BinQueues**: Per-bin FIFO queues
- **BatchSelector**: Forms batches from bins respecting KV/memory limits
- **StarvationGuard**: Prevents long waits with priority promotion

## Installation

```bash
# Install dependencies
pip install -e .
pip install numpy pandas matplotlib seaborn pyyaml nvidia-ml-py3 httpx tqdm

# Generate datasets
python data/download_datasets.py

# Quick setup script (on GPU)
./scripts/setup_gpu.sh
```

## GPU Execution (RunPod)

### Quick Setup

```bash
# 1. Upload code to RunPod
cd ~/Desktop/Project/MultiBin
tar -czf - --exclude='venv' --exclude='__pycache__' . | \
  ssh -i ~/.ssh/id_ed25519 okcvmgdpg3zvl4-644110e0@ssh.runpod.io \
  "mkdir -p /workspace/MultiBin && cd /workspace/MultiBin && tar -xzf -"

# 2. SSH to RunPod
ssh -i ~/.ssh/id_ed25519 okcvmgdpg3zvl4-644110e0@ssh.runpod.io

# 3. Setup (on RunPod)
cd /workspace/MultiBin
./scripts/setup_gpu.sh

# 4. Verify implementation matches paper
python verify_paper.py

# 5. Install vLLM and apply MBB patch
pip install vllm
cd /workspace && git clone https://github.com/vllm-project/vllm.git
cd vllm && git checkout -b mbb-integration
# Apply patches from scripts/vllm_patch.py manually

# 6. Generate datasets and manifests
python data/download_datasets.py
python bench/run_manifest.py

# 7. Run benchmark matrix
python scripts/run_benchmark_matrix.py

# 8. Run stress test (demonstrates ~2× improvement)
python scripts/stress_test.py

# 9. Generate report
python scripts/generate_report.py
```

### Individual Commands

```bash
# Test single workload
python bench/vllm_benchmark.py --manifest bench/manifests/baseline_mixed.yaml

# Compare results
python scripts/compare_results.py results/baseline_mixed_summary.json results/mbb_k3_mixed_summary.json

# Generate plots only
python scripts/generate_plots.py results/

# Run ablation study only
python scripts/run_benchmark_matrix.py --ablation-only
```

## Quick Start

### Run Tests

```bash
make test
```

### Run Benchmark Simulation

```bash
python bench/loadgen.py
```

### Integration with vLLM

```python
from adapters.vllm.hooks import create_mbb_components, should_use_mbb
import argparse

# Parse arguments with MBB flags
parser = argparse.ArgumentParser()
args = parser.parse_args([
    "--scheduler-policy=multi_bin",
    "--mbb-bins=3",
    "--mbb-bin-edges=128,512",
])

# Create MBB components
if should_use_mbb(args):
    components = create_mbb_components(args)
    predictor = components['predictor']
    bin_edges = components['bin_edges']
    bins = components['bins']
    selector = components['selector']
```

### Integration with SGLang

```python
from adapters.sglang.hooks import create_mbb_components, should_use_mbb

# Similar integration pattern as vLLM
if should_use_mbb(args):
    components = create_mbb_components(args)
```

## Core Components

### 1. Predictor (`mbb_core/predict.py`)

Output length estimation strategies:

- **HeuristicPredictor**: `pred = a * prompt_tokens + b`
- **OraclePredictor**: Uses ground truth output length (benchmarking only)
- **ModelBasedPredictor**: Placeholder for learned models

### 2. Bin Management (`mbb_core/bins.py`)

Bin configuration:

```python
from mbb_core.bins import BinEdges

# Create bins: [<128], [128..512], [>=512]
edges = BinEdges([128, 512])

# Assign request to bin
predicted_len = predictor(prompt_tokens)
bin_id = edges.bin_for(predicted_len)
```

### 3. Request Queues (`mbb_core/queues.py`)

Per-bin FIFO queues:

```python
from mbb_core.queues import BinQueues, Request

bins = BinQueues(n_bins=3)

# Enqueue request
bins.push(bin_id, request)

# Check non-empty bins
non_empty = bins.non_empty_bins()
```

### 4. Batch Selector (`mbb_core/selector.py`)

Batch formation policies:

- **RoundRobinSelector**: Alternates across bins
- **WeightedSelector**: Prioritizes bins by queue size
- **StarvationGuard**: Promotes aged requests

```python
from mbb_core.selector import BatchSelector

selector = BatchSelector(
    kv_budget_tokens=8192,
    max_batched_tokens=2048,
    starvation_ms=500,
    selection_policy="round_robin"
)

batch = selector.build_batch(current_time_ms)
```

### 5. Scheduler (`mbb_core/scheduler.py`)

High-level orchestration:

```python
from mbb_core import MBScheduler

scheduler = MBScheduler(
    num_bins=3,
    bin_edges=[128, 512],
    kv_budget_tokens=8192,
    max_batched_tokens=2048,
    starvation_ms=500
)

# Enqueue request
rid = scheduler.enqueue_request(prompt_tokens=100)

# Build batch
batch = scheduler.build_batch(current_time_ms)

# Get statistics
stats = scheduler.get_statistics()
```

## Benchmarking

### Workload Mixes

Three workload patterns are provided in `bench/loadgen.py`:

1. **short_heavy**: Small prompts (64-256 tok), short outputs (64-256 tok)
2. **mixed**: Varied prompts (64-2048 tok), mixed outputs (64-1024 tok)
3. **long_tail**: Mostly short (70%), some very long (10%)

### Running Benchmarks

```python
from bench.loadgen import WorkloadMix, RequestGenerator
from mbb_core import MBScheduler

# Create workload
mix = WorkloadMix.short_heavy_mix()
generator = RequestGenerator(mix, seed=42)
requests = generator.generate_requests()

# Create scheduler
scheduler = MBScheduler(num_bins=3)
simulator = LoadSimulator(scheduler)
simulator.simulate(requests)

metrics = simulator.get_metrics()
```

### Metrics Collected

- **Throughput**: Tokens per second (TPS)
- **Latency**: Time to first token (TTFT), time per output token (TPOT)
- **Percentiles**: P50, P95 for TTFT and TPOT
- **GPU Utilization**: Compute utilization percentage
- **Bin Occupancy**: Distribution across bins

## Design Choices

### Predictor

The default heuristic predictor uses:

```
predicted_output = 0.5 * prompt_tokens + 32
```

This can be tuned per model on historical data.

### Bin Edges

Default 3-bin configuration:

- **Bin 0**: Predicted output < 128 tokens
- **Bin 1**: 128 ≤ predicted < 512 tokens
- **Bin 2**: Predicted output ≥ 512 tokens

### Selection Policy

- **Round-robin**: Fair rotation across bins
- **Weighted**: Prefer bins with larger backlogs

### Starvation Guard

Requests waiting > 500ms are promoted to prevent starvation.

## Performance Expectations

Based on the paper, you should see:

- **+10-30% TPS** on mixed workloads vs continuous batching
- **Similar TTFT/TPOT** with proper bin configuration
- **Reduced GPU idle time** due to better batching

Actual gains depend on:
- Workload distribution
- Model size and context length
- Hardware (GPU type, memory)
- Number of bins and edges

## Development

### Code Style

```bash
# Format code
make format

# Lint code
make lint

# Run tests
make test

# Run everything
make all
```

### Project Structure

```
MultiBin/
├── mbb_core/              # Core MBB implementation
│   ├── predict.py         # Output length predictors
│   ├── bins.py            # Bin management
│   ├── queues.py          # Request queues
│   ├── selector.py        # Batch selection
│   ├── scheduler.py       # Main scheduler
│   └── tests/             # Unit tests
├── adapters/              # Framework adapters
│   ├── vllm/              # vLLM integration
│   └── sglang/            # SGLang integration
├── bench/                 # Benchmarking tools
│   ├── loadgen.py         # Load generation
│   └── metrics.py         # Metrics collection
├── report/                # Report templates
└── README.md              # This file
```

## Integration Guide

### For vLLM Developers

1. Add MBB flags in vLLM argument parsing
2. Hook into scheduler's request arrival path
3. Replace batch builder with MBB selector
4. Update metrics collection

See `adapters/vllm/hooks.py` for reference implementation.

### For SGLang Developers

1. Check for `--scheduler-policy=multi_bin` flag
2. Create MBB components during initialization
3. Integrate with SGLang's CPU scheduler
4. Maintain compatibility with radix/prefix caching

See `adapters/sglang/hooks.py` for reference implementation.

## Paper Reference

- **Multi-Bin Batching for Increasing LLM Inference Throughput**
- arXiv: [2412.04504](https://arxiv.org/abs/2412.04504)
- Authors: Discuss queueing-theoretic guarantees and empirical results

## Related Frameworks

- **vLLM**: [docs.vllm.ai](https://docs.vllm.ai)
- **SGLang**: [sgl-project.github.io](https://sgl-project.github.io)
- **PagedAttention**: KV cache management in vLLM
- **Continuous Batching**: Dynamic batching in modern LLM servers

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run `make all` to ensure tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- vLLM team for continuous batching and PagedAttention
- SGLang team for efficient scheduling
- Multi-Bin Batching paper authors

