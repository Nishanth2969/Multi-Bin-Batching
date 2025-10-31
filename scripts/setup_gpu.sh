#!/bin/bash
# Quick setup script for RunPod GPU

set -e

echo "=== Multi-Bin Batching Setup ==="

# Install dependencies
pip install --upgrade pip
pip install numpy pandas matplotlib seaborn pyyaml nvidia-ml-py3 httpx tqdm

# Install MultiBin
pip install -e .

# Generate datasets
python data/download_datasets.py

# Test core components
python -c "
from mbb_core.scheduler import MBScheduler
from mbb_core.bins import BinHistogram
print('✓ MBB core components OK')
"

# Install vLLM (if needed)
pip install vllm || echo "vLLM installation skipped (install manually if needed)"

# Generate manifests
python bench/run_manifest.py || echo "Manifest generation skipped"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "1. Test: python -c 'from mbb_core import MBScheduler; print(MBScheduler())'"
echo "2. Run benchmark: python bench/vllm_benchmark.py --manifest bench/manifests/baseline_mixed.yaml"
echo ""

