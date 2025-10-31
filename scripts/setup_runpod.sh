#!/bin/bash
# Setup script for RunPod A5000 environment

set -e

echo "================================"
echo "Multi-Bin Batching Setup"
echo "================================"

# Update system
echo "[1/8] Updating system packages..."
apt-get update -qq
apt-get install -y -qq git build-essential python3-dev

# Create workspace
echo "[2/8] Creating workspace..."
cd /workspace || cd ~
mkdir -p mbb_project
cd mbb_project

# Clone vLLM
if [ ! -d "vllm" ]; then
    echo "[3/8] Cloning vLLM repository..."
    git clone https://github.com/vllm-project/vllm.git
    cd vllm
    git checkout -b mbb-integration
    cd ..
else
    echo "[3/8] vLLM already cloned"
fi

# Clone MultiBin
if [ ! -d "MultiBin" ]; then
    echo "[4/8] Cloning MultiBin repository..."
    # TODO: Replace with actual repo URL when available
    git clone <MULTIBIN_REPO_URL> MultiBin || \
        echo "Please manually copy MultiBin code to $(pwd)/MultiBin"
else
    echo "[4/8] MultiBin already present"
fi

# Setup Python environment
echo "[5/8] Setting up Python environment..."
cd MultiBin
python3 -m pip install --upgrade pip
pip install -e .
pip install -r requirements.txt

# Install vLLM
echo "[6/8] Installing vLLM..."
cd ../vllm
pip install -e .

# Download model
echo "[7/8] Downloading Phi-3.5-mini model..."
python3 << 'EOF'
from huggingface_hub import snapshot_download
import os
os.makedirs("/workspace/models", exist_ok=True)
snapshot_download(
    repo_id="microsoft/Phi-3.5-mini-instruct",
    local_dir="/workspace/models/Phi-3.5-mini-instruct",
    local_dir_use_symlinks=False
)
print("Model downloaded successfully")
EOF

# Generate datasets
echo "[8/8] Generating benchmark datasets..."
cd ../MultiBin
python data/download_datasets.py

echo ""
echo "================================"
echo "Setup Complete!"
echo "================================"
echo ""
echo "Next steps:"
echo "1. cd /workspace/mbb_project/vllm"
echo "2. Apply MBB patches (see IMPLEMENTATION_PLAN.md)"
echo "3. cd /workspace/mbb_project/MultiBin"
echo "4. python bench/run_manifest.py  # Generate manifests"
echo "5. python bench/vllm_benchmark.py --manifest bench/manifests/baseline_mixed.yaml"
echo ""
echo "Model location: /workspace/models/Phi-3.5-mini-instruct"
echo "MultiBin code: /workspace/mbb_project/MultiBin"
echo "vLLM code: /workspace/mbb_project/vllm"
echo ""

