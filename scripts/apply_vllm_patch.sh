#!/bin/bash
# Auto-apply vLLM MBB integration patch

set -e

VLLM_DIR="${1:-/workspace/vllm}"
MULTIBIN_DIR="${2:-/workspace/MultiBin}"

if [ ! -d "$VLLM_DIR" ]; then
    echo "ERROR: vLLM directory not found: $VLLM_DIR"
    echo "Usage: $0 <vllm_dir> [multibin_dir]"
    exit 1
fi

echo "Applying MBB patch to vLLM..."
echo "vLLM dir: $VLLM_DIR"
echo "MultiBin dir: $MULTIBIN_DIR"

cd "$VLLM_DIR"

# Backup original files
echo "Creating backups..."
cp vllm/engine/arg_utils.py vllm/engine/arg_utils.py.bak
cp vllm/core/scheduler.py vllm/core/scheduler.py.bak
cp vllm/engine/llm_engine.py vllm/engine/llm_engine.py.bak

echo "✓ Backup complete"
echo "Please manually apply patches from scripts/vllm_patch.py"
echo "Or use: python $MULTIBIN_DIR/scripts/vllm_patch.py"

