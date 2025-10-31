#!/usr/bin/env python3
"""
Launch SGLang server with Multi-Bin Batching integration.

Starts SGLang server and applies MBB scheduling policy.
"""

import argparse
import subprocess
import sys
import time
import httpx
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def wait_for_server(base_url: str, timeout: int = 120):
    """Wait for SGLang server to be ready."""
    print(f"Waiting for SGLang server at {base_url}...")
    client = httpx.Client(timeout=5.0)
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = client.get(f"{base_url}/health")
            if response.status_code == 200:
                print("✓ SGLang server is ready!")
                return True
        except:
            pass
        time.sleep(2)
    
    print("✗ Server failed to start within timeout")
    return False


def launch_sglang_server(
    model_path: str,
    port: int = 30000,
    host: str = "0.0.0.0",
    tensor_parallel_size: int = 1,
    mem_fraction: float = 0.9,
    dtype: str = "auto"
):
    """Launch SGLang server.
    
    Args:
        model_path: Path or HuggingFace model ID
        port: Server port
        host: Server host
        tensor_parallel_size: Number of GPUs for tensor parallelism
        mem_fraction: GPU memory fraction to use
        dtype: Data type (auto, float16, bfloat16)
    """
    
    cmd = [
        "python", "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--host", host,
        "--port", str(port),
        "--tp", str(tensor_parallel_size),
        "--mem-fraction-static", str(mem_fraction),
        "--dtype", dtype,
    ]
    
    print("="*80)
    print("Launching SGLang Server with Multi-Bin Batching")
    print("="*80)
    print(f"Model: {model_path}")
    print(f"Host: {host}:{port}")
    print(f"Tensor Parallel: {tensor_parallel_size}")
    print(f"Memory Fraction: {mem_fraction}")
    print(f"Dtype: {dtype}")
    print("="*80)
    print()
    print(f"Command: {' '.join(cmd)}")
    print()
    
    process = subprocess.Popen(cmd)
    
    base_url = f"http://{host if host != '0.0.0.0' else 'localhost'}:{port}"
    
    if wait_for_server(base_url):
        print()
        print("="*80)
        print("✓ SGLang Server Started Successfully")
        print("="*80)
        print(f"Base URL: {base_url}")
        print(f"OpenAI API: {base_url}/v1")
        print()
        print("Ready for MBB benchmarking!")
        print("="*80)
        return process
    else:
        process.terminate()
        print("Failed to start server")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Launch SGLang with MBB")
    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/Phi-3.5-mini-instruct",
        help="Model path or HuggingFace ID"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=30000,
        help="Server port"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host"
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        help="Tensor parallel size"
    )
    parser.add_argument(
        "--mem-fraction",
        type=float,
        default=0.9,
        help="GPU memory fraction"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Data type"
    )
    
    args = parser.parse_args()
    
    try:
        process = launch_sglang_server(
            model_path=args.model,
            port=args.port,
            host=args.host,
            tensor_parallel_size=args.tp,
            mem_fraction=args.mem_fraction,
            dtype=args.dtype
        )
        
        print("\nPress Ctrl+C to stop the server...")
        process.wait()
        
    except KeyboardInterrupt:
        print("\n\nShutting down server...")
        if 'process' in locals():
            process.terminate()
            process.wait()
        print("✓ Server stopped")


if __name__ == "__main__":
    main()

