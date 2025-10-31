"""Quick test to verify Ollama is running and accessible."""

import asyncio
import httpx
import json
import time


async def test_ollama():
    """Test Ollama connectivity and basic generation."""
    
    print("Testing Ollama connection...")
    
    # Test 1: Check if Ollama is running
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get("http://127.0.0.1:11434/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                print(f"✓ Ollama is running")
                print(f"  Available models: {[m['name'] for m in models]}")
            else:
                print(f"✗ Ollama responded with status {response.status_code}")
                return False
    except Exception as e:
        print(f"✗ Cannot connect to Ollama: {e}")
        print("\n  Start Ollama with: ollama serve")
        return False
    
    # Test 2: Check if llama3.2:1b is available
    model_name = "llama3.2:1b"
    model_available = any(m['name'] == model_name for m in models)
    
    if not model_available:
        print(f"✗ Model {model_name} not found")
        print(f"\n  Pull model with: ollama pull {model_name}")
        return False
    else:
        print(f"✓ Model {model_name} is available")
    
    # Test 3: Run a simple generation
    print("\nTesting generation...")
    test_prompt = "What is 2+2?"
    
    t0 = time.perf_counter()
    ttft = None
    tokens = 0
    response_text = ""
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            async with client.stream(
                "POST",
                "http://127.0.0.1:11434/api/generate",
                json={
                    "model": model_name,
                    "prompt": test_prompt,
                    "stream": True,
                    "options": {"num_predict": 32}
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
        
        duration = time.perf_counter() - t0
        tpot = (duration / max(tokens, 1)) * 1000
        
        print(f"✓ Generation successful")
        print(f"  Prompt: {test_prompt}")
        print(f"  Response: {response_text[:100]}...")
        print(f"  TTFT: {ttft*1000:.1f} ms")
        print(f"  Tokens: {tokens}")
        print(f"  TPOT: {tpot:.1f} ms/token")
        print(f"  Total: {duration:.2f} s")
        
        return True
        
    except Exception as e:
        print(f"✗ Generation failed: {e}")
        return False


async def main():
    """Run tests."""
    print("="*60)
    print("Ollama Connectivity Test")
    print("="*60)
    
    success = await test_ollama()
    
    print("\n" + "="*60)
    if success:
        print("✓ All tests passed! Ready to run benchmarks.")
        print("\nNext step: python bench/ollama_benchmark.py --workload mixed")
    else:
        print("✗ Tests failed. Fix issues above before running benchmarks.")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())

