"""
Compare baseline vs MBB benchmark results.
"""

import json
import sys
from pathlib import Path


def load_results(filepath: str) -> dict:
    """Load results JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def compare(baseline: dict, mbb: dict):
    """Compare two result sets."""
    
    print("\n" + "="*80)
    print("BENCHMARK COMPARISON")
    print("="*80)
    
    print(f"\nBaseline: {baseline['run_id']}")
    print(f"MBB:      {mbb['run_id']}")
    
    print("\n" + "-"*80)
    print("THROUGHPUT")
    print("-"*80)
    
    baseline_tps = baseline.get('tps', 0)
    mbb_tps = mbb.get('tps', 0)
    improvement = ((mbb_tps - baseline_tps) / baseline_tps * 100) if baseline_tps > 0 else 0
    
    print(f"{'Metric':<30} {'Baseline':<15} {'MBB':<15} {'Change':<15}")
    print("-"*80)
    print(f"{'Tokens/second':<30} {baseline_tps:>14.2f} {mbb_tps:>14.2f} {improvement:>+13.1f}%")
    print(f"{'Total tokens':<30} {baseline.get('total_tokens', 0):>14} {mbb.get('total_tokens', 0):>14} {'':>15}")
    print(f"{'Duration (s)':<30} {baseline.get('duration_s', 0):>14.1f} {mbb.get('duration_s', 0):>14.1f} {'':>15}")
    
    print("\n" + "-"*80)
    print("LATENCY")
    print("-"*80)
    
    metrics = [
        ('TTFT P50 (ms)', 'ttft_p50_ms'),
        ('TTFT P95 (ms)', 'ttft_p95_ms'),
        ('TPOT P50 (ms)', 'tpot_p50_ms'),
        ('TPOT P95 (ms)', 'tpot_p95_ms'),
    ]
    
    print(f"{'Metric':<30} {'Baseline':<15} {'MBB':<15} {'Change':<15}")
    print("-"*80)
    
    for label, key in metrics:
        b_val = baseline.get(key, 0)
        m_val = mbb.get(key, 0)
        change = ((m_val - b_val) / b_val * 100) if b_val > 0 else 0
        print(f"{label:<30} {b_val:>14.2f} {m_val:>14.2f} {change:>+13.1f}%")
    
    print("\n" + "-"*80)
    print("GPU UTILIZATION")
    print("-"*80)
    
    if 'avg_sm_util' in baseline and 'avg_sm_util' in mbb:
        print(f"{'Metric':<30} {'Baseline':<15} {'MBB':<15} {'Change':<15}")
        print("-"*80)
        
        b_sm = baseline.get('avg_sm_util', 0)
        m_sm = mbb.get('avg_sm_util', 0)
        sm_change = m_sm - b_sm
        
        b_mem = baseline.get('avg_mem_used_mb', 0)
        m_mem = mbb.get('avg_mem_used_mb', 0)
        mem_change = m_mem - b_mem
        
        print(f"{'SM Utilization (%)':<30} {b_sm:>14.1f} {m_sm:>14.1f} {sm_change:>+13.1f}pp")
        print(f"{'Memory Used (MB)':<30} {b_mem:>14.0f} {m_mem:>14.0f} {mem_change:>+13.0f}")
    
    print("\n" + "-"*80)
    print("TOKEN PARITY")
    print("-"*80)
    
    b_tokens = baseline.get('total_output_tokens', 0)
    m_tokens = mbb.get('total_output_tokens', 0)
    parity = abs(b_tokens - m_tokens) / b_tokens * 100 if b_tokens > 0 else 0
    
    print(f"Baseline output tokens: {b_tokens}")
    print(f"MBB output tokens:      {m_tokens}")
    print(f"Difference:             {parity:.2f}%")
    
    if parity > 1.0:
        print(f"\n⚠️  WARNING: Token parity difference ({parity:.2f}%) exceeds 1% threshold!")
        print("   This may indicate decode parameter mismatch or non-determinism.")
    else:
        print(f"\n✓ Token parity check PASSED ({parity:.2f}% < 1%)")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if improvement > 0:
        print(f"\n✓ MBB achieves {improvement:.1f}% throughput improvement over baseline")
    else:
        print(f"\n✗ MBB shows {improvement:.1f}% throughput change (regression)")
    
    print("\nKey findings:")
    print(f"  - Throughput: {improvement:+.1f}%")
    print(f"  - TTFT P95: {((mbb.get('ttft_p95_ms', 0) - baseline.get('ttft_p95_ms', 0)) / baseline.get('ttft_p95_ms', 1) * 100):+.1f}%")
    print(f"  - SM Util: {(mbb.get('avg_sm_util', 0) - baseline.get('avg_sm_util', 0)):+.1f}pp")
    print(f"  - Token parity: {parity:.2f}%")
    
    print("\n" + "="*80 + "\n")


def main():
    """Main comparison script."""
    if len(sys.argv) != 3:
        print("Usage: python compare_results.py <baseline_summary.json> <mbb_summary.json>")
        sys.exit(1)
    
    baseline_path = sys.argv[1]
    mbb_path = sys.argv[2]
    
    try:
        baseline = load_results(baseline_path)
        mbb = load_results(mbb_path)
        compare(baseline, mbb)
    except FileNotFoundError as e:
        print(f"ERROR: File not found: {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

