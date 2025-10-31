"""
Generate plots for MBB benchmark report.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

sns.set_style("whitegrid")
sns.set_palette("husl")


def load_summary(filepath: str) -> dict:
    """Load summary JSON."""
    with open(filepath, 'r') as f:
        return json.load(f)


def load_detailed_csv(filepath: str) -> pd.DataFrame:
    """Load detailed metrics CSV."""
    return pd.read_csv(filepath)


def load_gpu_csv(filepath: str) -> pd.DataFrame:
    """Load GPU stats CSV."""
    return pd.read_csv(filepath)


def plot_tps_vs_bins(results_dir: str, output_path: str):
    """Plot throughput vs number of bins."""
    
    # Find all ablation results
    results_dir = Path(results_dir)
    summaries = []
    
    for k in [1, 2, 3, 4, 8]:
        pattern = f"ablation_k{k}_*_summary.json"
        files = list(results_dir.glob(pattern))
        for f in files:
            data = load_summary(str(f))
            summaries.append({
                'k': k,
                'tps': data.get('tps', 0),
                'policy': 'equalmass' if 'equalmass' in f.name else 'static'
            })
    
    if not summaries:
        print("No ablation results found")
        return
    
    df = pd.DataFrame(summaries)
    
    plt.figure(figsize=(10, 6))
    
    for policy in df['policy'].unique():
        subset = df[df['policy'] == policy]
        plt.plot(subset['k'], subset['tps'], marker='o', label=policy, linewidth=2)
    
    plt.xlabel('Number of Bins (k)', fontsize=12)
    plt.ylabel('Throughput (tokens/second)', fontsize=12)
    plt.title('Throughput vs. Number of Bins', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_gpu_utilization_timeline(gpu_csv_baseline: str, gpu_csv_mbb: str, output_path: str):
    """Plot GPU utilization over time."""
    
    baseline_df = load_gpu_csv(gpu_csv_baseline)
    mbb_df = load_gpu_csv(gpu_csv_mbb)
    
    # Convert timestamps to relative seconds
    baseline_df['time_s'] = (baseline_df['timestamp_ms'] - baseline_df['timestamp_ms'].min()) / 1000
    mbb_df['time_s'] = (mbb_df['timestamp_ms'] - mbb_df['timestamp_ms'].min()) / 1000
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # SM Utilization
    ax1.plot(baseline_df['time_s'], baseline_df['sm_utilization'], 
             label='Baseline', alpha=0.7, linewidth=1.5)
    ax1.plot(mbb_df['time_s'], mbb_df['sm_utilization'], 
             label='MBB', alpha=0.7, linewidth=1.5)
    ax1.set_ylabel('SM Utilization (%)', fontsize=11)
    ax1.set_title('GPU Utilization Timeline', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Memory Utilization
    ax2.plot(baseline_df['time_s'], baseline_df['memory_utilization'], 
             label='Baseline', alpha=0.7, linewidth=1.5)
    ax2.plot(mbb_df['time_s'], mbb_df['memory_utilization'], 
             label='MBB', alpha=0.7, linewidth=1.5)
    ax2.set_xlabel('Time (seconds)', fontsize=11)
    ax2.set_ylabel('Memory Utilization (%)', fontsize=11)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_latency_comparison(baseline_summary: str, mbb_summary: str, output_path: str):
    """Plot latency comparison."""
    
    baseline = load_summary(baseline_summary)
    mbb = load_summary(mbb_summary)
    
    metrics = ['ttft_p50_ms', 'ttft_p95_ms', 'tpot_p50_ms', 'tpot_p95_ms']
    labels = ['TTFT P50', 'TTFT P95', 'TPOT P50', 'TPOT P95']
    
    baseline_vals = [baseline.get(m, 0) for m in metrics]
    mbb_vals = [mbb.get(m, 0) for m in metrics]
    
    x = range(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.bar([i - width/2 for i in x], baseline_vals, width, label='Baseline', alpha=0.8)
    ax.bar([i + width/2 for i in x], mbb_vals, width, label='MBB', alpha=0.8)
    
    ax.set_ylabel('Latency (ms)', fontsize=12)
    ax.set_title('Latency Comparison: Baseline vs. MBB', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_throughput_comparison(baseline_summary: str, mbb_summary: str, output_path: str):
    """Plot throughput bar chart."""
    
    baseline = load_summary(baseline_summary)
    mbb = load_summary(mbb_summary)
    
    baseline_tps = baseline.get('tps', 0)
    mbb_tps = mbb.get('tps', 0)
    improvement = ((mbb_tps - baseline_tps) / baseline_tps * 100) if baseline_tps > 0 else 0
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    bars = ax.bar(['Baseline', 'MBB'], [baseline_tps, mbb_tps], 
                   color=['#3498db', '#2ecc71'], alpha=0.8, width=0.6)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Add improvement label
    ax.text(1, mbb_tps + (mbb_tps * 0.05), f'+{improvement:.1f}%',
            ha='center', fontsize=13, fontweight='bold', color='green')
    
    ax.set_ylabel('Throughput (tokens/second)', fontsize=12)
    ax.set_title('Throughput Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    """Generate all plots for report."""
    
    if len(sys.argv) < 2:
        print("Usage: python generate_plots.py <results_dir>")
        print("\nExpected files in results_dir:")
        print("  - baseline_*_summary.json")
        print("  - baseline_*_gpu.csv")
        print("  - mbb_*_summary.json")
        print("  - mbb_*_gpu.csv")
        print("  - ablation_k*_*_summary.json (for various k values)")
        sys.exit(1)
    
    results_dir = Path(sys.argv[1])
    output_dir = results_dir / "plots"
    output_dir.mkdir(exist_ok=True)
    
    print(f"Generating plots from: {results_dir}")
    print(f"Output directory: {output_dir}")
    
    # Find baseline and MBB files
    baseline_summaries = list(results_dir.glob("baseline_*_summary.json"))
    mbb_summaries = list(results_dir.glob("mbb_k3_*_summary.json"))
    
    if not baseline_summaries:
        print("ERROR: No baseline summary files found")
        sys.exit(1)
    
    if not mbb_summaries:
        print("ERROR: No MBB summary files found")
        sys.exit(1)
    
    baseline_summary = str(baseline_summaries[0])
    mbb_summary = str(mbb_summaries[0])
    
    baseline_gpu = baseline_summary.replace('_summary.json', '_gpu.csv')
    mbb_gpu = mbb_summary.replace('_summary.json', '_gpu.csv')
    
    # Generate plots
    print("\n[1/4] Generating throughput comparison...")
    plot_throughput_comparison(
        baseline_summary, 
        mbb_summary,
        str(output_dir / "throughput_comparison.png")
    )
    
    print("[2/4] Generating latency comparison...")
    plot_latency_comparison(
        baseline_summary,
        mbb_summary,
        str(output_dir / "latency_comparison.png")
    )
    
    if Path(baseline_gpu).exists() and Path(mbb_gpu).exists():
        print("[3/4] Generating GPU utilization timeline...")
        plot_gpu_utilization_timeline(
            baseline_gpu,
            mbb_gpu,
            str(output_dir / "gpu_utilization_timeline.png")
        )
    else:
        print("[3/4] Skipping GPU timeline (CSV files not found)")
    
    print("[4/4] Generating TPS vs bins plot...")
    plot_tps_vs_bins(
        str(results_dir),
        str(output_dir / "tps_vs_bins.png")
    )
    
    print(f"\n✓ All plots generated in: {output_dir}")
    print("\nGenerated files:")
    for plot in output_dir.glob("*.png"):
        print(f"  - {plot.name}")


if __name__ == "__main__":
    main()

