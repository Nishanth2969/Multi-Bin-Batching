"""
Integration hooks for SGLang scheduler.

This module provides integration points for adding multi-bin batching
to SGLang's scheduler.
"""

from typing import List, Optional, Dict, Any


def add_mbb_flags_to_parser(parser):
    """Add multi-bin batching flags to SGLang argument parser.
    
    Args:
        parser: argparse.ArgumentParser to extend
        
    Returns:
        Modified parser
    """
    parser.add_argument(
        "--scheduler-policy",
        type=str,
        default="continuous",
        choices=["continuous", "multi_bin"],
        help="Scheduler policy: continuous (default) or multi_bin"
    )
    
    parser.add_argument(
        "--mbb-bins",
        type=int,
        default=3,
        help="Number of bins for multi-bin batching"
    )
    
    parser.add_argument(
        "--mbb-bin-edges",
        type=str,
        default="128,512",
        help="Bin edges as comma-separated integers"
    )
    
    parser.add_argument(
        "--mbb-predictor",
        type=str,
        default="heuristic",
        choices=["heuristic", "model", "oracle"],
        help="Predictor type for output length estimation"
    )
    
    parser.add_argument(
        "--mbb-starvation-ms",
        type=int,
        default=500,
        help="Maximum wait time in ms before promoting"
    )
    
    return parser


def create_mbb_components(args) -> Dict[str, Any]:
    """Create multi-bin batching components from args.
    
    Args:
        args: Parsed arguments containing MBB configuration
        
    Returns:
        Dictionary with MBB components
    """
    try:
        from mbb_core.predict import HeuristicPredictor
        from mbb_core.bins import BinEdges
        from mbb_core.queues import BinQueues
        from mbb_core.selector import BatchSelector
    except ImportError:
        raise ImportError("mbb_core not found")
    
    edges_str = args.mbb_bin_edges
    if isinstance(edges_str, str):
        edges = [int(x.strip()) for x in edges_str.split(",")]
    else:
        edges = edges_str
    
    bin_edges = BinEdges(edges) if edges else BinEdges([128, 512])
    
    if args.mbb_predictor == "heuristic":
        predictor = HeuristicPredictor()
    else:
        predictor = HeuristicPredictor()
    
    bins = BinQueues(n_bins=bin_edges.num_bins())
    
    selector = BatchSelector(
        kv_budget_tokens=8192,
        max_batched_tokens=2048,
        starvation_ms=args.mbb_starvation_ms
    )
    selector.set_bins(bins)
    
    return {
        'predictor': predictor,
        'bin_edges': bin_edges,
        'bins': bins,
        'selector': selector
    }


def should_use_mbb(args) -> bool:
    """Check if multi-bin batching should be used."""
    return hasattr(args, 'scheduler_policy') and args.scheduler_policy == "multi_bin"


# Integration guide for SGLang developers:
#
# 1. Check for --scheduler-policy=multi_bin in SGLang initialization
# 2. Create MBB components using create_mbb_components()
# 3. Integrate into SGLang's CPU scheduler:
#    - Enqueue requests into bins based on predicted output length
#    - Use selector.build_batch() to form batches
#    - Maintain compatibility with SGLang's radix prefix caching
#
# See docs for detailed integration instructions.

