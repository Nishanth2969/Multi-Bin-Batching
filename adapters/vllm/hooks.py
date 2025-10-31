"""
Integration hooks for vLLM scheduler.

This module provides integration points for adding multi-bin batching
to vLLM's continuous batching scheduler.
"""

from typing import List, Optional, Dict, Any


def add_mbb_flags_to_parser(parser):
    """Add multi-bin batching flags to vLLM argument parser.
    
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
        help="Bin edges as comma-separated integers (e.g., '128,512')"
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
        help="Maximum wait time in ms before promoting starved requests"
    )
    
    parser.add_argument(
        "--mbb-selection-policy",
        type=str,
        default="round_robin",
        choices=["round_robin", "weighted"],
        help="Request selection policy within bins"
    )
    
    return parser


def create_mbb_components(args) -> Dict[str, Any]:
    """Create multi-bin batching components from args.
    
    Args:
        args: Parsed arguments containing MBB configuration
        
    Returns:
        Dictionary with MBB components
    """
    # Import locally to avoid CUDA dependency when used for testing
    try:
        from mbb_core.predict import HeuristicPredictor
        from mbb_core.bins import BinEdges
        from mbb_core.queues import BinQueues
        from mbb_core.selector import BatchSelector
    except ImportError:
        raise ImportError("mbb_core not found. Install with: pip install -e .")
    
    # Parse bin edges
    edges_str = args.mbb_bin_edges
    if isinstance(edges_str, str):
        edges = [int(x.strip()) for x in edges_str.split(",")]
    else:
        edges = edges_str
    
    # Create bin edges
    bin_edges = BinEdges(edges) if edges else BinEdges([128, 512])
    
    # Create predictor
    if args.mbb_predictor == "heuristic":
        predictor = HeuristicPredictor()
    elif args.mbb_predictor == "oracle":
        from mbb_core.predict import OraclePredictor
        predictor = OraclePredictor()
    elif args.mbb_predictor == "model":
        from mbb_core.predict import ModelBasedPredictor
        predictor = ModelBasedPredictor()
    else:
        predictor = HeuristicPredictor()
    
    # Create bin queues
    bins = BinQueues(n_bins=bin_edges.num_bins())
    
    # Create batch selector
    selector = BatchSelector(
        kv_budget_tokens=8192,  # This should come from vLLM config
        max_batched_tokens=2048,  # This should come from vLLM config
        starvation_ms=args.mbb_starvation_ms,
        selection_policy=args.mbb_selection_policy
    )
    selector.set_bins(bins)
    
    return {
        'predictor': predictor,
        'bin_edges': bin_edges,
        'bins': bins,
        'selector': selector
    }


def should_use_mbb(args) -> bool:
    """Check if multi-bin batching should be used.
    
    Args:
        args: Parsed arguments
        
    Returns:
        True if MBB should be enabled
    """
    return hasattr(args, 'scheduler_policy') and args.scheduler_policy == "multi_bin"


# Integration guide for vLLM developers:
#
# 1. In vLLM's scheduler initialization, check for --scheduler-policy=multi_bin
# 2. If enabled, call create_mbb_components(args) to get MBB components
# 3. In the scheduling loop:
#    a. On request arrival: enqueue into appropriate bin via MBB
#    b. When building batch: call selector.build_batch() instead of default logic
#    c. Update bins and selector state after each step
#
# See docs for detailed integration instructions.

