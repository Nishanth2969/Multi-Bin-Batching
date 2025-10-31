"""
Multi-Bin Batching (MBB) core implementation.

MBB improves LLM inference throughput by grouping requests into bins
based on predicted output length, then batching within each bin.
"""

from .predict import HeuristicPredictor, OraclePredictor, ModelBasedPredictor
from .bins import BinEdges, create_equal_width_bins, create_quantile_bins
from .queues import Request, BinQueues, RequestReassigner
from .selector import (
    RoundRobinSelector,
    WeightedSelector,
    StarvationGuard,
    BatchSelector
)
from .scheduler import MBScheduler

__all__ = [
    'HeuristicPredictor',
    'OraclePredictor',
    'ModelBasedPredictor',
    'BinEdges',
    'create_equal_width_bins',
    'create_quantile_bins',
    'Request',
    'BinQueues',
    'RequestReassigner',
    'RoundRobinSelector',
    'WeightedSelector',
    'StarvationGuard',
    'BatchSelector',
    'MBScheduler',
]

__version__ = "0.1.0"

