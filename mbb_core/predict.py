"""
Output length predictor for multi-bin batching.

Implements a simple heuristic predictor that estimates output length
based on prompt length.
"""

from dataclasses import dataclass


@dataclass
class HeuristicPredictor:
    """Heuristic predictor for output length based on prompt tokens."""
    
    a: float = 0.5
    b: int = 32
    Lmax: int = 4096

    def __call__(self, prompt_tokens: int) -> int:
        """Predict output length from prompt length.
        
        Args:
            prompt_tokens: Number of tokens in the prompt
            
        Returns:
            Predicted output length in tokens
        """
        pred = int(self.a * prompt_tokens + self.b)
        return max(1, min(self.Lmax, pred))
    
    def from_known_output(self, prompt_tokens: int, true_output_tokens: int) -> float:
        """Update predictor with ground truth.
        
        Simple online learning variant - returns suggested 'a' parameter
        based on observed output length.
        
        Args:
            prompt_tokens: Input prompt tokens
            true_output_tokens: Actual output length
            
        Returns:
            Suggested 'a' parameter
        """
        if prompt_tokens > 0:
            return (true_output_tokens - self.b) / prompt_tokens
        return self.a


@dataclass
class OraclePredictor:
    """Oracle predictor using known output lengths.
    
    For benchmarking and establishing upper bounds on MBB performance.
    """
    
    def __call__(self, prompt_tokens: int, true_output_tokens: int) -> int:
        return true_output_tokens


@dataclass
class AdaptivePredictor:
    """Adaptive predictor that learns from history."""
    
    def __init__(self, alpha: float = 0.1, min_samples: int = 10):
        """Initialize adaptive predictor.
        
        Args:
            alpha: Learning rate (0-1, higher = faster adaptation)
            min_samples: Minimum samples before using learned model
        """
        self.alpha = alpha
        self.min_samples = min_samples
        self.avg_ratio = 0.5
        self.samples = 0
        self.base_predictor = HeuristicPredictor()
    
    def __call__(self, prompt_tokens: int) -> int:
        """Predict output length using adaptive model."""
        if self.samples < self.min_samples:
            return self.base_predictor(prompt_tokens)
        
        pred = int(self.avg_ratio * prompt_tokens + 32)
        return max(1, min(4096, pred))
    
    def update(self, prompt_tokens: int, true_output_tokens: int):
        """Update predictor with observed output length.
        
        Args:
            prompt_tokens: Input prompt tokens
            true_output_tokens: Actual output length
        """
        if prompt_tokens > 0:
            ratio = true_output_tokens / prompt_tokens
            if self.samples == 0:
                self.avg_ratio = ratio
            else:
                self.avg_ratio = (1 - self.alpha) * self.avg_ratio + self.alpha * ratio
            self.samples += 1


@dataclass
class ModelBasedPredictor:
    """Model-based predictor using adaptive learning."""
    
    def __init__(self):
        """Initialize model-based predictor."""
        self.adaptive = AdaptivePredictor()
    
    def __call__(self, prompt_tokens: int, prompt_text: str = "") -> int:
        """Predict using adaptive learned model.
        
        Args:
            prompt_tokens: Prompt length in tokens
            prompt_text: Optional prompt text for richer features
            
        Returns:
            Predicted output length
        """
        return self.adaptive(prompt_tokens)
    
    def update(self, prompt_tokens: int, true_output_tokens: int):
        """Update model with observed data."""
        self.adaptive.update(prompt_tokens, true_output_tokens)

