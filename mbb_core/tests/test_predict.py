"""Tests for predictor components."""

import pytest
from mbb_core.predict import HeuristicPredictor, OraclePredictor


def test_heuristic_predictor_default():
    """Test default heuristic predictor."""
    pred = HeuristicPredictor()
    
    assert pred(0) == 32  # 0.5 * 0 + 32, minimum is 1 but we don't clip because 32 > 1
    assert pred(100) == 82  # 0.5 * 100 + 32
    assert pred(1000) == 532  # 0.5 * 1000 + 32


def test_heuristic_predictor_custom_params():
    """Test heuristic predictor with custom parameters."""
    pred = HeuristicPredictor(a=1.0, b=10, Lmax=100)
    
    assert pred(20) == 30
    assert pred(1000) == 100  # Capped at Lmax
    assert pred(200) == 100  # Also capped


def test_heuristic_predictor_bounds():
    """Test predictor respects min/max bounds."""
    pred = HeuristicPredictor(Lmax=128)
    
    # Should be capped at 128
    assert pred(1000) == 128
    
    # Should be at least 1
    assert pred(-1000) == 1


def test_heuristic_predictor_update():
    """Test predictor update logic."""
    pred = HeuristicPredictor(a=0.5, b=32)
    
    # Test parameter update suggestion
    suggested_a = pred.from_known_output(prompt_tokens=100, true_output_tokens=100)
    assert suggested_a == (100 - 32) / 100  # (true - b) / prompt


def test_oracle_predictor():
    """Test oracle predictor."""
    pred = OraclePredictor()
    
    assert pred(100, 50) == 50  # Returns true output length
    assert pred(200, 300) == 300


if __name__ == "__main__":
    pytest.main([__file__])

