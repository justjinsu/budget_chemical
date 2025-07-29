"""
Test sampler output shapes and basic properties.

Ensures that Monte Carlo sampler produces correctly shaped outputs
and weights sum to approximately 1.0.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from budget_chemical.monte_carlo.sampler import Sampler


class TestSamplerShapes:
    """Test suite for sampler output validation."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = {
            'seed': 42,
            'global_budget': {
                'low': 4.5e11,
                'mid': 5.0e11,
                'high': 5.5e11
            },
            'user_weights': {
                'responsibility': 0.30,
                'capability': 0.40,
                'equality': 0.30
            },
            'uncertainty': {
                'responsibility': {
                    'low': 0.0095,
                    'mid': 0.0110,
                    'high': 0.0128
                },
                'capability': {
                    'mu': 0.017,
                    'sd_pct': 0.05
                },
                'equality': {
                    'mu': 0.0067,
                    'sd_pct': 0.03
                }
            },
            'lambda_dist': {
                'type': 'beta',
                'a': 2.0,
                'b': 2.0
            },
            'curve_types': ['linear', 'exp', 'log']
        }
        self.sampler = Sampler(self.config)
    
    def test_sample_all_shapes(self):
        """Test that sample_all returns correctly shaped arrays."""
        n_draws = 100
        samples = self.sampler.sample_all(n_draws)
        
        # Check all required keys are present
        required_keys = [
            'global_budgets', 'responsibility_shares', 'capability_shares',
            'equality_shares', 'weights', 'lambdas', 'curve_types'
        ]
        
        for key in required_keys:
            assert key in samples, f"Missing key: {key}"
        
        # Check shapes
        assert samples['global_budgets'].shape == (n_draws,), "Global budgets shape incorrect"
        assert samples['responsibility_shares'].shape == (n_draws,), "Responsibility shares shape incorrect"
        assert samples['capability_shares'].shape == (n_draws,), "Capability shares shape incorrect"
        assert samples['equality_shares'].shape == (n_draws,), "Equality shares shape incorrect"
        assert samples['weights'].shape == (n_draws, 3), "Weights shape incorrect"
        assert samples['lambdas'].shape == (n_draws,), "Lambdas shape incorrect"
        assert len(samples['curve_types']) == n_draws, "Curve types length incorrect"
    
    def test_weights_sum_to_one(self):
        """Test that sampled weights sum to approximately 1.0."""
        n_draws = 200
        samples = self.sampler.sample_all(n_draws)
        
        weights = samples['weights']
        weight_sums = weights.sum(axis=1)
        
        # All weight sums should be very close to 1.0
        np.testing.assert_allclose(weight_sums, 1.0, rtol=1e-10,
                                  err_msg="Weights do not sum to 1.0")
        
        # Check that all weights are non-negative
        assert np.all(weights >= 0), "Some weights are negative"
    
    def test_global_budget_range(self):
        """Test that global budgets fall within expected range."""
        n_draws = 150
        samples = self.sampler.sample_all(n_draws)
        
        budgets = samples['global_budgets']
        
        # Should be within triangular distribution bounds
        assert np.all(budgets >= self.config['global_budget']['low']), "Budget below minimum"
        assert np.all(budgets <= self.config['global_budget']['high']), "Budget above maximum"
        
        # Mean should be close to mode
        expected_mean = (self.config['global_budget']['low'] + 
                        self.config['global_budget']['mid'] + 
                        self.config['global_budget']['high']) / 3
        actual_mean = np.mean(budgets)
        
        # Allow some tolerance due to sampling variation
        assert abs(actual_mean - expected_mean) / expected_mean < 0.1, (
            f"Global budget mean {actual_mean:.2e} differs significantly from expected {expected_mean:.2e}"
        )
    
    def test_lambda_bounds(self):
        """Test that lambda values are bounded in [0, 1]."""
        n_draws = 100
        samples = self.sampler.sample_all(n_draws)
        
        lambdas = samples['lambdas']
        
        assert np.all(lambdas >= 0), "Some lambda values are negative"
        assert np.all(lambdas <= 1), "Some lambda values exceed 1.0"
        
        # For beta(2,2), mean should be 0.5
        expected_mean = 0.5
        actual_mean = np.mean(lambdas)
        assert abs(actual_mean - expected_mean) < 0.1, (
            f"Lambda mean {actual_mean:.3f} differs from expected {expected_mean}"
        )
    
    def test_curve_types_validity(self):
        """Test that all curve types are valid."""
        n_draws = 80
        samples = self.sampler.sample_all(n_draws)
        
        curve_types = samples['curve_types']
        valid_types = set(self.config['curve_types'])
        
        for curve_type in curve_types:
            assert curve_type in valid_types, f"Invalid curve type: {curve_type}"
        
        # Should have reasonable distribution among types
        unique_types, counts = np.unique(curve_types, return_counts=True)
        assert len(unique_types) >= 2, "Too few curve types sampled"
    
    def test_allocation_factors_positive(self):
        """Test that allocation factors are positive."""
        n_draws = 120
        samples = self.sampler.sample_all(n_draws)
        
        # All allocation factors should be positive
        assert np.all(samples['responsibility_shares'] > 0), "Negative responsibility shares"
        assert np.all(samples['capability_shares'] > 0), "Negative capability shares"
        assert np.all(samples['equality_shares'] > 0), "Negative equality shares"
    
    def test_reproducibility(self):
        """Test that sampler produces reproducible results with same seed."""
        n_draws = 50
        
        # First sampling
        sampler1 = Sampler(self.config)
        samples1 = sampler1.sample_all(n_draws)
        
        # Second sampling with same seed
        sampler2 = Sampler(self.config)
        samples2 = sampler2.sample_all(n_draws)
        
        # Should be identical
        np.testing.assert_array_equal(samples1['global_budgets'], samples2['global_budgets'])
        np.testing.assert_array_equal(samples1['weights'], samples2['weights'])
        np.testing.assert_array_equal(samples1['lambdas'], samples2['lambdas'])
        assert samples1['curve_types'] == samples2['curve_types']
    
    def test_different_n_draws(self):
        """Test sampler with different numbers of draws."""
        test_sizes = [1, 10, 50, 500]
        
        for n_draws in test_sizes:
            samples = self.sampler.sample_all(n_draws)
            
            # Check shapes are correct
            assert len(samples['global_budgets']) == n_draws
            assert samples['weights'].shape[0] == n_draws
            assert len(samples['curve_types']) == n_draws
            
            # Check weights still sum to 1
            weight_sums = samples['weights'].sum(axis=1)
            np.testing.assert_allclose(weight_sums, 1.0, rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__])