"""
Test budget compliance of emission pathways.

Ensures that generated pathways exactly meet allocated budgets within tolerance.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'model_D_MC'))

from model_D_MC.pathwayCalculation import PathwayCalculator
from model_D_MC.mc_sampler import Sampler


class TestPathwayBudgetMatch:
    """Test suite for pathway budget compliance."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.years = np.arange(2024, 2051)
        self.config = {
            'timeline': {
                'start_year': 2024,
                'mid_year': 2035,
                'end_year': 2050
            },
            'solver': {
                'tolerance': 1e-3,
                'max_iter': 60
            }
        }
        self.calculator = PathwayCalculator(self.years, self.config['solver'])
        self.tolerance = 1e-2  # 1% budget error tolerance
    
    def test_linear_pathway_budget_match(self):
        """Test that linear pathways meet budget constraints."""
        test_cases = [
            {'start_emission': 100e6, 'budget': 1e9, 'frontload': 0.5},
            {'start_emission': 50e6, 'budget': 500e6, 'frontload': 0.3},
            {'start_emission': 200e6, 'budget': 2e9, 'frontload': 0.7}
        ]
        
        for case in test_cases:
            pathway = self.calculator.build_path(
                start_emission=case['start_emission'],
                budget=case['budget'],
                curve_type='linear',
                frontload=case['frontload']
            )
            
            # Calculate actual budget (area under curve)
            actual_budget = np.trapz(pathway, dx=1)
            budget_error = abs(actual_budget - case['budget']) / case['budget']
            
            assert budget_error < self.tolerance, (
                f"Linear pathway budget error {budget_error:.4f} exceeds tolerance {self.tolerance}"
            )
            
            # Check final emission is near zero
            assert pathway[-1] < 1.0, f"Final emission {pathway[-1]} should be near zero"
            
            # Check non-negative emissions
            assert np.all(pathway >= 0), "Pathway contains negative emissions"
    
    def test_exponential_pathway_budget_match(self):
        """Test that exponential pathways meet budget constraints."""
        test_cases = [
            {'start_emission': 75e6, 'budget': 800e6, 'frontload': 0.4},
            {'start_emission': 120e6, 'budget': 1.2e9, 'frontload': 0.6}
        ]
        
        for case in test_cases:
            pathway = self.calculator.build_path(
                start_emission=case['start_emission'],
                budget=case['budget'],
                curve_type='exp',
                frontload=case['frontload']
            )
            
            actual_budget = np.trapz(pathway, dx=1)
            budget_error = abs(actual_budget - case['budget']) / case['budget']
            
            assert budget_error < self.tolerance, (
                f"Exponential pathway budget error {budget_error:.4f} exceeds tolerance"
            )
    
    def test_logarithmic_pathway_budget_match(self):
        """Test that logarithmic pathways meet budget constraints."""
        test_cases = [
            {'start_emission': 80e6, 'budget': 900e6, 'frontload': 0.5},
            {'start_emission': 150e6, 'budget': 1.5e9, 'frontload': 0.8}
        ]
        
        for case in test_cases:
            pathway = self.calculator.build_path(
                start_emission=case['start_emission'],
                budget=case['budget'],
                curve_type='log',
                frontload=case['frontload']
            )
            
            actual_budget = np.trapz(pathway, dx=1)
            budget_error = abs(actual_budget - case['budget']) / case['budget']
            
            assert budget_error < self.tolerance, (
                f"Logarithmic pathway budget error {budget_error:.4f} exceeds tolerance"
            )
    
    def test_pathway_validation(self):
        """Test pathway validation functionality."""
        start_emission = 100e6
        budget = 1e9
        
        pathway = self.calculator.build_path(
            start_emission=start_emission,
            budget=budget,
            curve_type='linear',
            frontload=0.5
        )
        
        validation = self.calculator.validate_pathway(pathway, budget)
        
        assert 'is_valid' in validation
        assert 'budget_error_pct' in validation
        assert 'actual_budget' in validation
        assert 'target_budget' in validation
        
        # Should be valid for well-formed pathway
        assert validation['is_valid'], "Pathway should be valid"
        assert validation['budget_error_pct'] < 0.1, "Budget error should be small"
    
    def test_random_pathway_scenarios(self):
        """Test pathways with random parameters."""
        np.random.seed(42)  # For reproducible tests
        
        n_tests = 10
        for _ in range(n_tests):
            start_emission = np.random.uniform(50e6, 200e6)
            budget = np.random.uniform(500e6, 2e9)
            frontload = np.random.uniform(0.2, 0.8)
            curve_type = np.random.choice(['linear', 'exp', 'log'])
            
            try:
                pathway = self.calculator.build_path(
                    start_emission=start_emission,
                    budget=budget,
                    curve_type=curve_type,
                    frontload=frontload
                )
                
                actual_budget = np.trapz(pathway, dx=1)
                budget_error = abs(actual_budget - budget) / budget
                
                # Allow slightly higher tolerance for random tests
                assert budget_error < 0.05, (
                    f"Random test failed: {curve_type} pathway with "
                    f"budget error {budget_error:.4f}"
                )
                
            except Exception as e:
                pytest.fail(f"Random test failed with parameters: "
                           f"start={start_emission:.0f}, budget={budget:.0f}, "
                           f"frontload={frontload:.2f}, curve={curve_type}. "
                           f"Error: {e}")


if __name__ == "__main__":
    pytest.main([__file__])