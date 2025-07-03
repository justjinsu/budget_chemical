"""
Tests for pathway generation functionality.

These tests verify that emission pathways are generated correctly
and respect budget constraints through trapezoidal integration.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from ..pathway import PathwayGenerator, BudgetOverflowError


class TestPathwayGenerator:
    """Test suite for PathwayGenerator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.baseline_emissions = 50.0
        self.allocated_budget = 400.0  # Mt CO2e for 2035-2050
        self.generator = PathwayGenerator(self.baseline_emissions, self.allocated_budget)
    
    def test_generator_initialization(self):
        """Test pathway generator initialization."""
        assert self.generator.baseline_emissions == 50.0
        assert self.generator.allocated_budget == 400.0
        assert self.generator.start_year == 2035
        assert self.generator.end_year == 2050
        assert len(self.generator.years) == 16  # 2035-2050 inclusive
    
    def test_linear_to_zero_pathway(self):
        """Test linear to zero pathway generation."""
        pathway = self.generator.linear_to_zero()
        
        # Check structure
        assert isinstance(pathway, pd.DataFrame)
        assert len(pathway) == 16
        assert list(pathway.columns) == ['year', 'emission', 'cumulative', 'budget_left']
        
        # Check boundary conditions
        assert pathway.iloc[0]['emission'] == self.baseline_emissions
        assert pathway.iloc[-1]['emission'] == 0.0
        
        # Check monotonic decrease
        assert all(pathway['emission'].iloc[i] >= pathway['emission'].iloc[i+1] 
                  for i in range(len(pathway)-1))
    
    def test_constant_rate_pathway(self):
        """Test constant rate pathway generation."""
        rate = 5.0  # 5% per year
        pathway = self.generator.constant_rate(rate)
        
        # Check structure
        assert isinstance(pathway, pd.DataFrame)
        assert len(pathway) == 16
        
        # Check initial condition
        assert pathway.iloc[0]['emission'] == self.baseline_emissions
        
        # Check reduction rate
        for i in range(1, len(pathway)):
            expected = pathway.iloc[i-1]['emission'] * (1 - rate/100)
            assert abs(pathway.iloc[i]['emission'] - expected) < 1e-10
    
    def test_iea_proxy_pathway(self):
        """Test IEA proxy pathway generation."""
        pathway = self.generator.iea_proxy()
        
        # Check structure
        assert isinstance(pathway, pd.DataFrame)
        assert len(pathway) == 16
        
        # Check initial condition
        assert pathway.iloc[0]['emission'] == self.baseline_emissions
        
        # Check decreasing trend
        assert pathway.iloc[-1]['emission'] < pathway.iloc[0]['emission']
        
        # Check minimum emission threshold
        min_emission = 0.05 * self.baseline_emissions
        assert all(pathway['emission'] >= min_emission)
    
    def test_custom_pathway_linear(self):
        """Test custom pathway with linear interpolation."""
        waypoints = {
            2035: 50.0,
            2040: 30.0,
            2045: 15.0,
            2050: 0.0
        }
        
        pathway = self.generator.custom_pathway(waypoints, method='linear')
        
        # Check structure
        assert isinstance(pathway, pd.DataFrame)
        assert len(pathway) == 16
        
        # Check waypoints
        assert pathway.iloc[0]['emission'] == 50.0
        assert pathway.iloc[-1]['emission'] == 0.0
        
        # Check intermediate waypoint (2040 is index 5)
        year_2040_idx = pathway[pathway['year'] == 2040].index[0]
        assert abs(pathway.iloc[year_2040_idx]['emission'] - 30.0) < 1e-10
    
    def test_trapezoidal_integration_budget_constraint(self):
        """Test that trapezoidal integration respects budget constraints."""
        pathway = self.generator.linear_to_zero()
        
        # Manual trapezoidal integration
        emissions = pathway['emission'].values
        years = pathway['year'].values
        
        # Trapezoidal rule: sum of (y[i] + y[i+1]) * (x[i+1] - x[i]) / 2
        total_emissions = 0.0
        for i in range(len(emissions) - 1):
            dt = years[i+1] - years[i]  # Should be 1 year
            area = (emissions[i] + emissions[i+1]) * dt / 2
            total_emissions += area
        
        # Should be approximately equal to simple sum for annual data
        simple_sum = np.sum(emissions)
        
        # For annual data, trapezoidal is very close to simple sum
        assert abs(total_emissions - simple_sum) < 1e-10
        
        # Check budget constraint
        assert total_emissions <= self.allocated_budget
    
    def test_cumulative_emissions_consistency(self):
        """Test that cumulative emissions are calculated correctly."""
        pathway = self.generator.linear_to_zero()
        
        # Check cumulative calculation
        expected_cumulative = np.cumsum(pathway['emission'])
        np.testing.assert_allclose(pathway['cumulative'], expected_cumulative, rtol=1e-10)
        
        # Check budget_left calculation
        expected_budget_left = self.allocated_budget - pathway['cumulative']
        np.testing.assert_allclose(pathway['budget_left'], expected_budget_left, rtol=1e-10)
    
    def test_budget_overflow_detection(self):
        """Test that budget overflow is detected correctly."""
        # Create scenario that will overflow
        small_budget = 10.0  # Much smaller than needed
        small_generator = PathwayGenerator(self.baseline_emissions, small_budget)
        
        with pytest.raises(BudgetOverflowError):
            small_generator.linear_to_zero()
    
    def test_budget_overflow_constant_rate(self):
        """Test budget overflow detection for constant rate pathway."""
        # Use very small reduction rate with small budget
        small_budget = 50.0
        small_generator = PathwayGenerator(self.baseline_emissions, small_budget)
        
        with pytest.raises(BudgetOverflowError):
            small_generator.constant_rate(1.0)  # Only 1% reduction per year
    
    def test_pathway_validation_valid(self):
        """Test pathway validation for valid pathway."""
        pathway = self.generator.linear_to_zero()
        
        assert self.generator.validate_pathway(pathway)
    
    def test_pathway_validation_invalid_columns(self):
        """Test pathway validation with missing columns."""
        invalid_pathway = pd.DataFrame({
            'year': [2035, 2036],
            'emission': [50.0, 40.0]
            # Missing 'cumulative' and 'budget_left'
        })
        
        assert not self.generator.validate_pathway(invalid_pathway)
    
    def test_pathway_validation_negative_emissions(self):
        """Test pathway validation with negative emissions."""
        invalid_pathway = pd.DataFrame({
            'year': [2035, 2036],
            'emission': [50.0, -10.0],  # Negative emission
            'cumulative': [50.0, 40.0],
            'budget_left': [350.0, 360.0]
        })
        
        assert not self.generator.validate_pathway(invalid_pathway)
    
    def test_pathway_validation_budget_exceeded(self):
        """Test pathway validation when budget is exceeded."""
        # Create pathway that exceeds budget
        large_emissions = [1000.0, 1000.0]  # Way too large
        invalid_pathway = pd.DataFrame({
            'year': [2035, 2036],
            'emission': large_emissions,
            'cumulative': np.cumsum(large_emissions),
            'budget_left': self.allocated_budget - np.cumsum(large_emissions)
        })
        
        assert not self.generator.validate_pathway(invalid_pathway)
    
    def test_pathway_summary_statistics(self):
        """Test pathway summary statistics calculation."""
        pathway = self.generator.linear_to_zero()
        summary = self.generator.get_pathway_summary(pathway)
        
        # Check required keys
        required_keys = [
            'total_emissions', 'peak_emission', 'final_emission',
            'peak_to_final_reduction_pct', 'budget_utilization_pct', 'overshoot_year'
        ]
        
        for key in required_keys:
            assert key in summary
        
        # Check values
        assert summary['total_emissions'] == pathway['emission'].sum()
        assert summary['peak_emission'] == pathway['emission'].max()
        assert summary['final_emission'] == pathway['emission'].iloc[-1]
        assert summary['overshoot_year'] is None  # No overshoot for valid pathway
        
        # Check percentages
        assert 0 <= summary['budget_utilization_pct'] <= 100
        assert 0 <= summary['peak_to_final_reduction_pct'] <= 100
    
    def test_overshoot_year_detection(self):
        """Test overshoot year detection."""
        # Create pathway that will overshoot
        small_budget = 50.0
        
        # Create pathway manually that overshoots
        pathway = pd.DataFrame({
            'year': [2035, 2036, 2037],
            'emission': [40.0, 30.0, 20.0],
            'cumulative': [40.0, 70.0, 90.0],  # Exceeds 50 Mt budget
            'budget_left': [10.0, -20.0, -40.0]  # Goes negative
        })
        
        small_generator = PathwayGenerator(50.0, small_budget)
        summary = small_generator.get_pathway_summary(pathway)
        
        assert summary['overshoot_year'] == 2036  # First year with negative budget_left
    
    def test_zero_emission_pathway(self):
        """Test pathway with zero baseline emissions."""
        zero_generator = PathwayGenerator(0.0, 100.0)
        pathway = zero_generator.linear_to_zero()
        
        # All emissions should be zero
        assert all(pathway['emission'] == 0.0)
        assert all(pathway['cumulative'] == 0.0)
        assert all(pathway['budget_left'] == 100.0)
    
    def test_custom_pathway_spline(self):
        """Test custom pathway with spline interpolation."""
        waypoints = {
            2035: 50.0,
            2040: 30.0,
            2045: 15.0,
            2050: 0.0
        }
        
        pathway = self.generator.custom_pathway(waypoints, method='spline')
        
        # Check structure
        assert isinstance(pathway, pd.DataFrame)
        assert len(pathway) == 16
        
        # Check waypoints are honored
        assert pathway.iloc[0]['emission'] == 50.0
        assert pathway.iloc[-1]['emission'] == 0.0
        
        # Check all emissions are non-negative
        assert all(pathway['emission'] >= 0)


if __name__ == '__main__':
    pytest.main([__file__])