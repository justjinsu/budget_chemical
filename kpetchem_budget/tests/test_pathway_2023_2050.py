"""
Tests for pathway generation with 2023-2050 timeline.

These tests verify that emission pathways correctly cover the 2023-2050 period
and that trapezoidal integration respects budget constraints.
"""

import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pathway import PathwayGenerator, BudgetOverflowError, mark_milestones
from data_layer import get_timeline_years


class TestPathway2023_2050:
    """Test suite for 2023-2050 pathway generation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.baseline_emissions = 50.0
        self.allocated_budget = 800.0  # Mt CO2e for 2023-2050
        self.generator = PathwayGenerator(
            self.baseline_emissions, 
            self.allocated_budget,
            start_year=2023,
            net_zero_year=2050
        )
    
    def test_timeline_coverage_2023_2050(self):
        """Test that pathways cover exactly 2023-2050 period."""
        pathway = self.generator.linear_to_zero()
        
        # Check timeline coverage
        assert pathway['year'].min() == 2023
        assert pathway['year'].max() == 2050
        assert len(pathway) == 28  # 2023-2050 inclusive
        
        # Check year sequence
        expected_years = get_timeline_years()
        assert np.array_equal(pathway['year'].values, expected_years)
    
    def test_pathway_integral_2023_2050(self):
        """Test trapezoidal integration over 2023-2050 period respects budget."""
        pathways_to_test = [
            ('linear', lambda: self.generator.linear_to_zero()),
            ('constant', lambda: self.generator.constant_rate(10.0)),
            ('logistic', lambda: self.generator.logistic_decline()),
            ('iea_proxy', lambda: self.generator.iea_proxy())
        ]
        
        for name, pathway_func in pathways_to_test:
            pathway = pathway_func()
            
            # Manual trapezoidal integration over 2023-2050
            emissions = pathway['emission'].values
            years = pathway['year'].values
            
            # Trapezoidal rule for cumulative emissions
            total_emissions_trap = 0.0
            for i in range(len(emissions) - 1):
                dt = years[i+1] - years[i]  # Should be 1 year
                area = (emissions[i] + emissions[i+1]) * dt / 2
                total_emissions_trap += area
            
            # For annual data, trapezoidal ≈ simple sum
            total_emissions_simple = np.sum(emissions)
            
            # Both should be within budget (with 1% tolerance)
            budget_tolerance = self.allocated_budget * 0.01
            
            assert total_emissions_trap <= self.allocated_budget + budget_tolerance, \
                f"{name} pathway trapezoidal integral exceeds budget"
            
            assert total_emissions_simple <= self.allocated_budget + budget_tolerance, \
                f"{name} pathway simple sum exceeds budget"
            
            # Trapezoidal and simple sum should be close for annual data
            assert abs(total_emissions_trap - total_emissions_simple) < 1e-10, \
                f"{name} pathway: trapezoidal and simple sum should be nearly equal"
    
    def test_milestone_columns_2035_2050(self):
        """Test that milestone markers correctly identify 2035 and 2050."""
        pathway = self.generator.linear_to_zero()
        
        # Check milestone markers exist
        assert 'is2035' in pathway.columns
        assert 'is2050' in pathway.columns
        
        # Check exactly one True value for each milestone
        assert pathway['is2035'].sum() == 1
        assert pathway['is2050'].sum() == 1
        
        # Check correct years are marked
        year_2035_row = pathway[pathway['is2035']].iloc[0]
        year_2050_row = pathway[pathway['is2050']].iloc[0]
        
        assert year_2035_row['year'] == 2035
        assert year_2050_row['year'] == 2050
    
    def test_mark_milestones_function(self):
        """Test standalone mark_milestones function."""
        # Create test DataFrame
        years = np.arange(2023, 2051)
        df = pd.DataFrame({
            'year': years,
            'emission': np.linspace(50, 0, len(years))
        })
        
        # Add milestones
        marked_df = mark_milestones(df)
        
        # Check milestone columns added
        assert 'is2035' in marked_df.columns
        assert 'is2050' in marked_df.columns
        
        # Check exactly one True for each
        assert marked_df['is2035'].sum() == 1
        assert marked_df['is2050'].sum() == 1
        
        # Check correct years
        assert marked_df[marked_df['is2035']]['year'].iloc[0] == 2035
        assert marked_df[marked_df['is2050']]['year'].iloc[0] == 2050
    
    def test_budget_overflow_detection_2023_2050(self):
        """Test budget overflow detection for 2023-2050 period."""
        # Create generator with very small budget
        small_budget = 100.0  # Too small for 50 Mt/yr over 28 years
        small_generator = PathwayGenerator(
            self.baseline_emissions,
            small_budget,
            start_year=2023,
            net_zero_year=2050
        )
        
        # Should raise BudgetOverflowError
        try:
            small_generator.linear_to_zero()
            assert False, "Should have raised BudgetOverflowError"
        except BudgetOverflowError as e:
            assert "2023-2050" in str(e) or "Mt CO2e" in str(e)
    
    def test_net_zero_year_variations(self):
        """Test pathway generation with different net-zero years."""
        net_zero_years = [2045, 2050, 2055]
        
        for net_zero_year in net_zero_years:
            generator = PathwayGenerator(
                self.baseline_emissions,
                self.allocated_budget,
                start_year=2023,
                net_zero_year=net_zero_year
            )
            
            pathway = generator.linear_to_zero()
            
            # Check timeline still covers 2023-2050
            assert pathway['year'].min() == 2023
            assert pathway['year'].max() == 2050
            
            # Check net-zero constraint
            if net_zero_year <= 2050:
                net_zero_emission = pathway[pathway['year'] == net_zero_year]['emission'].iloc[0]
                assert net_zero_emission == 0.0 or net_zero_emission < 1e-6
                
                # All years after net-zero should be zero
                post_zero_years = pathway[pathway['year'] > net_zero_year]
                if not post_zero_years.empty:
                    assert (post_zero_years['emission'] < 1e-6).all()
    
    def test_pathway_summary_includes_milestones(self):
        """Test that pathway summary includes 2035 and 2050 milestone data."""
        pathway = self.generator.linear_to_zero()
        summary = self.generator.get_pathway_summary(pathway)
        
        # Check milestone keys exist
        milestone_keys = [
            'emissions_2023', 'emissions_2035', 'emissions_2050',
            'cumulative_2035', 'cumulative_2050',
            'reduction_2023_to_2035_pct', 'reduction_2035_to_2050_pct'
        ]
        
        for key in milestone_keys:
            assert key in summary, f"Missing milestone key: {key}"
        
        # Check values are reasonable
        assert summary['emissions_2023'] == self.baseline_emissions
        assert summary['emissions_2035'] < summary['emissions_2023']
        assert summary['emissions_2050'] <= summary['emissions_2035']
        assert 0 <= summary['reduction_2023_to_2035_pct'] <= 100
        assert 0 <= summary['reduction_2035_to_2050_pct'] <= 100
    
    def test_logistic_decline_new_pathway(self):
        """Test the new logistic decline pathway family."""
        pathway = self.generator.logistic_decline(k_factor=1.0, midpoint_year=2035)
        
        # Check basic structure
        assert len(pathway) == 28
        assert pathway['year'].min() == 2023
        assert pathway['year'].max() == 2050
        
        # Check logistic behavior
        assert pathway.iloc[0]['emission'] == self.baseline_emissions  # Starts at baseline
        assert pathway.iloc[-1]['emission'] < pathway.iloc[0]['emission']  # Decreases
        
        # Check budget constraint
        total_emissions = pathway['emission'].sum()
        assert total_emissions <= self.allocated_budget
        
        # Check milestone markers
        assert pathway['is2035'].sum() == 1
        assert pathway['is2050'].sum() == 1
    
    def test_pathway_validation_2023_2050(self):
        """Test pathway validation for 2023-2050 requirements."""
        pathway = self.generator.linear_to_zero()
        
        # Should pass validation
        assert self.generator.validate_pathway(pathway)
        
        # Test invalid pathways
        
        # Missing milestone columns
        invalid_pathway = pathway.drop(columns=['is2035', 'is2050'])
        assert not self.generator.validate_pathway(invalid_pathway)
        
        # Wrong timeline
        wrong_timeline = pathway.copy()
        wrong_timeline.loc[0, 'year'] = 2022  # Starts before 2023
        assert not self.generator.validate_pathway(wrong_timeline)
        
        # Multiple milestone markers
        multiple_milestones = pathway.copy()
        multiple_milestones.loc[1, 'is2035'] = True  # Two 2035 markers
        assert not self.generator.validate_pathway(multiple_milestones)
    
    def test_compare_pathways_includes_all_families(self):
        """Test pathway comparison with all four families."""
        pathways = {
            'Linear': self.generator.linear_to_zero(),
            'Constant': self.generator.constant_rate(8.0),
            'Logistic': self.generator.logistic_decline(),
            'IEA': self.generator.iea_proxy()
        }
        
        comparison = self.generator.compare_pathways(pathways)
        
        # Check all pathways included
        assert len(comparison) == 4
        assert set(comparison.index) == set(pathways.keys())
        
        # Check milestone columns present
        milestone_cols = [
            'emissions_2035', 'emissions_2050',
            'cumulative_2035', 'cumulative_2050'
        ]
        
        for col in milestone_cols:
            assert col in comparison.columns
            # All values should be non-negative
            assert (comparison[col] >= 0).all()
    
    def test_budget_utilization_metrics(self):
        """Test budget utilization calculations for 2023-2050."""
        pathway = self.generator.linear_to_zero()
        summary = self.generator.get_pathway_summary(pathway)
        
        # Check budget utilization
        assert 'budget_utilization_pct' in summary
        assert 0 <= summary['budget_utilization_pct'] <= 100
        
        # Manual calculation
        total_emissions = pathway['emission'].sum()
        expected_utilization = (total_emissions / self.allocated_budget) * 100
        
        assert abs(summary['budget_utilization_pct'] - expected_utilization) < 1e-6


if __name__ == '__main__':
    # Simple test runner
    test_instance = TestPathway2023_2050()
    
    methods = [method for method in dir(test_instance) if method.startswith('test_')]
    
    passed = 0
    failed = 0
    
    for method_name in methods:
        try:
            test_instance.setup_method()
            method = getattr(test_instance, method_name)
            method()
            print(f"✓ {method_name}")
            passed += 1
        except Exception as e:
            print(f"❌ {method_name}: {e}")
            failed += 1
    
    print(f"\nResults: {passed} passed, {failed} failed")