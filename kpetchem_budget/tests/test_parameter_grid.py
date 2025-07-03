"""
Tests for parameter grid and Monte Carlo sampling.

These tests verify the parameter space is correctly structured
with 144 deterministic cases and appropriate Monte Carlo variations.
"""

import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from parameter_space import (
    ParameterGrid, MonteCarloSampler, ParameterCase, MonteCarloSample,
    get_budget_line_params, calculate_total_simulations
)


class TestParameterGrid:
    """Test suite for parameter grid generation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.grid = ParameterGrid()
    
    def test_parameter_grid_size(self):
        """Test that parameter grid generates exactly 768 cases."""
        assert self.grid.total_cases == 768
        
        # Verify manual calculation: 4 × 4 × 4 × 3 × 4 = 768
        expected = (
            len(self.grid.allocation_rules) *    # 4
            len(self.grid.budget_lines) *        # 4 (now includes 1.7C-50%)
            len(self.grid.start_years) *         # 4 (now [2023, 2025, 2030, 2035])
            len(self.grid.net_zero_years) *      # 3
            len(self.grid.pathway_families)      # 4
        )
        assert expected == 768
    
    def test_parameter_grid_axes(self):
        """Test parameter grid axes have correct values."""
        # Check allocation rules
        expected_allocation = ['population', 'gdp', 'national_ghg', 'iea_sector']
        assert self.grid.allocation_rules == expected_allocation
        
        # Check budget lines (now includes 1.7C-50%)
        expected_budget = ['1.5C-67%', '1.5C-50%', '1.7C-50%', '2.0C-67%']
        assert self.grid.budget_lines == expected_budget
        
        # Check start years (now multiple options)
        expected_start_years = [2023, 2025, 2030, 2035]
        assert self.grid.start_years == expected_start_years
        
        # Check net-zero years
        expected_net_zero = [2045, 2050, 2055]
        assert self.grid.net_zero_years == expected_net_zero
        
        # Check pathway families
        expected_pathways = ['linear', 'constant_rate', 'logistic', 'iea_proxy']
        assert self.grid.pathway_families == expected_pathways
    
    def test_parameter_case_generation(self):
        """Test parameter case generation produces valid cases."""
        cases = list(self.grid.generate_cases())
        
        # Check total count
        assert len(cases) == 768
        
        # Check first case structure
        first_case = cases[0]
        assert isinstance(first_case, ParameterCase)
        assert hasattr(first_case, 'allocation_rule')
        assert hasattr(first_case, 'budget_line')
        assert hasattr(first_case, 'start_year')
        assert hasattr(first_case, 'net_zero_year')
        assert hasattr(first_case, 'pathway_family')
        assert hasattr(first_case, 'case_id')
        
        # Check case IDs are sequential
        case_ids = [case.case_id for case in cases]
        assert case_ids == list(range(768))
        
        # Check start years include multiple options
        start_years = set(case.start_year for case in cases)
        assert start_years == {2023, 2025, 2030, 2035}
    
    def test_parameter_grid_to_dataframe(self):
        """Test conversion to DataFrame."""
        df = self.grid.to_dataframe()
        
        assert len(df) == 768
        
        # Check required columns
        expected_cols = [
            'allocation_rule', 'budget_line', 'start_year', 
            'net_zero_year', 'pathway_family', 'case_id'
        ]
        assert all(col in df.columns for col in expected_cols)
        
        # Check unique combinations
        combo_cols = ['allocation_rule', 'budget_line', 'start_year', 'net_zero_year', 'pathway_family']
        unique_combos = df[combo_cols].drop_duplicates()
        assert len(unique_combos) == 768  # All combinations should be unique
    
    def test_get_case_by_id(self):
        """Test retrieval of specific cases by ID."""
        # Test first case
        case_0 = self.grid.get_case_by_id(0)
        assert case_0.case_id == 0
        assert case_0.start_year in [2023, 2025, 2030, 2035]
        
        # Test last case
        case_767 = self.grid.get_case_by_id(767)
        assert case_767.case_id == 767
        assert case_767.start_year in [2023, 2025, 2030, 2035]
        
        # Test invalid ID
        try:
            self.grid.get_case_by_id(9999)
            assert False, "Should raise ValueError"
        except ValueError:
            pass
    
    def test_budget_line_parameter_mapping(self):
        """Test budget line string to parameter conversion."""
        # Test all budget line mappings
        assert get_budget_line_params('1.5C-67%') == (1.5, 0.67)
        assert get_budget_line_params('1.5C-50%') == (1.5, 0.50)
        assert get_budget_line_params('1.7C-50%') == (1.7, 0.50)
        assert get_budget_line_params('2.0C-67%') == (2.0, 0.67)
        
        # Test invalid budget line
        try:
            get_budget_line_params('invalid')
            assert False, "Should raise ValueError"
        except ValueError:
            pass


class TestMonteCarloSampler:
    """Test suite for Monte Carlo sampling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.sampler = MonteCarloSampler(n_samples=100, random_seed=42)
    
    def test_monte_carlo_sample_count(self):
        """Test that Monte Carlo generates correct number of samples."""
        samples = list(self.sampler.generate_samples(case_id=0))
        assert len(samples) == 100
        
        # Test different sample count
        small_sampler = MonteCarloSampler(n_samples=10)
        small_samples = list(small_sampler.generate_samples(case_id=0))
        assert len(small_samples) == 10
    
    def test_monte_carlo_sample_structure(self):
        """Test Monte Carlo sample structure and ranges."""
        samples = list(self.sampler.generate_samples(case_id=0))
        
        for i, sample in enumerate(samples[:10]):  # Test first 10
            assert isinstance(sample, MonteCarloSample)
            assert sample.case_id == 0
            assert sample.sample_id == i
            
            # Check parameter ranges (updated for new structure)
            assert 0.5 <= sample.global_budget_error <= 1.5
            assert 0.5 <= sample.production_share_error <= 2.0
            assert 0.15 <= sample.logistic_k_factor <= 0.35
    
    def test_monte_carlo_reproducibility(self):
        """Test that Monte Carlo sampling is reproducible with fixed seed."""
        # Generate samples twice with same seed
        sampler1 = MonteCarloSampler(n_samples=10, random_seed=42)
        sampler2 = MonteCarloSampler(n_samples=10, random_seed=42)
        
        samples1 = list(sampler1.generate_samples(case_id=0))
        samples2 = list(sampler2.generate_samples(case_id=0))
        
        # Should be identical
        for s1, s2 in zip(samples1, samples2):
            assert s1.global_budget_error == s2.global_budget_error
            assert s1.production_share_error == s2.production_share_error
            assert s1.logistic_k_factor == s2.logistic_k_factor
    
    def test_monte_carlo_variability(self):
        """Test that Monte Carlo samples show proper variability."""
        samples = list(self.sampler.generate_samples(case_id=0))
        
        # Extract parameter arrays
        budget_errors = [s.global_budget_error for s in samples]
        share_errors = [s.production_share_error for s in samples]
        k_factors = [s.logistic_k_factor for s in samples]
        
        # Check that we get variability (not all identical)
        assert len(set(budget_errors)) > 50  # Should have many unique values
        assert len(set(share_errors)) > 50
        assert len(set(k_factors)) > 50
        
        # Check distribution properties
        assert 0.9 <= np.mean(budget_errors) <= 1.1  # Should center around 1.0
        assert 0.8 <= np.mean(share_errors) <= 1.3  # Log-normal around 1.0
        assert 0.20 <= np.mean(k_factors) <= 0.30  # Triangular center around 0.25
    
    def test_generate_all_samples(self):
        """Test generation of all samples for parameter grid."""
        grid = ParameterGrid()
        sampler = MonteCarloSampler(n_samples=5, random_seed=42)
        
        all_samples = list(sampler.generate_all_samples(grid))
        
        # Should get 768 × 5 = 3840 samples
        assert len(all_samples) == 3840
        
        # Each item should be (ParameterCase, MonteCarloSample) tuple
        for case, mc_sample in all_samples[:10]:  # Check first 10
            assert isinstance(case, ParameterCase)
            assert isinstance(mc_sample, MonteCarloSample)
            assert case.case_id == mc_sample.case_id
    
    def test_seed_reset(self):
        """Test random seed reset functionality."""
        # Generate samples with initial seed
        samples1 = list(self.sampler.generate_samples(case_id=0))[:5]
        
        # Reset to same seed and generate again
        self.sampler.reset_seed(42)
        samples2 = list(self.sampler.generate_samples(case_id=0))[:5]
        
        # Should be identical
        for s1, s2 in zip(samples1, samples2):
            assert s1.global_budget_error == s2.global_budget_error
            
        # Reset to different seed
        self.sampler.reset_seed(123)
        samples3 = list(self.sampler.generate_samples(case_id=0))[:5]
        
        # Should be different
        different = any(
            s1.global_budget_error != s3.global_budget_error
            for s1, s3 in zip(samples1, samples3)
        )
        assert different


class TestParameterSpaceCalculations:
    """Test suite for parameter space calculations."""
    
    def test_total_simulation_calculation(self):
        """Test calculation of total simulation count."""
        grid = ParameterGrid()
        
        # Test with default 100 samples
        total = calculate_total_simulations(grid, n_mc_samples=100)
        assert total == 76800  # 768 × 100
        
        # Test with different sample count
        total_50 = calculate_total_simulations(grid, n_mc_samples=50)
        assert total_50 == 38400  # 768 × 50
    
    def test_parameter_coverage(self):
        """Test that parameter space covers all required combinations."""
        grid = ParameterGrid()
        df = grid.to_dataframe()
        
        # Check each parameter dimension
        unique_allocation = df['allocation_rule'].unique()
        unique_budget = df['budget_line'].unique()
        unique_start = df['start_year'].unique()
        unique_net_zero = df['net_zero_year'].unique()
        unique_pathway = df['pathway_family'].unique()
        
        # Should match expected values
        assert len(unique_allocation) == 4
        assert len(unique_budget) == 4  # Now includes 1.7C-50%
        assert len(unique_start) == 4  # Now multiple start years
        assert len(unique_net_zero) == 3
        assert len(unique_pathway) == 4
        
        # Check specific values
        assert set(unique_start) == {2023, 2025, 2030, 2035}
        assert set(unique_net_zero) == {2045, 2050, 2055}
        assert 'population' in unique_allocation
        assert 'iea_sector' in unique_allocation
        assert '1.5C-50%' in unique_budget
        assert '1.7C-50%' in unique_budget
        assert 'linear' in unique_pathway
        assert 'logistic' in unique_pathway
    
    def test_start_year_flexibility(self):
        """Test that start year includes multiple options."""
        grid = ParameterGrid()
        cases = list(grid.generate_cases())
        
        # Cases should have multiple start_year options
        start_years = [case.start_year for case in cases]
        unique_start_years = set(start_years)
        assert unique_start_years == {2023, 2025, 2030, 2035}
        assert len(unique_start_years) == 4  # Four unique values
    
    def test_parameter_case_uniqueness(self):
        """Test that all parameter cases are unique."""
        grid = ParameterGrid()
        cases = list(grid.generate_cases())
        
        # Create tuples of all parameters except case_id
        param_tuples = [
            (case.allocation_rule, case.budget_line, case.start_year, 
             case.net_zero_year, case.pathway_family)
            for case in cases
        ]
        
        # All should be unique
        assert len(param_tuples) == len(set(param_tuples))


class TestParameterSpaceValidation:
    """Test suite for parameter space validation."""
    
    def test_validate_parameter_space(self):
        """Test the overall parameter space validation function."""
        from parameter_space import validate_parameter_space
        
        results = validate_parameter_space()
        
        # Check all validation criteria
        assert results['grid_size_correct'] == True
        assert results['all_combinations_unique'] == True
        assert results['mc_samples_in_range'] == True
        assert results['total_simulations_correct'] == True
        
        # Verify total simulations is 76,800
        assert 76800 in str(results), "Should validate 76,800 total simulations"
    
    def test_enhanced_monte_carlo_parameters(self):
        """Test enhanced Monte Carlo parameter distributions."""
        from parameter_space import MonteCarloSampler
        
        sampler = MonteCarloSampler(n_samples=1000, random_seed=42)
        samples = list(sampler.generate_samples(case_id=0))
        
        # Extract all parameters
        global_budget_errors = [s.global_budget_error for s in samples]
        production_share_errors = [s.production_share_error for s in samples]
        logistic_k_factors = [s.logistic_k_factor for s in samples]
        
        # Test global budget error distribution (Normal around 1.0)
        mean_budget = np.mean(global_budget_errors)
        assert 0.95 <= mean_budget <= 1.05, f"Budget error mean should be ~1.0, got {mean_budget}"
        
        # Test production share error distribution (Log-normal around 1.0)
        mean_share = np.mean(production_share_errors)
        assert 0.9 <= mean_share <= 1.2, f"Share error mean should be ~1.0, got {mean_share}"
        
        # Test logistic k-factor distribution (Triangular 0.15, 0.25, 0.35)
        mean_k = np.mean(logistic_k_factors)
        assert 0.22 <= mean_k <= 0.28, f"K-factor mean should be ~0.25, got {mean_k}"
        
        # Test ranges are respected
        assert all(0.5 <= x <= 1.5 for x in global_budget_errors)
        assert all(0.5 <= x <= 2.0 for x in production_share_errors)
        assert all(0.15 <= x <= 0.35 for x in logistic_k_factors)
    
    def test_full_system_integration(self):
        """Test full system integration with realistic parameters."""
        from parameter_space import ParameterGrid, MonteCarloSampler, calculate_total_simulations
        
        # Test with realistic small sample
        grid = ParameterGrid()
        sampler = MonteCarloSampler(n_samples=10, random_seed=42)
        
        # Generate small subset for performance
        sample_count = 0
        for case in grid.generate_cases():
            if sample_count >= 100:  # Test only first 100 cases
                break
            
            mc_samples = list(sampler.generate_samples(case.case_id))
            assert len(mc_samples) == 10
            
            # Verify case has all required attributes
            assert hasattr(case, 'budget_line')
            assert hasattr(case, 'allocation_rule')
            assert hasattr(case, 'start_year')
            assert hasattr(case, 'net_zero_year')
            assert hasattr(case, 'pathway_family')
            assert hasattr(case, 'case_id')
            
            # Verify MC sample structure
            for mc_sample in mc_samples:
                assert hasattr(mc_sample, 'global_budget_error')
                assert hasattr(mc_sample, 'production_share_error')
                assert hasattr(mc_sample, 'logistic_k_factor')
                assert hasattr(mc_sample, 'case_id')
                assert hasattr(mc_sample, 'sample_id')
            
            sample_count += 1
        
        # Verify we tested substantial number of cases
        assert sample_count == 100, f"Expected to test 100 cases, tested {sample_count}"


if __name__ == '__main__':
    # Simple test runner
    test_classes = [TestParameterGrid, TestMonteCarloSampler, TestParameterSpaceCalculations, TestParameterSpaceValidation]
    
    total_passed = 0
    total_failed = 0
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}")
        print("=" * len(test_class.__name__))
        
        test_instance = test_class()
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
        
        total_passed += passed
        total_failed += failed
        print(f"\n{passed} passed, {failed} failed")
    
    print(f"\n\nOverall Results: {total_passed} passed, {total_failed} failed")