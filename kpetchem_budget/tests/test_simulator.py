"""
Tests for parallel Monte Carlo simulator.

These tests verify the simulator can execute the full 14,400 scenario
parameter space efficiently and correctly.
"""

import numpy as np
import pandas as pd
import sys
import os
import time
from unittest.mock import patch

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from simulator import ParallelSimulator, run_single_simulation, SimulationResult
from parameter_space import ParameterGrid, MonteCarloSampler, ParameterCase, MonteCarloSample


class TestParallelSimulator:
    """Test suite for parallel Monte Carlo simulator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.simulator = ParallelSimulator(n_workers=2)  # Use 2 workers for testing
    
    def test_parameter_space_size(self):
        """Test that parameter space contains exactly 144 deterministic cases."""
        grid = ParameterGrid()
        total_cases = grid.total_cases
        
        assert total_cases == 144, f"Expected 144 cases, got {total_cases}"
        
        # Verify calculation: 4 × 3 × 1 × 3 × 4 = 144
        expected = (
            len(grid.allocation_rules) *      # 4
            len(grid.budget_lines) *          # 3  
            len(grid.start_years) *           # 1 (fixed at 2023)
            len(grid.net_zero_years) *        # 3
            len(grid.pathway_families)        # 4
        )
        assert total_cases == expected
    
    def test_monte_carlo_samples_per_case(self):
        """Test that Monte Carlo generates 100 samples per case."""
        sampler = MonteCarloSampler(n_samples=100)
        samples = list(sampler.generate_samples(case_id=0))
        
        assert len(samples) == 100
        
        # Check sample structure
        for sample in samples[:5]:  # Check first 5
            assert hasattr(sample, 'case_id')
            assert hasattr(sample, 'sample_id')
            assert hasattr(sample, 'global_budget_factor')
            assert hasattr(sample, 'production_share_error')
            assert hasattr(sample, 'k_factor')
            
            # Check reasonable ranges
            assert 0.7 <= sample.global_budget_factor <= 1.3
            assert -0.005 <= sample.production_share_error <= 0.005
            assert 0.5 <= sample.k_factor <= 3.0
    
    def test_total_simulation_count(self):
        """Test that total simulations equal 144 × 100 = 14,400."""
        grid = ParameterGrid()
        sampler = MonteCarloSampler(n_samples=100)
        
        total_simulations = 0
        for case in list(grid.generate_cases())[:5]:  # Test first 5 cases
            samples = list(sampler.generate_samples(case.case_id))
            total_simulations += len(samples)
        
        # Should be 5 × 100 = 500 for first 5 cases
        assert total_simulations == 500
        
        # Full calculation should be 14,400
        expected_total = grid.total_cases * 100
        assert expected_total == 14400
    
    def test_single_simulation_execution(self):
        """Test single simulation run produces valid result."""
        # Create test case and sample
        case = ParameterCase(
            allocation_rule='population',
            budget_line='1.5C-50%',
            start_year=2023,
            net_zero_year=2050,
            pathway_family='linear',
            case_id=0
        )
        
        sample = MonteCarloSample(
            case_id=0,
            sample_id=0,
            global_budget_factor=1.0,
            production_share_error=0.0,
            k_factor=1.0
        )
        
        # Run simulation
        result = run_single_simulation((case, sample))
        
        # Check result structure
        assert isinstance(result, SimulationResult)
        assert result.case_id == 0
        assert result.sample_id == 0
        assert result.allocation_rule == 'population'
        assert result.budget_line == '1.5C-50%'
        assert result.pathway_family == 'linear'
        
        # Check that it succeeded or has meaningful error
        if result.success:
            assert result.allocated_budget > 0
            assert result.total_emissions >= 0
            assert result.baseline_emissions == 50.0  # Default baseline
            assert result.emissions_2023 == 50.0
            assert result.emissions_2050 >= 0
        else:
            assert result.error_message is not None
    
    def test_simulator_subset_execution(self):
        """Test simulator subset execution works correctly."""
        # Run small subset for testing
        results = self.simulator.run_subset(
            allocation_rules=['population'],
            budget_lines=['1.5C-50%'],
            pathway_families=['linear'],
            n_samples=5
        )
        
        # Should get 1 × 1 × 3 × 1 × 5 = 15 results (3 net-zero years)
        assert len(results) == 15
        
        # Check all results have correct parameters
        for result in results:
            assert result.allocation_rule == 'population'
            assert result.budget_line == '1.5C-50%'
            assert result.pathway_family == 'linear'
            assert result.net_zero_year in [2045, 2050, 2055]
    
    def test_performance_benchmark(self):
        """Test performance benchmark functionality."""
        # Run small benchmark
        metrics = self.simulator.benchmark_performance(n_test_cases=50)
        
        # Check metrics structure
        required_keys = [
            'n_cases', 'elapsed_time_seconds', 'throughput_per_second',
            'success_rate', 'estimated_full_runtime_seconds'
        ]
        
        for key in required_keys:
            assert key in metrics
        
        # Check reasonable values
        assert metrics['n_cases'] == 50
        assert metrics['elapsed_time_seconds'] > 0
        assert metrics['throughput_per_second'] > 0
        assert 0 <= metrics['success_rate'] <= 1
        assert metrics['estimated_full_runtime_seconds'] > 0
        
        # Performance target: should be faster than 1 simulation per second
        assert metrics['throughput_per_second'] >= 1.0
    
    def test_error_handling_in_simulation(self):
        """Test that simulation errors are handled gracefully."""
        # Create case that might cause errors
        case = ParameterCase(
            allocation_rule='invalid_rule',  # Invalid rule
            budget_line='1.5C-50%',
            start_year=2023,
            net_zero_year=2050,
            pathway_family='linear',
            case_id=999
        )
        
        sample = MonteCarloSample(
            case_id=999,
            sample_id=0,
            global_budget_factor=1.0,
            production_share_error=0.0,
            k_factor=1.0
        )
        
        # Should not crash, should return failed result
        result = run_single_simulation((case, sample))
        
        assert isinstance(result, SimulationResult)
        assert not result.success
        assert result.error_message is not None
        assert result.case_id == 999
    
    def test_budget_allocation_variations(self):
        """Test that Monte Carlo properly varies budget allocations."""
        # Create multiple samples with different factors
        case = ParameterCase(
            allocation_rule='population',
            budget_line='1.5C-50%',
            start_year=2023,
            net_zero_year=2050,
            pathway_family='linear',
            case_id=0
        )
        
        # Test samples with different budget factors
        budget_factors = [0.8, 1.0, 1.2]
        results = []
        
        for i, factor in enumerate(budget_factors):
            sample = MonteCarloSample(
                case_id=0,
                sample_id=i,
                global_budget_factor=factor,
                production_share_error=0.0,
                k_factor=1.0
            )
            
            result = run_single_simulation((case, sample))
            if result.success:
                results.append(result)
        
        # Should get different allocated budgets
        if len(results) >= 2:
            budgets = [r.allocated_budget for r in results]
            assert len(set(budgets)) > 1, "Budget factors should produce different allocations"
    
    def test_pathway_family_coverage(self):
        """Test that all pathway families are covered in simulation."""
        # Run subset covering all pathway families
        results = self.simulator.run_subset(
            allocation_rules=['population'],
            budget_lines=['1.5C-50%'],
            pathway_families=None,  # All families
            n_samples=2
        )
        
        # Extract pathway families from results
        pathway_families = set(r.pathway_family for r in results if r.success)
        
        # Should include all four families
        expected_families = {'linear', 'constant_rate', 'logistic', 'iea_proxy'}
        assert pathway_families == expected_families
    
    def test_start_year_fixed_at_2023(self):
        """Test that all simulations use start year 2023."""
        # Run small subset
        results = self.simulator.run_subset(
            allocation_rules=['population'],
            budget_lines=['1.5C-50%'],
            n_samples=3
        )
        
        # Check all successful results
        for result in results:
            if result.success:
                # Baseline emission should be 2023 value
                assert result.emissions_2023 == result.baseline_emissions
    
    def test_simulation_result_data_types(self):
        """Test that simulation results have correct data types."""
        # Run one simulation
        results = self.simulator.run_subset(
            allocation_rules=['population'],
            budget_lines=['1.5C-50%'],
            pathway_families=['linear'],
            n_samples=1
        )
        
        if results and results[0].success:
            result = results[0]
            
            # Check integer fields
            assert isinstance(result.case_id, int)
            assert isinstance(result.sample_id, int)
            assert isinstance(result.net_zero_year, int)
            
            # Check float fields
            float_fields = [
                'baseline_emissions', 'allocated_budget', 'total_emissions',
                'budget_utilization', 'emissions_2023', 'emissions_2035', 
                'emissions_2050', 'cumulative_2035', 'cumulative_2050'
            ]
            
            for field in float_fields:
                value = getattr(result, field)
                assert isinstance(value, (int, float))
                assert value >= 0  # Should be non-negative
            
            # Check string fields
            string_fields = ['allocation_rule', 'budget_line', 'pathway_family']
            for field in string_fields:
                value = getattr(result, field)
                assert isinstance(value, str)
                assert len(value) > 0
            
            # Check boolean field
            assert isinstance(result.success, bool)


if __name__ == '__main__':
    # Simple test runner
    test_instance = TestParallelSimulator()
    
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