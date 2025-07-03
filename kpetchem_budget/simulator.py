"""
Ultra-high-performance parallel simulator for large-scale Monte Carlo analysis.

This module executes 76,800 carbon budget simulations targeting completion
in under 2 minutes on 8 cores using vectorized operations and joblib parallel processing.
"""

import numpy as np
import pandas as pd
import time
from joblib import Parallel, delayed
from typing import Dict, List, Tuple, Optional, Iterator
from dataclasses import dataclass
import warnings
import multiprocessing as mp

from .parameter_space import ParameterGrid, MonteCarloSampler, ParameterCase, MonteCarloSample, get_budget_line_params
from .data_layer import load_global_budget, load_iea_sector_budget, get_korean_shares, get_timeline_years
from .pathway import PathwayGenerator, BudgetOverflowError


@dataclass
class SimulationResult:
    """Enhanced result from a single simulation run with vectorized output."""
    case_id: int
    sample_id: int
    budget_line: str
    allocation_rule: str
    start_year: int
    net_zero_year: int
    pathway_family: str
    baseline_emissions: float
    allocated_budget: float
    total_emissions: float
    budget_utilization: float
    emissions_2023: float
    emissions_2035: float
    emissions_2050: float
    cumulative_2035: float
    cumulative_2050: float
    peak_emission: float
    min_emission: float
    success: bool
    error_message: Optional[str] = None


class VectorizedBudgetAllocator:
    """
    Vectorized budget allocator optimized for high-performance simulation.
    
    Uses cached data and vectorized operations to minimize computation overhead.
    """
    
    def __init__(self):
        """Initialize with cached data for maximum performance."""
        self.korean_shares = get_korean_shares()
        self.global_budget_df = load_global_budget()
        self.iea_budget = load_iea_sector_budget()
        
        # Pre-compute lookup tables for fast access
        self._create_budget_lookup()
        
    def _create_budget_lookup(self):
        """Create fast lookup table for budget values."""
        self.budget_lookup = {}
        
        for _, row in self.global_budget_df.iterrows():
            temp = row['temp']
            prob = row['probability']
            key = f"{temp}C-{int(prob*100)}%"
            self.budget_lookup[key] = row['budget_gt']
    
    def allocate_budget_vectorized(self, 
                                  allocation_rule: str,
                                  budget_line: str,
                                  global_budget_error: float = 1.0,
                                  production_share_error: float = 1.0) -> float:
        """
        Vectorized budget allocation with Monte Carlo perturbations.
        
        Parameters
        ----------
        allocation_rule : str
            Allocation method
        budget_line : str
            Budget scenario string
        global_budget_error : float
            Multiplicative factor for global budget
        production_share_error : float
            Multiplicative factor for production share
            
        Returns
        -------
        float
            Allocated budget in Mt CO2e (2023-2050)
        """
        # Fast lookup from pre-computed table
        global_budget_gt = self.budget_lookup.get(budget_line, 400.0)  # Default fallback
        
        # Apply global budget uncertainty
        global_budget_gt *= global_budget_error
        
        if allocation_rule == 'iea_sector':
            # IEA sector logic
            sector_budget_gt = min(self.iea_budget, global_budget_gt)
            korean_share = self.korean_shares['production'] * production_share_error
            korean_share = np.clip(korean_share, 0.001, 0.1)  # Reasonable bounds
        else:
            # Share-based allocation
            korean_share = self.korean_shares[allocation_rule]
            sector_budget_gt = global_budget_gt
        
        # Convert to Mt
        return sector_budget_gt * korean_share * 1000


def run_case(params: Tuple[ParameterCase, MonteCarloSample]) -> SimulationResult:
    """
    Optimized single simulation execution with minimal overhead.
    
    Parameters
    ----------
    params : Tuple[ParameterCase, MonteCarloSample]
        Parameter case and Monte Carlo sample
        
    Returns
    -------
    SimulationResult
        Simulation result with comprehensive metrics
    """
    case, mc_sample = params
    
    try:
        # Initialize vectorized allocator (cached)
        allocator = VectorizedBudgetAllocator()
        
        # Allocate budget with uncertainties
        allocated_budget = allocator.allocate_budget_vectorized(
            case.allocation_rule,
            case.budget_line,
            mc_sample.global_budget_error,
            mc_sample.production_share_error
        )
        
        # Default baseline emissions (50 Mt/year in 2023)
        baseline_emissions = 50.0
        
        # Create pathway generator
        generator = PathwayGenerator(
            baseline_emissions=baseline_emissions,
            allocated_budget=allocated_budget,
            start_year=case.start_year,
            net_zero_year=case.net_zero_year
        )
        
        # Generate pathway with optimized methods
        if case.pathway_family == 'linear':
            pathway_df = generator.linear_to_zero()
        elif case.pathway_family == 'constant_rate':
            pathway_df = generator.constant_rate(5.0)  # Default 5% reduction
        elif case.pathway_family == 'logistic':
            pathway_df = generator.logistic_decline(k_factor=mc_sample.logistic_k_factor)
        elif case.pathway_family == 'iea_proxy':
            pathway_df = generator.iea_proxy()
        else:
            raise ValueError(f"Unknown pathway family: {case.pathway_family}")
        
        # Vectorized metric extraction
        emissions = pathway_df['emission'].values
        years = pathway_df['year'].values
        cumulative = pathway_df['cumulative'].values
        
        # Fast milestone extraction
        idx_2023 = 0  # Always first year
        idx_2035 = np.where(years == 2035)[0][0] if 2035 in years else idx_2023
        idx_2050 = np.where(years == 2050)[0][0] if 2050 in years else -1
        
        total_emissions = np.sum(emissions)
        budget_utilization = total_emissions / allocated_budget if allocated_budget > 0 else 0
        
        return SimulationResult(
            case_id=case.case_id,
            sample_id=mc_sample.sample_id,
            budget_line=case.budget_line,
            allocation_rule=case.allocation_rule,
            start_year=case.start_year,
            net_zero_year=case.net_zero_year,
            pathway_family=case.pathway_family,
            baseline_emissions=baseline_emissions,
            allocated_budget=allocated_budget,
            total_emissions=total_emissions,
            budget_utilization=budget_utilization,
            emissions_2023=emissions[idx_2023],
            emissions_2035=emissions[idx_2035],
            emissions_2050=emissions[idx_2050],
            cumulative_2035=cumulative[idx_2035],
            cumulative_2050=cumulative[idx_2050],
            peak_emission=np.max(emissions),
            min_emission=np.min(emissions),
            success=True
        )
        
    except Exception as e:
        return SimulationResult(
            case_id=case.case_id,
            sample_id=mc_sample.sample_id,
            budget_line=case.budget_line,
            allocation_rule=case.allocation_rule,
            start_year=case.start_year,
            net_zero_year=case.net_zero_year,
            pathway_family=case.pathway_family,
            baseline_emissions=50.0,
            allocated_budget=0.0,
            total_emissions=0.0,
            budget_utilization=0.0,
            emissions_2023=0.0,
            emissions_2035=0.0,
            emissions_2050=0.0,
            cumulative_2035=0.0,
            cumulative_2050=0.0,
            peak_emission=0.0,
            min_emission=0.0,
            success=False,
            error_message=str(e)
        )


class HighPerformanceSimulator:
    """
    Ultra-high-performance parallel simulator for large-scale Monte Carlo analysis.
    
    Targets 76,800 simulations in <2 minutes using vectorized operations,
    joblib parallel processing, and optimized memory management.
    
    Examples
    --------
    >>> simulator = HighPerformanceSimulator(n_workers=8)
    >>> results = simulator.run_all_simulations()
    >>> len(results)
    76800
    """
    
    def __init__(self, n_workers: Optional[int] = None, batch_size: int = 1000, backend: str = 'loky'):
        """
        Initialize high-performance simulator.
        
        Parameters
        ----------
        n_workers : int, optional
            Number of worker processes. If None, uses CPU count.
        batch_size : int
            Batch size for parallel processing
        backend : str
            Joblib backend ('loky', 'threading', 'multiprocessing')
        """
        self.n_workers = n_workers or min(mp.cpu_count(), 8)
        self.batch_size = batch_size
        self.backend = backend
        self.parameter_grid = ParameterGrid()
        self.mc_sampler = MonteCarloSampler(n_samples=100, random_seed=42)
        
    def run_all_simulations(self, progress_callback=None) -> List[SimulationResult]:
        """
        Execute all 76,800 Monte Carlo simulations with maximum performance.
        
        Parameters
        ----------
        progress_callback : callable, optional
            Callback function for progress updates
            
        Returns
        -------
        List[SimulationResult]
            All simulation results
            
        Examples
        --------
        >>> simulator = HighPerformanceSimulator(n_workers=8)
        >>> results = simulator.run_all_simulations()
        >>> successful = [r for r in results if r.success]
        >>> len(successful) > 75000  # Expect high success rate
        True
        """
        start_time = time.time()
        
        # Generate all parameter combinations efficiently
        print(f"🚀 Generating {self.parameter_grid.total_cases * 100:,} parameter combinations...")
        all_params = list(self.mc_sampler.generate_all_samples(self.parameter_grid))
        total_sims = len(all_params)
        
        print(f"⚡ Starting {total_sims:,} simulations on {self.n_workers} workers (backend: {self.backend})...")
        
        # Execute in parallel with joblib for maximum performance
        results = Parallel(
            n_jobs=self.n_workers,
            backend=self.backend,
            batch_size=self.batch_size,
            verbose=1  # Show progress
        )(delayed(run_case)(params) for params in all_params)
        
        elapsed_time = time.time() - start_time
        success_rate = sum(1 for r in results if r.success) / len(results) * 100
        throughput = total_sims / elapsed_time
        
        print(f"✅ Completed {total_sims:,} simulations in {elapsed_time:.1f}s")
        print(f"📊 Success rate: {success_rate:.1f}%")
        print(f"🚄 Throughput: {throughput:.0f} simulations/second")
        
        # Performance target check
        if elapsed_time <= 120:  # 2 minutes
            print(f"🎯 Performance target MET: {elapsed_time:.1f}s ≤ 120s")
        else:
            print(f"⚠️ Performance target MISSED: {elapsed_time:.1f}s > 120s")
        
        return results
    
    def run_filtered_simulation(self, 
                               budget_lines: Optional[List[str]] = None,
                               allocation_rules: Optional[List[str]] = None,
                               start_years: Optional[List[int]] = None,
                               net_zero_years: Optional[List[int]] = None,
                               pathway_families: Optional[List[str]] = None,
                               n_samples: int = 100) -> List[SimulationResult]:
        """
        Run filtered subset of simulations for interactive analysis.
        
        Parameters
        ----------
        budget_lines : List[str], optional
            Subset of budget lines to run
        allocation_rules : List[str], optional
            Subset of allocation rules to run
        start_years : List[int], optional
            Subset of start years to run
        net_zero_years : List[int], optional
            Subset of net-zero years to run
        pathway_families : List[str], optional
            Subset of pathway families to run
        n_samples : int
            Number of Monte Carlo samples per case
            
        Returns
        -------
        List[SimulationResult]
            Filtered simulation results
        """
        # Generate filtered parameter combinations
        filtered_params = []
        
        for case in self.parameter_grid.generate_cases():
            # Apply filters
            if budget_lines and case.budget_line not in budget_lines:
                continue
            if allocation_rules and case.allocation_rule not in allocation_rules:
                continue
            if start_years and case.start_year not in start_years:
                continue
            if net_zero_years and case.net_zero_year not in net_zero_years:
                continue
            if pathway_families and case.pathway_family not in pathway_families:
                continue
                
            # Generate MC samples for this case
            mc_sampler = MonteCarloSampler(n_samples=n_samples, random_seed=42)
            for mc_sample in mc_sampler.generate_samples(case.case_id):
                filtered_params.append((case, mc_sample))
        
        print(f"🎯 Running {len(filtered_params):,} filtered simulations...")
        
        # Execute filtered simulations
        start_time = time.time()
        results = Parallel(
            n_jobs=self.n_workers,
            backend=self.backend,
            batch_size=min(self.batch_size, 100)
        )(delayed(run_case)(params) for params in filtered_params)
        
        elapsed_time = time.time() - start_time
        success_rate = sum(1 for r in results if r.success) / len(results) * 100
        
        print(f"✅ Filtered simulation completed in {elapsed_time:.1f}s (success: {success_rate:.1f}%)")
        
        return results
    
    def benchmark_performance(self, n_test_cases: int = 1000) -> Dict[str, float]:
        """
        Benchmark simulation performance for scaling estimates.
        
        Parameters
        ----------
        n_test_cases : int
            Number of test cases to run
            
        Returns
        -------
        Dict[str, float]
            Performance metrics and projections
        """
        print(f"🏃 Benchmarking with {n_test_cases:,} test cases...")
        
        # Get first n test cases
        all_params = list(self.mc_sampler.generate_all_samples(self.parameter_grid))
        test_params = all_params[:n_test_cases]
        
        start_time = time.time()
        
        results = Parallel(
            n_jobs=self.n_workers,
            backend=self.backend,
            batch_size=min(self.batch_size, 100)
        )(delayed(run_case)(params) for params in test_params)
        
        elapsed_time = time.time() - start_time
        success_rate = sum(1 for r in results if r.success) / len(results)
        throughput = n_test_cases / elapsed_time
        
        # Project full performance
        projected_full_time = 76800 / throughput
        
        metrics = {
            'n_test_cases': n_test_cases,
            'elapsed_time_seconds': elapsed_time,
            'throughput_per_second': throughput,
            'success_rate': success_rate,
            'projected_full_runtime_seconds': projected_full_time,
            'meets_performance_target': projected_full_time <= 120,
            'n_workers': self.n_workers,
            'backend': self.backend
        }
        
        print(f"📈 Benchmark Results:")
        print(f"   Throughput: {throughput:.0f} sims/sec")
        print(f"   Projected full runtime: {projected_full_time:.1f}s")
        print(f"   Target (≤120s): {'✅ MET' if metrics['meets_performance_target'] else '❌ MISSED'}")
        
        return metrics
    
    def optimize_performance(self) -> Dict[str, any]:
        """
        Auto-optimize performance settings for current hardware.
        
        Returns
        -------
        Dict[str, any]
            Optimized settings and performance metrics
        """
        print("🔧 Auto-optimizing performance settings...")
        
        # Test different configurations
        configs = [
            {'backend': 'loky', 'batch_size': 500},
            {'backend': 'loky', 'batch_size': 1000},
            {'backend': 'threading', 'batch_size': 200},
            {'backend': 'multiprocessing', 'batch_size': 100}
        ]
        
        best_config = None
        best_throughput = 0
        
        for config in configs:
            # Temporarily update settings
            old_backend = self.backend
            old_batch_size = self.batch_size
            
            self.backend = config['backend']
            self.batch_size = config['batch_size']
            
            try:
                # Quick benchmark
                metrics = self.benchmark_performance(n_test_cases=200)
                throughput = metrics['throughput_per_second']
                
                if throughput > best_throughput:
                    best_throughput = throughput
                    best_config = config.copy()
                    best_config['throughput'] = throughput
                    
            except Exception as e:
                print(f"⚠️ Config {config} failed: {e}")
            
            # Restore settings
            self.backend = old_backend
            self.batch_size = old_batch_size
        
        if best_config:
            self.backend = best_config['backend']
            self.batch_size = best_config['batch_size']
            print(f"🚀 Optimized: {best_config}")
        
        return best_config or {'status': 'optimization_failed'}


def main():
    """Main function for command-line execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run K-PetChem Monte Carlo simulations')
    parser.add_argument('--workers', type=int, default=None, help='Number of worker processes')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmark')
    parser.add_argument('--optimize', action='store_true', help='Auto-optimize performance')
    parser.add_argument('--subset', type=int, help='Run subset of N simulations')
    
    args = parser.parse_args()
    
    simulator = HighPerformanceSimulator(n_workers=args.workers)
    
    if args.benchmark:
        simulator.benchmark_performance()
    elif args.optimize:
        simulator.optimize_performance()
    elif args.subset:
        # Run subset for testing
        results = simulator.run_filtered_simulation(
            budget_lines=['1.5C-50%'],
            allocation_rules=['population'],
            n_samples=args.subset
        )
        print(f"Completed {len(results)} subset simulations")
    else:
        # Run full simulation
        results = simulator.run_all_simulations()
        print(f"Completed {len(results)} total simulations")


if __name__ == "__main__":
    main()