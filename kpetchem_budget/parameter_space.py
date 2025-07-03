"""
Parameter space definition for large-scale Monte Carlo carbon budget simulations.

This module defines the expanded deterministic parameter grid (768 cases) and Monte Carlo
sampling for comprehensive uncertainty quantification in Korean petrochemical carbon budgets.
"""

import numpy as np
import pandas as pd
from itertools import product
from typing import Dict, List, Tuple, Iterator
from dataclasses import dataclass
import warnings


@dataclass
class ParameterCase:
    """Single parameter case for simulation."""
    budget_line: str
    allocation_rule: str
    start_year: int
    net_zero_year: int
    pathway_family: str
    case_id: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'budget_line': self.budget_line,
            'allocation_rule': self.allocation_rule,
            'start_year': self.start_year,
            'net_zero_year': self.net_zero_year,
            'pathway_family': self.pathway_family,
            'case_id': self.case_id
        }


@dataclass
class MonteCarloSample:
    """Single Monte Carlo sample with enhanced parameter variations."""
    case_id: int
    sample_id: int
    global_budget_error: float      # Normal distribution around baseline budget
    production_share_error: float   # Log-normal production share multiplier
    logistic_k_factor: float        # Triangular distribution for logistic steepness


class ParameterGrid:
    """
    Expanded deterministic parameter grid for carbon budget scenarios.
    
    Generates 768 deterministic cases from Cartesian product of:
    - 4 global budget lines
    - 4 allocation rules  
    - 4 start years
    - 3 net-zero years
    - 4 pathway families
    
    Examples
    --------
    >>> grid = ParameterGrid()
    >>> len(list(grid.generate_cases()))
    768
    >>> case = next(grid.generate_cases())
    >>> case.start_year in [2023, 2025, 2030, 2035]
    True
    """
    
    def __init__(self):
        """Initialize expanded parameter grid."""
        # Global budget scenarios (4 options)
        self.budget_lines = [
            '1.5C-67%',  # 1.5°C with 67% probability
            '1.5C-50%',  # 1.5°C with 50% probability  
            '1.7C-50%',  # 1.7°C with 50% probability
            '2.0C-67%'   # 2.0°C with 67% probability
        ]
        
        # Allocation criteria (4 options)
        self.allocation_rules = [
            'population',
            'gdp', 
            'national_ghg',
            'iea_sector'
        ]
        
        # Start years (4 options)
        self.start_years = [2023, 2025, 2030, 2035]
        
        # Net-zero target years (3 options)
        self.net_zero_years = [2045, 2050, 2055]
        
        # Pathway families (4 options)
        self.pathway_families = [
            'linear',
            'constant_rate',
            'logistic', 
            'iea_proxy'
        ]
        
        # Calculate total cases: 4 × 4 × 4 × 3 × 4 = 768
        self.total_cases = (
            len(self.budget_lines) * 
            len(self.allocation_rules) *
            len(self.start_years) *
            len(self.net_zero_years) * 
            len(self.pathway_families)
        )
        
        assert self.total_cases == 768, f"Expected 768 cases, got {self.total_cases}"
        
    def generate_cases(self) -> Iterator[ParameterCase]:
        """
        Generate all parameter cases from Cartesian product.
        
        Yields
        ------
        ParameterCase
            Individual parameter case
            
        Examples
        --------
        >>> grid = ParameterGrid()
        >>> cases = list(grid.generate_cases())
        >>> len(cases)
        768
        >>> set(case.start_year for case in cases) == {2023, 2025, 2030, 2035}
        True
        """
        case_id = 0
        
        for budget_line, allocation_rule, start_year, net_zero_year, pathway_family in product(
            self.budget_lines,
            self.allocation_rules,
            self.start_years, 
            self.net_zero_years,
            self.pathway_families
        ):
            yield ParameterCase(
                budget_line=budget_line,
                allocation_rule=allocation_rule,
                start_year=start_year,
                net_zero_year=net_zero_year,
                pathway_family=pathway_family,
                case_id=case_id
            )
            case_id += 1
            
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert parameter grid to DataFrame.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with all parameter combinations
            
        Examples
        --------
        >>> grid = ParameterGrid()
        >>> df = grid.to_dataframe()
        >>> len(df)
        768
        >>> 'budget_line' in df.columns
        True
        """
        cases = [case.to_dict() for case in self.generate_cases()]
        return pd.DataFrame(cases)
    
    def get_case_by_id(self, case_id: int) -> ParameterCase:
        """
        Get specific case by ID.
        
        Parameters
        ----------
        case_id : int
            Case identifier (0-767)
            
        Returns
        -------
        ParameterCase
            Parameter case
            
        Examples
        --------
        >>> grid = ParameterGrid()
        >>> case = grid.get_case_by_id(0)
        >>> case.case_id
        0
        """
        for case in self.generate_cases():
            if case.case_id == case_id:
                return case
        raise ValueError(f"Case ID {case_id} not found (valid range: 0-767)")


class MonteCarloSampler:
    """
    Enhanced Monte Carlo uncertainty sampler for parameter perturbations.
    
    Generates N=100 Monte Carlo samples per deterministic case by perturbing:
    - Global budget (Normal distribution with 10% std dev)
    - Korean production share (Log-normal distribution, σ=0.20)  
    - Logistic k-factor (Triangular distribution: 0.15, 0.25, 0.35)
    
    Examples
    --------
    >>> sampler = MonteCarloSampler(n_samples=100, random_seed=42)
    >>> samples = list(sampler.generate_samples(case_id=0))
    >>> len(samples)
    100
    """
    
    def __init__(self, n_samples: int = 100, random_seed: int = 42):
        """
        Initialize Monte Carlo sampler.
        
        Parameters
        ----------
        n_samples : int
            Number of Monte Carlo samples per case (default 100)
        random_seed : int
            Random seed for reproducibility (default 42)
            
        Examples
        --------
        >>> sampler = MonteCarloSampler(n_samples=50)
        >>> sampler.n_samples
        50
        """
        self.n_samples = n_samples
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)
        
    def generate_samples(self, case_id: int) -> Iterator[MonteCarloSample]:
        """
        Generate Monte Carlo samples for a given case.
        
        Parameters
        ----------
        case_id : int
            Deterministic case identifier
            
        Yields
        ------
        MonteCarloSample
            Individual Monte Carlo sample
            
        Examples
        --------
        >>> sampler = MonteCarloSampler(n_samples=10)
        >>> samples = list(sampler.generate_samples(case_id=5))
        >>> len(samples)
        10
        >>> all(s.case_id == 5 for s in samples)
        True
        """
        for sample_id in range(self.n_samples):
            # Global budget error: Normal distribution (μ=1.0, σ=0.1)
            # Truncated to ensure non-negative budgets
            global_budget_error = self.rng.normal(1.0, 0.1)
            global_budget_error = np.clip(global_budget_error, 0.5, 1.5)  # Reasonable bounds
            
            # Production share error: Log-normal distribution (σ=0.20)
            # Represents multiplicative uncertainty in Korean production share
            production_share_error = self.rng.lognormal(0.0, 0.20)
            production_share_error = np.clip(production_share_error, 0.5, 2.0)  # Reasonable bounds
            
            # Logistic k-factor: Triangular distribution (0.15, 0.25, 0.35)
            # Controls steepness of logistic decline curves
            logistic_k_factor = self.rng.triangular(0.15, 0.25, 0.35)
            
            yield MonteCarloSample(
                case_id=case_id,
                sample_id=sample_id,
                global_budget_error=global_budget_error,
                production_share_error=production_share_error,
                logistic_k_factor=logistic_k_factor
            )
    
    def generate_all_samples(self, parameter_grid: ParameterGrid) -> Iterator[Tuple[ParameterCase, MonteCarloSample]]:
        """
        Generate all Monte Carlo samples for entire parameter grid.
        
        Parameters
        ----------
        parameter_grid : ParameterGrid
            Parameter grid to sample from
            
        Yields
        ------
        Tuple[ParameterCase, MonteCarloSample]
            Parameter case and corresponding Monte Carlo sample
            
        Examples
        --------
        >>> grid = ParameterGrid()
        >>> sampler = MonteCarloSampler(n_samples=10)
        >>> samples = list(sampler.generate_all_samples(grid))
        >>> len(samples)
        7680
        """
        for case in parameter_grid.generate_cases():
            for mc_sample in self.generate_samples(case.case_id):
                yield case, mc_sample
    
    def reset_seed(self, new_seed: int) -> None:
        """
        Reset random seed.
        
        Parameters
        ----------
        new_seed : int
            New random seed
            
        Examples
        --------
        >>> sampler = MonteCarloSampler()
        >>> sampler.reset_seed(123)
        >>> sampler.random_seed
        123
        """
        self.random_seed = new_seed
        self.rng = np.random.RandomState(new_seed)


def get_budget_line_params(budget_line: str) -> Tuple[float, float]:
    """
    Convert budget line string to temperature and probability parameters.
    
    Parameters
    ----------
    budget_line : str
        Budget line identifier ('1.5C-67%', '1.5C-50%', '1.7C-50%', '2.0C-67%')
        
    Returns
    -------
    Tuple[float, float]
        Temperature (°C) and probability values
        
    Examples
    --------
    >>> get_budget_line_params('1.5C-67%')
    (1.5, 0.67)
    >>> get_budget_line_params('1.7C-50%')
    (1.7, 0.50)
    """
    mapping = {
        '1.5C-67%': (1.5, 0.67),
        '1.5C-50%': (1.5, 0.50),
        '1.7C-50%': (1.7, 0.50),
        '2.0C-67%': (2.0, 0.67)
    }
    
    if budget_line not in mapping:
        raise ValueError(f"Unknown budget line: {budget_line}. Valid options: {list(mapping.keys())}")
    
    return mapping[budget_line]


def calculate_total_simulations(parameter_grid: ParameterGrid, n_mc_samples: int = 100) -> int:
    """
    Calculate total number of simulations.
    
    Parameters
    ----------
    parameter_grid : ParameterGrid
        Parameter grid
    n_mc_samples : int
        Number of Monte Carlo samples per case
        
    Returns
    -------
    int
        Total number of simulations
        
    Examples
    --------
    >>> grid = ParameterGrid()
    >>> calculate_total_simulations(grid, n_mc_samples=100)
    76800
    """
    return parameter_grid.total_cases * n_mc_samples


def deterministic_grid() -> pd.DataFrame:
    """
    Generate the full deterministic parameter grid as DataFrame.
    
    Returns
    -------
    pd.DataFrame
        Complete 768-case parameter grid
        
    Examples
    --------
    >>> df = deterministic_grid()
    >>> len(df)
    768
    >>> df.columns.tolist()
    ['budget_line', 'allocation_rule', 'start_year', 'net_zero_year', 'pathway_family', 'case_id']
    """
    grid = ParameterGrid()
    return grid.to_dataframe()


def mc_draws(case_id: int, n_samples: int = 100, random_seed: int = 42) -> pd.DataFrame:
    """
    Generate Monte Carlo draws for a specific case.
    
    Parameters
    ----------
    case_id : int
        Case identifier
    n_samples : int
        Number of Monte Carlo samples
    random_seed : int
        Random seed for reproducibility
        
    Returns
    -------
    pd.DataFrame
        Monte Carlo samples as DataFrame
        
    Examples
    --------
    >>> draws = mc_draws(case_id=0, n_samples=10)
    >>> len(draws)
    10
    >>> 'global_budget_error' in draws.columns
    True
    """
    sampler = MonteCarloSampler(n_samples=n_samples, random_seed=random_seed)
    samples = list(sampler.generate_samples(case_id))
    
    data = []
    for sample in samples:
        data.append({
            'case_id': sample.case_id,
            'sample_id': sample.sample_id,
            'global_budget_error': sample.global_budget_error,
            'production_share_error': sample.production_share_error,
            'logistic_k_factor': sample.logistic_k_factor
        })
    
    return pd.DataFrame(data)


def validate_parameter_space() -> Dict[str, bool]:
    """
    Validate parameter space construction and constraints.
    
    Returns
    -------
    Dict[str, bool]
        Validation results
        
    Examples
    --------
    >>> results = validate_parameter_space()
    >>> results['grid_size_correct']
    True
    """
    grid = ParameterGrid()
    sampler = MonteCarloSampler(n_samples=10, random_seed=42)
    
    results = {
        'grid_size_correct': grid.total_cases == 768,
        'all_combinations_unique': True,
        'mc_samples_in_range': True,
        'total_simulations_correct': calculate_total_simulations(grid) == 76800
    }
    
    # Test unique combinations
    df = grid.to_dataframe()
    combo_cols = ['budget_line', 'allocation_rule', 'start_year', 'net_zero_year', 'pathway_family']
    unique_combos = df[combo_cols].drop_duplicates()
    results['all_combinations_unique'] = len(unique_combos) == 768
    
    # Test MC sample ranges
    test_samples = list(sampler.generate_samples(case_id=0))
    for sample in test_samples[:10]:  # Check first 10
        if not (0.5 <= sample.global_budget_error <= 1.5):
            results['mc_samples_in_range'] = False
            break
        if not (0.5 <= sample.production_share_error <= 2.0):
            results['mc_samples_in_range'] = False
            break
        if not (0.15 <= sample.logistic_k_factor <= 0.35):
            results['mc_samples_in_range'] = False
            break
    
    return results


if __name__ == "__main__":
    # Quick validation
    validation = validate_parameter_space()
    print("Parameter Space Validation:")
    for check, passed in validation.items():
        status = "✓" if passed else "✗"
        print(f"{status} {check}: {passed}")
    
    if all(validation.values()):
        print(f"\n🎉 Parameter space validated: 768 cases × 100 samples = 76,800 total simulations")
    else:
        print(f"\n❌ Parameter space validation failed")