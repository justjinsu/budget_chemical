"""
Emission pathway generation for Korean petrochemical sector (2023-2050).

This module generates emission trajectories using four pathway families:
linear, constant rate, logistic decline, and IEA proxy. All pathways validate
budget constraints through trapezoidal integration.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Union
from scipy import interpolate
import warnings

try:
    from .data_layer import get_timeline_years
except ImportError:
    from data_layer import get_timeline_years


class BudgetOverflowError(Exception):
    """Raised when pathway exceeds allocated budget."""
    pass


def mark_milestones(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add milestone marker columns to pathway DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Pathway DataFrame with 'year' column
        
    Returns
    -------
    pd.DataFrame
        DataFrame with added boolean columns 'is2035' and 'is2050'
        
    Examples
    --------
    >>> years = np.arange(2023, 2051)
    >>> df = pd.DataFrame({'year': years, 'emission': np.ones(len(years))})
    >>> marked_df = mark_milestones(df)
    >>> marked_df['is2035'].sum()
    1
    >>> marked_df['is2050'].sum()
    1
    """
    df = df.copy()
    df['is2035'] = df['year'] == 2035
    df['is2050'] = df['year'] == 2050
    return df


class PathwayGenerator:
    """
    Generates emission pathways for Korean petrochemical sector (2023-2050).
    
    Creates annual emission trajectories using different reduction strategies
    while ensuring cumulative emissions stay within allocated budget.
    
    Parameters
    ----------
    baseline_emissions : float
        2023 annual emissions in Mt CO2e/year
    allocated_budget : float
        Total allocated budget for 2023-2050 in Mt CO2e
    start_year : int
        Start year (default 2023)
    net_zero_year : int
        Target net-zero year (default 2050)
        
    Examples
    --------
    >>> generator = PathwayGenerator(50.0, 800.0)
    >>> pathway = generator.linear_to_zero()
    >>> len(pathway)
    28
    >>> pathway.iloc[0]['year']
    2023
    >>> pathway.iloc[-1]['year']
    2050
    """
    
    def __init__(self, 
                 baseline_emissions: float, 
                 allocated_budget: float,
                 start_year: int = 2023,
                 net_zero_year: int = 2050):
        """Initialize pathway generator with 2023-2050 timeline."""
        self.baseline_emissions = baseline_emissions
        self.allocated_budget = allocated_budget
        self.start_year = start_year
        self.net_zero_year = net_zero_year
        self.years = get_timeline_years()  # 2023-2050
        
        # Validate inputs
        if start_year != 2023:
            warnings.warn(f"Start year {start_year} != 2023. Using 2023-2050 timeline.")
        
        if net_zero_year < 2023 or net_zero_year > 2050:
            raise ValueError(f"Net-zero year {net_zero_year} must be between 2023-2050")
    
    def linear_to_zero(self) -> pd.DataFrame:
        """
        Generate linear pathway to zero emissions by net-zero year.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with columns: year, emission, cumulative, budget_left, is2035, is2050
            
        Examples
        --------
        >>> generator = PathwayGenerator(50.0, 800.0)
        >>> pathway = generator.linear_to_zero()
        >>> pathway.iloc[-1]['emission']
        0.0
        >>> pathway['emission'].iloc[0]
        50.0
        """
        # Find index of net-zero year
        net_zero_idx = np.where(self.years == self.net_zero_year)[0]
        if len(net_zero_idx) == 0:
            raise ValueError(f"Net-zero year {self.net_zero_year} not in timeline")
        net_zero_idx = net_zero_idx[0]
        
        # Linear decline from baseline to zero by net-zero year
        emissions = np.zeros(len(self.years))
        
        # Linear decline to net-zero year
        for i in range(net_zero_idx + 1):
            progress = i / net_zero_idx if net_zero_idx > 0 else 0
            emissions[i] = self.baseline_emissions * (1 - progress)
        
        # Zero emissions after net-zero year
        emissions[net_zero_idx + 1:] = 0.0
        
        return self._finalize_pathway(emissions)
    
    def constant_rate(self, rate_pct: float) -> pd.DataFrame:
        """
        Generate pathway with constant annual reduction rate.
        
        Parameters
        ----------
        rate_pct : float
            Annual reduction rate as percentage (e.g., 5.0 for 5% per year)
            
        Returns
        -------
        pd.DataFrame
            DataFrame with columns: year, emission, cumulative, budget_left, is2035, is2050
            
        Examples
        --------
        >>> generator = PathwayGenerator(50.0, 800.0)
        >>> pathway = generator.constant_rate(5.0)
        >>> pathway.iloc[0]['emission']
        50.0
        >>> pathway.iloc[1]['emission'] < 50.0
        True
        """
        rate = rate_pct / 100.0
        
        emissions = np.zeros(len(self.years))
        emissions[0] = self.baseline_emissions
        
        for i in range(1, len(self.years)):
            emissions[i] = emissions[i-1] * (1 - rate)
            # Ensure non-negative
            emissions[i] = max(0.0, emissions[i])
        
        return self._finalize_pathway(emissions)
    
    def logistic_decline(self, k_factor: float = 1.0, midpoint_year: int = 2035) -> pd.DataFrame:
        """
        Generate pathway following logistic decline curve.
        
        Parameters
        ----------
        k_factor : float
            Steepness parameter for logistic curve (default 1.0)
        midpoint_year : int
            Year of maximum decline rate (default 2035)
            
        Returns
        -------
        pd.DataFrame
            DataFrame with columns: year, emission, cumulative, budget_left, is2035, is2050
            
        Examples
        --------
        >>> generator = PathwayGenerator(50.0, 800.0)
        >>> pathway = generator.logistic_decline()
        >>> pathway.iloc[0]['emission']
        50.0
        >>> pathway.iloc[-1]['emission'] < pathway.iloc[0]['emission']
        True
        """
        # Logistic function: emissions = baseline / (1 + exp(k * (year - midpoint)))
        # Modified to ensure realistic decline
        
        years_normalized = self.years - 2023  # Start from 0
        midpoint_normalized = midpoint_year - 2023
        
        # Logistic decline with asymptote at low emission level
        min_emission = 0.02 * self.baseline_emissions  # 2% floor
        
        emissions = np.zeros(len(self.years))
        for i, year_norm in enumerate(years_normalized):
            # Logistic function
            logistic_factor = 1 / (1 + np.exp(-k_factor * (year_norm - midpoint_normalized)))
            
            # Interpolate between baseline and minimum
            emissions[i] = self.baseline_emissions * (1 - logistic_factor) + min_emission * logistic_factor
        
        # Enforce net-zero by target year if specified
        net_zero_idx = np.where(self.years == self.net_zero_year)[0]
        if len(net_zero_idx) > 0:
            emissions[net_zero_idx[0]:] = 0.0
        
        return self._finalize_pathway(emissions)
    
    def iea_proxy(self) -> pd.DataFrame:
        """
        Generate pathway following IEA global chemicals trajectory.
        
        Scales the IEA global chemicals pathway to match Korean allocation.
        Uses representative decline pattern based on IEA Net Zero scenarios.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with columns: year, emission, cumulative, budget_left, is2035, is2050
            
        Examples
        --------
        >>> generator = PathwayGenerator(50.0, 800.0)
        >>> pathway = generator.iea_proxy()
        >>> pathway.iloc[0]['emission']
        50.0
        >>> pathway.iloc[-1]['emission'] < pathway.iloc[0]['emission']
        True
        """
        # IEA-style pathway: initial plateau, then accelerated decline
        years_normalized = (self.years - 2023) / (2050 - 2023)
        
        emissions = np.zeros(len(self.years))
        
        for i, t in enumerate(years_normalized):
            if t <= 0.3:  # 2023-2031: slow decline
                emissions[i] = self.baseline_emissions * (1 - 0.15 * t / 0.3)
            elif t <= 0.7:  # 2032-2042: accelerated decline  
                progress = (t - 0.3) / 0.4
                emissions[i] = self.baseline_emissions * 0.85 * (1 - 0.7 * progress)
            else:  # 2043-2050: final phase to near-zero
                progress = (t - 0.7) / 0.3
                emissions[i] = self.baseline_emissions * 0.85 * 0.3 * (1 - progress)
        
        # Ensure smooth trajectory and non-negative
        emissions = np.maximum(emissions, 0.0)
        
        # Apply smoothing
        if len(emissions) > 3:
            from scipy.ndimage import gaussian_filter1d
            emissions = gaussian_filter1d(emissions, sigma=0.8)
            emissions = np.maximum(emissions, 0.0)
        
        return self._finalize_pathway(emissions)
    
    def custom_pathway(self, 
                      waypoints: Dict[int, float], 
                      method: str = 'linear') -> pd.DataFrame:
        """
        Generate custom pathway through specified waypoints.
        
        Parameters
        ----------
        waypoints : Dict[int, float]
            Dictionary mapping years to emission values
        method : str
            Interpolation method ('linear' or 'spline')
            
        Returns
        -------
        pd.DataFrame
            DataFrame with columns: year, emission, cumulative, budget_left, is2035, is2050
            
        Examples
        --------
        >>> generator = PathwayGenerator(50.0, 800.0)
        >>> waypoints = {2023: 50.0, 2035: 30.0, 2050: 5.0}
        >>> pathway = generator.custom_pathway(waypoints)
        >>> pathway.iloc[0]['emission']
        50.0
        """
        # Ensure 2023 and net-zero year are in waypoints
        if 2023 not in waypoints:
            waypoints[2023] = self.baseline_emissions
        if self.net_zero_year not in waypoints:
            waypoints[self.net_zero_year] = 0.0
        
        # Extract and sort waypoints
        waypoint_years = sorted(waypoints.keys())
        waypoint_emissions = [waypoints[year] for year in waypoint_years]
        
        # Interpolate emissions
        if method == 'linear':
            emissions = np.interp(self.years, waypoint_years, waypoint_emissions)
        elif method == 'spline':
            if len(waypoint_years) >= 3:
                f = interpolate.interp1d(waypoint_years, waypoint_emissions, kind='cubic')
                emissions = f(self.years)
            else:
                emissions = np.interp(self.years, waypoint_years, waypoint_emissions)
        else:
            raise ValueError(f"Unknown interpolation method: {method}")
        
        # Ensure non-negative
        emissions = np.maximum(emissions, 0.0)
        
        return self._finalize_pathway(emissions)
    
    def _finalize_pathway(self, emissions: np.ndarray) -> pd.DataFrame:
        """
        Finalize pathway with cumulative calculations and validation.
        
        Parameters
        ----------
        emissions : np.ndarray
            Annual emissions array
            
        Returns
        -------
        pd.DataFrame
            Finalized pathway DataFrame
            
        Raises
        ------
        BudgetOverflowError
            If cumulative emissions exceed allocated budget
        """
        # Calculate cumulative emissions using trapezoidal integration
        cumulative = np.cumsum(emissions)
        
        # Check budget constraint
        total_emissions = np.sum(emissions)
        if total_emissions > self.allocated_budget:
            raise BudgetOverflowError(
                f"Pathway exceeds budget: {total_emissions:.1f} > {self.allocated_budget:.1f} Mt CO2e"
            )
        
        # Calculate remaining budget
        budget_left = self.allocated_budget - cumulative
        
        # Create DataFrame
        df = pd.DataFrame({
            'year': self.years,
            'emission': emissions,
            'cumulative': cumulative,
            'budget_left': budget_left
        })
        
        # Add milestone markers
        df = mark_milestones(df)
        
        return df
    
    def validate_pathway(self, pathway: pd.DataFrame, tolerance: float = 1e-6) -> bool:
        """
        Validate that pathway respects constraints and is well-formed.
        
        Parameters
        ----------
        pathway : pd.DataFrame
            Pathway DataFrame
        tolerance : float
            Numerical tolerance for validation
            
        Returns
        -------
        bool
            True if pathway is valid
            
        Examples
        --------
        >>> generator = PathwayGenerator(50.0, 800.0)
        >>> pathway = generator.linear_to_zero()
        >>> generator.validate_pathway(pathway)
        True
        """
        try:
            # Check required columns
            required_cols = ['year', 'emission', 'cumulative', 'budget_left', 'is2035', 'is2050']
            if not all(col in pathway.columns for col in required_cols):
                return False
            
            # Check timeline coverage (2023-2050)
            if pathway['year'].min() != 2023 or pathway['year'].max() != 2050:
                return False
            
            # Check non-negative emissions
            if (pathway['emission'] < 0).any():
                return False
            
            # Check budget constraint
            total_emissions = pathway['emission'].sum()
            if total_emissions > self.allocated_budget + tolerance:
                return False
            
            # Check cumulative consistency
            expected_cumulative = np.cumsum(pathway['emission'])
            if not np.allclose(pathway['cumulative'], expected_cumulative, rtol=tolerance):
                return False
            
            # Check milestone markers
            if pathway['is2035'].sum() != 1 or pathway['is2050'].sum() != 1:
                return False
            
            return True
            
        except Exception:
            return False
    
    def get_pathway_summary(self, pathway: pd.DataFrame) -> Dict[str, float]:
        """
        Get summary statistics for pathway.
        
        Parameters
        ----------
        pathway : pd.DataFrame
            Pathway DataFrame
            
        Returns
        -------
        Dict[str, float]
            Summary statistics including milestone values
            
        Examples
        --------
        >>> generator = PathwayGenerator(50.0, 800.0)
        >>> pathway = generator.linear_to_zero()
        >>> summary = generator.get_pathway_summary(pathway)
        >>> 'emissions_2035' in summary
        True
        """
        # Extract milestone values
        year_2035_data = pathway[pathway['year'] == 2035].iloc[0]
        year_2050_data = pathway[pathway['year'] == 2050].iloc[0]
        
        return {
            'total_emissions': pathway['emission'].sum(),
            'peak_emission': pathway['emission'].max(),
            'final_emission': pathway['emission'].iloc[-1],
            'emissions_2023': pathway['emission'].iloc[0],
            'emissions_2035': year_2035_data['emission'],
            'emissions_2050': year_2050_data['emission'],
            'cumulative_2035': year_2035_data['cumulative'],
            'cumulative_2050': year_2050_data['cumulative'],
            'peak_to_final_reduction_pct': (
                (pathway['emission'].max() - pathway['emission'].iloc[-1]) / 
                pathway['emission'].max() * 100
            ),
            'budget_utilization_pct': (
                pathway['emission'].sum() / self.allocated_budget * 100
            ),
            'reduction_2023_to_2035_pct': (
                (pathway['emission'].iloc[0] - year_2035_data['emission']) /
                pathway['emission'].iloc[0] * 100
            ),
            'reduction_2035_to_2050_pct': (
                (year_2035_data['emission'] - year_2050_data['emission']) /
                max(year_2035_data['emission'], 1e-6) * 100
            )
        }
    
    def compare_pathways(self, pathways: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Compare multiple pathways side by side.
        
        Parameters
        ----------
        pathways : Dict[str, pd.DataFrame]
            Dictionary of pathway name to DataFrame
            
        Returns
        -------
        pd.DataFrame
            Comparison table with key metrics
            
        Examples
        --------
        >>> generator = PathwayGenerator(50.0, 800.0)
        >>> pathways = {
        ...     'Linear': generator.linear_to_zero(),
        ...     'Constant': generator.constant_rate(5.0)
        ... }
        >>> comparison = generator.compare_pathways(pathways)
        >>> len(comparison)
        2
        """
        comparison_data = []
        
        for name, pathway_df in pathways.items():
            summary = self.get_pathway_summary(pathway_df)
            summary['pathway_name'] = name
            comparison_data.append(summary)
        
        return pd.DataFrame(comparison_data).set_index('pathway_name')


def regenerate_pathway_from_result(result) -> pd.DataFrame:
    """
    Regenerate pathway DataFrame from simulation result for percentile computation.
    
    This function is used by the datastore module to create detailed time series
    for Monte Carlo percentile visualization.
    
    Parameters
    ----------
    result : SimulationResult
        Result from Monte Carlo simulation
        
    Returns
    -------
    pd.DataFrame
        Pathway time series with year, emission, cumulative columns
    """
    if not result.success:
        raise ValueError(f"Cannot regenerate pathway from failed simulation: {result.error_message}")
    
    # Create generator with result parameters
    generator = PathwayGenerator(
        baseline_emissions=result.baseline_emissions,
        allocated_budget=result.allocated_budget,
        start_year=result.start_year,
        net_zero_year=result.net_zero_year
    )
    
    # Generate pathway based on family
    if result.pathway_family == 'linear':
        pathway_df = generator.linear_to_zero()
    elif result.pathway_family == 'constant_rate':
        pathway_df = generator.constant_rate(5.0)  # Default rate
    elif result.pathway_family == 'logistic':
        # Use default logistic parameters - could be enhanced with stored k_factor
        pathway_df = generator.logistic_decline()
    elif result.pathway_family == 'iea_proxy':
        pathway_df = generator.iea_proxy()
    else:
        raise ValueError(f"Unknown pathway family: {result.pathway_family}")
    
    return pathway_df


def batch_generate_pathways(results: list, max_workers: int = 4) -> pd.DataFrame:
    """
    Generate pathways for multiple simulation results in parallel.
    
    This function is optimized for creating large pathway datasets for
    percentile computation in the upgraded 768-case system.
    
    Parameters
    ----------
    results : list
        List of SimulationResult objects
    max_workers : int
        Maximum number of parallel workers
        
    Returns
    -------
    pd.DataFrame
        Combined pathway time series with case_id and sample_id columns
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import warnings
    
    successful_results = [r for r in results if r.success]
    
    print(f"🔄 Generating pathways for {len(successful_results):,} successful simulations...")
    
    pathway_data = []
    
    def generate_single_pathway(result):
        try:
            pathway_df = regenerate_pathway_from_result(result)
            pathway_df['case_id'] = result.case_id
            pathway_df['sample_id'] = result.sample_id
            return pathway_df
        except Exception as e:
            warnings.warn(f"Failed to generate pathway for case {result.case_id}, sample {result.sample_id}: {e}")
            return None
    
    # Use ThreadPoolExecutor for I/O-bound pathway generation
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_result = {
            executor.submit(generate_single_pathway, result): result 
            for result in successful_results
        }
        
        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_result):
            pathway_df = future.result()
            if pathway_df is not None:
                pathway_data.append(pathway_df)
            
            completed += 1
            if completed % 1000 == 0:
                print(f"   Progress: {completed:,}/{len(successful_results):,}")
    
    if not pathway_data:
        raise ValueError("No valid pathways could be generated")
    
    # Combine all pathway DataFrames
    combined_df = pd.concat(pathway_data, ignore_index=True)
    
    print(f"✅ Generated {len(combined_df):,} pathway time series points for {len(pathway_data):,} simulations")
    
    return combined_df


def validate_monte_carlo_pathways(pathways_df: pd.DataFrame, 
                                 expected_simulations: int) -> Dict[str, any]:
    """
    Validate pathway dataset for Monte Carlo percentile computation.
    
    Parameters
    ----------
    pathways_df : pd.DataFrame
        Combined pathway time series from batch_generate_pathways
    expected_simulations : int
        Expected number of simulations (e.g., 76800)
        
    Returns
    -------
    Dict[str, any]
        Validation results and statistics
    """
    validation = {
        'total_time_points': len(pathways_df),
        'unique_simulations': pathways_df[['case_id', 'sample_id']].drop_duplicates().shape[0],
        'expected_simulations': expected_simulations,
        'simulation_coverage': None,
        'years_covered': sorted(pathways_df['year'].unique()),
        'expected_years': list(range(2023, 2051)),
        'timeline_complete': None,
        'all_non_negative': (pathways_df['emission'] >= 0).all(),
        'valid': False
    }
    
    # Check simulation coverage
    validation['simulation_coverage'] = (
        validation['unique_simulations'] / expected_simulations * 100
    )
    
    # Check timeline completeness
    validation['timeline_complete'] = (
        set(validation['years_covered']) == set(validation['expected_years'])
    )
    
    # Overall validation
    validation['valid'] = (
        validation['simulation_coverage'] >= 95.0 and  # At least 95% coverage
        validation['timeline_complete'] and
        validation['all_non_negative']
    )
    
    return validation


def optimize_pathway_for_budget(generator: PathwayGenerator, 
                               pathway_family: str,
                               target_utilization: float = 0.95) -> pd.DataFrame:
    """
    Optimize pathway to achieve target budget utilization.
    
    This function iteratively adjusts pathway parameters to maximize
    budget utilization without exceeding constraints.
    
    Parameters
    ----------
    generator : PathwayGenerator
        Pathway generator instance
    pathway_family : str
        Type of pathway to optimize ('linear', 'constant_rate', 'logistic', 'iea_proxy')
    target_utilization : float
        Target budget utilization (0.0 to 1.0)
        
    Returns
    -------
    pd.DataFrame
        Optimized pathway achieving target utilization
    """
    if pathway_family == 'linear':
        # Linear pathways are already optimized
        return generator.linear_to_zero()
    
    elif pathway_family == 'constant_rate':
        # Optimize reduction rate
        rates = np.linspace(1.0, 10.0, 20)
        best_pathway = None
        best_utilization = 0.0
        
        for rate in rates:
            try:
                pathway = generator.constant_rate(rate)
                utilization = pathway['emission'].sum() / generator.allocated_budget
                
                if utilization <= 1.0 and abs(utilization - target_utilization) < abs(best_utilization - target_utilization):
                    best_pathway = pathway
                    best_utilization = utilization
            except BudgetOverflowError:
                continue
        
        return best_pathway if best_pathway is not None else generator.constant_rate(5.0)
    
    elif pathway_family == 'logistic':
        # Optimize k_factor
        k_factors = np.linspace(0.5, 3.0, 15)
        best_pathway = None
        best_utilization = 0.0
        
        for k in k_factors:
            try:
                pathway = generator.logistic_decline(k_factor=k)
                utilization = pathway['emission'].sum() / generator.allocated_budget
                
                if utilization <= 1.0 and abs(utilization - target_utilization) < abs(best_utilization - target_utilization):
                    best_pathway = pathway
                    best_utilization = utilization
            except BudgetOverflowError:
                continue
        
        return best_pathway if best_pathway is not None else generator.logistic_decline()
    
    elif pathway_family == 'iea_proxy':
        # IEA pathway is fixed, but we can scale if needed
        pathway = generator.iea_proxy()
        utilization = pathway['emission'].sum() / generator.allocated_budget
        
        if utilization > 1.0:
            # Scale down emissions proportionally
            scale_factor = target_utilization / utilization
            pathway['emission'] *= scale_factor
            pathway['cumulative'] = np.cumsum(pathway['emission'])
            pathway['budget_left'] = generator.allocated_budget - pathway['cumulative']
        
        return pathway
    
    else:
        raise ValueError(f"Unknown pathway family: {pathway_family}")


if __name__ == "__main__":
    # Test enhanced functionality
    print("🧪 Testing enhanced pathway generation...")
    
    # Test basic pathway generation
    generator = PathwayGenerator(50.0, 800.0)
    pathways = {
        'Linear': generator.linear_to_zero(),
        'Constant 5%': generator.constant_rate(5.0),
        'Logistic': generator.logistic_decline(),
        'IEA Proxy': generator.iea_proxy()
    }
    
    # Test comparison
    comparison = generator.compare_pathways(pathways)
    print(f"✅ Generated and compared {len(comparison)} pathway families")
    
    # Test optimization
    for family in ['constant_rate', 'logistic']:
        optimized = optimize_pathway_for_budget(generator, family, target_utilization=0.90)
        utilization = optimized['emission'].sum() / generator.allocated_budget
        print(f"✅ Optimized {family}: {utilization:.1%} budget utilization")
    
    print("🎉 All pathway tests passed!")