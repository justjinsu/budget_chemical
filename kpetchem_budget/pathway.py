"""
Emission pathway generation for Korean petrochemical sector.

This module generates emission trajectories for 2035-2050 using different
reduction strategies while respecting budget constraints.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Union
from scipy import interpolate


class BudgetOverflowError(Exception):
    """Raised when pathway exceeds allocated budget."""
    pass


class PathwayGenerator:
    """
    Generates emission pathways for Korean petrochemical sector.
    
    Creates annual emission trajectories from 2035-2050 using different
    reduction strategies while ensuring cumulative emissions stay within
    allocated budget.
    
    Parameters
    ----------
    baseline_emissions : float
        Current annual emissions in Mt CO2e/year
    allocated_budget : float
        Total allocated budget for 2035-2050 in Mt CO2e
        
    Examples
    --------
    >>> generator = PathwayGenerator(50.0, 400.0)
    >>> pathway = generator.linear_to_zero()
    >>> 'year' in pathway.columns
    True
    """
    
    def __init__(self, baseline_emissions: float, allocated_budget: float):
        self.baseline_emissions = baseline_emissions
        self.allocated_budget = allocated_budget
        self.start_year = 2035
        self.end_year = 2050
        self.years = np.arange(self.start_year, self.end_year + 1)
        
    def linear_to_zero(self) -> pd.DataFrame:
        """
        Generate linear pathway to zero emissions by 2050.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with columns: year, emission, cumulative, budget_left
            
        Examples
        --------
        >>> generator = PathwayGenerator(50.0, 400.0)
        >>> pathway = generator.linear_to_zero()
        >>> pathway.iloc[-1]['emission']
        0.0
        >>> len(pathway)
        16
        """
        # Linear decline from baseline to zero
        emissions = np.linspace(self.baseline_emissions, 0.0, len(self.years))
        
        # Calculate cumulative emissions using trapezoidal rule
        cumulative = np.cumsum(emissions)
        
        # Check budget constraint
        total_emissions = np.sum(emissions)
        if total_emissions > self.allocated_budget:
            raise BudgetOverflowError(
                f"Linear pathway exceeds budget: {total_emissions:.1f} > {self.allocated_budget:.1f} Mt"
            )
        
        # Calculate remaining budget
        budget_left = self.allocated_budget - cumulative
        
        return pd.DataFrame({
            'year': self.years,
            'emission': emissions,
            'cumulative': cumulative,
            'budget_left': budget_left
        })
    
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
            DataFrame with columns: year, emission, cumulative, budget_left
            
        Examples
        --------
        >>> generator = PathwayGenerator(50.0, 400.0)
        >>> pathway = generator.constant_rate(5.0)
        >>> pathway.iloc[0]['emission']
        50.0
        >>> pathway.iloc[1]['emission'] < 50.0
        True
        """
        # Convert percentage to decimal
        rate = rate_pct / 100.0
        
        # Generate emissions with constant reduction rate
        emissions = np.zeros(len(self.years))
        emissions[0] = self.baseline_emissions
        
        for i in range(1, len(self.years)):
            emissions[i] = emissions[i-1] * (1 - rate)
        
        # Calculate cumulative emissions
        cumulative = np.cumsum(emissions)
        
        # Check budget constraint
        total_emissions = np.sum(emissions)
        if total_emissions > self.allocated_budget:
            raise BudgetOverflowError(
                f"Constant rate pathway exceeds budget: {total_emissions:.1f} > {self.allocated_budget:.1f} Mt"
            )
        
        # Calculate remaining budget
        budget_left = self.allocated_budget - cumulative
        
        return pd.DataFrame({
            'year': self.years,
            'emission': emissions,
            'cumulative': cumulative,
            'budget_left': budget_left
        })
    
    def iea_proxy(self) -> pd.DataFrame:
        """
        Generate pathway following IEA global chemicals trajectory.
        
        Scales the IEA global chemicals pathway to match Korean allocation.
        Uses a representative decline pattern based on IEA Net Zero scenarios.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with columns: year, emission, cumulative, budget_left
            
        Examples
        --------
        >>> generator = PathwayGenerator(50.0, 400.0)
        >>> pathway = generator.iea_proxy()
        >>> pathway.iloc[0]['emission']
        50.0
        >>> pathway.iloc[-1]['emission'] < pathway.iloc[0]['emission']
        True
        """
        # IEA-style pathway: steep initial decline, then gradual
        # Approximate IEA chemical sector trajectory
        years_normalized = (self.years - self.start_year) / (self.end_year - self.start_year)
        
        # IEA proxy: exponential decline with plateau
        # Steeper decline early, then slower
        decline_factor = 0.85  # Retains 85% by 2050
        emissions = self.baseline_emissions * np.exp(-2.0 * years_normalized) * decline_factor
        
        # Ensure smooth trajectory
        emissions = np.maximum(emissions, 0.05 * self.baseline_emissions)  # Minimum 5% of baseline
        
        # Scale to fit within budget if needed
        preliminary_total = np.sum(emissions)
        if preliminary_total > self.allocated_budget:
            # Scale down proportionally
            scale_factor = self.allocated_budget / preliminary_total
            emissions = emissions * scale_factor
        
        # Calculate cumulative emissions
        cumulative = np.cumsum(emissions)
        
        # Calculate remaining budget
        budget_left = self.allocated_budget - cumulative
        
        return pd.DataFrame({
            'year': self.years,
            'emission': emissions,
            'cumulative': cumulative,
            'budget_left': budget_left
        })
    
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
            Interpolation method: 'linear' or 'spline'
            
        Returns
        -------
        pd.DataFrame
            DataFrame with columns: year, emission, cumulative, budget_left
            
        Examples
        --------
        >>> generator = PathwayGenerator(50.0, 400.0)
        >>> waypoints = {2035: 50.0, 2040: 30.0, 2050: 5.0}
        >>> pathway = generator.custom_pathway(waypoints)
        >>> pathway.iloc[0]['emission']
        50.0
        """
        # Ensure start and end years are in waypoints
        if self.start_year not in waypoints:
            waypoints[self.start_year] = self.baseline_emissions
        if self.end_year not in waypoints:
            waypoints[self.end_year] = 0.0
        
        # Extract waypoint data
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
        
        # Ensure non-negative emissions
        emissions = np.maximum(emissions, 0.0)
        
        # Calculate cumulative emissions
        cumulative = np.cumsum(emissions)
        
        # Check budget constraint
        total_emissions = np.sum(emissions)
        if total_emissions > self.allocated_budget:
            raise BudgetOverflowError(
                f"Custom pathway exceeds budget: {total_emissions:.1f} > {self.allocated_budget:.1f} Mt"
            )
        
        # Calculate remaining budget
        budget_left = self.allocated_budget - cumulative
        
        return pd.DataFrame({
            'year': self.years,
            'emission': emissions,
            'cumulative': cumulative,
            'budget_left': budget_left
        })
    
    def validate_pathway(self, pathway: pd.DataFrame, tolerance: float = 1e-6) -> bool:
        """
        Validate that pathway respects budget constraints.
        
        Parameters
        ----------
        pathway : pd.DataFrame
            Pathway DataFrame with emission data
        tolerance : float
            Tolerance for budget constraint checking
            
        Returns
        -------
        bool
            True if pathway is valid
            
        Examples
        --------
        >>> generator = PathwayGenerator(50.0, 400.0)
        >>> pathway = generator.linear_to_zero()
        >>> generator.validate_pathway(pathway)
        True
        """
        try:
            # Check required columns
            required_cols = ['year', 'emission', 'cumulative', 'budget_left']
            if not all(col in pathway.columns for col in required_cols):
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
            Summary statistics
            
        Examples
        --------
        >>> generator = PathwayGenerator(50.0, 400.0)
        >>> pathway = generator.linear_to_zero()
        >>> summary = generator.get_pathway_summary(pathway)
        >>> 'total_emissions' in summary
        True
        """
        return {
            'total_emissions': pathway['emission'].sum(),
            'peak_emission': pathway['emission'].max(),
            'final_emission': pathway['emission'].iloc[-1],
            'peak_to_final_reduction_pct': (
                (pathway['emission'].max() - pathway['emission'].iloc[-1]) / 
                pathway['emission'].max() * 100
            ),
            'budget_utilization_pct': (
                pathway['emission'].sum() / self.allocated_budget * 100
            ),
            'overshoot_year': self._find_overshoot_year(pathway)
        }
    
    def _find_overshoot_year(self, pathway: pd.DataFrame) -> Optional[int]:
        """
        Find year when budget is exceeded (if any).
        
        Parameters
        ----------
        pathway : pd.DataFrame
            Pathway DataFrame
            
        Returns
        -------
        Optional[int]
            Year of overshoot, or None if no overshoot
        """
        overshoot_mask = pathway['budget_left'] < 0
        if overshoot_mask.any():
            return int(pathway.loc[overshoot_mask, 'year'].iloc[0])
        return None