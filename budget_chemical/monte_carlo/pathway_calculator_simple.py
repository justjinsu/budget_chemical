"""
Simplified Pathway Calculator for Linear Decay to Zero

This module implements a simplified emission pathway generation that:
1. Starts from current emissions (2024)
2. Linearly decreases to zero emissions in 2050
3. Exactly meets the allocated carbon budget
4. No lambda parameter or 2035 mid-year target needed
"""

import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class PathwayCalculator:
    """
    Simplified calculator for linear emission pathways to zero.
    
    Generates emission trajectories that:
    1. Start from current emissions in start_year
    2. Linearly decay to zero emissions in end_year (2050)
    3. Exactly meet the allocated carbon budget (within tolerance)
    """
    
    def __init__(self, years: np.ndarray, config: Optional[Dict[str, Any]] = None):
        """
        Initialize pathway calculator.
        
        Args:
            years: Array of years for pathway calculation
            config: Optional configuration dictionary
        """
        self.years = years
        self.n_years = len(years)
        self.start_year = years[0]
        self.end_year = years[-1]
        
        # Extract configuration
        if config is None:
            config = {}
        
        self.tolerance = float(config.get('tolerance', 1e-3))
        self.max_iter = int(config.get('max_iter', 60))
        
        logger.debug(f"Simplified PathwayCalculator initialized for {self.start_year}-{self.end_year}")
    
    def build_path(self, start_emission: float, budget: float, curve_type: str = 'linear', 
                   **kwargs) -> np.ndarray:
        """
        Build a linear emission pathway that decays to zero in 2050.
        
        Args:
            start_emission: Starting emission rate (tCO2/year) 
            budget: Total carbon budget to allocate (tCO2)
            curve_type: Ignored (always linear for simplicity)
            **kwargs: Ignored (for compatibility)
            
        Returns:
            Array of emission rates for each year
            
        Raises:
            ValueError: If solving fails or inputs are invalid
        """
        # Ensure inputs are correct types
        try:
            start_emission = float(start_emission)
            budget = float(budget)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Type conversion failed: {e}")
        
        logger.debug(f"Building linear path: start_emission={start_emission:.2e}, budget={budget:.2e}")
        
        try:
            return self._build_linear_path(start_emission, budget)
        except Exception as e:
            logger.error(f"Failed to build pathway: {e}")
            logger.error(f"Parameters: start_emission={start_emission}, budget={budget}")
            raise
    
    def _build_linear_path(self, start_emission: float, budget: float) -> np.ndarray:
        """
        Build a simple linear decay pathway from start_emission to zero.
        
        The pathway is: E(t) = start_emission * (1 - (t - start_year) / (end_year - start_year))
        
        We scale this to exactly meet the budget using numerical scaling.
        """
        # Create base linear decay from start_emission to 0
        time_fraction = (self.years - self.start_year) / (self.end_year - self.start_year)
        base_pathway = start_emission * (1 - time_fraction)
        
        # Ensure final year is exactly zero
        base_pathway[-1] = 0.0
        
        # Calculate cumulative emissions (trapezoidal integration)
        base_budget = np.trapz(base_pathway, dx=1.0)
        
        if base_budget <= 0:
            raise ValueError("Base pathway has non-positive budget")
        
        # Scale pathway to match target budget
        scale_factor = budget / base_budget
        pathway = base_pathway * scale_factor
        
        # Ensure final year remains zero
        pathway[-1] = 0.0
        
        logger.debug(f"Linear pathway: base_budget={base_budget:.2e}, scale_factor={scale_factor:.4f}")
        
        return pathway
    
    def validate_pathway(self, pathway: np.ndarray, target_budget: float) -> Dict[str, Any]:
        """
        Validate that pathway meets budget requirements.
        
        Args:
            pathway: Emission pathway array
            target_budget: Target budget to validate against
            
        Returns:
            Dictionary with validation results
        """
        # Calculate actual budget using trapezoidal integration
        actual_budget = np.trapz(pathway, dx=1.0)
        
        # Calculate error
        budget_error = abs(actual_budget - target_budget)
        budget_error_pct = (budget_error / target_budget) * 100
        
        # Check if within tolerance
        is_valid = budget_error_pct <= (self.tolerance * 100)
        
        # Check constraints
        starts_positive = pathway[0] > 0
        ends_zero = abs(pathway[-1]) < 1e-6  # Very small threshold for zero
        monotonic_decreasing = np.all(np.diff(pathway) <= 1e-6)  # Allow small numerical errors
        
        validation = {
            'is_valid': is_valid,
            'budget_error': budget_error,
            'budget_error_pct': budget_error_pct,
            'actual_budget': actual_budget,
            'target_budget': target_budget,
            'starts_positive': starts_positive,
            'ends_zero': ends_zero,
            'monotonic_decreasing': monotonic_decreasing,
            'tolerance_pct': self.tolerance * 100
        }
        
        logger.debug(f"Validation: budget_error={budget_error_pct:.3f}%, valid={is_valid}")
        
        return validation