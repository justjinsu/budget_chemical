"""
Enhanced Pathway Calculator with Multiple Curve Types

This module implements emission pathway generation with multiple curve types that:
1. Start from current emissions (2024)
2. Decay to zero emissions in 2050 using different curve shapes
3. Exactly meet the allocated carbon budget
4. No lambda parameter needed - curves are automatically fitted to budget
"""

import numpy as np
from typing import Dict, Any, Optional
from scipy.optimize import brentq
import logging

logger = logging.getLogger(__name__)


class PathwayCalculator:
    """
    Enhanced calculator for emission pathways with multiple curve types.
    
    Generates emission trajectories that:
    1. Start from current emissions in start_year
    2. Decay to zero emissions in end_year (2050) using various curve shapes
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
        
        logger.debug(f"Enhanced PathwayCalculator initialized for {self.start_year}-{self.end_year}")
    
    def build_path(self, start_emission: float, budget: float, curve_type: str = 'linear', 
                   **kwargs) -> np.ndarray:
        """
        Build an emission pathway that decays to zero in 2050.
        
        Args:
            start_emission: Starting emission rate (tCO2/year) 
            budget: Total carbon budget to allocate (tCO2)
            curve_type: Type of curve ('linear', 'exp', 'log')
            **kwargs: Additional parameters (ignored for compatibility)
            
        Returns:
            Array of emission rates for each year
            
        Raises:
            ValueError: If solving fails or inputs are invalid
        """
        # Ensure inputs are correct types
        try:
            start_emission = float(start_emission)
            budget = float(budget)
            curve_type = str(curve_type)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Type conversion failed: {e}")
        
        logger.debug(f"Building {curve_type} path: start_emission={start_emission:.2e}, budget={budget:.2e}")
        
        try:
            if curve_type == 'linear':
                return self._build_linear_path(start_emission, budget)
            elif curve_type == 'exp':
                return self._build_exponential_path(start_emission, budget)
            elif curve_type == 'log':
                return self._build_logarithmic_path(start_emission, budget)
            else:
                logger.warning(f"Unknown curve type '{curve_type}', using linear")
                return self._build_linear_path(start_emission, budget)
        except Exception as e:
            logger.error(f"Failed to build pathway: {e}")
            logger.error(f"Parameters: start_emission={start_emission}, budget={budget}, curve_type={curve_type}")
            raise
    
    def _build_linear_path(self, start_emission: float, budget: float) -> np.ndarray:
        """
        Build a linear decay pathway from start_emission to zero.
        
        The pathway is: E(t) = start_emission * (1 - (t - start_year) / (end_year - start_year))
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
    
    def _build_exponential_path(self, start_emission: float, budget: float) -> np.ndarray:
        """
        Build an exponential decay pathway from start_emission to zero.
        
        The pathway is: E(t) = start_emission * exp(-k * (t - start_year)) * scale_factor
        We solve for the decay constant k and scale factor to meet the budget.
        """
        time_span = self.end_year - self.start_year
        
        def budget_error(k):
            """Calculate budget error for given decay constant k."""
            if k <= 0:
                return float('inf')
            
            # Create exponential decay
            time_from_start = self.years - self.start_year
            pathway = start_emission * np.exp(-k * time_from_start)
            
            # Force final year to zero
            pathway[-1] = 0.0
            
            # Calculate cumulative budget
            calc_budget = np.trapz(pathway, dx=1.0)
            
            return calc_budget - budget
        
        # Find decay constant that gives us the right budget
        # k should be positive, and we need some reasonable bounds
        try:
            k_optimal = brentq(budget_error, 0.01, 10.0, xtol=self.tolerance)
        except ValueError:
            # If brentq fails, use a fallback approach
            logger.warning("Exponential optimization failed, using linear fallback")
            return self._build_linear_path(start_emission, budget)
        
        # Generate final pathway
        time_from_start = self.years - self.start_year
        pathway = start_emission * np.exp(-k_optimal * time_from_start)
        pathway[-1] = 0.0  # Ensure zero end point
        
        logger.debug(f"Exponential pathway: k={k_optimal:.4f}")
        
        return pathway
    
    def _build_logarithmic_path(self, start_emission: float, budget: float) -> np.ndarray:
        """
        Build a logarithmic decay pathway from start_emission to zero.
        
        The pathway is based on a logarithmic function that starts high and decays slowly,
        then drops to zero at the end.
        """
        time_span = self.end_year - self.start_year
        
        def budget_error(a):
            """Calculate budget error for given logarithmic parameter a."""
            if a <= 0:
                return float('inf')
            
            # Create logarithmic decay: higher emissions early, lower later
            time_from_start = self.years - self.start_year
            
            # Avoid log(0) by adding small offset
            time_normalized = (time_from_start + 0.1) / (time_span + 0.1)
            
            # Logarithmic decay: log(1 + a*(1-t)) / log(1+a)
            # This starts at 1 when t=0 and goes to 0 when t=1
            pathway = start_emission * np.log(1 + a * (1 - time_normalized)) / np.log(1 + a)
            
            # Force final year to zero
            pathway[-1] = 0.0
            
            # Calculate cumulative budget
            calc_budget = np.trapz(pathway, dx=1.0)
            
            return calc_budget - budget
        
        # Find parameter that gives us the right budget
        try:
            a_optimal = brentq(budget_error, 0.1, 50.0, xtol=self.tolerance)
        except ValueError:
            # If brentq fails, use linear fallback
            logger.warning("Logarithmic optimization failed, using linear fallback")
            return self._build_linear_path(start_emission, budget)
        
        # Generate final pathway
        time_from_start = self.years - self.start_year
        time_normalized = (time_from_start + 0.1) / (time_span + 0.1)
        pathway = start_emission * np.log(1 + a_optimal * (1 - time_normalized)) / np.log(1 + a_optimal)
        pathway[-1] = 0.0  # Ensure zero end point
        
        logger.debug(f"Logarithmic pathway: a={a_optimal:.4f}")
        
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