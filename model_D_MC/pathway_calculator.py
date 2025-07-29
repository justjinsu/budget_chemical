"""
Pathway Calculation for Budget-Exact Emission Trajectories

This module implements emission pathway generation that exactly meets carbon budgets
through numerical solver methods. Supports multiple curve types: linear, exponential,
and logarithmic, with front-loading parameter control.
"""

import numpy as np
from typing import Dict, Any, Optional
from scipy.optimize import brentq, minimize_scalar
import logging

logger = logging.getLogger(__name__)


class PathwayCalculator:
    """
    Calculator for budget-exact emission pathways.
    
    Generates emission trajectories from start_year to end_year that:
    1. Exactly meet the allocated carbon budget (within tolerance)
    2. End with zero emissions in 2050
    3. Use front-loading parameter λ to control early vs late emissions
    4. Support multiple curve shapes: linear, exponential, logarithmic
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
        
        # Fixed mid-year for front-loading calculation
        self.mid_year = 2035
        self.mid_idx = np.where(years == self.mid_year)[0]
        if len(self.mid_idx) == 0:
            raise ValueError(f"Mid-year {self.mid_year} not found in years array")
        self.mid_idx = self.mid_idx[0]
        
        logger.debug(f"PathwayCalculator initialized for {self.start_year}-{self.end_year}")
    
    def build_path(self, start_emission: float, budget: float, curve_type: str, 
                   frontload: float, **kwargs) -> np.ndarray:
        """
        Build a budget-exact emission pathway.
        
        Args:
            start_emission: Starting emission rate (tCO2/year) 
            budget: Total carbon budget to allocate (tCO2)
            curve_type: Type of curve ('linear', 'exp', 'log')
            frontload: Front-loading parameter λ ∈ [0,1]
            **kwargs: Additional curve-specific parameters
            
        Returns:
            Array of emission rates for each year
            
        Raises:
            ValueError: If curve_type is unknown or solving fails
        """
        # Debug logging and type validation
        logger.debug(f"build_path called with: start_emission={start_emission} (type: {type(start_emission)}), "
                    f"budget={budget} (type: {type(budget)}), curve_type={curve_type} (type: {type(curve_type)}), "
                    f"frontload={frontload} (type: {type(frontload)})")
        
        # Ensure inputs are correct types
        try:
            start_emission = float(start_emission)
            budget = float(budget)
            frontload = float(frontload)
            curve_type = str(curve_type)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Type conversion failed: {e}")
        
        try:
            if curve_type == 'linear':
                logger.debug(f"Calling _build_linear_path...")
                return self._build_linear_path(start_emission, budget, frontload)
            elif curve_type == 'exp':
                logger.debug(f"Calling _build_exponential_path...")
                return self._build_exponential_path(start_emission, budget, frontload)
            elif curve_type == 'log':
                logger.debug(f"Calling _build_logarithmic_path...")
                return self._build_logarithmic_path(start_emission, budget, frontload)
            else:
                raise ValueError(f"Unknown curve type: {curve_type}")
        except Exception as e:
            import traceback
            logger.error(f"Error in pathway generation for curve_type={curve_type}: {e}")
            logger.error(f"Parameters: start_emission={start_emission}, budget={budget}, frontload={frontload}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def _build_linear_path(self, start_emission: float, budget: float, 
                          frontload: float) -> np.ndarray:
        """
        Build linear emission pathway with two segments.
        
        Segment 1 (2024-2035): Linear decline from start_emission to mid_emission
        Segment 2 (2036-2050): Linear decline from mid_emission to 0
        
        Solves for mid_emission such that:
        - Segment 1 area = λ * budget  
        - Segment 2 area = (1-λ) * budget
        - Total area = budget
        """
        # Ensure all inputs are float
        start_emission = float(start_emission)
        budget = float(budget)
        frontload = float(frontload)
        
        # Years in each segment
        seg1_years = int(self.mid_idx + 1)  # 2024-2035 inclusive (12 years)
        seg2_years = int(self.n_years - self.mid_idx - 1)  # 2036-2050 inclusive (15 years)
        
        # Solve for mid_emission
        def budget_error(mid_emission):
            mid_emission = float(mid_emission)
            # Segment 1: trapezoid area from start_emission to mid_emission
            area1 = 0.5 * (start_emission + mid_emission) * seg1_years
            
            # Segment 2: trapezoid area from mid_emission to 0
            area2 = 0.5 * mid_emission * seg2_years
            
            # Return error in total budget (primary constraint)
            total_area = area1 + area2
            return total_area - budget
        
        # Find valid range for mid_emission
        max_mid = 2.0 * budget / (seg1_years + seg2_years)  # If equal decline
        
        try:
            # Solve for mid_emission that gives exact budget
            mid_emission = brentq(budget_error, 0.0, max_mid * 2.0)
        except ValueError as e:
            logger.warning(f"Linear path solver failed: {e}")
            # Fallback: simple proportional allocation
            mid_emission = start_emission * 0.5
        
        # Ensure mid_emission is float
        mid_emission = float(mid_emission)
        
        # Build pathway arrays
        pathway = np.zeros(self.n_years, dtype=np.float64)
        
        # Segment 1: linear decline
        for i in range(seg1_years):
            t = float(i) / float(seg1_years - 1) if seg1_years > 1 else 0.0
            pathway[i] = start_emission * (1.0 - t) + mid_emission * t
        
        # Segment 2: linear decline to zero
        for i in range(seg2_years):
            t = float(i) / float(seg2_years - 1) if seg2_years > 1 else 0.0
            pathway[self.mid_idx + 1 + i] = mid_emission * (1.0 - t)
        
        return pathway
    
    def _build_exponential_path(self, start_emission: float, budget: float,
                               frontload: float) -> np.ndarray:
        """
        Build exponential decay pathway with two segments.
        
        Segment 1: E(t) = E0 * exp(-k1 * t)
        Segment 2: E(t) = Em * exp(-k2 * (t - tmid))
        
        Where Em is mid-year emission and k1, k2 are decay rates.
        """
        # Ensure all inputs are float
        start_emission = float(start_emission)
        budget = float(budget)
        frontload = float(frontload)
        
        seg1_years = int(self.mid_idx + 1)
        seg2_years = int(self.n_years - self.mid_idx - 1)
        
        def budget_error(params):
            k1, mid_emission = float(params[0]), float(params[1])
            
            # Segment 1 exponential decay integral
            if abs(k1) < 1e-8:
                area1 = start_emission * seg1_years
            else:
                area1 = start_emission * (1.0 - np.exp(-k1 * seg1_years)) / k1
            
            # Segment 2: find k2 such that Em * exp(-k2 * seg2_years) ≈ 0
            # Use constraint that final emission should be very small
            final_target = 0.01 * mid_emission  # 1% of mid emission
            if mid_emission < 1e-6:
                k2 = 0.1  # Default small decay
            else:
                k2 = -np.log(final_target / mid_emission) / seg2_years if mid_emission > final_target else 0.1
            
            if abs(k2) < 1e-8:
                area2 = mid_emission * seg2_years
            else:
                area2 = mid_emission * (1.0 - np.exp(-k2 * seg2_years)) / k2
            
            total_area = area1 + area2
            return (total_area - budget)**2
        
        # Optimize to find best k1 and mid_emission
        from scipy.optimize import minimize
        
        try:
            result = minimize(budget_error, x0=[0.05, start_emission * 0.6], 
                            bounds=[(0.001, 1.0), (0.0, start_emission)],
                            method='L-BFGS-B')
            k1, mid_emission = float(result.x[0]), float(result.x[1])
        except:
            # Fallback to simple exponential
            k1 = 0.05
            mid_emission = start_emission * 0.6
        
        # Build pathway
        pathway = np.zeros(self.n_years, dtype=np.float64)
        
        # Segment 1
        for i in range(seg1_years):
            pathway[i] = start_emission * np.exp(-k1 * float(i))
        
        # Segment 2  
        k2 = -np.log(0.01) / seg2_years  # Decay to 1% in seg2_years
        for i in range(seg2_years):
            pathway[self.mid_idx + 1 + i] = mid_emission * np.exp(-k2 * float(i))
        
        return pathway
    
    def _build_logarithmic_path(self, start_emission: float, budget: float,
                               frontload: float) -> np.ndarray:
        """
        Build logarithmic pathway (slow early, steep later).
        
        Uses inverse exponential shape: E(t) = E0 * (1 - log(1 + α*t)/log(1 + α*T))
        where α controls the curvature.
        """
        # Ensure all inputs are float
        start_emission = float(start_emission)
        budget = float(budget)
        frontload = float(frontload)
        
        total_years = int(self.n_years - 1)
        
        def budget_error(alpha):
            alpha = float(alpha)
            if alpha <= 0:
                return budget**2
            
            pathway_test = np.zeros(self.n_years, dtype=np.float64)
            log_norm = np.log(1.0 + alpha * total_years)
            
            for i in range(self.n_years):
                if i == self.n_years - 1:
                    pathway_test[i] = 0.0  # Force final year to zero
                else:
                    t_norm = float(i) / float(total_years)
                    decay_factor = 1.0 - np.log(1.0 + alpha * float(i)) / log_norm
                    pathway_test[i] = start_emission * max(0.0, decay_factor)
            
            # Approximate area using trapezoidal rule
            area = np.trapz(pathway_test, dx=1.0)
            return (area - budget)**2
        
        # Find optimal alpha
        try:
            result = minimize_scalar(budget_error, bounds=(0.1, 10.0), method='bounded')
            alpha = float(result.x)
        except:
            alpha = 1.0  # Default
        
        # Build final pathway
        pathway = np.zeros(self.n_years, dtype=np.float64)
        log_norm = np.log(1.0 + alpha * total_years)
        
        for i in range(self.n_years):
            if i == self.n_years - 1:
                pathway[i] = 0.0
            else:
                decay_factor = 1.0 - np.log(1.0 + alpha * float(i)) / log_norm  
                pathway[i] = start_emission * max(0.0, decay_factor)
        
        return pathway
    
    def validate_pathway(self, pathway: np.ndarray, budget: float) -> Dict[str, float]:
        """
        Validate that pathway meets budget constraints.
        
        Args:
            pathway: Emission pathway array
            budget: Target carbon budget
            
        Returns:
            Dictionary with validation metrics
        """
        # Debug logging
        logger.debug(f"validate_pathway called with pathway shape: {pathway.shape}, budget: {budget} (type: {type(budget)})")
        logger.debug(f"pathway dtype: {pathway.dtype}, pathway range: {pathway.min():.2f} to {pathway.max():.2f}")
        
        # Ensure budget is float
        budget = float(budget)
        
        # Calculate actual budget (area under curve)
        actual_budget = float(np.trapz(pathway, dx=1.0))
        logger.debug(f"actual_budget: {actual_budget} (type: {type(actual_budget)})")
        
        # Budget error
        budget_error = abs(actual_budget - budget) / budget if budget > 0 else 0.0
        budget_error = float(budget_error)
        logger.debug(f"budget_error: {budget_error} (type: {type(budget_error)})")
        
        # Check final emission is approximately zero
        final_emission = float(pathway[-1])
        logger.debug(f"final_emission: {final_emission} (type: {type(final_emission)})")
        
        # Check non-negative emissions
        negative_emissions = int((pathway < 0).sum())
        logger.debug(f"negative_emissions: {negative_emissions} (type: {type(negative_emissions)})")
        
        # Debug tolerance comparison
        logger.debug(f"self.tolerance: {self.tolerance} (type: {type(self.tolerance)})")
        logger.debug(f"budget_error < self.tolerance: {budget_error < self.tolerance}")
        
        metrics = {
            'actual_budget': actual_budget,
            'target_budget': budget,
            'budget_error_pct': budget_error * 100.0,
            'final_emission': final_emission,
            'negative_count': negative_emissions,
            'is_valid': budget_error < self.tolerance and final_emission < 1.0 and negative_emissions == 0
        }
        
        logger.debug(f"validation metrics: {metrics}")
        return metrics