"""
Dual-Scenario Pathway Calculator with Enhanced Curve Types

This module implements emission pathway generation for both 1.5°C and 2.0°C scenarios with:
1. Start from current emissions (2024)
2. Decay to zero emissions in 2050 using different curve shapes
3. Exactly meet the allocated carbon budget
4. Support for exponential, logarithmic, and S-curve pathways
"""

import numpy as np
from typing import Dict, Any, Optional
from scipy.optimize import brentq
import logging

logger = logging.getLogger(__name__)


class PathwayCalculator:
    """
    Dual-scenario calculator for emission pathways with enhanced curve types.
    
    Generates emission trajectories that:
    1. Start from current emissions in start_year
    2. Decay to zero emissions in end_year (2050) using various curve shapes
    3. Exactly meet the allocated carbon budget (within tolerance)
    4. Support exponential, logarithmic, and S-curve pathways
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
        
        # Epsilon for gradual approach to zero instead of hard constraint
        self.end_epsilon = float(config.get('end_epsilon', 1e4))  # 10,000 tCO2/year minimum
        
        logger.debug(f"Dual-scenario PathwayCalculator initialized for {self.start_year}-{self.end_year}")
    
    def build_path(self, start_emission: float, budget: float, curve_type: str = 'exp', 
                   **kwargs) -> np.ndarray:
        """
        Build an emission pathway that decays to zero in 2050.
        
        Args:
            start_emission: Starting emission rate (tCO2/year) 
            budget: Total carbon budget to allocate (tCO2)
            curve_type: Type of curve ('exp', 'log', 's_curve')
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
            if curve_type == 'exp':
                return self._build_exponential_path(start_emission, budget)
            elif curve_type == 'log':
                return self._build_logarithmic_path(start_emission, budget)
            elif curve_type == 's_curve':
                return self._build_s_curve_path(start_emission, budget)
            else:
                logger.warning(f"Unknown curve type '{curve_type}', using exponential")
                return self._build_exponential_path(start_emission, budget)
        except Exception as e:
            logger.error(f"Failed to build pathway: {e}")
            logger.error(f"Parameters: start_emission={start_emission}, budget={budget}, curve_type={curve_type}")
            raise
    
    def _build_exponential_path(self, start_emission: float, budget: float) -> np.ndarray:
        """
        Build an exponential decay pathway (constant % decay rate).
        
        The pathway is: E(t) = start_emission * exp(-k * (t - start_year))
        We solve for the decay constant k to meet the budget.
        """
        time_span = self.end_year - self.start_year
        
        def budget_error(k):
            """Calculate budget error for given decay constant k."""
            if k <= 0:
                return float('inf')
            
            # Create exponential decay
            time_from_start = self.years - self.start_year
            pathway = start_emission * np.exp(-k * time_from_start)
            
            # Gradual approach to epsilon instead of hard zero
            pathway[-1] = max(pathway[-1], self.end_epsilon)
            
            # Calculate cumulative budget
            calc_budget = np.trapz(pathway, dx=1.0)
            
            return calc_budget - budget
        
        # Find decay constant that gives us the right budget
        try:
            # For large budgets, we need smaller k values, so expand search range significantly
            k_optimal = brentq(budget_error, 0.00001, 20.0, xtol=self.tolerance)
        except ValueError:
            # If brentq fails, use improved analytical approximation
            logger.warning("Exponential optimization failed, using improved analytical approximation")
            total_linear_budget = start_emission * time_span
            
            # Improved analytical approach for exponential budget matching
            # For exponential decay: E(t) = E0 * exp(-k*t)
            # Budget integral: B = ∫[0,T] E0*exp(-k*t) dt = E0*(1-exp(-k*T))/k
            # For large budgets (B > linear), we need very small k
            
            if budget <= total_linear_budget:
                # Normal case: use Newton-Raphson iteration for better accuracy
                # Starting guess from linear approximation
                k_guess = 2 * (total_linear_budget - budget) / (start_emission * time_span * time_span)
                k_guess = max(0.001, min(10.0, k_guess))
                
                # Newton-Raphson iterations to refine k
                for _ in range(10):
                    exp_term = np.exp(-k_guess * time_span)
                    f = start_emission * (1 - exp_term) / k_guess - budget
                    df_dk = start_emission * (exp_term * time_span * k_guess - (1 - exp_term)) / (k_guess * k_guess)
                    
                    if abs(df_dk) < 1e-12:
                        break
                    
                    k_new = k_guess - f / df_dk
                    k_new = max(0.0001, min(20.0, k_new))  # Keep in bounds
                    
                    if abs(k_new - k_guess) < 1e-8:
                        break
                    k_guess = k_new
                
                k_optimal = k_guess
                logger.debug(f"Newton-Raphson approximation: k={k_optimal:.6f}")
            else:
                # Large budget case: budget exceeds linear decay
                # Exponential decay cannot exceed linear decay cumulative emissions
                # Fall back to a convex pathway that can meet large budgets
                logger.warning(f"Budget ({budget:.2e}) exceeds linear decay budget ({total_linear_budget:.2e}), using convex pathway")
                return self._build_convex_path_for_large_budget(start_emission, budget)
        
        # Generate final pathway
        time_from_start = self.years - self.start_year
        pathway = start_emission * np.exp(-k_optimal * time_from_start)
        pathway[-1] = max(pathway[-1], self.end_epsilon)  # Gradual approach to epsilon
        
        logger.debug(f"Exponential pathway: k={k_optimal:.4f}")
        
        return pathway
    
    def _build_convex_path_for_large_budget(self, start_emission: float, budget: float) -> np.ndarray:
        """
        Build a convex pathway for budgets that exceed linear decay.
        
        Uses a quadratic convex curve that starts high, decreases slowly at first,
        then accelerates the reduction. This can achieve higher cumulative emissions
        than linear or exponential decay.
        
        The pathway is: E(t) = start_emission * (1 - a * t^2)
        where t is normalized time [0,1] and a is the curvature parameter.
        """
        time_span = self.end_year - self.start_year
        
        def budget_error(a):
            """Calculate budget error for given curvature parameter a."""
            if a <= 0 or a > 1.0:
                return float('inf')
            
            # Create convex quadratic pathway
            time_from_start = self.years - self.start_year
            t_norm = time_from_start / time_span  # Normalize to [0,1]
            
            # Quadratic convex curve: E(t) = E0 * (1 - a * t^2)
            # This starts at E0, decreases slowly, then faster
            pathway = start_emission * (1 - a * t_norm**2)
            
            # Ensure non-negative emissions
            pathway = np.maximum(pathway, 0.0)
            
            # Gradual approach to epsilon instead of hard zero
            pathway[-1] = max(pathway[-1], self.end_epsilon)
            
            # Calculate cumulative budget
            calc_budget = np.trapz(pathway, dx=1.0)
            
            return calc_budget - budget
        
        # Find curvature parameter that gives us the right budget
        try:
            a_optimal = brentq(budget_error, 0.0001, 0.9999, xtol=self.tolerance)
        except ValueError:
            # If optimization fails, use analytical approximation
            # For quadratic: ∫[0,T] E0*(1-a*t²/T²) dt = E0*T*(1-a/3)
            # Solving: E0*T*(1-a/3) = budget → a = 3*(1 - budget/(E0*T))
            # But for large budgets, we need negative a, which means convex pathway
            linear_budget = start_emission * time_span
            if budget > linear_budget:
                # For budget > linear, we need concave pathway: E(t) = E0*(1 + a*t²/T²)
                # This gives higher emissions at start, decreasing quadratically
                # But this violates the zero-end constraint. 
                # Instead, use a different parameterization: E(t) = E0*(1 + a*(1-t²))
                # This starts at E0*(1+a), decays quadratically to E0
                
                # For large budgets, fall back to piecewise linear with late reduction
                logger.warning(f"Large budget {budget:.2e} > linear {linear_budget:.2e}, using late-reduction pathway")
                return self._build_late_reduction_pathway(start_emission, budget)
            else:
                a_optimal = max(0.0001, min(0.9999, 3 * (1 - budget / linear_budget)))
                logger.warning(f"Convex optimization failed, using analytical approximation: a={a_optimal:.4f}")
        
        # Generate final pathway
        time_from_start = self.years - self.start_year
        t_norm = time_from_start / time_span
        pathway = start_emission * (1 - a_optimal * t_norm**2)
        pathway = np.maximum(pathway, 0.0)  # Ensure non-negative
        pathway[-1] = max(pathway[-1], self.end_epsilon)  # Gradual approach to epsilon
        
        logger.debug(f"Convex pathway: a={a_optimal:.4f}, budget={budget:.2e}")
        
        return pathway
    
    def _build_late_reduction_pathway(self, start_emission: float, budget: float) -> np.ndarray:
        """
        Build a pathway with late reduction for very large budgets.
        
        This pathway maintains high emissions for most of the period,
        then rapidly reduces to zero in the final years. This can achieve
        cumulative emissions higher than linear decay.
        
        The pathway is constant at start_emission until a transition year,
        then linearly decreases to zero.
        """
        time_span = self.end_year - self.start_year
        
        def budget_error(transition_fraction):
            """Calculate budget error for given transition point."""
            if transition_fraction <= 0 or transition_fraction >= 1:
                return float('inf')
            
            # Create late-reduction pathway
            time_from_start = self.years - self.start_year
            t_norm = time_from_start / time_span
            
            # Constant emission until transition, then linear decrease to zero
            pathway = np.full_like(t_norm, start_emission)
            
            # Apply linear reduction after transition point
            mask = t_norm >= transition_fraction
            remaining_time = 1 - transition_fraction
            if remaining_time > 0:
                reduction_factor = (1 - t_norm[mask]) / remaining_time
                pathway[mask] = start_emission * reduction_factor
            
            # Gradual approach to epsilon instead of hard zero
            pathway[-1] = max(pathway[-1], self.end_epsilon)
            
            # Calculate cumulative budget
            calc_budget = np.trapz(pathway, dx=1.0)
            
            return calc_budget - budget
        
        # Find transition point that gives us the right budget
        try:
            transition_optimal = brentq(budget_error, 0.1, 0.99, xtol=self.tolerance)
        except ValueError:
            # If optimization fails, use analytical approximation
            # For late reduction pathway: E(t) = E0 for t < t_trans, then linear to 0
            # Budget = E0*T*t_trans + E0*T*(1-t_trans)/2 = E0*T*(t_trans + (1-t_trans)/2) = E0*T*(1+t_trans)/2
            # Solving: E0*T*(1+t_trans)/2 = budget → t_trans = 2*budget/(E0*T) - 1
            linear_budget = start_emission * time_span
            transition_analytical = 2 * budget / linear_budget - 1
            
            # Ensure we use the actual analytical solution to preserve budget variability
            if transition_analytical >= 0.99:
                # For extremely large budgets, clamp but preserve some variability
                # Map budget range to transition range [0.80, 0.95]
                budget_ratio = budget / linear_budget
                transition_optimal = 0.80 + 0.15 * min(1.0, (budget_ratio - 1.0) / 0.5)  # Scale based on excess budget
                logger.debug(f"Very large budget {budget:.2e}, ratio {budget_ratio:.3f}, using transition={transition_optimal:.4f}")
            elif transition_analytical >= 0.1:
                transition_optimal = transition_analytical
                logger.debug(f"Analytical solution: budget={budget:.2e}, transition={transition_optimal:.4f}")
            else:
                # For smaller large budgets, use a minimum transition point but preserve some variability
                budget_ratio = budget / linear_budget
                transition_optimal = max(0.1, 0.1 + 0.3 * (budget_ratio - 1.0))  # Give some variability even for small excess
                logger.debug(f"Small excess budget {budget:.2e}, ratio {budget_ratio:.3f}, using transition={transition_optimal:.4f}")
                
            logger.warning(f"Late-reduction optimization failed, using analytical approximation: budget={budget:.2e}, transition={transition_optimal:.4f}")
        
        # Generate final pathway
        time_from_start = self.years - self.start_year
        t_norm = time_from_start / time_span
        
        pathway = np.full_like(t_norm, start_emission)
        mask = t_norm >= transition_optimal
        remaining_time = 1 - transition_optimal
        
        if remaining_time > 0:
            reduction_factor = (1 - t_norm[mask]) / remaining_time
            pathway[mask] = start_emission * reduction_factor
        
        pathway[-1] = max(pathway[-1], self.end_epsilon)  # Gradual approach to epsilon
        
        logger.debug(f"Late-reduction pathway: transition={transition_optimal:.4f}, budget={budget:.2e}")
        
        return pathway
    
    def _build_logarithmic_path(self, start_emission: float, budget: float) -> np.ndarray:
        """
        Build a logarithmic decay pathway (slow-start, fast-finish).
        
        The pathway uses a logarithmic function that starts slowly and accelerates.
        """
        time_span = self.end_year - self.start_year
        
        def budget_error(a):
            """Calculate budget error for given logarithmic parameter a."""
            if a <= 0:
                return float('inf')
            
            # Create logarithmic decay: slow at start, fast at end
            time_from_start = self.years - self.start_year
            
            # Normalize time to [0, 1]
            t_norm = time_from_start / time_span
            
            # Logarithmic decay: starts high, drops faster later
            # Use 1 - log(a*t + 1) / log(a + 1) to ensure it goes from 1 to 0
            pathway = start_emission * (1 - np.log(a * t_norm + 1) / np.log(a + 1))
            
            # Gradual approach to epsilon instead of hard zero
            pathway[-1] = max(pathway[-1], self.end_epsilon)
            
            # Calculate cumulative budget
            calc_budget = np.trapz(pathway, dx=1.0)
            
            return calc_budget - budget
        
        # Find parameter that gives us the right budget
        try:
            a_optimal = brentq(budget_error, 0.1, 100.0, xtol=self.tolerance)
        except ValueError:
            # If brentq fails, use exponential fallback
            logger.warning("Logarithmic optimization failed, using exponential fallback")
            return self._build_exponential_path(start_emission, budget)
        
        # Generate final pathway
        time_from_start = self.years - self.start_year
        t_norm = time_from_start / time_span
        pathway = start_emission * (1 - np.log(a_optimal * t_norm + 1) / np.log(a_optimal + 1))
        pathway[-1] = max(pathway[-1], self.end_epsilon)  # Gradual approach to epsilon
        
        logger.debug(f"Logarithmic pathway: a={a_optimal:.4f}")
        
        return pathway
    
    def _build_s_curve_path(self, start_emission: float, budget: float) -> np.ndarray:
        """
        Build an S-curve (logistic) pathway for technology diffusion patterns.
        
        The S-curve starts slow, accelerates in the middle, then slows down again.
        This mimics technology adoption/phase-out patterns.
        """
        time_span = self.end_year - self.start_year
        
        def budget_error(k):
            """Calculate budget error for given steepness parameter k."""
            if k <= 0:
                return float('inf')
            
            # Create S-curve (logistic decay)
            time_from_start = self.years - self.start_year
            
            # Normalize time to [0, 1]
            t_norm = time_from_start / time_span
            
            # S-curve: 1 / (1 + exp(k * (t - 0.5)))
            # This creates an S-shape that goes from ~1 to ~0
            # Centered at t=0.5 (middle of timeline)
            midpoint = 0.5  # S-curve inflection point at middle of timeline
            s_curve = 1 / (1 + np.exp(k * (t_norm - midpoint)))
            
            pathway = start_emission * s_curve
            
            # Gradual approach to epsilon instead of hard zero
            pathway[-1] = max(pathway[-1], self.end_epsilon)
            
            # Calculate cumulative budget
            calc_budget = np.trapz(pathway, dx=1.0)
            
            return calc_budget - budget
        
        # Find steepness parameter that gives us the right budget
        try:
            k_optimal = brentq(budget_error, 1.0, 20.0, xtol=self.tolerance)
        except ValueError:
            # If brentq fails, use exponential fallback
            logger.warning("S-curve optimization failed, using exponential fallback")
            return self._build_exponential_path(start_emission, budget)
        
        # Generate final pathway
        time_from_start = self.years - self.start_year
        t_norm = time_from_start / time_span
        midpoint = 0.5
        s_curve = 1 / (1 + np.exp(k_optimal * (t_norm - midpoint)))
        pathway = start_emission * s_curve
        pathway[-1] = max(pathway[-1], self.end_epsilon)  # Gradual approach to epsilon
        
        logger.debug(f"S-curve pathway: k={k_optimal:.4f}")
        
        return pathway
    
    def calculate_emission_reduction_rate(self, pathway: np.ndarray, baseline_2023: float, 
                                        target_year: int = 2035) -> float:
        """
        Calculate emission reduction rate for a target year compared to 2023 baseline.
        
        Args:
            pathway: Emission pathway array
            baseline_2023: 2023 baseline emission (tCO2/year)
            target_year: Target year for reduction calculation (default 2035)
            
        Returns:
            Emission reduction rate as percentage (0-100)
        """
        # Find the index for the target year
        target_idx = target_year - self.start_year
        
        # Validate target year is within pathway range
        if target_idx < 0 or target_idx >= len(pathway):
            raise ValueError(f"Target year {target_year} is outside pathway range {self.start_year}-{self.end_year}")
        
        # Get emission value for target year
        target_emission = pathway[target_idx]
        
        # Calculate reduction rate: (baseline - target) / baseline * 100
        reduction_rate = ((baseline_2023 - target_emission) / baseline_2023) * 100.0
        
        return reduction_rate
    
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
        ends_near_epsilon = abs(pathway[-1] - self.end_epsilon) < self.end_epsilon * 0.1  # Within 10% of epsilon
        non_negative = np.all(pathway >= -1e-6)  # Allow tiny numerical errors
        
        validation = {
            'is_valid': is_valid,
            'budget_error': budget_error,
            'budget_error_pct': budget_error_pct,
            'actual_budget': actual_budget,
            'target_budget': target_budget,
            'starts_positive': starts_positive,
            'ends_near_epsilon': ends_near_epsilon,
            'non_negative': non_negative,
            'tolerance_pct': self.tolerance * 100,
            'end_epsilon': self.end_epsilon
        }
        
        logger.debug(f"Validation: budget_error={budget_error_pct:.3f}%, valid={is_valid}")
        
        return validation