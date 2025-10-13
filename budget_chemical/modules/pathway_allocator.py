"""
Module 2: Annual Pathway Allocator

This module allocates a given carbon budget across years using various curve methods.
It operates independently from budget calculation (Module 1).

Key Features:
- Multiple curve types: exponential, logarithmic, S-curve, linear, plateau
- Feasibility constraints: maximum reduction rates, realistic transitions
- True net-zero pathways (no epsilon tricks)
- Budget matching with high precision
- Validation and quality metrics

Class-based design for clean separation between backend logic and frontend usage.
"""

import numpy as np
from scipy.optimize import brentq, minimize_scalar, OptimizeResult
from typing import Dict, Any, Optional, Tuple, Callable, List
import logging
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class PathwayAllocator:
    """
    Allocate carbon budget across years using various emission reduction curves.

    This class handles the annual allocation of a total carbon budget,
    ensuring the pathway matches the budget while respecting feasibility constraints.
    """

    def __init__(
        self,
        start_year: int = 2024,
        end_year: int = 2050,
        start_emission: float = 185e6,  # tCO2/year
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the pathway allocator.

        Args:
            start_year: First year of the pathway
            end_year: Last year (target net-zero year)
            start_emission: Initial emission rate (tCO2/year)
            config: Optional configuration dictionary with constraints
        """
        self.start_year = start_year
        self.end_year = end_year
        self.start_emission = start_emission
        self.n_years = end_year - start_year + 1
        self.years = np.arange(start_year, end_year + 1)

        # Load configuration
        self.config = config or {}
        self.tolerance = self.config.get('tolerance', 0.001)  # 0.1% budget match
        self.max_iter = self.config.get('max_iter', 100)
        self.end_epsilon = self.config.get('end_epsilon', 1.0)  # TRUE net-zero (1 tCO2)

        # Feasibility constraints
        self.max_annual_reduction = self.config.get('max_annual_reduction_rate', 0.10)  # 10%/year
        self.min_final_emission = self.config.get('min_final_emission', 1.0)  # tCO2/year
        self.allow_residual = self.config.get('allow_residual', False)

        # Available curve types
        self.curve_types = {
            'exponential': self._exponential_curve,
            'logarithmic': self._logarithmic_curve,
            's_curve': self._s_curve,
            'linear': self._linear_curve,
            'plateau': self._plateau_curve,
            'convex': self._convex_curve,
            'early_action': self._early_action_curve,
            'delayed_action': self._delayed_action_curve
        }

        logger.info(f"PathwayAllocator initialized: {start_year}-{end_year}, "
                   f"start={start_emission:.2e} tCO2/year")
        logger.info(f"Constraints: max_reduction={self.max_annual_reduction:.1%}/year, "
                   f"end_epsilon={self.end_epsilon:.0f} tCO2/year")

    def allocate_budget(
        self,
        budget: float,
        curve_type: str = 'exponential',
        validate: bool = True,
        tier: str = 'single',
        industry_fraction: float = 0.37,
        petrochem_fraction: float = 0.10,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Allocate budget to generate annual emission pathway.

        Args:
            budget: Total carbon budget (tCO2)
            curve_type: Type of reduction curve
            validate: Whether to validate feasibility
            tier: 'single', 'two_tier' (national + industry), or 'three_tier' (national + industry + petrochem)
            industry_fraction: Fraction of national budget for industry (default: 0.37)
            petrochem_fraction: Fraction of industry budget for petrochemical (default: 0.10)
            **kwargs: Additional parameters for specific curves

        Returns:
            Dictionary containing:
                - pathway: Annual emissions (tCO2/year) - for single tier, this is the main pathway
                - pathway_national: National-level pathway (if tier='two_tier' or 'three_tier')
                - pathway_industry: Industry-level pathway (if tier='two_tier' or 'three_tier')
                - pathway_petrochem: Petrochemical-level pathway (if tier='three_tier')
                - years: Year array
                - budget_allocated: Total budget
                - budget_national: National budget (if tier='two_tier' or 'three_tier')
                - budget_industry: Industry budget (if tier='two_tier' or 'three_tier')
                - budget_petrochem: Petrochemical budget (if tier='three_tier')
                - cumulative: Cumulative emissions
                - validation: Validation results
                - parameters: Curve parameters used
                - milestones: Key year emissions (2035, 2040, 2045, 2050)
        """
        if curve_type not in self.curve_types:
            raise ValueError(f"Unknown curve type '{curve_type}'. "
                           f"Available: {list(self.curve_types.keys())}")

        logger.info(f"Allocating budget {budget:.2e} tCO2 using {curve_type} curve (tier={tier})")

        if tier == 'three_tier':
            # Generate national-level pathway
            curve_func = self.curve_types[curve_type]
            pathway_national, params = self._optimize_pathway(budget, curve_func, **kwargs)

            # Calculate industry budget and pathway
            budget_industry = budget * industry_fraction
            pathway_industry, _ = self._optimize_pathway(budget_industry, curve_func, **kwargs)

            # Calculate petrochemical budget and pathway
            budget_petrochem = budget_industry * petrochem_fraction
            pathway_petrochem, _ = self._optimize_pathway(budget_petrochem, curve_func, **kwargs)

            # Calculate cumulative emissions
            cumulative_national = np.trapz(pathway_national, dx=1)
            cumulative_industry = np.trapz(pathway_industry, dx=1)
            cumulative_petrochem = np.trapz(pathway_petrochem, dx=1)

            # Validate if requested
            validation = {}
            if validate:
                validation_national = self._validate_pathway(pathway_national, budget)
                validation_industry = self._validate_pathway(pathway_industry, budget_industry)
                validation_petrochem = self._validate_pathway(pathway_petrochem, budget_petrochem)
                validation = {
                    'national': validation_national,
                    'industry': validation_industry,
                    'petrochem': validation_petrochem
                }

            # Calculate milestones
            milestones = self._calculate_milestones_three_tier(
                pathway_national, pathway_industry, pathway_petrochem
            )

            result = {
                'pathway': pathway_national,  # For backward compatibility
                'pathway_national': pathway_national,
                'pathway_industry': pathway_industry,
                'pathway_petrochem': pathway_petrochem,
                'years': self.years,
                'budget_allocated': budget,
                'budget_national': budget,
                'budget_industry': budget_industry,
                'budget_petrochem': budget_petrochem,
                'cumulative_emissions': cumulative_national,
                'cumulative_industry': cumulative_industry,
                'cumulative_petrochem': cumulative_petrochem,
                'curve_type': curve_type,
                'parameters': params,
                'validation': validation,
                'milestones': milestones,
                'tier': 'three_tier',
                'industry_fraction': industry_fraction,
                'petrochem_fraction': petrochem_fraction,
                'metadata': {
                    'start_year': self.start_year,
                    'end_year': self.end_year,
                    'start_emission': self.start_emission,
                    'n_years': self.n_years
                }
            }

            if validate:
                if not validation['national']['is_valid']:
                    logger.warning(f"National pathway validation failed: {validation['national']['issues']}")
                if not validation['industry']['is_valid']:
                    logger.warning(f"Industry pathway validation failed: {validation['industry']['issues']}")
                if not validation['petrochem']['is_valid']:
                    logger.warning(f"Petrochemical pathway validation failed: {validation['petrochem']['issues']}")

            logger.info(f"Three-tier pathway generated: National={cumulative_national:.2e} tCO2, "
                       f"Industry={cumulative_industry:.2e} tCO2, Petrochem={cumulative_petrochem:.2e} tCO2")

        elif tier == 'two_tier':
            # Generate national-level pathway
            curve_func = self.curve_types[curve_type]
            pathway_national, params = self._optimize_pathway(budget, curve_func, **kwargs)

            # Calculate industry budget and pathway
            budget_industry = budget * industry_fraction
            pathway_industry, _ = self._optimize_pathway(budget_industry, curve_func, **kwargs)

            # Calculate cumulative emissions
            cumulative_national = np.trapz(pathway_national, dx=1)
            cumulative_industry = np.trapz(pathway_industry, dx=1)

            # Validate if requested
            validation = {}
            if validate:
                validation_national = self._validate_pathway(pathway_national, budget)
                validation_industry = self._validate_pathway(pathway_industry, budget_industry)
                validation = {
                    'national': validation_national,
                    'industry': validation_industry
                }

            # Calculate milestones
            milestones = self._calculate_milestones(pathway_national, pathway_industry)

            result = {
                'pathway': pathway_national,  # For backward compatibility
                'pathway_national': pathway_national,
                'pathway_industry': pathway_industry,
                'years': self.years,
                'budget_allocated': budget,
                'budget_national': budget,
                'budget_industry': budget_industry,
                'cumulative_emissions': cumulative_national,
                'cumulative_industry': cumulative_industry,
                'curve_type': curve_type,
                'parameters': params,
                'validation': validation,
                'milestones': milestones,
                'tier': 'two_tier',
                'industry_fraction': industry_fraction,
                'metadata': {
                    'start_year': self.start_year,
                    'end_year': self.end_year,
                    'start_emission': self.start_emission,
                    'n_years': self.n_years
                }
            }

            if validate:
                if not validation['national']['is_valid']:
                    logger.warning(f"National pathway validation failed: {validation['national']['issues']}")
                if not validation['industry']['is_valid']:
                    logger.warning(f"Industry pathway validation failed: {validation['industry']['issues']}")

            logger.info(f"Two-tier pathway generated: National={cumulative_national:.2e} tCO2, "
                       f"Industry={cumulative_industry:.2e} tCO2")

        else:  # single tier
            # Generate pathway using specified curve
            curve_func = self.curve_types[curve_type]
            pathway, params = self._optimize_pathway(budget, curve_func, **kwargs)

            # Calculate cumulative emissions
            cumulative = np.trapz(pathway, dx=1)

            # Validate if requested
            validation = {}
            if validate:
                validation = self._validate_pathway(pathway, budget)

            # Calculate milestones (single pathway)
            milestones = self._calculate_milestones(pathway, None)

            result = {
                'pathway': pathway,
                'years': self.years,
                'budget_allocated': budget,
                'cumulative_emissions': cumulative,
                'curve_type': curve_type,
                'parameters': params,
                'validation': validation,
                'milestones': milestones,
                'tier': 'single',
                'metadata': {
                    'start_year': self.start_year,
                    'end_year': self.end_year,
                    'start_emission': self.start_emission,
                    'n_years': self.n_years
                }
            }

            if validate and not validation['is_valid']:
                logger.warning(f"Pathway validation failed: {validation['issues']}")

            logger.info(f"Pathway generated: cumulative={cumulative:.2e} tCO2, "
                       f"budget_match={(cumulative/budget - 1)*100:.2f}%")

        return result

    def _optimize_pathway(
        self,
        budget: float,
        curve_func: Callable,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Optimize curve parameters to match budget.

        Args:
            budget: Target carbon budget
            curve_func: Curve generation function
            **kwargs: Additional curve parameters

        Returns:
            Tuple of (pathway, parameters)
        """
        # Try to find parameter that matches budget
        def budget_error(param):
            try:
                pathway = curve_func(param, **kwargs)
                cumulative = np.trapz(pathway, dx=1)
                return cumulative - budget
            except:
                return 1e15  # Large error for invalid parameters

        # Search for optimal parameter
        try:
            # Try Brentq method first (faster for single parameter)
            param_opt = brentq(
                budget_error,
                1e-6,  # Lower bound
                100.0,  # Upper bound
                xtol=1e-8,
                maxiter=self.max_iter
            )
            success = True
        except (ValueError, RuntimeError) as e:
            logger.warning(f"Brentq optimization failed: {e}, trying minimize_scalar")

            # Fallback to minimize_scalar
            result = minimize_scalar(
                lambda p: abs(budget_error(p)),
                bounds=(1e-6, 100.0),
                method='bounded',
                options={'maxiter': self.max_iter}
            )
            param_opt = result.x
            success = result.success

        if not success:
            logger.error("Optimization failed, using fallback linear pathway")
            return self._linear_curve(1.0), {'fallback': True}

        # Generate optimal pathway
        pathway = curve_func(param_opt, **kwargs)

        # Newton-Raphson refinement for better accuracy
        for _ in range(5):
            error = budget_error(param_opt)
            if abs(error / budget) < self.tolerance:
                break

            # Numerical derivative
            h = param_opt * 1e-6
            deriv = (budget_error(param_opt + h) - error) / h

            if abs(deriv) > 1e-10:
                param_opt -= error / deriv
                pathway = curve_func(param_opt, **kwargs)

        params = {'optimized_parameter': float(param_opt)}

        return pathway, params

    # ==================== Curve Functions ====================

    def _exponential_curve(self, decay_rate: float, **kwargs) -> np.ndarray:
        """
        Exponential decay curve: E(t) = E0 * exp(-k*t) + epsilon

        Args:
            decay_rate: Decay parameter k

        Returns:
            Emission pathway array
        """
        t = np.linspace(0, 1, self.n_years)  # Normalized time [0, 1]
        pathway = self.start_emission * np.exp(-decay_rate * t) + self.end_epsilon
        pathway[-1] = self.end_epsilon  # Ensure final value
        return pathway

    def _logarithmic_curve(self, shape_param: float, **kwargs) -> np.ndarray:
        """
        Logarithmic decay: E(t) = E0 * (1 - log(1 + k*t) / log(1 + k)) + epsilon

        Provides steeper initial reductions than exponential.

        Args:
            shape_param: Shape parameter k

        Returns:
            Emission pathway array
        """
        t = np.linspace(0, 1, self.n_years)

        # Logarithmic decay formula
        decay_factor = np.log(1 + shape_param * t) / np.log(1 + shape_param)
        pathway = self.start_emission * (1 - decay_factor) + self.end_epsilon

        # Ensure monotonic decrease
        pathway = np.maximum.accumulate(pathway[::-1])[::-1]
        pathway[-1] = self.end_epsilon

        return pathway

    def _s_curve(self, inflection_point: float, steepness: float = 10.0, **kwargs) -> np.ndarray:
        """
        S-curve (logistic function): Slow start, rapid middle, slow end.

        E(t) = E0 * (1 - 1/(1 + exp(-steepness*(t - inflection)))) + epsilon

        Args:
            inflection_point: Point of maximum reduction rate (0-1)
            steepness: Steepness of the S-curve

        Returns:
            Emission pathway array
        """
        t = np.linspace(0, 1, self.n_years)

        # Logistic decay
        decay_factor = 1 / (1 + np.exp(-steepness * (t - inflection_point)))
        pathway = self.start_emission * (1 - decay_factor) + self.end_epsilon
        pathway[-1] = self.end_epsilon

        return pathway

    def _linear_curve(self, _: float = 1.0, **kwargs) -> np.ndarray:
        """
        Linear reduction: E(t) = E0 * (1 - t) + epsilon

        Returns:
            Emission pathway array
        """
        t = np.linspace(0, 1, self.n_years)
        pathway = self.start_emission * (1 - t) + self.end_epsilon
        pathway[-1] = self.end_epsilon
        return pathway

    def _plateau_curve(self, plateau_year: float, decay_rate: float = 5.0, **kwargs) -> np.ndarray:
        """
        Plateau curve: Maintain emissions until plateau_year, then exponential decay.

        Useful for scenarios with residual hard-to-abate emissions.

        Args:
            plateau_year: Year fraction where decay begins (0-1)
            decay_rate: Rate of decay after plateau

        Returns:
            Emission pathway array
        """
        t = np.linspace(0, 1, self.n_years)

        pathway = np.zeros(self.n_years)
        plateau_idx = int(plateau_year * self.n_years)

        # Plateau phase
        pathway[:plateau_idx] = self.start_emission

        # Decay phase
        t_decay = np.linspace(0, 1, self.n_years - plateau_idx)
        pathway[plateau_idx:] = (self.start_emission * np.exp(-decay_rate * t_decay) +
                                self.end_epsilon)

        pathway[-1] = self.end_epsilon
        return pathway

    def _convex_curve(self, curvature: float, **kwargs) -> np.ndarray:
        """
        Convex curve: E(t) = E0 * (1 - t^curvature) + epsilon

        Front-loaded reduction (curvature > 1)

        Args:
            curvature: Curvature parameter (>1 for front-loading)

        Returns:
            Emission pathway array
        """
        t = np.linspace(0, 1, self.n_years)
        pathway = self.start_emission * (1 - t**curvature) + self.end_epsilon
        pathway[-1] = self.end_epsilon
        return pathway

    def _early_action_curve(self, intensity: float, **kwargs) -> np.ndarray:
        """
        Early action: Aggressive initial reductions, then plateau.

        E(t) = E0 * (0.3 + 0.7 * (1-t)^intensity) + epsilon

        Args:
            intensity: Intensity of early reductions

        Returns:
            Emission pathway array
        """
        t = np.linspace(0, 1, self.n_years)
        pathway = self.start_emission * (0.3 + 0.7 * (1 - t)**intensity) + self.end_epsilon
        pathway[-1] = self.end_epsilon
        return pathway

    def _delayed_action_curve(self, delay_fraction: float, **kwargs) -> np.ndarray:
        """
        Delayed action: Maintain emissions initially, then rapid reduction.

        Args:
            delay_fraction: Fraction of time period to delay (0-0.8)

        Returns:
            Emission pathway array
        """
        delay_fraction = np.clip(delay_fraction, 0, 0.8)
        t = np.linspace(0, 1, self.n_years)

        delay_idx = int(delay_fraction * self.n_years)
        pathway = np.ones(self.n_years) * self.start_emission

        # Rapid reduction after delay
        t_reduction = np.linspace(0, 1, self.n_years - delay_idx)
        pathway[delay_idx:] = (self.start_emission *
                               (1 - t_reduction)**2 + self.end_epsilon)

        pathway[-1] = self.end_epsilon
        return pathway

    # ==================== Milestone Calculation ====================

    def _calculate_milestones(
        self,
        pathway_national: np.ndarray,
        pathway_industry: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """
        Calculate emissions at key milestone years: 2035, 2040, 2045, 2050.

        Args:
            pathway_national: National-level emission pathway
            pathway_industry: Industry-level emission pathway (optional)

        Returns:
            Dictionary with milestone emissions for each pathway
        """
        milestone_years = [2035, 2040, 2045, 2050]
        milestones = {'years': milestone_years}

        # National milestones
        national_emissions = []
        for year in milestone_years:
            if self.start_year <= year <= self.end_year:
                idx = year - self.start_year
                national_emissions.append(float(pathway_national[idx]))
            else:
                national_emissions.append(None)

        milestones['national'] = national_emissions

        # Industry milestones (if provided)
        if pathway_industry is not None:
            industry_emissions = []
            for year in milestone_years:
                if self.start_year <= year <= self.end_year:
                    idx = year - self.start_year
                    industry_emissions.append(float(pathway_industry[idx]))
                else:
                    industry_emissions.append(None)

            milestones['industry'] = industry_emissions

        return milestones

    def _calculate_milestones_three_tier(
        self,
        pathway_national: np.ndarray,
        pathway_industry: np.ndarray,
        pathway_petrochem: np.ndarray
    ) -> Dict[str, Any]:
        """
        Calculate emissions at key milestone years for three-tier structure.

        Args:
            pathway_national: National-level emission pathway
            pathway_industry: Industry-level emission pathway
            pathway_petrochem: Petrochemical-level emission pathway

        Returns:
            Dictionary with milestone emissions for each pathway
        """
        milestone_years = [2035, 2040, 2045, 2050]
        milestones = {'years': milestone_years}

        # National milestones
        national_emissions = []
        for year in milestone_years:
            if self.start_year <= year <= self.end_year:
                idx = year - self.start_year
                national_emissions.append(float(pathway_national[idx]))
            else:
                national_emissions.append(None)
        milestones['national'] = national_emissions

        # Industry milestones
        industry_emissions = []
        for year in milestone_years:
            if self.start_year <= year <= self.end_year:
                idx = year - self.start_year
                industry_emissions.append(float(pathway_industry[idx]))
            else:
                industry_emissions.append(None)
        milestones['industry'] = industry_emissions

        # Petrochemical milestones
        petrochem_emissions = []
        for year in milestone_years:
            if self.start_year <= year <= self.end_year:
                idx = year - self.start_year
                petrochem_emissions.append(float(pathway_petrochem[idx]))
            else:
                petrochem_emissions.append(None)
        milestones['petrochem'] = petrochem_emissions

        return milestones

    # ==================== Validation ====================

    def _validate_pathway(
        self,
        pathway: np.ndarray,
        budget: float
    ) -> Dict[str, Any]:
        """
        Validate pathway for feasibility and budget compliance.

        Args:
            pathway: Emission pathway array
            budget: Target budget

        Returns:
            Validation results dictionary
        """
        issues = []

        # Check budget match
        cumulative = np.trapz(pathway, dx=1)
        budget_error = abs(cumulative - budget) / budget

        if budget_error > self.tolerance:
            issues.append(f"Budget mismatch: {budget_error*100:.2f}% error")

        # Check monotonic decrease
        if not np.all(np.diff(pathway) <= 0):
            issues.append("Pathway not monotonically decreasing")

        # Check annual reduction rates
        annual_reductions = -np.diff(pathway) / pathway[:-1]
        max_reduction = np.max(annual_reductions)

        if max_reduction > self.max_annual_reduction:
            issues.append(f"Exceeds max reduction rate: {max_reduction:.1%}/year "
                        f"(limit: {self.max_annual_reduction:.1%}/year)")

        # Check final emission
        if pathway[-1] > self.min_final_emission and not self.allow_residual:
            issues.append(f"Final emission {pathway[-1]:.0f} tCO2/year exceeds "
                        f"minimum {self.min_final_emission:.0f} tCO2/year")

        # Check for negative values
        if np.any(pathway < 0):
            issues.append("Pathway contains negative emissions")

        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'budget_error_pct': float(budget_error * 100),
            'max_annual_reduction_pct': float(max_reduction * 100),
            'final_emission': float(pathway[-1]),
            'cumulative_emissions': float(cumulative)
        }

    def export_pathway(
        self,
        result: Dict[str, Any],
        output_path: Path,
        format: str = 'csv'
    ) -> None:
        """
        Export pathway results to file.

        Args:
            result: Result dictionary from allocate_budget()
            output_path: Output file path
            format: 'csv' or 'json'
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == 'csv':
            # Create DataFrame
            df = pd.DataFrame({
                'year': result['years'],
                'emissions_tco2': result['pathway'],
                'cumulative_tco2': np.cumsum(result['pathway'])
            })
            df.to_csv(output_path, index=False)
            logger.info(f"Pathway exported to {output_path}")

        elif format == 'json':
            # Prepare JSON-serializable dict
            output_dict = {
                'pathway': result['pathway'].tolist(),
                'years': result['years'].tolist(),
                'budget_allocated': result['budget_allocated'],
                'cumulative_emissions': result['cumulative_emissions'],
                'curve_type': result['curve_type'],
                'parameters': result['parameters'],
                'validation': result['validation'],
                'metadata': result['metadata']
            }

            with open(output_path, 'w') as f:
                json.dump(output_dict, f, indent=2)
            logger.info(f"Pathway exported to {output_path}")

        else:
            raise ValueError(f"Unknown format '{format}'. Use 'csv' or 'json'")

    def compare_curves(
        self,
        budget: float,
        curve_types: Optional[List[str]] = None,
        tier: str = 'single',
        industry_fraction: float = 0.37,
        petrochem_fraction: float = 0.10
    ) -> pd.DataFrame:
        """
        Compare multiple curve types for the same budget.

        Args:
            budget: Target carbon budget
            curve_types: List of curve types to compare (or None for all)
            tier: 'single', 'two_tier', or 'three_tier' pathway mode
            industry_fraction: Fraction of national budget for industry
            petrochem_fraction: Fraction of industry budget for petrochemical

        Returns:
            DataFrame with comparison metrics including milestones
        """
        if curve_types is None:
            curve_types = list(self.curve_types.keys())

        results = []

        for curve_type in curve_types:
            try:
                result = self.allocate_budget(
                    budget, curve_type, validate=True,
                    tier=tier, industry_fraction=industry_fraction,
                    petrochem_fraction=petrochem_fraction
                )

                pathway = result['pathway']

                # Get validation (handle single, two-tier, and three-tier)
                if tier in ['two_tier', 'three_tier']:
                    validation = result['validation']['national']
                else:
                    validation = result['validation']

                # Calculate metrics
                initial_reduction = (pathway[0] - pathway[1]) / pathway[0] * 100
                avg_reduction = np.mean(-np.diff(pathway) / pathway[:-1]) * 100
                final_emission = pathway[-1]

                # Extract milestones
                milestones = result['milestones']
                row = {
                    'curve_type': curve_type,
                    'is_valid': validation['is_valid'],
                    'budget_error_pct': validation['budget_error_pct'],
                    'initial_reduction_pct': initial_reduction,
                    'avg_annual_reduction_pct': avg_reduction,
                    'final_emission_tco2': final_emission,
                    'cumulative_tco2': result['cumulative_emissions']
                }

                # Add milestone emissions (national)
                for i, year in enumerate(milestones['years']):
                    row[f'emission_{year}'] = milestones['national'][i]

                # Add industry milestones if two-tier or three-tier
                if tier in ['two_tier', 'three_tier'] and 'industry' in milestones:
                    for i, year in enumerate(milestones['years']):
                        row[f'industry_{year}'] = milestones['industry'][i]

                # Add petrochemical milestones if three-tier
                if tier == 'three_tier' and 'petrochem' in milestones:
                    for i, year in enumerate(milestones['years']):
                        row[f'petrochem_{year}'] = milestones['petrochem'][i]

                results.append(row)
            except Exception as e:
                logger.warning(f"Failed to generate {curve_type} pathway: {e}")

        return pd.DataFrame(results)
