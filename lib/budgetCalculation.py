"""
Budget Calculation using BKIR Formula

This module implements the BKIR (Budget Korea Industrial/Petrochemical) formula
for allocating global carbon budgets to Korea and subsequently to industrial sectors.
The allocation is based on three fairness pillars: Responsibility, Capability, and Equality.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class BudgetAllocation:
    """
    Budget allocation calculator using BKIR formula.
    
    Implements the allocation formula:
    BKIR(j) = B_global(j) × (w_r × δ_r + w_c × δ_c + w_e × δ_e)
    
    Where:
    - B_global(j): Global carbon budget for scenario j
    - w_r, w_c, w_e: User-defined weights for responsibility, capability, equality
    - δ_r, δ_c, δ_e: Normalized country factors (shares)
    """
    
    def __init__(self, data_df: Optional[pd.DataFrame] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize budget allocation calculator.
        
        Args:
            data_df: Optional DataFrame containing country-level data
            config: Optional configuration dictionary
        """
        self.data_df = data_df
        self.config = config or {}
        
        # Default shares for Korea (can be overridden by data)
        self.default_shares = {
            'responsibility': 0.011,  # Historical cumulative emissions share
            'capability': 0.017,      # GDP share (PPP-adjusted)
            'equality': 0.0067        # Population share
        }
        
        # Base emissions (tCO2/year) from config or defaults
        base_emissions_cfg = self.config.get('base_emissions', {})
        self.base_emissions_data = {
            'industry': base_emissions_cfg.get('industry', 185.0e6),
            'petrochem': base_emissions_cfg.get('petrochem', 50.0e6)
        }
        
        logger.info("BudgetAllocation initialized")
    
    def calculate_bkir_budget(self, global_budget: float, responsibility_share: float,
                             capability_share: float, equality_share: float,
                             weights: np.ndarray) -> float:
        """
        Calculate Korea's budget using BKIR formula.
        
        Args:
            global_budget: Global carbon budget (tCO2)
            responsibility_share: Korea's responsibility factor (δ_r)
            capability_share: Korea's capability factor (δ_c)
            equality_share: Korea's equality factor (δ_e)
            weights: User weights [w_r, w_c, w_e]
            
        Returns:
            Korea's allocated carbon budget (tCO2)
        """
        # Extract individual weights
        w_r, w_c, w_e = weights
        
        # Calculate weighted factor sum
        weighted_factors = (w_r * responsibility_share + 
                           w_c * capability_share + 
                           w_e * equality_share)
        
        # Apply BKIR formula
        korea_budget = global_budget * weighted_factors
        
        return korea_budget
    
    def allocate_to_sectors(self, korea_budget: float, 
                           industry_fraction: float,
                           petrochem_fraction: float) -> Tuple[float, float]:
        """
        Allocate Korea's budget to industry and petrochemical sectors.
        
        Args:
            korea_budget: Korea's total carbon budget (tCO2)
            industry_fraction: Fraction allocated to industry sector
            petrochem_fraction: Fraction of industry budget allocated to petrochemicals
            
        Returns:
            Tuple of (industry_budget, petrochem_budget) in tCO2
        """
        industry_budget = korea_budget * industry_fraction
        petrochem_budget = industry_budget * petrochem_fraction
        
        logger.debug(f"Korea budget: {korea_budget:.2e} tCO2")
        logger.debug(f"Industry budget: {industry_budget:.2e} tCO2")
        logger.debug(f"Petrochemical budget: {petrochem_budget:.2e} tCO2")
        
        return industry_budget, petrochem_budget
    
    def get_base_emissions(self, sector: str) -> float:
        """
        Get base emissions for specified sector.
        
        Args:
            sector: Sector name ('industry' or 'petrochem')
            
        Returns:
            Base emission rate (tCO2/year)
            
        Raises:
            ValueError: If sector is unknown
        """
        if sector not in self.base_emissions_data:
            raise ValueError(f"Unknown sector: {sector}. Available: {list(self.base_emissions_data.keys())}")
        
        return self.base_emissions_data[sector]
    
    def get_allocation_shares(self, country_code: str = 'KOR') -> Dict[str, float]:
        """
        Get allocation shares for a country.
        
        Args:
            country_code: ISO country code
            
        Returns:
            Dictionary with responsibility, capability, and equality shares
        """
        if self.data_df is not None:
            # Try to extract shares from data
            try:
                shares = self._extract_shares_from_data(country_code)
                logger.info(f"Using data-derived shares for {country_code}")
                return shares
            except Exception as e:
                logger.warning(f"Could not extract shares from data: {e}")
        
        # Use default shares
        logger.info(f"Using default shares for {country_code}")
        return self.default_shares.copy()
    
    def _extract_shares_from_data(self, country_code: str) -> Dict[str, float]:
        """
        Extract allocation shares from data DataFrame.
        
        This is a placeholder implementation. In practice, this would
        calculate shares based on historical emissions, GDP, and population data.
        
        Args:
            country_code: ISO country code
            
        Returns:
            Dictionary with calculated shares
        """
        # Placeholder: calculate shares from actual data
        # This would involve aggregating historical emissions, GDP, population
        # and calculating country's share relative to global totals
        
        # For now, return defaults with small variations
        base_shares = self.default_shares.copy()
        
        # Add small country-specific adjustments (placeholder)
        adjustments = {
            'KOR': {'responsibility': 0.001, 'capability': 0.002, 'equality': -0.0005},
            'JPN': {'responsibility': 0.01, 'capability': 0.05, 'equality': -0.002},
            'CHN': {'responsibility': 0.15, 'capability': 0.12, 'equality': 0.18}
        }
        
        if country_code in adjustments:
            for key, adj in adjustments[country_code].items():
                base_shares[key] += adj
        
        return base_shares
    
    def validate_allocation(self, budget: float, emissions_path: np.ndarray) -> Dict[str, Any]:
        """
        Validate that an emission pathway respects the allocated budget.
        
        Args:
            budget: Allocated carbon budget (tCO2)
            emissions_path: Array of annual emissions (tCO2/year)
            
        Returns:
            Dictionary with validation results
        """
        # Calculate cumulative emissions (trapezoidal integration)
        cumulative_emissions = np.trapz(emissions_path, dx=1)
        
        # Calculate budget utilization
        utilization = cumulative_emissions / budget if budget > 0 else 0
        
        # Check for overshoot
        overshoot = max(0, cumulative_emissions - budget)
        overshoot_pct = (overshoot / budget * 100) if budget > 0 else 0
        
        # Calculate reduction rate (start to end)
        start_emission = emissions_path[0] if len(emissions_path) > 0 else 0
        end_emission = emissions_path[-1] if len(emissions_path) > 0 else 0
        reduction_rate = ((start_emission - end_emission) / start_emission * 100) if start_emission > 0 else 0
        
        validation_results = {
            'allocated_budget': budget,
            'cumulative_emissions': cumulative_emissions,
            'budget_utilization': utilization,
            'overshoot_amount': overshoot,
            'overshoot_percentage': overshoot_pct,
            'emission_reduction_rate': reduction_rate,
            'is_compliant': overshoot < budget * 0.01,  # Allow 1% tolerance
            'start_emission': start_emission,
            'end_emission': end_emission,
            'pathway_length': len(emissions_path)
        }
        
        return validation_results
    
    def calculate_sector_budgets_batch(self, global_budgets: np.ndarray,
                                     responsibility_shares: np.ndarray,
                                     capability_shares: np.ndarray,
                                     equality_shares: np.ndarray,
                                     weights: np.ndarray,
                                     industry_fraction: float,
                                     petrochem_fraction: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate sector budgets for multiple Monte Carlo draws.
        
        Args:
            global_budgets: Array of global budget samples
            responsibility_shares: Array of responsibility share samples
            capability_shares: Array of capability share samples
            equality_shares: Array of equality share samples
            weights: Array of weight samples (n_draws, 3)
            industry_fraction: Industry allocation fraction
            petrochem_fraction: Petrochemical allocation fraction
            
        Returns:
            Tuple of (industry_budgets, petrochem_budgets) arrays
        """
        n_draws = len(global_budgets)
        
        # Calculate Korea budgets using vectorized BKIR formula
        weighted_factors = (weights[:, 0] * responsibility_shares +
                           weights[:, 1] * capability_shares +
                           weights[:, 2] * equality_shares)
        
        korea_budgets = global_budgets * weighted_factors
        
        # Calculate sector budgets
        industry_budgets = korea_budgets * industry_fraction
        petrochem_budgets = industry_budgets * petrochem_fraction
        
        logger.info(f"Calculated budgets for {n_draws} Monte Carlo draws")
        logger.info(f"Korea budget range: {korea_budgets.min():.2e} - {korea_budgets.max():.2e} tCO2")
        logger.info(f"Industry budget range: {industry_budgets.min():.2e} - {industry_budgets.max():.2e} tCO2")
        logger.info(f"Petrochemical budget range: {petrochem_budgets.min():.2e} - {petrochem_budgets.max():.2e} tCO2")
        
        return industry_budgets, petrochem_budgets
    
    def get_budget_statistics(self, budgets: np.ndarray) -> Dict[str, float]:
        """
        Calculate summary statistics for budget arrays.
        
        Args:
            budgets: Array of budget values
            
        Returns:
            Dictionary with statistical measures
        """
        stats = {
            'mean': float(np.mean(budgets)),
            'median': float(np.median(budgets)),
            'std': float(np.std(budgets)),
            'min': float(np.min(budgets)),
            'max': float(np.max(budgets)),
            'p05': float(np.percentile(budgets, 5)),
            'p25': float(np.percentile(budgets, 25)),
            'p75': float(np.percentile(budgets, 75)),
            'p95': float(np.percentile(budgets, 95)),
            'cv': float(np.std(budgets) / np.mean(budgets)) if np.mean(budgets) > 0 else 0
        }
        
        return stats