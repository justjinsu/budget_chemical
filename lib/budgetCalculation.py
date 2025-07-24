import pandas as pd
import numpy as np


class budgetAllocation:
    def __init__(self, data_df=None, budget=None):
        """
        Initialize the budgetAllocation class for BKIR formula implementation.
        
        Parameters:
        data_df (pd.DataFrame): Optional DataFrame containing country-level data
        budget: Optional budget parameter (kept for compatibility)
        """
        self.data_df = data_df
        self.budget = budget

    def calculate_BKIR(self, global_budget: np.ndarray, responsibility_factor: np.ndarray, 
                       capability_factor: np.ndarray, equality_factor: np.ndarray, 
                       weights: np.ndarray) -> np.ndarray:
        """
        Calculate Korea Industrial/Petrochemical budget using BKIR formula:
        BKIR(j) = Bglobal(j) × (w(r)δ(r) + w(c)δ(c) + w(e)δ(e))
        
        Parameters:
        global_budget: Array of global budget scenarios (n_draws,)
        responsibility_factor: Array of responsibility factors δr (n_draws,)
        capability_factor: Array of capability factors δc (n_draws,) 
        equality_factor: Array of equality factors δe (n_draws,)
        weights: Array of user weights (n_draws, 3) for [r, c, e]
        
        Returns:
        np.ndarray: BKIR budget allocations (n_draws,)
        """
        # Extract individual weight components
        w_r = weights[:, 0]  # responsibility weights
        w_c = weights[:, 1]  # capability weights  
        w_e = weights[:, 2]  # equality weights
        
        # Calculate weighted factor sum
        weighted_factors = (w_r * responsibility_factor + 
                           w_c * capability_factor + 
                           w_e * equality_factor)
        
        # Apply BKIR formula
        bkir_budget = global_budget * weighted_factors
        
        return bkir_budget

    def allocate_industry_petrochem(self, bkir_budget: np.ndarray, 
                                   industry_fraction: float, 
                                   petrochem_fraction: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Allocate BKIR budget to industry and petrochemical sectors.
        
        Parameters:
        bkir_budget: Array of BKIR budget allocations (n_draws,)
        industry_fraction: Fraction for industry sector (target: 0.37)
        petrochem_fraction: Fraction for petrochemical sector (target: 0.10)
        
        Returns:
        tuple: (industry_budget, petrochem_budget) arrays
        """
        industry_budget = bkir_budget * industry_fraction
        petrochem_budget = industry_budget * petrochem_fraction
        
        return industry_budget, petrochem_budget

    def get_base_emissions(self, sector: str = "petrochem") -> float:
        """
        Get base emissions for the specified sector.
        
        Parameters:
        sector: Sector name ("petrochem" or "industry")
        
        Returns:
        float: Base emissions in tCO2
        """
        base_emissions = {
            'petrochem': 50.0e6,  # 50 MtCO2 for Korean petrochemical sector
            'industry': 185.0e6   # 185 MtCO2 for Korean industry sector
        }
        
        return base_emissions.get(sector, 50.0e6)

    # Legacy methods for compatibility
    def allocate_by_keyword(self, keyword, year, invert=False):
        """Legacy method for backward compatibility"""
        # Placeholder implementation
        return pd.DataFrame()
    
    def get_shares(self):
        """Legacy method for backward compatibility"""
        return {
            'responsibility': 0.011,  # Default responsibility share
            'equality': 0.0067        # Default equality share
        }