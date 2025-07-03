"""
Budget allocation logic for Korean petrochemical sector.

This module implements four allocation criteria to determine Korea's share
of the global carbon budget for the petrochemical industry.
"""

import pandas as pd
import numpy as np
from typing import Dict, Union, Optional
try:
    # Try relative import first (for package usage)
    from .data_layer import (
        load_global_budget, 
        load_iea_sector_budget, 
        load_demo_industry_data,
        get_korean_shares
    )
except ImportError:
    # Fall back to direct import (for Streamlit)
    from data_layer import (
        load_global_budget, 
        load_iea_sector_budget, 
        load_demo_industry_data,
        get_korean_shares
    )


class BudgetAllocator:
    """
    Allocates global carbon budgets to Korean petrochemical sector.
    
    Supports four allocation criteria:
    1. Population share
    2. GDP share  
    3. Historical GHG share
    4. IEA sector pathway share
    
    Parameters
    ----------
    baseline_emissions : float
        Current Korean petrochemical emissions in Mt CO2e/year
        
    Examples
    --------
    >>> allocator = BudgetAllocator(baseline_emissions=50.0)
    >>> budget = allocator.allocate_budget('population', temp=1.5, probability=0.5)
    >>> budget > 0
    True
    """
    
    def __init__(self, baseline_emissions: float = 50.0):
        self.baseline_emissions = baseline_emissions
        self.korean_shares = get_korean_shares()
        
    def allocate_budget(self, 
                       method: str, 
                       temp: float = 1.5, 
                       probability: float = 0.5,
                       approach: str = 'GDP',
                       period: int = 2023) -> float:
        """
        Allocate budget using specified method.
        
        Parameters
        ----------
        method : str
            Allocation method: 'population', 'gdp', 'historical_ghg', or 'iea_sector'
        temp : float
            Temperature target in degrees C
        probability : float
            Probability value for budget selection
        approach : str
            Approach for budget selection
        period : int
            Period year for budget selection
            
        Returns
        -------
        float
            Allocated budget in Mt CO2e for 2035-2050
            
        Examples
        --------
        >>> allocator = BudgetAllocator(50.0)
        >>> budget = allocator.allocate_budget('population')
        >>> isinstance(budget, float)
        True
        """
        if method == 'iea_sector':
            return self._allocate_iea_sector(temp, probability, approach, period)
        elif method in ['population', 'gdp', 'historical_ghg']:
            return self._allocate_share_based(method, temp, probability, approach, period)
        else:
            raise ValueError(f"Unknown allocation method: {method}")
    
    def _allocate_share_based(self, 
                             method: str, 
                             temp: float, 
                             probability: float,
                             approach: str, 
                             period: int) -> float:
        """
        Allocate budget based on Korean share of global indicator.
        
        Parameters
        ----------
        method : str
            Share type: 'population', 'gdp', or 'historical_ghg'
        temp : float
            Temperature target
        probability : float
            Probability value
        approach : str
            Approach for budget selection
        period : int
            Period year
            
        Returns
        -------
        float
            Allocated budget in Mt CO2e
        """
        # Get global budget
        global_budget_df = load_global_budget()
        
        # Filter for specified criteria
        filtered = global_budget_df[
            (global_budget_df['temp'] == temp) &
            (global_budget_df['probability'] == probability) &
            (global_budget_df['approach'] == approach) &
            (global_budget_df['period'] == period)
        ]
        
        if filtered.empty:
            raise ValueError(f"No budget found for criteria: temp={temp}, prob={probability}")
        
        global_budget_gt = filtered['budget_gt'].iloc[0]
        
        # Get Korean share
        korean_share = self.korean_shares[method]
        
        # Convert to Mt and return
        return global_budget_gt * korean_share * 1000  # Gt to Mt
    
    def _allocate_iea_sector(self, 
                            temp: float, 
                            probability: float,
                            approach: str, 
                            period: int) -> float:
        """
        Allocate budget using IEA sector pathway.
        
        Uses minimum of IEA sector allocation (6 Gt) and user-selected global budget,
        then multiplies by Korean production share.
        
        Parameters
        ----------
        temp : float
            Temperature target
        probability : float
            Probability value
        approach : str
            Approach for budget selection
        period : int
            Period year
            
        Returns
        -------
        float
            Allocated budget in Mt CO2e
        """
        # Get IEA sector budget (6 Gt)
        iea_sector_budget = load_iea_sector_budget()  # Returns 6.0 Gt
        
        # Get global budget
        global_budget_df = load_global_budget()
        filtered = global_budget_df[
            (global_budget_df['temp'] == temp) &
            (global_budget_df['probability'] == probability) &
            (global_budget_df['approach'] == approach) &
            (global_budget_df['period'] == period)
        ]
        
        if filtered.empty:
            raise ValueError(f"No budget found for criteria: temp={temp}, prob={probability}")
        
        global_budget_gt = filtered['budget_gt'].iloc[0]
        
        # Use minimum of IEA sector budget and global budget
        sector_budget_gt = min(iea_sector_budget, global_budget_gt)
        
        # Apply Korean production share
        korean_share = self.korean_shares['production']
        
        # Convert to Mt and return
        return sector_budget_gt * korean_share * 1000  # Gt to Mt
    
    def get_allocation_summary(self, 
                              temp: float = 1.5, 
                              probability: float = 0.5,
                              approach: str = 'GDP',
                              period: int = 2023) -> pd.DataFrame:
        """
        Get summary of all allocation methods.
        
        Parameters
        ----------
        temp : float
            Temperature target
        probability : float
            Probability value
        approach : str
            Approach for budget selection
        period : int
            Period year
            
        Returns
        -------
        pd.DataFrame
            Summary with columns: method, korean_share, allocated_budget_mt
            
        Examples
        --------
        >>> allocator = BudgetAllocator(50.0)
        >>> summary = allocator.get_allocation_summary()
        >>> 'method' in summary.columns
        True
        >>> len(summary)
        4
        """
        methods = ['population', 'gdp', 'historical_ghg', 'iea_sector']
        results = []
        
        for method in methods:
            try:
                budget = self.allocate_budget(method, temp, probability, approach, period)
                # Map method to correct share key
                if method == 'iea_sector':
                    share = self.korean_shares.get('production', 0.0)
                else:
                    share = self.korean_shares.get(method, 0.0)
                
                results.append({
                    'method': method,
                    'korean_share': share,
                    'allocated_budget_mt': budget
                })
            except Exception as e:
                results.append({
                    'method': method,
                    'korean_share': 0.0,
                    'allocated_budget_mt': 0.0
                })
        
        return pd.DataFrame(results)
    
    def validate_shares(self, tolerance: float = 1e-6) -> bool:
        """
        Validate that Korean shares are reasonable.
        
        Parameters
        ----------
        tolerance : float
            Tolerance for validation checks
            
        Returns
        -------
        bool
            True if shares are valid
            
        Examples
        --------
        >>> allocator = BudgetAllocator(50.0)
        >>> allocator.validate_shares()
        True
        """
        for method, share in self.korean_shares.items():
            if not (0 <= share <= 1):
                return False
            if abs(share - 0) < tolerance and method != 'production':
                return False
        return True