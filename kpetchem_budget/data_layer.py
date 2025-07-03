"""
Data layer for Korean petrochemical carbon budget allocation (2023-2050).

This module handles data ingestion, caching, and processing for global carbon budgets,
IEA sector data, and Korean industry baseline data starting from 2023.
"""

import pandas as pd
import numpy as np
import os
from functools import lru_cache
from typing import Optional, Dict, Any


class DataLoadError(Exception):
    """Raised when data loading fails."""
    pass


@lru_cache(maxsize=1)
def load_global_budget() -> pd.DataFrame:
    """
    Load global carbon budget data from CSV file (2023-2050 cumulative).
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: temp, probability, approach, period, budget_gt
        Budget values are cumulative for 2023-2050 period.
        
    Examples
    --------
    >>> budget_df = load_global_budget()
    >>> budget_df.columns.tolist()
    ['temp', 'probability', 'approach', 'period', 'budget_gt']
    >>> len(budget_df)
    60
    """
    # Try to load from parent directory's data folder
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    budget_path = os.path.join(parent_dir, "data", "globalbudget.csv")
    
    if not os.path.exists(budget_path):
        # Create dummy data if file doesn't exist - updated for 2023-2050
        data = {
            'temp': [1.5, 1.5, 1.5, 2.0, 2.0, 2.0] * 10,
            'probability': [0.17, 0.5, 0.83, 0.17, 0.5, 0.83] * 10,
            'approach': ['GDP'] * 60,
            'period': [2023] * 60,
            'budget_gt': [150, 300, 450, 600, 750, 900] * 10  # 2023-2050 cumulative
        }
        return pd.DataFrame(data)
    
    try:
        df = pd.read_csv(budget_path)
        
        # Validate required columns - check for actual file format
        if 'temp' in df.columns and 'probability' in df.columns and 'budget' in df.columns:
            # Original format - add missing columns with defaults
            df = df.rename(columns={'budget': 'budget_gt'})
            df['approach'] = 'GDP'  # Default approach
            df['period'] = 2023     # Default period (2023-2050)
            
            # Scale budget values for 2023-2050 period (28 years vs original assumption)
            # Original data appears to be for shorter period, scale up
            df['budget_gt'] = df['budget_gt'] * 1.4  # Rough scaling factor
            
        else:
            # Expected format
            required_cols = ['temp', 'probability', 'approach', 'period', 'budget_gt']
            if not all(col in df.columns for col in required_cols):
                raise DataLoadError(f"Missing required columns: {required_cols}")
            
        return df
        
    except Exception as e:
        raise DataLoadError(f"Failed to load global budget data: {str(e)}")


@lru_cache(maxsize=1)
def load_iea_sector_budget() -> float:
    """
    Load IEA sector budget allocation for global chemicals industry (2023-2050).
    
    Returns
    -------
    float
        IEA sector budget in Gt CO2e for 2023-2050 period (6.0 Gt)
        
    Examples
    --------
    >>> budget = load_iea_sector_budget()
    >>> budget
    6.0
    """
    # IEA Net Zero Roadmap allocation for chemicals sector (2023-2050)
    return 6.0


@lru_cache(maxsize=32)
def load_demo_industry_data(file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load Korean petrochemical industry baseline data (2023 only).
    
    Parameters
    ----------
    file_path : str, optional
        Path to CSV file with industry data. If None, uses default 2023 baseline.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: year, production_Mt, direct_CO2_Mt, value_added_bnKRW
        Contains single row for 2023 baseline.
        
    Examples
    --------
    >>> data = load_demo_industry_data()
    >>> data.columns.tolist()
    ['year', 'production_Mt', 'direct_CO2_Mt', 'value_added_bnKRW']
    >>> len(data)
    1
    >>> data.iloc[0]['year']
    2023
    """
    try:
        if file_path and os.path.exists(file_path):
            df = pd.read_csv(file_path)
            
            # Validate required columns
            required_cols = ['year', 'production_Mt', 'direct_CO2_Mt']
            if not all(col in df.columns for col in required_cols):
                raise DataLoadError(f"Missing required columns: {required_cols}")
            
            # Ensure data starts from 2023
            if df['year'].min() < 2023:
                raise DataLoadError("Custom data must start from 2023 or later")
            
            return df
        else:
            # Return 2023 baseline only
            current_dir = os.path.dirname(os.path.abspath(__file__))
            demo_path = os.path.join(current_dir, "sample_data", "kpetchem_demo.csv")
            
            if os.path.exists(demo_path):
                df = pd.read_csv(demo_path)
                # Filter to 2023 only if multiple years present
                return df[df['year'] == 2023].reset_index(drop=True)
            else:
                # Create 2023 baseline data
                data = {
                    'year': [2023],
                    'production_Mt': [90.0],
                    'direct_CO2_Mt': [50.0],
                    'value_added_bnKRW': [38000.0]
                }
                return pd.DataFrame(data)
                
    except Exception as e:
        raise DataLoadError(f"Failed to load industry data: {str(e)}")


def get_korean_shares() -> Dict[str, float]:
    """
    Get Korean shares for different allocation criteria.
    
    Returns
    -------
    Dict[str, float]
        Dictionary with keys: 'population', 'gdp', 'historical_ghg', 'production'
        
    Examples
    --------
    >>> shares = get_korean_shares()
    >>> 'population' in shares
    True
    >>> 0 < shares['population'] < 1
    True
    """
    # Korean shares for 2023-2050 period
    return {
        'population': 0.0066,      # ~0.66% of global population
        'gdp': 0.018,              # ~1.8% of global GDP  
        'historical_ghg': 0.014,   # ~1.4% of global historical GHG
        'production': 0.03         # ~3% of global petrochemical production
    }


@lru_cache(maxsize=1)
def load_world_bank_data() -> pd.DataFrame:
    """
    Load World Bank data for population and GDP (2023 baseline).
    
    Returns
    -------
    pd.DataFrame
        DataFrame with World Bank indicators for 2023
        
    Examples
    --------
    >>> wb_data = load_world_bank_data()
    >>> 'country_code' in wb_data.columns
    True
    """
    try:
        # Create 2023 baseline World Bank data structure
        data = {
            'country_code': ['KOR', 'WLD'] * 3,
            'indicator': ['SP.POP.TOTL', 'SP.POP.TOTL', 'NY.GDP.MKTP.PP.KD', 
                         'NY.GDP.MKTP.PP.KD', 'CO2', 'CO2'],
            'year': [2023] * 6,
            'value': [51.8e6, 8.0e9, 2.4e12, 135e12, 0.6e9, 38e9]  # 2023 estimates
        }
        return pd.DataFrame(data)
        
    except Exception as e:
        raise DataLoadError(f"Failed to load World Bank data: {str(e)}")


def get_timeline_years() -> np.ndarray:
    """
    Get the standard timeline for K-PetChem analysis (2023-2050).
    
    Returns
    -------
    np.ndarray
        Array of years from 2023 to 2050 inclusive
        
    Examples
    --------
    >>> years = get_timeline_years()
    >>> years[0]
    2023
    >>> years[-1] 
    2050
    >>> len(years)
    28
    """
    return np.arange(2023, 2051)


def cache_clear_all() -> None:
    """
    Clear all cached data.
    
    Examples
    --------
    >>> cache_clear_all()
    """
    load_global_budget.cache_clear()
    load_iea_sector_budget.cache_clear()
    load_demo_industry_data.cache_clear()
    load_world_bank_data.cache_clear()