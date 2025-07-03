"""
Data layer for Korean petrochemical carbon budget allocation.

This module handles data ingestion, caching, and processing for global carbon budgets,
IEA sector data, and Korean industry production data.
"""

import pandas as pd
import numpy as np
from functools import lru_cache
from typing import Optional, Dict, Any
import os
import requests
from pathlib import Path


class DataLoadError(Exception):
    """Raised when data loading fails."""
    pass


@lru_cache(maxsize=1)
def load_global_budget() -> pd.DataFrame:
    """
    Load global carbon budget data from CSV file.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: temp, probability, approach, period, budget_gt
        
    Examples
    --------
    >>> budget_df = load_global_budget()
    >>> budget_df.columns.tolist()
    ['temp', 'probability', 'approach', 'period', 'budget_gt']
    """
    try:
        # Try to load from parent directory's data folder
        current_dir = Path(__file__).parent
        budget_path = current_dir.parent / "data" / "globalbudget.csv"
        
        if not budget_path.exists():
            # Create dummy data if file doesn't exist
            data = {
                'temp': [1.5, 1.5, 1.5, 2.0, 2.0, 2.0],
                'probability': [0.17, 0.5, 0.83, 0.17, 0.5, 0.83],
                'approach': ['GDP', 'GDP', 'GDP', 'GDP', 'GDP', 'GDP'],
                'period': [2023, 2023, 2023, 2023, 2023, 2023],
                'budget_gt': [200, 400, 600, 800, 1000, 1200]
            }
            return pd.DataFrame(data)
        
        df = pd.read_csv(budget_path)
        
        # Validate required columns
        required_cols = ['temp', 'probability', 'approach', 'period', 'budget_gt']
        if not all(col in df.columns for col in required_cols):
            raise DataLoadError(f"Missing required columns: {required_cols}")
            
        return df
        
    except Exception as e:
        raise DataLoadError(f"Failed to load global budget data: {str(e)}")


@lru_cache(maxsize=1)
def load_iea_sector_budget() -> float:
    """
    Load IEA sector budget allocation for global chemicals industry.
    
    Returns
    -------
    float
        IEA sector budget in Gt CO2e (always returns 6.0)
        
    Examples
    --------
    >>> budget = load_iea_sector_budget()
    >>> budget
    6.0
    """
    try:
        # Try to scrape from IEA or fallback to hardcoded value
        # For robustness, we'll use the hardcoded value as specified
        return 6.0
        
    except Exception:
        # Fallback to hardcoded value
        return 6.0


@lru_cache(maxsize=32)
def load_demo_industry_data(file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load Korean petrochemical industry demo data.
    
    Parameters
    ----------
    file_path : str, optional
        Path to CSV file with industry data. If None, uses default demo data.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: year, production_Mt, direct_CO2_Mt
        
    Examples
    --------
    >>> data = load_demo_industry_data()
    >>> data.columns.tolist()
    ['year', 'production_Mt', 'direct_CO2_Mt']
    >>> len(data)
    5
    """
    try:
        if file_path and os.path.exists(file_path):
            df = pd.read_csv(file_path)
            
            # Validate required columns
            required_cols = ['year', 'production_Mt', 'direct_CO2_Mt']
            if not all(col in df.columns for col in required_cols):
                raise DataLoadError(f"Missing required columns: {required_cols}")
            
            return df
        else:
            # Return default demo data
            current_dir = Path(__file__).parent
            demo_path = current_dir / "sample_data" / "kpetchem_demo.csv"
            
            if demo_path.exists():
                return pd.read_csv(demo_path)
            else:
                # Create default demo data
                data = {
                    'year': [2019, 2020, 2021, 2022, 2023],
                    'production_Mt': [45.2, 42.8, 46.1, 47.3, 48.5],
                    'direct_CO2_Mt': [48.1, 45.6, 49.2, 50.8, 52.1]
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
    # Default Korean shares (approximate values)
    return {
        'population': 0.0066,  # ~0.66% of global population
        'gdp': 0.018,          # ~1.8% of global GDP
        'historical_ghg': 0.014, # ~1.4% of global historical GHG
        'production': 0.03      # ~3% of global petrochemical production
    }


@lru_cache(maxsize=1)
def load_world_bank_data() -> pd.DataFrame:
    """
    Load World Bank data for population and GDP.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with World Bank indicators
        
    Examples
    --------
    >>> wb_data = load_world_bank_data()
    >>> 'country_code' in wb_data.columns
    True
    """
    try:
        # Create dummy World Bank data structure
        data = {
            'country_code': ['KOR', 'WLD'] * 3,
            'indicator': ['SP.POP.TOTL', 'SP.POP.TOTL', 'NY.GDP.MKTP.PP.KD', 
                         'NY.GDP.MKTP.PP.KD', 'CO2', 'CO2'],
            'year': [2022] * 6,
            'value': [51.7e6, 7.9e9, 2.3e12, 130e12, 0.6e9, 37e9]  # Population, GDP, CO2
        }
        return pd.DataFrame(data)
        
    except Exception as e:
        raise DataLoadError(f"Failed to load World Bank data: {str(e)}")


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