"""
Korean Petrochemical Carbon Budget Allocation Toolkit

Advanced Monte Carlo simulation engine for carbon budget allocation and 
emission pathway optimization for the Korean petrochemical sector (2023-2050).

Features:
- 4 allocation criteria (population, GDP, national GHG, IEA sector)
- 4 pathway generators (linear, constant rate, logistic, IEA proxy)
- Monte Carlo uncertainty quantification (14,400 scenarios)
- High-performance parallel execution
- Interactive Streamlit dashboard
"""

__version__ = "2.0.0"
__author__ = "Korean Petrochemical Carbon Budget Team"

# Core modules
from .data_layer import (
    load_global_budget,
    load_iea_sector_budget, 
    load_demo_industry_data,
    get_korean_shares
)
from .parameter_space import ParameterGrid, MonteCarloSampler
from .simulator import ParallelSimulator
from .datastore import ParquetWarehouse
from .pathway import (
    PathwayGenerator,
    BudgetOverflowError,
    mark_milestones
)

# Dashboard components
from .dashboard.app import main as run_dashboard

__all__ = [
    "load_global_budget",
    "load_iea_sector_budget",
    "load_demo_industry_data", 
    "get_korean_shares",
    "ParameterGrid",
    "MonteCarloSampler",
    "ParallelSimulator",
    "ParquetWarehouse",
    "PathwayGenerator",
    "BudgetOverflowError",
    "mark_milestones",
    "run_dashboard"
]