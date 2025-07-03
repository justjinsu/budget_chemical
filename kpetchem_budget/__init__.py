"""
Korean Petrochemical Carbon Budget Allocation Package

This package provides tools for allocating global carbon budgets to the Korean 
petrochemical sector using multiple allocation criteria and generating emission 
pathways for 2035-2050.
"""

__version__ = "1.0.0"
__author__ = "Carbon Budget Team"

from .data_layer import load_global_budget, load_iea_sector_budget, load_demo_industry_data
from .allocator import BudgetAllocator
from .pathway import PathwayGenerator, BudgetOverflowError
from .app import main

__all__ = [
    "load_global_budget",
    "load_iea_sector_budget", 
    "load_demo_industry_data",
    "BudgetAllocator",
    "PathwayGenerator",
    "BudgetOverflowError",
    "main"
]