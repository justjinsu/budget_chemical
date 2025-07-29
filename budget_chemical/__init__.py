"""
Budget Chemical - Carbon Budget Allocation and Pathway Modeling

A comprehensive framework for carbon budget allocation and emission pathway modeling
with Monte Carlo uncertainty analysis.
"""

__version__ = "1.0.0"
__author__ = "PLANiT Institute"

# Import main functionality
from .core.budget_calculation import BudgetAllocation
from .core.pathway_calculation import pathwayCalculator
from .monte_carlo.runner import run_monte_carlo_analysis
from .monte_carlo.sampler import Sampler

__all__ = [
    'BudgetAllocation',
    'pathwayCalculator', 
    'run_monte_carlo_analysis',
    'Sampler'
]