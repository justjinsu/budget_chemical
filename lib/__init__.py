"""
Budget Chemical Library

Core modules for carbon budget allocation and emission pathway modeling.
"""

from .budgetCalculation import BudgetAllocation
from .pathwayCalculation import PathwayCalculator
from .dataAPI import *
from .utils import *

__version__ = "0.1.0"
__author__ = "PLANiT Institute"

__all__ = [
    'BudgetAllocation',
    'PathwayCalculator'
]