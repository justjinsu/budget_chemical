"""
Budget Chemical Modules

Module 1: Korean Budget Calculator - Calculate Korea's carbon budget
Module 2: Pathway Allocator - Allocate budget across years using various curves
"""

from .budget_calculator import KoreaBudgetCalculator
from .pathway_allocator import PathwayAllocator

__all__ = ['KoreaBudgetCalculator', 'PathwayAllocator']
