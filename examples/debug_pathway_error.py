#!/usr/bin/env python3
"""
Debug script to reproduce the pathway calculator error.
"""

import numpy as np
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'lib'))

from lib.pathway_calculator import PathwayCalculator

def test_pathway_calculator():
    """Test pathway calculator with the problematic parameters."""
    
    # Create years array (2024-2050)
    years = np.arange(2024, 2051)
    print(f"Years array: {years[0]} to {years[-1]} ({len(years)} years)")
    
    # Initialize pathway calculator
    config = {'tolerance': 1e-3, 'max_iter': 60}
    pathway_calc = PathwayCalculator(years, config)
    
    # Test parameters that are causing the error
    start_emission = 185000000.0  # float
    budget = 1500000000.0  # float (example budget)
    curve_type = 'linear'  # string
    frontload = np.float64(0.6)  # numpy.float64
    
    print(f"Testing with parameters:")
    print(f"  start_emission: {start_emission} (type: {type(start_emission)})")
    print(f"  budget: {budget} (type: {type(budget)})")
    print(f"  curve_type: '{curve_type}' (type: {type(curve_type)})")
    print(f"  frontload: {frontload} (type: {type(frontload)})")
    print()
    
    # Test each curve type
    curve_types = ['linear', 'exp', 'log']
    
    for curve in curve_types:
        print(f"Testing {curve} curve...")
        try:
            pathway = pathway_calc.build_path(
                start_emission=start_emission,
                budget=budget,
                curve_type=curve,
                frontload=frontload
            )
            print(f"  ✓ Success: pathway shape {pathway.shape}")
            print(f"  First 3 values: {pathway[:3]}")
            print(f"  Last 3 values: {pathway[-3:]}")
            
        except Exception as e:
            print(f"  ✗ Error: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
        print()
    
    # Test with different parameter types to isolate the issue
    print("Testing parameter type variations...")
    
    # Test with pure Python float instead of numpy.float64
    print("Testing with pure Python float for frontload...")
    try:
        pathway = pathway_calc.build_path(
            start_emission=start_emission,
            budget=budget,
            curve_type='linear',
            frontload=float(frontload)  # Convert to pure Python float
        )
        print("  ✓ Success with Python float")
    except Exception as e:
        print(f"  ✗ Error with Python float: {e}")
    
    # Test with string values that might cause comparison issues
    print("Testing with potential string contamination...")
    try:
        # This should definitely fail if there's a string comparison issue
        pathway = pathway_calc.build_path(
            start_emission="185000000.0",  # String instead of float
            budget=budget,
            curve_type='linear',
            frontload=frontload
        )
        print("  ✓ Unexpected success with string start_emission")
    except Exception as e:
        print(f"  ✗ Expected error with string start_emission: {e}")

if __name__ == "__main__":
    test_pathway_calculator()