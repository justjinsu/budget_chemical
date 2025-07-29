#!/usr/bin/env python3
"""
Simple test script to reproduce the pathway calculator error.

This script attempts to reproduce the error: 
"'<' not supported between instances of 'float' and 'str'"

Usage: python reproduce_error.py
"""

import numpy as np
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'lib'))

from lib.pathway_calculator import PathwayCalculator

def reproduce_error():
    """
    Attempt to reproduce the comparison error by forcing string contamination
    in the pathway calculation process.
    """
    
    print("Attempting to reproduce the pathway calculator error...")
    print("="*60)
    
    # Setup
    years = np.arange(2024, 2051)
    config = {'tolerance': 1e-3, 'max_iter': 60}
    pathway_calc = PathwayCalculator(years, config)
    
    # Test parameters - these match your error description
    start_emission = 185000000.0  # float
    budget = 1500000000.0  # float (example budget)
    curve_type = 'linear'  # string
    frontload = np.float64(0.6)  # numpy.float64
    
    print(f"Test parameters:")
    print(f"  start_emission: {start_emission} ({type(start_emission)})")
    print(f"  budget: {budget} ({type(budget)})")
    print(f"  curve_type: '{curve_type}' ({type(curve_type)})")
    print(f"  frontload: {frontload} ({type(frontload)})")
    print()
    
    # Method 1: Direct test with normal parameters
    print("Method 1: Testing with normal parameters...")
    try:
        pathway = pathway_calc.build_path(
            start_emission=start_emission,
            budget=budget,
            curve_type=curve_type,
            frontload=frontload
        )
        print(f"  ✓ Success: pathway shape {pathway.shape}, dtype {pathway.dtype}")
        
        # Test validation (this is where the error might occur)
        validation = pathway_calc.validate_pathway(pathway, budget)
        print(f"  ✓ Validation success: {validation['is_valid']}")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        if "'<' not supported between instances of" in str(e):
            print("  >>> This is the target error!")
            return True
    
    # Method 2: Force string contamination by monkey-patching
    print("\nMethod 2: Forcing string contamination...")
    try:
        # Save original method
        original_zeros = np.zeros
        
        # Create a contaminated zeros function
        def contaminated_zeros(*args, **kwargs):
            arr = original_zeros(*args, **kwargs)
            if len(arr) > 10:  # Only contaminate large arrays (pathways)
                arr = np.array(['0.0'] * len(arr), dtype=object)  # Force object array with strings
            return arr
        
        # Monkey patch
        np.zeros = contaminated_zeros
        
        try:
            pathway = pathway_calc.build_path(
                start_emission=start_emission,
                budget=budget,
                curve_type=curve_type,
                frontload=frontload
            )
            print(f"  ✓ Pathway creation success: shape {pathway.shape}, dtype {pathway.dtype}")
            
            # This should fail when trying to compare
            validation = pathway_calc.validate_pathway(pathway, budget)
            print(f"  ✓ Unexpected validation success: {validation}")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            if "'<' not supported between instances of" in str(e):
                print("  >>> Successfully reproduced the target error!")
                return True
        finally:
            # Restore original
            np.zeros = original_zeros
    
    except Exception as e:
        print(f"  ✗ Monkey-patch error: {e}")
    
    # Method 3: Test all curve types with extreme parameters
    print("\nMethod 3: Testing all curve types with extreme parameters...")
    extreme_test_cases = [
        ('linear', 0.0, 'zero budget'),
        ('exp', -1000.0, 'negative budget'), 
        ('log', 1e-10, 'tiny budget'),
        ('log', 1e20, 'huge budget')
    ]
    
    for curve, test_budget, description in extreme_test_cases:
        print(f"  Testing {curve} with {description}...")
        try:
            pathway = pathway_calc.build_path(
                start_emission=start_emission,
                budget=test_budget,
                curve_type=curve,
                frontload=frontload
            )
            validation = pathway_calc.validate_pathway(pathway, test_budget)
            print(f"    ✓ Success")
            
        except Exception as e:
            print(f"    ✗ Error: {e}")
            if "'<' not supported between instances of" in str(e):
                print("    >>> Found the target error!")
                return True
    
    print("\nConclusion: Could not reproduce the exact error with these tests.")
    print("The error likely occurs in a specific combination of input parameters")
    print("or data types that we haven't tested yet.")
    
    return False

if __name__ == "__main__":
    success = reproduce_error()
    if success:
        print("\n🎯 Successfully reproduced the error!")
    else:
        print("\n❌ Could not reproduce the error.")
        print("\nTo help debug further, please provide:")
        print("1. The exact configuration file being used")
        print("2. The specific input data that causes the error")
        print("3. The complete error traceback")