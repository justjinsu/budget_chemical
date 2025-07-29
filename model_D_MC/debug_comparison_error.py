#!/usr/bin/env python3
"""
Debug script to test specific comparison operations that might cause the error.
"""

import numpy as np
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from pathway_calculator import PathwayCalculator

def test_array_comparison_with_strings():
    """Test array comparison operations with mixed types."""
    
    print("Testing array comparison operations...")
    
    # Test normal numeric array comparison
    try:
        pathway = np.array([1.0, 2.0, -0.5, 3.0])
        negative_count = int((pathway < 0).sum())
        print(f"✓ Normal array comparison: {negative_count} negative values")
    except Exception as e:
        print(f"✗ Error in normal array comparison: {e}")
    
    # Test array with string elements
    try:
        pathway_mixed = np.array([1.0, '2.0', -0.5, '3.0'])
        print(f"Mixed array dtype: {pathway_mixed.dtype}")
        negative_count = int((pathway_mixed < 0).sum())
        print(f"Mixed array comparison result: {negative_count}")
    except Exception as e:
        print(f"✗ Error in mixed array comparison: {e}")
        if "'<' not supported between instances of" in str(e):
            print("  >>> This could be our error!")
    
    # Test object array with mixed types
    try:
        pathway_obj = np.array([1.0, '2.0', -0.5, '3.0'], dtype=object)
        print(f"Object array dtype: {pathway_obj.dtype}")
        negative_count = int((pathway_obj < 0).sum())
        print(f"Object array comparison result: {negative_count}")
    except Exception as e:
        print(f"✗ Error in object array comparison: {e}")
        if "'<' not supported between instances of" in str(e):
            print("  >>> This could be our error!")

def test_max_comparison_with_strings():
    """Test max() comparison with mixed types."""
    
    print("\nTesting max() comparison operations...")
    
    # Test normal max comparison
    try:
        result = max(0.0, 1.5)
        print(f"✓ Normal max comparison: {result}")
    except Exception as e:
        print(f"✗ Error in normal max comparison: {e}")
    
    # Test max with string
    try:
        result = max(0.0, '1.5')
        print(f"Max with string result: {result}")
    except Exception as e:
        print(f"✗ Error in max with string: {e}")
        if "'<' not supported between instances of" in str(e):
            print("  >>> This could be our error!")
    
    # Test max with numpy string
    try:
        result = max(0.0, np.str_('1.5'))
        print(f"Max with numpy string result: {result}")
    except Exception as e:
        print(f"✗ Error in max with numpy string: {e}")
        if "'<' not supported between instances of" in str(e):
            print("  >>> This could be our error!")

def test_corrupted_pathway_scenario():
    """Test what happens if pathway calculation produces mixed types."""
    
    print("\nTesting corrupted pathway scenario...")
    
    # Create a pathway calculator
    years = np.arange(2024, 2051)
    config = {'tolerance': 1e-3, 'max_iter': 60}
    pathway_calc = PathwayCalculator(years, config)
    
    # Manually create a corrupted pathway array (this simulates what might go wrong)
    corrupted_pathway = np.array([185000000.0, 170000000.0, '155000000.0', 140000000.0])
    
    try:
        # This should trigger an error when validate_pathway tries to compare
        validation = pathway_calc.validate_pathway(corrupted_pathway, 1500000000.0)
        print(f"Validation unexpectedly succeeded: {validation}")
    except Exception as e:
        print(f"✗ Validation error as expected: {e}")
        if "'<' not supported between instances of" in str(e):
            print("  >>> This is the error we're looking for!")

def test_edge_cases_in_solvers():
    """Test edge cases that might cause type issues in solver methods."""
    
    print("\nTesting edge cases in solver methods...")
    
    years = np.arange(2024, 2051)
    config = {'tolerance': 1e-3, 'max_iter': 60}
    pathway_calc = PathwayCalculator(years, config)
    
    # Test with extreme values that might cause numerical issues
    test_cases = [
        {
            'name': 'Very small budget',
            'start_emission': 185000000.0,
            'budget': 1.0,  # Extremely small budget
            'curve_type': 'log',
            'frontload': 0.5
        },
        {
            'name': 'Zero budget',
            'start_emission': 185000000.0,
            'budget': 0.0,
            'curve_type': 'log',
            'frontload': 0.5
        },
        {
            'name': 'Negative budget',
            'start_emission': 185000000.0,
            'budget': -1000000.0,
            'curve_type': 'log',
            'frontload': 0.5
        }
    ]
    
    for test_case in test_cases:
        print(f"Testing: {test_case['name']}")
        try:
            pathway = pathway_calc.build_path(**{k: v for k, v in test_case.items() if k != 'name'})
            print(f"  ✓ Success: pathway shape {pathway.shape}, dtype {pathway.dtype}")
            # Check if pathway contains any non-numeric values
            if pathway.dtype.kind not in 'if':  # not integer or float
                print(f"  ! Warning: unexpected dtype {pathway.dtype}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            if "'<' not supported between instances of" in str(e):
                print("  >>> This is the error we're looking for!")

if __name__ == "__main__":
    test_array_comparison_with_strings()
    test_max_comparison_with_strings()  
    test_corrupted_pathway_scenario()
    test_edge_cases_in_solvers()