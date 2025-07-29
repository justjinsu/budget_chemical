#!/usr/bin/env python3
"""
Debug script to identify the root cause of the type comparison error.
"""

import numpy as np
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from pathway_calculator import PathwayCalculator

def test_parameter_contamination():
    """Test various scenarios where parameters might be contaminated with strings."""
    
    print("Testing parameter contamination scenarios...")
    
    years = np.arange(2024, 2051)
    config = {'tolerance': 1e-3, 'max_iter': 60}
    pathway_calc = PathwayCalculator(years, config)
    
    # Base parameters
    base_params = {
        'start_emission': 185000000.0,
        'budget': 1500000000.0,
        'curve_type': 'log',  # Use log curve as it has the most complex logic
        'frontload': 0.6
    }
    
    # Test scenarios where one parameter at a time is a string
    string_contamination_tests = [
        ('start_emission as string', {'start_emission': '185000000.0'}),
        ('budget as string', {'budget': '1500000000.0'}),
        ('frontload as string', {'frontload': '0.6'}),
        ('start_emission as numpy string', {'start_emission': np.str_('185000000.0')}),
        ('budget as numpy string', {'budget': np.str_('1500000000.0')}),
        ('frontload as numpy string', {'frontload': np.str_('0.6')}),
    ]
    
    for test_name, param_override in string_contamination_tests:
        print(f"\nTesting: {test_name}")
        test_params = base_params.copy()
        test_params.update(param_override)
        
        print(f"  Parameters: {test_params}")
        
        try:
            pathway = pathway_calc.build_path(**test_params)
            print(f"  ✓ Success: pathway shape {pathway.shape}, dtype {pathway.dtype}")
            
            # Check if the resulting pathway has any string contamination
            if pathway.dtype.kind not in 'if':  # not integer or float
                print(f"  ! Warning: pathway has non-numeric dtype {pathway.dtype}")
                print(f"  First few values: {pathway[:5]}")
                print(f"  Value types: {[type(x) for x in pathway[:3]]}")
                
        except Exception as e:
            print(f"  ✗ Error: {type(e).__name__}: {e}")
            if "'<' not supported between instances of" in str(e) or "'>' not supported between instances of" in str(e):
                print("  >>> This is the error we're looking for!")
                import traceback
                traceback.print_exc()

def test_numpy_array_creation_with_mixed_types():
    """Test how numpy arrays behave when created with mixed types."""
    
    print("\n" + "="*60)
    print("TESTING NUMPY ARRAY CREATION WITH MIXED TYPES")
    print("="*60)
    
    # Test different ways arrays might get contaminated
    test_cases = [
        ("Pure floats", [1.0, 2.0, 3.0]),
        ("Mixed float/string", [1.0, '2.0', 3.0]),
        ("Mixed int/string", [1, '2', 3]),
        ("Mixed numpy types", [np.float64(1.0), '2.0', np.float64(3.0)]),
    ]
    
    for test_name, values in test_cases:
        print(f"\nTesting: {test_name}")
        try:
            arr = np.array(values)
            print(f"  Array: {arr}")
            print(f"  Dtype: {arr.dtype}")
            
            # Try comparison that might fail
            try:
                result = arr < 2.5
                print(f"  Comparison result: {result}")
            except Exception as e:
                print(f"  Comparison error: {e}")
                
        except Exception as e:
            print(f"  Array creation error: {e}")

def test_pathway_array_assignment():
    """Test different ways pathway arrays might get contaminated during assignment."""
    
    print("\n" + "="*60)
    print("TESTING PATHWAY ARRAY ASSIGNMENT")
    print("="*60)
    
    n_years = 27
    
    # Test normal pathway creation
    print("Normal pathway creation:")
    pathway = np.zeros(n_years, dtype=np.float64)
    pathway[0] = 185000000.0
    pathway[1] = 170000000.0
    print(f"  Pathway dtype: {pathway.dtype}")
    print(f"  First values: {pathway[:3]}")
    
    # Test what happens if we assign string values
    print("\nAssigning string values:")
    pathway_mixed = np.zeros(n_years, dtype=np.float64)
    pathway_mixed[0] = 185000000.0
    try:
        pathway_mixed[1] = '170000000.0'  # This might convert automatically
        print(f"  After string assignment - dtype: {pathway_mixed.dtype}")
        print(f"  First values: {pathway_mixed[:3]}")
        print(f"  Value types: {[type(x) for x in pathway_mixed[:3]]}")
    except Exception as e:
        print(f"  String assignment error: {e}")
    
    # Test object array assignment
    print("\nObject array assignment:")
    pathway_obj = np.zeros(n_years, dtype=object)
    pathway_obj[0] = 185000000.0
    pathway_obj[1] = '170000000.0'
    print(f"  Object array dtype: {pathway_obj.dtype}")
    print(f"  First values: {pathway_obj[:3]}")
    
    # Test comparison on object array
    try:
        result = pathway_obj < 0
        print(f"  Object array comparison: {result}")
    except Exception as e:
        print(f"  Object array comparison error: {e}")
        if "'<' not supported between instances of" in str(e):
            print("  >>> This matches our target error!")

if __name__ == "__main__":
    test_parameter_contamination()
    test_numpy_array_creation_with_mixed_types()
    test_pathway_array_assignment()