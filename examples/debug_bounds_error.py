#!/usr/bin/env python3
"""
Debug script to test the bounds error in scipy optimize.
"""

import numpy as np
from scipy.optimize import minimize

def test_bounds_with_mixed_types():
    """Test scipy minimize with mixed types in bounds."""
    
    def dummy_objective(x):
        return (x[0] - 1)**2 + (x[1] - 2)**2
    
    # Test with normal float bounds
    print("Testing with normal float bounds...")
    try:
        bounds = [(0.001, 1.0), (0.0, 185000000.0)]
        result = minimize(dummy_objective, x0=[0.05, 100000.0], bounds=bounds, method='L-BFGS-B')
        print(f"✓ Success with float bounds: {result.x}")
    except Exception as e:
        print(f"✗ Error with float bounds: {e}")
    
    # Test with string in bounds (this should cause the error)
    print("Testing with string in bounds...")
    try:
        bounds = [(0.001, 1.0), (0.0, '185000000.0')]  # String upper bound
        result = minimize(dummy_objective, x0=[0.05, 100000.0], bounds=bounds, method='L-BFGS-B')
        print(f"✓ Unexpected success with string bounds: {result.x}")
    except Exception as e:
        print(f"✗ Expected error with string bounds: {e}")
    
    # Test with numpy.float64 in bounds
    print("Testing with numpy.float64 in bounds...")
    try:
        bounds = [(0.001, 1.0), (0.0, np.float64(185000000.0))]
        result = minimize(dummy_objective, x0=[0.05, 100000.0], bounds=bounds, method='L-BFGS-B')
        print(f"✓ Success with numpy.float64 bounds: {result.x}")
    except Exception as e:
        print(f"✗ Error with numpy.float64 bounds: {e}")
    
    # Test with numpy.float64 converted from string 
    print("Testing with numpy.float64 from string conversion...")
    try:
        start_emission_str = '185000000.0'
        start_emission_np = np.float64(start_emission_str)
        bounds = [(0.001, 1.0), (0.0, start_emission_np)]
        result = minimize(dummy_objective, x0=[0.05, 100000.0], bounds=bounds, method='L-BFGS-B')
        print(f"✓ Success with converted numpy.float64: {result.x}")
    except Exception as e:
        print(f"✗ Error with converted numpy.float64: {e}")

def test_direct_comparison():
    """Test direct comparison operations that might cause the error."""
    
    print("\nTesting direct comparisons...")
    
    # Float vs string comparison
    try:
        result = 1.0 < '2.0'
        print(f"Float < string: {result}")
    except Exception as e:
        print(f"Error in float < string: {e}")
    
    # numpy.float64 vs string comparison  
    try:
        result = np.float64(1.0) < '2.0'
        print(f"numpy.float64 < string: {result}")
    except Exception as e:
        print(f"Error in numpy.float64 < string: {e}")
    
    # Array comparison with mixed types
    try:
        arr1 = np.array([1.0, 2.0])
        arr2 = np.array(['1.5', '2.5'])
        result = arr1 < arr2
        print(f"Array comparison result: {result}")
    except Exception as e:
        print(f"Error in array comparison: {e}")

if __name__ == "__main__":
    test_bounds_with_mixed_types()
    test_direct_comparison()