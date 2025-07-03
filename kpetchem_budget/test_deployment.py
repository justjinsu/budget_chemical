"""
Test script to verify deployment works correctly.
Run this to test before deploying to Streamlit Cloud.
"""

import sys
import os

def test_imports():
    """Test all imports work."""
    print("Testing imports...")
    
    try:
        import streamlit as st
        print("✓ streamlit")
        
        import pandas as pd
        print("✓ pandas")
        
        import numpy as np
        print("✓ numpy")
        
        import matplotlib.pyplot as plt
        print("✓ matplotlib")
        
        import scipy
        print("✓ scipy")
        
        # Test our modules
        import data_layer
        print("✓ data_layer")
        
        import allocator
        print("✓ allocator")
        
        import pathway
        print("✓ pathway")
        
        # Test main app
        import app
        print("✓ app")
        
        print("\n🎉 All imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_functionality():
    """Test basic functionality."""
    print("\nTesting functionality...")
    
    try:
        from data_layer import load_global_budget, load_iea_sector_budget
        from allocator import BudgetAllocator
        from pathway import PathwayGenerator
        
        # Test data loading
        budget_df = load_global_budget()
        iea_budget = load_iea_sector_budget()
        print(f"✓ Data loaded: {len(budget_df)} budget rows")
        
        # Test allocation
        allocator = BudgetAllocator(50.0)
        budget = allocator.allocate_budget('population', temp=1.5, probability=0.5)
        print(f"✓ Budget allocated: {budget:.1f} Mt")
        
        # Test pathway
        generator = PathwayGenerator(50.0, budget)
        pathway = generator.linear_to_zero()
        print(f"✓ Pathway generated: {len(pathway)} points")
        
        print("\n🎉 All functionality works!")
        return True
        
    except Exception as e:
        print(f"❌ Functionality error: {e}")
        return False

if __name__ == "__main__":
    print("DEPLOYMENT TEST")
    print("=" * 30)
    
    imports_ok = test_imports()
    functionality_ok = test_functionality()
    
    if imports_ok and functionality_ok:
        print("\n✅ READY FOR DEPLOYMENT!")
        print("\nDeploy with:")
        print("- Repository: your-username/budget_chemical")
        print("- Main file path: budget_chemical/kpetchem_budget/app.py")
    else:
        print("\n❌ NOT READY - Fix errors above first")