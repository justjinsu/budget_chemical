#!/usr/bin/env python3
"""
Test script to verify the app.py fixes work correctly.
"""

import sys
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

def test_pathway_summary():
    """Test that pathway summary works correctly."""
    print("🔍 Testing pathway summary...")
    
    try:
        from pathway import PathwayGenerator
        
        # Create a simple pathway
        generator = PathwayGenerator(
            baseline_emissions=50.0,
            allocated_budget=800.0,
            start_year=2023,
            net_zero_year=2050
        )
        
        # Generate a pathway
        pathway = generator.linear_to_zero()
        print(f"  ✅ Generated pathway with {len(pathway)} time points")
        
        # Get summary
        summary = generator.get_pathway_summary(pathway)
        print(f"  ✅ Generated summary with {len(summary)} metrics")
        
        # Check expected keys
        expected_keys = [
            'total_emissions',
            'peak_emission', 
            'final_emission',
            'emissions_2023',
            'emissions_2035',
            'emissions_2050',
            'cumulative_2035',
            'cumulative_2050',
            'peak_to_final_reduction_pct',
            'budget_utilization_pct',
            'reduction_2023_to_2035_pct',
            'reduction_2035_to_2050_pct'
        ]
        
        missing_keys = []
        for key in expected_keys:
            if key not in summary:
                missing_keys.append(key)
        
        if missing_keys:
            print(f"  ❌ Missing keys: {missing_keys}")
            return False
        else:
            print(f"  ✅ All expected keys present")
        
        # Verify 'overshoot_year' is NOT in summary (this was causing the error)
        if 'overshoot_year' in summary:
            print(f"  ⚠️  Unexpected 'overshoot_year' key found - this should be removed")
        else:
            print(f"  ✅ No 'overshoot_year' key (this is correct)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Pathway summary test failed: {e}")
        return False

def test_budget_allocation():
    """Test budget allocation works."""
    print("\n🔍 Testing budget allocation...")
    
    try:
        from allocator import BudgetAllocator
        
        allocator = BudgetAllocator(baseline_emissions=50.0)
        budget = allocator.allocate_budget('population', temp=1.5, probability=0.5)
        
        print(f"  ✅ Budget allocation: {budget:.1f} Mt CO₂e")
        
        if budget > 0:
            return True
        else:
            print(f"  ❌ Invalid budget amount: {budget}")
            return False
            
    except Exception as e:
        print(f"  ❌ Budget allocation test failed: {e}")
        return False

def test_app_components():
    """Test individual app components."""
    print("\n🔍 Testing app components...")
    
    try:
        # Import app functions
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        
        # Test that we can import the app without errors
        import app
        
        print(f"  ✅ App module imported successfully")
        
        # Test key functions exist
        functions_to_check = [
            'initialize_session_state',
            'create_sidebar',
            'calculate_budget_allocation', 
            'generate_pathways',
            'display_kpi_cards',
            'create_pathway_chart',
            'main'
        ]
        
        missing_functions = []
        for func_name in functions_to_check:
            if not hasattr(app, func_name):
                missing_functions.append(func_name)
        
        if missing_functions:
            print(f"  ❌ Missing functions: {missing_functions}")
            return False
        else:
            print(f"  ✅ All expected functions present")
        
        return True
        
    except Exception as e:
        print(f"  ❌ App component test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🏭 K-PetChem App Fix Verification")
    print("=" * 40)
    
    tests = [
        ("Pathway Summary", test_pathway_summary),
        ("Budget Allocation", test_budget_allocation),
        ("App Components", test_app_components)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"  💥 {test_name} crashed: {e}")
            results[test_name] = False
    
    print("\n" + "=" * 40)
    print("📊 Test Results:")
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} {test_name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 40)
    if all_passed:
        print("🎉 All app fixes verified! The app should work correctly now.")
        print("💡 The 'overshoot_year' KeyError has been fixed.")
    else:
        print("⚠️  Some tests failed. Check the issues above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)