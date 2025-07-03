#!/usr/bin/env python3
"""
Test script to verify all imports work correctly for K-PetChem toolkit.
"""

import sys
import os
from pathlib import Path

def test_basic_imports():
    """Test basic Python module imports."""
    print("🔍 Testing basic imports...")
    
    basic_modules = {
        'pandas': 'pd',
        'numpy': 'np', 
        'matplotlib.pyplot': 'plt',
        'scipy': None,
        'streamlit': 'st'
    }
    
    for module, alias in basic_modules.items():
        try:
            if alias:
                exec(f"import {module} as {alias}")
            else:
                exec(f"import {module}")
            print(f"  ✅ {module}")
        except ImportError as e:
            print(f"  ❌ {module}: {e}")
            return False
    
    return True

def test_kpetchem_imports():
    """Test K-PetChem specific module imports."""
    print("\n🔍 Testing K-PetChem module imports...")
    
    # Add current directory to path
    current_dir = Path(__file__).parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    
    kpetchem_modules = [
        'data_layer',
        'parameter_space', 
        'pathway',
        'simulator',
        'datastore',
        'allocator'
    ]
    
    success_count = 0
    for module in kpetchem_modules:
        try:
            __import__(module)
            print(f"  ✅ {module}")
            success_count += 1
        except ImportError as e:
            print(f"  ❌ {module}: {e}")
    
    return success_count == len(kpetchem_modules)

def test_advanced_imports():
    """Test advanced functionality imports."""
    print("\n🔍 Testing advanced functionality...")
    
    try:
        from parameter_space import ParameterGrid, MonteCarloSampler
        grid = ParameterGrid()
        print(f"  ✅ ParameterGrid: {grid.total_cases} cases")
        
        sampler = MonteCarloSampler(n_samples=10)
        print(f"  ✅ MonteCarloSampler: {sampler.n_samples} samples")
        
    except Exception as e:
        print(f"  ❌ Advanced parameter space: {e}")
        return False
    
    try:
        from simulator import HighPerformanceSimulator
        simulator = HighPerformanceSimulator(n_workers=2)
        print(f"  ✅ HighPerformanceSimulator: {simulator.n_workers} workers")
        
    except Exception as e:
        print(f"  ❌ High performance simulator: {e}")
        return False
    
    try:
        from datastore import SimulationDataStore
        store = SimulationDataStore()
        print(f"  ✅ SimulationDataStore")
        
    except Exception as e:
        print(f"  ❌ Simulation datastore: {e}")
        return False
    
    return True

def test_file_structure():
    """Test that required files exist."""
    print("\n🔍 Testing file structure...")
    
    required_files = [
        'data_layer.py',
        'parameter_space.py',
        'pathway.py', 
        'simulator.py',
        'datastore.py',
        'allocator.py',
        'dashboard/app.py',
        'dashboard/app_upgraded.py',
        'sample_data/kpetchem_demo.csv'
    ]
    
    missing_files = []
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} (missing)")
            missing_files.append(file_path)
    
    return len(missing_files) == 0

def test_sample_functionality():
    """Test basic functionality works."""
    print("\n🔍 Testing sample functionality...")
    
    try:
        # Test data loading
        from data_layer import load_global_budget, get_timeline_years
        budget_data = load_global_budget()
        timeline = get_timeline_years()
        print(f"  ✅ Data loading: {len(budget_data)} budget scenarios, {len(timeline)} years")
        
        # Test parameter space
        from parameter_space import ParameterGrid
        grid = ParameterGrid()
        cases = list(grid.generate_cases())
        print(f"  ✅ Parameter space: {len(cases)} cases generated")
        
        # Test pathway generation
        from pathway import PathwayGenerator
        generator = PathwayGenerator(50.0, 800.0)
        pathway = generator.linear_to_zero()
        print(f"  ✅ Pathway generation: {len(pathway)} time points")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Sample functionality failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🏭 K-PetChem Carbon Budget Toolkit - Import Test")
    print("=" * 55)
    
    print(f"📁 Current directory: {Path.cwd()}")
    print(f"🐍 Python version: {sys.version}")
    
    tests = [
        ("Basic imports", test_basic_imports),
        ("File structure", test_file_structure), 
        ("K-PetChem imports", test_kpetchem_imports),
        ("Advanced imports", test_advanced_imports),
        ("Sample functionality", test_sample_functionality)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"  💥 {test_name} crashed: {e}")
            results[test_name] = False
    
    print("\n" + "=" * 55)
    print("📊 Test Results:")
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} {test_name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 55)
    if all_passed:
        print("🎉 All tests passed! The toolkit should work correctly.")
        print("💡 You can now run: python run_dashboard.py")
    else:
        print("⚠️  Some tests failed. Check the issues above.")
        print("💡 See TROUBLESHOOTING.md for help.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)