#!/usr/bin/env python3
"""
Debug script to test the full workflow that leads to the error.
"""

import numpy as np
import sys
import yaml
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'lib'))

from pathway_calculator import PathwayCalculator
from budgetCalculation import BudgetAllocation
from mc_sampler import Sampler

def load_config():
    """Load the actual configuration file."""
    config_path = Path(__file__).parent / 'mc_config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def test_full_workflow():
    """Test the complete workflow that might cause the error."""
    
    print("Loading configuration...")
    config = load_config()
    
    # Create years array 
    years = np.arange(config['timeline']['start_year'], config['timeline']['end_year'] + 1)
    print(f"Years: {years[0]} to {years[-1]} ({len(years)} years)")
    
    # Initialize components exactly like mc_runner.py
    sampler = Sampler(config)
    pathway_calc = PathwayCalculator(years, config.get('solver', {}))
    budget_allocator = BudgetAllocation(config=config)
    
    # Extract parameters
    industry_fraction = config['industry_fraction']
    petrochem_fraction = config['petrochem_fraction']
    
    print(f"Industry fraction: {industry_fraction} (type: {type(industry_fraction)})")
    print(f"Petrochem fraction: {petrochem_fraction} (type: {type(petrochem_fraction)})")
    
    # Sample parameters (just 1 draw for debugging)
    print("\nSampling parameters...")
    samples = sampler.sample_all(1)
    
    print("Sample types:")
    for key, value in samples.items():
        if hasattr(value, 'dtype'):
            print(f"  {key}: {type(value)} with dtype {value.dtype}")
        else:
            print(f"  {key}: {type(value)}")
    
    # Calculate budgets
    print("\nCalculating budgets...")
    industry_budgets, petrochem_budgets = budget_allocator.calculate_sector_budgets_batch(
        samples['global_budgets'],
        samples['responsibility_shares'],
        samples['capability_shares'],
        samples['equality_shares'],
        samples['weights'],
        industry_fraction,
        petrochem_fraction
    )
    
    print(f"Industry budget: {industry_budgets[0]} (type: {type(industry_budgets[0])})")
    print(f"Petrochem budget: {petrochem_budgets[0]} (type: {type(petrochem_budgets[0])})")
    
    # Get base emissions
    print("\nGetting base emissions...")
    industry_base_emission = budget_allocator.get_base_emissions('industry')
    petrochem_base_emission = budget_allocator.get_base_emissions('petrochem')
    
    print(f"Industry base emission: {industry_base_emission} (type: {type(industry_base_emission)})")
    print(f"Petrochem base emission: {petrochem_base_emission} (type: {type(petrochem_base_emission)})")
    
    # Extract specific parameters for pathway generation
    curve_type = samples['curve_types'][0]
    frontload = samples['lambdas'][0]
    
    print(f"Curve type: '{curve_type}' (type: {type(curve_type)})")
    print(f"Frontload: {frontload} (type: {type(frontload)})")
    
    # Now test pathway generation exactly like mc_runner.py
    print("\nTesting pathway generation...")
    
    try:
        print("Building industry pathway...")
        industry_path = pathway_calc.build_path(
            start_emission=industry_base_emission,
            budget=industry_budgets[0],
            curve_type=curve_type,
            frontload=frontload
        )
        print(f"✓ Industry pathway successful: shape {industry_path.shape}")
        
    except Exception as e:
        print(f"✗ Industry pathway failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        print("Building petrochem pathway...")
        petrochem_path = pathway_calc.build_path(
            start_emission=petrochem_base_emission,
            budget=petrochem_budgets[0],
            curve_type=curve_type,
            frontload=frontload
        )
        print(f"✓ Petrochem pathway successful: shape {petrochem_path.shape}")
        
    except Exception as e:
        print(f"✗ Petrochem pathway failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

def test_specific_values():
    """Test with specific values that might cause issues."""
    
    print("\n" + "="*60)
    print("TESTING WITH SPECIFIC PROBLEMATIC VALUES")
    print("="*60)
    
    config = load_config()
    years = np.arange(config['timeline']['start_year'], config['timeline']['end_year'] + 1)
    pathway_calc = PathwayCalculator(years, config.get('solver', {}))
    
    # Test various data type combinations that might cause the error
    test_cases = [
        {
            'name': 'All strings',
            'start_emission': '185000000.0',
            'budget': '1500000000.0',
            'curve_type': 'linear',
            'frontload': '0.6'
        },
        {
            'name': 'Mixed types - string budget',
            'start_emission': 185000000.0,
            'budget': '1500000000.0',
            'curve_type': 'linear',
            'frontload': 0.6
        },
        {
            'name': 'Mixed types - string start_emission',
            'start_emission': '185000000.0',
            'budget': 1500000000.0,
            'curve_type': 'linear',
            'frontload': 0.6
        }
    ]
    
    for test_case in test_cases:
        print(f"\nTesting: {test_case['name']}")
        try:
            pathway = pathway_calc.build_path(**{k: v for k, v in test_case.items() if k != 'name'})
            print(f"  ✓ Success")
        except Exception as e:
            print(f"  ✗ Error: {type(e).__name__}: {e}")
            if "'<' not supported between instances of 'float' and 'str'" in str(e):
                print("  >>> This is the error we're looking for!")

if __name__ == "__main__":
    test_full_workflow()
    test_specific_values()