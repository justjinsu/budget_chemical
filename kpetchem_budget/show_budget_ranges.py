#!/usr/bin/env python3
"""
Simple demonstration of Korean carbon budget allocation ranges.
"""

import pandas as pd
import sys
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from data_layer import get_korean_shares
from allocator import BudgetAllocator
from parameter_space import get_budget_line_params

def main():
    print("🏭 K-PETCHEM CARBON BUDGET MODEL - CORE LOGIC")
    print("=" * 55)
    
    # 1. Show Korean shares
    print("\n🇰🇷 KOREA'S GLOBAL SHARES:")
    shares = get_korean_shares()
    for key, value in shares.items():
        name = key.replace('_', ' ').title()
        print(f"   {name:20}: {value*100:5.2f}%")
    
    # 2. Show budget range
    print("\n📊 BUDGET ALLOCATION RANGE:")
    print("   Scenario      Population   GDP      Historical  IEA Sector")
    print("   " + "-" * 60)
    
    allocator = BudgetAllocator(baseline_emissions=50.0)
    scenarios = ['1.5C-67%', '1.5C-50%', '1.7C-50%', '2.0C-67%']
    methods = ['population', 'gdp', 'historical_ghg', 'iea_sector']
    
    all_budgets = []
    
    for scenario in scenarios:
        temp, prob = get_budget_line_params(scenario)
        budgets = []
        
        for method in methods:
            try:
                budget = allocator.allocate_budget(method, temp=temp, probability=prob)
                budgets.append(budget)
            except:
                budgets.append(0)
        
        all_budgets.extend(budgets)
        print(f"   {scenario:12} {budgets[0]:8.0f}   {budgets[1]:8.0f}   {budgets[2]:8.0f}     {budgets[3]:8.0f}")
    
    # 3. Show range analysis
    valid_budgets = [b for b in all_budgets if b > 0]
    if valid_budgets:
        min_budget = min(valid_budgets)
        max_budget = max(valid_budgets)
        range_budget = max_budget - min_budget
        
        print(f"\n📈 RANGE ANALYSIS:")
        print(f"   Minimum: {min_budget:8.0f} Mt CO₂e")
        print(f"   Maximum: {max_budget:8.0f} Mt CO₂e") 
        print(f"   Range:   {range_budget:8.0f} Mt CO₂e (±{range_budget/2:.0f})")
        print(f"   Ratio:   {max_budget/min_budget:.1f}x difference")
    
    print(f"\n💡 KEY INSIGHTS:")
    print(f"   • Different allocation principles = Different budgets")
    print(f"   • Range reflects fairness debates in climate policy")
    print(f"   • Higher temperature targets = More budget available")
    print(f"   • Model quantifies these trade-offs systematically")

if __name__ == "__main__":
    main()