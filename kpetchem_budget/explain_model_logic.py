#!/usr/bin/env python3
"""
K-PetChem Carbon Budget Model: Complete Logic Explanation

This script demonstrates the full logic of the Korean Petrochemical Carbon Budget model,
showing how different allocation standards work and the range of results they produce.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from data_layer import load_global_budget, get_korean_shares, load_iea_sector_budget
from allocator import BudgetAllocator
from pathway import PathwayGenerator
from parameter_space import get_budget_line_params

def explain_model_concept():
    """Explain the fundamental concept of the model."""
    print("🏭 K-PETCHEM CARBON BUDGET MODEL LOGIC")
    print("=" * 60)
    print()
    print("📋 FUNDAMENTAL CONCEPT:")
    print("   The model allocates GLOBAL carbon budgets to Korea's petrochemical sector")
    print("   using different fairness principles, then generates emission pathways (2023-2050)")
    print()
    print("🎯 CORE LOGIC:")
    print("   1. START: Global carbon budget (e.g., 400 Gt CO₂e for 1.5°C-50%)")
    print("   2. ALLOCATE: Apply Korea's 'fair share' using different allocation rules")
    print("   3. CONVERT: Korean allocation (Gt CO₂e) → Korean petrochemical budget (Mt CO₂e)")
    print("   4. PATHWAY: Generate emission trajectories that stay within budget")
    print("   5. ANALYZE: Compare pathways and assess feasibility")
    print()

def show_korean_shares():
    """Show Korea's shares under different allocation principles."""
    shares = get_korean_shares()
    
    print("🇰🇷 KOREA'S GLOBAL SHARES (2023 baseline):")
    print("-" * 40)
    
    share_explanations = {
        'population': ('Population Share', 'Korea has ~0.66% of global population'),
        'gdp': ('GDP Share', 'Korea generates ~1.8% of global GDP'),
        'historical_ghg': ('Historical GHG Share', 'Korea contributed ~1.4% of historical emissions'),
        'production': ('Production Share', 'Korea produces ~3% of global petrochemicals')
    }
    
    for key, (name, explanation) in share_explanations.items():
        share_pct = shares[key] * 100
        print(f"  {name:20}: {share_pct:5.2f}% | {explanation}")
    
    print()
    print("💡 FAIRNESS PRINCIPLES:")
    print("   • Population: Equal per-capita emissions rights")
    print("   • GDP: Emissions proportional to economic capacity") 
    print("   • Historical: Responsibility for past emissions")
    print("   • Production: Sector-specific activity allocation")
    print()

def demonstrate_allocation_range():
    """Demonstrate the range of budget allocations."""
    print("📊 BUDGET ALLOCATION DEMONSTRATION")
    print("=" * 50)
    
    # Test different budget scenarios
    budget_scenarios = ['1.5C-67%', '1.5C-50%', '1.7C-50%', '2.0C-67%']
    allocation_methods = ['population', 'gdp', 'historical_ghg', 'iea_sector']
    
    allocator = BudgetAllocator(baseline_emissions=50.0)
    
    results = []
    
    print("🌡️ TESTING BUDGET SCENARIOS:")
    for budget_line in budget_scenarios:
        temp, prob = get_budget_line_params(budget_line)
        print(f"\n📋 {budget_line} (Temperature: {temp}°C, Probability: {int(prob*100)}%)")
        print("   Method                Korean Budget (Mt CO₂e)   Range vs Population")
        print("   " + "-" * 65)
        
        scenario_results = {}
        for method in allocation_methods:
            try:
                budget = allocator.allocate_budget(method, temp=temp, probability=prob)
                scenario_results[method] = budget
                
                # Calculate ratio vs population method
                if method == 'population':
                    pop_budget = budget
                    ratio_text = "baseline"
                else:
                    ratio = budget / pop_budget if 'pop_budget' in locals() else 1.0
                    ratio_text = f"{ratio:.1f}x"
                
                print(f"   {method:20} {budget:10.1f} Mt CO₂e        {ratio_text:>10}")
                
            except Exception as e:
                print(f"   {method:20} ERROR: {str(e)}")
        
        # Calculate range
        if scenario_results:
            min_budget = min(scenario_results.values())
            max_budget = max(scenario_results.values())
            range_mt = max_budget - min_budget
            print(f"\n   📈 RANGE: {min_budget:.1f} - {max_budget:.1f} Mt CO₂e (±{range_mt/2:.1f} Mt)")
        
        results.append({
            'scenario': budget_line,
            'temperature': temp,
            'probability': prob,
            **scenario_results
        })
    
    return pd.DataFrame(results)

def show_pathway_implications(budget_df):
    """Show what the budget allocations mean for emission pathways."""
    print("\n🛤️ PATHWAY IMPLICATIONS")
    print("=" * 30)
    
    baseline_emissions = 50.0  # Mt CO₂e/year in 2023
    
    print(f"📊 Starting Point: {baseline_emissions} Mt CO₂e/year (2023 Korean petrochemical emissions)")
    print()
    
    # Take a representative case
    if not budget_df.empty:
        case = budget_df[budget_df['scenario'] == '1.5C-50%'].iloc[0]
        
        print(f"🎯 EXAMPLE: {case['scenario']} scenario")
        print("-" * 40)
        
        for method in ['population', 'gdp', 'historical_ghg', 'iea_sector']:
            if method in case and pd.notna(case[method]):
                allocated_budget = case[method]
                
                # Calculate what this means for pathways
                generator = PathwayGenerator(
                    baseline_emissions=baseline_emissions,
                    allocated_budget=allocated_budget,
                    start_year=2023,
                    net_zero_year=2050
                )
                
                try:
                    pathway = generator.linear_to_zero()
                    summary = generator.get_pathway_summary(pathway)
                    
                    emissions_2035 = summary['emissions_2035']
                    emissions_2050 = summary['emissions_2050']
                    total_emissions = summary['total_emissions']
                    utilization = summary['budget_utilization_pct']
                    
                    reduction_2035 = (baseline_emissions - emissions_2035) / baseline_emissions * 100
                    
                    print(f"\n  {method.upper()} ALLOCATION:")
                    print(f"    Budget: {allocated_budget:.1f} Mt CO₂e (2023-2050)")
                    print(f"    2035: {emissions_2035:.1f} Mt/year ({reduction_2035:.1f}% reduction)")
                    print(f"    2050: {emissions_2050:.1f} Mt/year (net-zero)")
                    print(f"    Total: {total_emissions:.1f} Mt CO₂e ({utilization:.1f}% of budget)")
                    
                except Exception as e:
                    print(f"  {method.upper()}: Cannot generate pathway - {str(e)}")

def create_comparison_visualization(budget_df):
    """Create a visualization comparing all allocation methods."""
    print("\n📈 CREATING COMPARISON VISUALIZATION...")
    
    if budget_df.empty:
        print("  ❌ No data available for visualization")
        return
    
    # Prepare data for plotting
    methods = ['population', 'gdp', 'historical_ghg', 'iea_sector']
    scenarios = budget_df['scenario'].tolist()
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Budget allocations by scenario
    x = np.arange(len(scenarios))
    width = 0.2
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, method in enumerate(methods):
        values = [budget_df[budget_df['scenario'] == scenario][method].iloc[0] 
                 if not budget_df[budget_df['scenario'] == scenario][method].isna().iloc[0]
                 else 0 for scenario in scenarios]
        
        ax1.bar(x + i * width, values, width, label=method.replace('_', ' ').title(), 
                color=colors[i], alpha=0.8)
    
    ax1.set_xlabel('Budget Scenario')
    ax1.set_ylabel('Korean Budget Allocation (Mt CO₂e)')
    ax1.set_title('Korean Petrochemical Carbon Budget\nby Allocation Method')
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels(scenarios, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Korean shares comparison
    shares = get_korean_shares()
    share_names = [name.replace('_', ' ').title() for name in shares.keys()]
    share_values = [shares[key] * 100 for key in shares.keys()]
    
    ax2.bar(share_names, share_values, color=colors, alpha=0.8)
    ax2.set_ylabel('Korea\'s Global Share (%)')
    ax2.set_title('Korea\'s Global Shares\nby Indicator')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_file = current_dir / 'korean_budget_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✅ Visualization saved to: {output_file}")
    
    plt.show()

def explain_monte_carlo_expansion():
    """Explain the Monte Carlo uncertainty expansion."""
    print("\n🎲 MONTE CARLO UNCERTAINTY EXPANSION")
    print("=" * 45)
    print()
    print("🔍 PURPOSE: The model expands beyond deterministic calculations to explore uncertainty")
    print()
    print("📊 768 DETERMINISTIC CASES:")
    print("   • 4 Budget scenarios (1.5C-67%, 1.5C-50%, 1.7C-50%, 2.0C-67%)")
    print("   • 4 Allocation rules (Population, GDP, Historical GHG, IEA Sector)")
    print("   • 4 Start years (2023, 2025, 2030, 2035)")
    print("   • 3 Net-zero years (2045, 2050, 2055)")
    print("   • 4 Pathway families (Linear, Constant rate, Logistic, IEA proxy)")
    print("   = 4 × 4 × 4 × 3 × 4 = 768 combinations")
    print()
    print("🎯 100 MONTE CARLO SAMPLES PER CASE:")
    print("   • Global budget uncertainty (±10% normal distribution)")
    print("   • Korean production share uncertainty (log-normal)")
    print("   • Pathway shape parameters (triangular distribution)")
    print("   = 768 × 100 = 76,800 total simulations")
    print()
    print("📈 OUTPUTS:")
    print("   • Percentile bands (5th-95th percentile)")
    print("   • Composite visualization with uncertainty ribbons")
    print("   • Milestone uncertainty (2035, 2050 emission ranges)")

def main():
    """Main demonstration function."""
    print("🚀 Starting K-PetChem Model Logic Explanation...\n")
    
    # 1. Explain fundamental concept
    explain_model_concept()
    
    # 2. Show Korean shares
    show_korean_shares()
    
    # 3. Demonstrate allocation range
    budget_df = demonstrate_allocation_range()
    
    # 4. Show pathway implications
    show_pathway_implications(budget_df)
    
    # 5. Create visualization
    create_comparison_visualization(budget_df)
    
    # 6. Explain Monte Carlo expansion
    explain_monte_carlo_expansion()
    
    print("\n" + "=" * 60)
    print("🎉 MODEL LOGIC EXPLANATION COMPLETE")
    print()
    print("💡 KEY INSIGHTS:")
    print("   • Different allocation principles give VERY different budgets")
    print("   • Range can be 2-3x between strictest and most generous")
    print("   • This reflects real policy debates about fairness")
    print("   • Monte Carlo adds uncertainty quantification")
    print("   • Result: Comprehensive range of possible futures")
    print()
    print("🎯 PRACTICAL USE:")
    print("   • Compare allocation principles")
    print("   • Assess pathway feasibility")
    print("   • Quantify uncertainty ranges")
    print("   • Support policy discussions")
    
    return budget_df

if __name__ == "__main__":
    results = main()