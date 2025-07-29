#!/usr/bin/env python3
"""
Run Emission Pathway Comparison Analysis

This script compares three emission pathways:
1. 1.5°C scenario industry paths (Monte Carlo results)
2. 2.0°C scenario industry paths (Monte Carlo results) 
3. Korean government's linear plan (10% by 2030, 80% by 2050)

Usage:
    python run_pathway_analysis.py
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid display issues

from budget_chemical.analysis.pathway_comparison import PathwayComparison

def main():
    print("="*70)
    print("EMISSION PATHWAY COMPARISON ANALYSIS")
    print("="*70)
    
    # Initialize the analyzer
    analyzer = PathwayComparison()
    
    # Load all pathway data
    print("\n1. Loading emission pathway data...")
    analyzer.load_all_pathways()
    
    # Calculate comprehensive statistics
    print("\n2. Calculating pathway statistics...")
    analyzer.calculate_pathway_statistics()
    
    # Print detailed summary to console
    print("\n3. Analysis Results:")
    analyzer.print_summary()
    
    # Create comprehensive visualization
    print("\n4. Creating visualization...")
    viz_path = analyzer.outputs_dir / "pathway_comparison_analysis.png"
    analyzer.create_comparison_visualization(save_path=viz_path)
    print(f"   📊 Visualization saved: {viz_path}")
    
    # Save detailed results to JSON
    print("\n5. Saving detailed results...")
    results_path = "pathway_comparison_results.json"
    analyzer.save_results(results_path)
    print(f"   💾 Results saved: {analyzer.outputs_dir / results_path}")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"📁 Check the '{analyzer.outputs_dir}' directory for:")
    print(f"   • {viz_path.name} (comprehensive visualization)")
    print(f"   • {results_path} (detailed statistics in JSON)")
    print("\nKey Findings Summary:")
    
    # Print key insights
    stats = analyzer.statistics
    if '1.5C' in stats and '2.0C' in stats and 'Korean_Gov' in stats:
        print(f"   • 1.5°C scenario: {stats['1.5C']['reduction_2035_stats']['median']:.1f}% reduction by 2035")
        print(f"   • 2.0°C scenario: {stats['2.0C']['reduction_2035_stats']['median']:.1f}% reduction by 2035") 
        print(f"   • Korean Gov plan: {stats['Korean_Gov']['reduction_2035_pct']:.1f}% reduction by 2035")
        print(f"   • Korean Gov plan is between the 1.5°C and 2.0°C scenarios")

if __name__ == "__main__":
    main()