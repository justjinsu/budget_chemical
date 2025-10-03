#!/usr/bin/env python3
"""
Frontend Script: Annual Pathway Allocator (Module 2)

Generate annual emission pathways from a given carbon budget.
Can be used standalone or with Module 1 (Budget Calculator).

Usage:
    # Standalone
    python run_pathway_allocator.py --budget 2.5e9 --curve exponential

    # With Module 1 results
    python run_pathway_allocator.py --from-budget-file outputs/budget_samples_*.csv
"""

import sys
import argparse
from pathlib import Path
import logging
import yaml

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from budget_chemical.modules.pathway_allocator import PathwayAllocator
import pandas as pd
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def plot_pathway(result, output_dir):
    """Create visualization of pathway."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    years = result['years']
    pathway = result['pathway']

    # Plot 1: Annual emissions
    ax1.plot(years, pathway / 1e6, 'b-', linewidth=2, label='Annual Emissions')
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Emissions (Mt CO₂/year)')
    ax1.set_title(f'Emission Pathway - {result["curve_type"].title()} Curve')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Cumulative emissions
    cumulative = pd.Series(pathway).cumsum().values
    budget_line = [result['budget_allocated']] * len(years)

    ax2.plot(years, cumulative / 1e9, 'g-', linewidth=2, label='Cumulative Emissions')
    ax2.axhline(y=result['budget_allocated'] / 1e9, color='r', linestyle='--',
               linewidth=2, label=f'Budget ({result["budget_allocated"]/1e9:.2f} Gt)')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Cumulative Emissions (Gt CO₂)')
    ax2.set_title('Cumulative Emissions vs Budget')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.fill_between(years, 0, cumulative / 1e9, alpha=0.3, color='green')

    plt.tight_layout()

    plot_file = output_dir / f'pathway_{result["curve_type"]}.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Plot saved to {plot_file}")


def main():
    """Main entry point."""

    parser = argparse.ArgumentParser(description='Annual Pathway Allocator (Module 2)')
    parser.add_argument('--budget', type=float, help='Carbon budget (tCO2)')
    parser.add_argument('--curve', type=str, default='exponential',
                       choices=['exponential', 'logarithmic', 's_curve', 'linear',
                               'plateau', 'convex', 'early_action', 'delayed_action'],
                       help='Curve type')
    parser.add_argument('--start-year', type=int, default=2024, help='Start year')
    parser.add_argument('--end-year', type=int, default=2050, help='End year')
    parser.add_argument('--start-emission', type=float, default=185e6, help='Start emission (tCO2/year)')
    parser.add_argument('--from-budget-file', type=str, help='Use budget from Module 1 CSV')
    parser.add_argument('--compare-all', action='store_true', help='Compare all curve types')
    parser.add_argument('--output-dir', type=str, default='outputs/pathways', help='Output directory')
    parser.add_argument('--config', type=str, help='Config file (YAML)')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config if provided
    config = {}
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)

    # Initialize allocator
    allocator = PathwayAllocator(
        start_year=args.start_year,
        end_year=args.end_year,
        start_emission=args.start_emission,
        config=config
    )

    # Determine budget
    if args.from_budget_file:
        logger.info(f"Loading budget from {args.from_budget_file}")
        df = pd.read_csv(args.from_budget_file)
        budget = df['industry'].median()  # Use median industry budget
        logger.info(f"Using median industry budget: {budget:.2e} tCO2")
    elif args.budget:
        budget = args.budget
    else:
        logger.error("Must specify --budget or --from-budget-file")
        sys.exit(1)

    # Compare all curves or single curve
    if args.compare_all:
        logger.info(f"Comparing all curve types for budget {budget:.2e} tCO2")

        comparison = allocator.compare_curves(budget)
        print("\n" + "="*80)
        print("CURVE COMPARISON RESULTS")
        print("="*80)
        print(comparison.to_string(index=False))
        print("="*80 + "\n")

        # Save comparison
        comparison_file = output_dir / 'curve_comparison.csv'
        comparison.to_csv(comparison_file, index=False)
        logger.info(f"Comparison saved to {comparison_file}")

        # Generate pathways for all valid curves
        for _, row in comparison.iterrows():
            if row['is_valid']:
                curve_type = row['curve_type']
                result = allocator.allocate_budget(budget, curve_type)

                # Export
                allocator.export_pathway(
                    result,
                    output_dir / f'pathway_{curve_type}.csv',
                    format='csv'
                )
                allocator.export_pathway(
                    result,
                    output_dir / f'pathway_{curve_type}.json',
                    format='json'
                )

                # Plot
                plot_pathway(result, output_dir)

    else:
        # Single curve
        logger.info(f"Generating {args.curve} pathway for budget {budget:.2e} tCO2")

        result = allocator.allocate_budget(budget, args.curve, validate=True)

        # Print summary
        print("\n" + "="*80)
        print(f"PATHWAY ALLOCATION RESULTS - {args.curve.upper()} CURVE")
        print("="*80)
        print(f"Budget Allocated: {result['budget_allocated']:.2e} tCO2")
        print(f"Cumulative Emissions: {result['cumulative_emissions']:.2e} tCO2")
        print(f"Budget Match: {(result['cumulative_emissions']/result['budget_allocated'] - 1)*100:.3f}%")
        print(f"\nValidation: {'✓ VALID' if result['validation']['is_valid'] else '✗ INVALID'}")
        if not result['validation']['is_valid']:
            print(f"Issues: {', '.join(result['validation']['issues'])}")
        print(f"\nStart Emission: {result['pathway'][0]/1e6:.2f} Mt CO₂/year")
        print(f"Final Emission: {result['pathway'][-1]/1e3:.2f} kt CO₂/year")
        print(f"Reduction: {(1 - result['pathway'][-1]/result['pathway'][0])*100:.1f}%")
        print("="*80 + "\n")

        # Export
        allocator.export_pathway(result, output_dir / f'pathway_{args.curve}.csv', format='csv')
        allocator.export_pathway(result, output_dir / f'pathway_{args.curve}.json', format='json')

        # Plot
        plot_pathway(result, output_dir)

    logger.info(f"All results saved to {output_dir.absolute()}")


if __name__ == "__main__":
    main()
