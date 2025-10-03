#!/usr/bin/env python3
"""
Frontend Script: Korean Carbon Budget Calculator

This script provides a clean interface to Module 1 (Budget Calculator).
All backend logic is encapsulated in the KoreaBudgetCalculator class.

Usage:
    python run_budget_calculator.py config/mc_config.yaml
"""

import sys
import yaml
from pathlib import Path
import logging

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from budget_chemical.modules.budget_calculator import KoreaBudgetCalculator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for budget calculator."""

    # Check command line arguments
    if len(sys.argv) != 2:
        print("Usage: python run_budget_calculator.py <config_file>")
        print("Example: python run_budget_calculator.py config/mc_config.yaml")
        sys.exit(1)

    config_path = sys.argv[1]

    # Load configuration
    logger.info(f"Loading configuration from {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Create output directory
    output_dir = Path(config.get('output_dir', 'outputs/budget_only'))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize budget calculator (backend class)
    logger.info("Initializing Korean Budget Calculator...")
    calculator = KoreaBudgetCalculator(config)

    # Run Monte Carlo simulation
    n_draws = config.get('n_draws', 1000)
    logger.info(f"Running Monte Carlo simulation with {n_draws} draws...")

    results = calculator.run_monte_carlo(
        n_draws=n_draws,
        scenario_mode='mixed'
    )

    # Print summary to console
    print("\n" + calculator.get_summary())

    # Export results
    logger.info("Exporting results...")
    calculator.export_results(output_dir, save_raw=True)

    # Also run separate scenario analyses
    logger.info("\nRunning separate scenario analyses...")

    for scenario in ['1p5C', '2p0C']:
        scenario_name = "1.5°C" if scenario == "1p5C" else "2.0°C"
        logger.info(f"\nAnalyzing {scenario_name} scenario...")

        scenario_results = calculator.run_scenario_analysis(
            n_draws=n_draws,
            scenario=scenario
        )

        # Create scenario-specific output directory
        scenario_dir = output_dir / f'scenario_{scenario_name.replace("°", "")}'
        scenario_dir.mkdir(exist_ok=True)

        # Save scenario results
        import json
        timestamp = scenario_results['metadata']['timestamp']
        stats_file = scenario_dir / f'budget_statistics_{scenario}_{timestamp}.json'
        with open(stats_file, 'w') as f:
            json.dump({
                'statistics': scenario_results['statistics'],
                'metadata': scenario_results['metadata']
            }, f, indent=2, default=str)

        logger.info(f"{scenario_name} results saved to {scenario_dir}")

    logger.info(f"\nAll results saved to: {output_dir.absolute()}")
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_dir.absolute()}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
