"""
Monte Carlo Runner for Carbon Budget Allocation

Main orchestration module that coordinates sampling, budget allocation,
pathway generation, and output creation for the carbon budget model.
"""

import os
import sys
import yaml
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

# Add current directory first, then lib directory to path
sys.path.insert(0, str(Path(__file__).parent))  # model_D_MC directory
sys.path.insert(0, str(Path(__file__).parent.parent / 'lib'))  # lib directory

from mc_sampler import Sampler
from pathway_calculator import PathwayCalculator  # This module (model_D_MC)
from budgetCalculation import BudgetAllocation  # From lib directory
from mc_metrics import compute_fan_quantiles, calculate_summary_stats
from viz import create_fan_charts, save_uncertainty_plots

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Configuration loaded from {config_path}")
    return config


def setup_output_directory(config: Dict[str, Any]) -> Path:
    """
    Setup output directory for results.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Path to output directory
    """
    output_dir = Path(config.get('output_dir', 'outputs'))
    output_dir.mkdir(exist_ok=True)
    
    logger.info(f"Output directory: {output_dir.absolute()}")
    return output_dir


def create_years_array(config: Dict[str, Any]) -> np.ndarray:
    """
    Create years array for pathway calculation.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Array of years
    """
    timeline = config['timeline']
    start_year = timeline['start_year']
    end_year = timeline['end_year']
    
    years = np.arange(start_year, end_year + 1)
    logger.info(f"Timeline: {start_year}-{end_year} ({len(years)} years)")
    
    return years


def run_monte_carlo_analysis(config_path: str) -> None:
    """
    Run complete Monte Carlo analysis for carbon budget allocation.
    
    Args:
        config_path: Path to configuration file
    """
    start_time = datetime.now()
    logger.info("Starting Monte Carlo carbon budget analysis")
    
    # Load configuration
    config = load_config(config_path)
    
    # Setup output directory
    output_dir = setup_output_directory(config)
    
    # Create years array
    years = create_years_array(config)
    
    # Initialize components
    sampler = Sampler(config)
    pathway_calc = PathwayCalculator(years, config.get('solver', {}))
    budget_allocator = BudgetAllocation(config=config)
    
    # Extract configuration parameters
    n_draws = config['n_draws']
    industry_fraction = config['industry_fraction']
    petrochem_fraction = config['petrochem_fraction']
    progress_interval = config.get('logging', {}).get('progress_interval', 500)
    
    logger.info(f"Running {n_draws} Monte Carlo draws")
    
    # Sample all uncertain parameters
    samples = sampler.sample_all(n_draws)
    
    # Calculate budgets for all draws
    industry_budgets, petrochem_budgets = budget_allocator.calculate_sector_budgets_batch(
        samples['global_budgets'],
        samples['responsibility_shares'],
        samples['capability_shares'],
        samples['equality_shares'],
        samples['weights'],
        industry_fraction,
        petrochem_fraction
    )
    
    # Get base emissions
    industry_base_emission = budget_allocator.get_base_emissions('industry')
    petrochem_base_emission = budget_allocator.get_base_emissions('petrochem')
    
    # Initialize pathway storage
    industry_paths = np.zeros((n_draws, len(years)))
    petrochem_paths = np.zeros((n_draws, len(years)))
    
    # Validation tracking
    validation_results = []
    failed_draws = 0
    
    logger.info("Generating emission pathways...")
    
    # Generate pathways for each Monte Carlo draw
    for i in range(n_draws):
        if (i + 1) % progress_interval == 0:
            logger.info(f"Completed {i + 1}/{n_draws} draws ({100*(i+1)/n_draws:.1f}%)")
        
        try:
            # Industry pathway
            industry_path = pathway_calc.build_path(
                start_emission=industry_base_emission,
                budget=industry_budgets[i],
                curve_type=samples['curve_types'][i],
                frontload=samples['lambdas'][i]
            )
            industry_paths[i, :] = industry_path
            
            # Petrochemical pathway
            petrochem_path = pathway_calc.build_path(
                start_emission=petrochem_base_emission,
                budget=petrochem_budgets[i],
                curve_type=samples['curve_types'][i],
                frontload=samples['lambdas'][i]
            )
            petrochem_paths[i, :] = petrochem_path
            
            # Validate pathways
            industry_validation = pathway_calc.validate_pathway(industry_path, industry_budgets[i])
            petrochem_validation = pathway_calc.validate_pathway(petrochem_path, petrochem_budgets[i])
            
            validation_results.append({
                'draw': i,
                'industry_valid': industry_validation['is_valid'],
                'petrochem_valid': petrochem_validation['is_valid'],
                'industry_budget_error': industry_validation['budget_error_pct'],
                'petrochem_budget_error': petrochem_validation['budget_error_pct']
            })
            
        except Exception as e:
            import traceback
            logger.error(f"Failed to generate pathway for draw {i}: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            failed_draws += 1
            # Fill with fallback linear decay
            industry_paths[i, :] = industry_base_emission * np.linspace(1, 0, len(years))
            petrochem_paths[i, :] = petrochem_base_emission * np.linspace(1, 0, len(years))
    
    logger.info(f"Pathway generation completed. Failed draws: {failed_draws}/{n_draws}")
    
    # Calculate quantiles for fan charts
    logger.info("Computing quantiles for fan charts...")
    quantiles = config['outputs']['quantiles']
    
    industry_quantiles = compute_fan_quantiles(industry_paths, quantiles)
    petrochem_quantiles = compute_fan_quantiles(petrochem_paths, quantiles)
    
    # Save raw pathways if requested
    if config['outputs'].get('save_raw_paths', False):
        logger.info("Saving raw pathway data...")
        
        # Industry pathways
        industry_df = pd.DataFrame(industry_paths, columns=[f'year_{year}' for year in years])
        industry_df.to_csv(output_dir / 'industry_emission_paths.csv', index=False)
        
        # Petrochemical pathways
        petrochem_df = pd.DataFrame(petrochem_paths, columns=[f'year_{year}' for year in years])
        petrochem_df.to_csv(output_dir / 'petrochem_emission_paths.csv', index=False)
    
    # Save quantile data
    logger.info("Saving quantile data...")
    
    # Create quantile DataFrames
    quantile_cols = [f'p{int(q*100):02d}' for q in quantiles]
    
    industry_quantile_df = pd.DataFrame(industry_quantiles.T, columns=quantile_cols)
    industry_quantile_df['year'] = years
    industry_quantile_df.to_csv(output_dir / 'industry_fan_quantiles.csv', index=False)
    
    petrochem_quantile_df = pd.DataFrame(petrochem_quantiles.T, columns=quantile_cols)
    petrochem_quantile_df['year'] = years
    petrochem_quantile_df.to_csv(output_dir / 'petrochem_fan_quantiles.csv', index=False)
    
    # Calculate summary statistics
    logger.info("Calculating summary statistics...")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    summary = {
        'analysis_metadata': {
            'timestamp': timestamp,
            'config_file': str(config_path),
            'n_draws': n_draws,
            'failed_draws': failed_draws,
            'success_rate': (n_draws - failed_draws) / n_draws,
            'runtime_seconds': (datetime.now() - start_time).total_seconds()
        },
        'timeline': {
            'start_year': int(years[0]),
            'end_year': int(years[-1]),
            'mid_year': config['timeline']['mid_year'],
            'n_years': len(years)
        },
        'budget_statistics': {
            'industry': budget_allocator.get_budget_statistics(industry_budgets),
            'petrochem': budget_allocator.get_budget_statistics(petrochem_budgets)
        },
        'pathway_statistics': {
            'industry': calculate_summary_stats(industry_paths, years),
            'petrochem': calculate_summary_stats(petrochem_paths, years)
        },
        'validation': {
            'total_draws': len(validation_results),
            'industry_valid_count': sum(1 for v in validation_results if v['industry_valid']),
            'petrochem_valid_count': sum(1 for v in validation_results if v['petrochem_valid']),
            'mean_industry_error': np.mean([v['industry_budget_error'] for v in validation_results]),
            'mean_petrochem_error': np.mean([v['petrochem_budget_error'] for v in validation_results])
        }
    }
    
    # Save summary
    with open(output_dir / f'summary_{timestamp}.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Create visualizations if requested
    if config['outputs'].get('create_plots', True):
        logger.info("Creating visualizations...")
        
        # Fan charts
        create_fan_charts(
            years=years,
            industry_quantiles=industry_quantiles,
            petrochem_quantiles=petrochem_quantiles,
            quantile_levels=quantiles,
            output_dir=output_dir,
            config=config
        )
        
        # Uncertainty plots
        save_uncertainty_plots(
            samples=samples,
            budgets={'industry': industry_budgets, 'petrochem': petrochem_budgets},
            output_dir=output_dir
        )
    
    # Print summary to console
    print_summary_stats(summary)
    
    logger.info(f"Analysis completed in {(datetime.now() - start_time).total_seconds():.1f} seconds")
    logger.info(f"Results saved to: {output_dir.absolute()}")


def print_summary_stats(summary: Dict[str, Any]) -> None:
    """
    Print key summary statistics to console.
    
    Args:
        summary: Summary statistics dictionary
    """
    print("\n" + "="*60)
    print("CARBON BUDGET ALLOCATION - MONTE CARLO RESULTS")
    print("="*60)
    
    meta = summary['analysis_metadata']
    print(f"Analysis completed: {meta['timestamp']}")
    print(f"Monte Carlo draws: {meta['n_draws']} (success rate: {meta['success_rate']:.1%})")
    print(f"Runtime: {meta['runtime_seconds']:.1f} seconds")
    
    timeline = summary['timeline']
    print(f"Timeline: {timeline['start_year']}-{timeline['end_year']} ({timeline['n_years']} years)")
    
    print("\nBUDGET ALLOCATION RESULTS:")
    print("-" * 30)
    
    for sector in ['industry', 'petrochem']:
        stats = summary['budget_statistics'][sector]
        print(f"\n{sector.upper()} SECTOR:")
        print(f"  Budget range: {stats['p05']:.2e} - {stats['p95']:.2e} tCO2 (90% CI)")
        print(f"  Median budget: {stats['median']:.2e} tCO2")
        print(f"  Coefficient of variation: {stats['cv']:.3f}")
    
    print("\nVALIDATION RESULTS:")
    print("-" * 20)
    val = summary['validation']
    print(f"Industry pathways valid: {val['industry_valid_count']}/{val['total_draws']} ({val['industry_valid_count']/val['total_draws']:.1%})")
    print(f"Petrochemical pathways valid: {val['petrochem_valid_count']}/{val['total_draws']} ({val['petrochem_valid_count']/val['total_draws']:.1%})")
    print(f"Mean budget errors: Industry {val['mean_industry_error']:.3f}%, Petrochemical {val['mean_petrochem_error']:.3f}%")
    
    print("\n" + "="*60)


def main():
    """Main entry point for command-line execution."""
    if len(sys.argv) != 2:
        print("Usage: python mc_runner.py <config_file>")
        print("Example: python mc_runner.py mc_config.yaml")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    try:
        run_monte_carlo_analysis(config_path)
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()