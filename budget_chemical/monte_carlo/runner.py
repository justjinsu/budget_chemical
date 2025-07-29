"""
Monte Carlo Runner for Carbon Budget Allocation

Main orchestration module that coordinates sampling, budget allocation,
pathway generation, and output creation for the carbon budget model.
"""

import sys
import yaml
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import numpy as np
import pandas as pd

from .sampler import Sampler
from .pathway_calculator import PathwayCalculator
from ..core.budget_calculation import BudgetAllocation
from .metrics import compute_fan_quantiles, calculate_summary_stats
from .visualization import create_fan_charts, save_uncertainty_plots

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
    
    # Check processing mode
    scenario_mode = config.get('scenario_mode', 'mixed')
    
    if scenario_mode == 'separate':
        run_separate_scenarios(config_path, config, start_time)
    elif scenario_mode == 'separate_methods':
        run_separate_methods(config_path, config, start_time)
    else:
        run_mixed_scenarios(config_path, config, start_time)


def run_separate_methods(config_path: str, config: Dict[str, Any], start_time: datetime) -> None:
    """
    Run Monte Carlo analysis with separate climate scenario and decay method combinations.
    
    Args:
        config_path: Path to configuration file
        config: Configuration dictionary
        start_time: Analysis start time
    """
    logger.info("Running separate climate scenario and decay method analysis")
    
    # Setup output directory
    output_dir = setup_output_directory(config)
    
    # Create years array
    years = create_years_array(config)
    
    # Process each combination separately
    climate_scenarios = ['1p5C', '2p0C']
    curve_types = config['curve_types']
    scenario_labels = {'1p5C': '1.5C', '2p0C': '2.0C'}
    
    all_summaries = {}
    
    for climate_scenario in climate_scenarios:
        for curve_type in curve_types:
            combination_key = f"{climate_scenario}_{curve_type}"
            climate_label = scenario_labels[climate_scenario]
            
            logger.info(f"Processing {climate_label} scenario with {curve_type} decay...")
            
            # Create combination-specific output directory
            combo_output_dir = output_dir / f'scenario_{climate_label}_{curve_type}'
            combo_output_dir.mkdir(exist_ok=True)
            
            # Run analysis for this combination
            summary = run_single_method_analysis(
                config, years, climate_scenario, curve_type, combo_output_dir, start_time
            )
            all_summaries[combination_key] = summary
    
    # Create combined summary
    create_methods_combined_summary(all_summaries, output_dir, start_time)
    
    logger.info(f"Separate methods analysis completed in {(datetime.now() - start_time).total_seconds():.1f} seconds")
    logger.info(f"Results saved to: {output_dir.absolute()}")


def run_separate_scenarios(config_path: str, config: Dict[str, Any], start_time: datetime) -> None:
    """
    Run Monte Carlo analysis with separate scenario processing.
    
    Args:
        config_path: Path to configuration file
        config: Configuration dictionary
        start_time: Analysis start time
    """
    logger.info("Running separate scenario analysis")
    
    # Setup output directory
    output_dir = setup_output_directory(config)
    
    # Create years array
    years = create_years_array(config)
    
    # Process each scenario separately
    scenarios = ['1p5C', '2p0C']
    scenario_labels = {'1p5C': '1.5C', '2p0C': '2.0C'}
    
    all_summaries = {}
    
    for scenario in scenarios:
        logger.info(f"Processing {scenario_labels[scenario]} scenario...")
        
        # Create scenario-specific output directory
        scenario_output_dir = output_dir / f'scenario_{scenario_labels[scenario]}'
        scenario_output_dir.mkdir(exist_ok=True)
        
        # Run analysis for this scenario
        summary = run_single_scenario_analysis(
            config, years, scenario, scenario_output_dir, start_time
        )
        all_summaries[scenario] = summary
    
    # Create combined summary
    create_combined_summary(all_summaries, output_dir, start_time)
    
    logger.info(f"Separate scenario analysis completed in {(datetime.now() - start_time).total_seconds():.1f} seconds")
    logger.info(f"Results saved to: {output_dir.absolute()}")


def run_mixed_scenarios(config_path: str, config: Dict[str, Any], start_time: datetime) -> None:
    """
    Run Monte Carlo analysis with mixed scenario sampling (original behavior).
    
    Args:
        config_path: Path to configuration file  
        config: Configuration dictionary
        start_time: Analysis start time
    """
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
                curve_type=samples['curve_types'][i]
            )
            industry_paths[i, :] = industry_path
            
            # Petrochemical pathway
            petrochem_path = pathway_calc.build_path(
                start_emission=petrochem_base_emission,
                budget=petrochem_budgets[i],
                curve_type=samples['curve_types'][i]
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
    
    # Calculate scenario-specific statistics
    scenario_stats = {}
    for scenario in ['1p5C', '2p0C']:
        scenario_mask = np.array(samples['scenarios']) == scenario
        scenario_count = np.sum(scenario_mask)
        
        if scenario_count > 0:
            scenario_stats[scenario] = {
                'count': int(scenario_count),
                'proportion': float(scenario_count / n_draws),
                'industry_budget_stats': budget_allocator.get_budget_statistics(industry_budgets[scenario_mask]),
                'petrochem_budget_stats': budget_allocator.get_budget_statistics(petrochem_budgets[scenario_mask])
            }
    
    summary = {
        'analysis_metadata': {
            'timestamp': timestamp,
            'config_file': str(config_path),
            'n_draws': n_draws,
            'failed_draws': failed_draws,
            'success_rate': (n_draws - failed_draws) / n_draws,
            'runtime_seconds': (datetime.now() - start_time).total_seconds(),
            'dual_scenario_analysis': True
        },
        'timeline': {
            'start_year': int(years[0]),
            'end_year': int(years[-1]),
            'n_years': len(years)
        },
        'scenario_statistics': scenario_stats,
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
    logger.info(f"Dual-scenario analysis with {len(scenario_stats)} climate scenarios")


def print_summary_stats(summary: Dict[str, Any]) -> None:
    """
    Print key summary statistics to console.
    
    Args:
        summary: Summary statistics dictionary
    """
    print("\n" + "="*60)
    print("DUAL-SCENARIO CARBON BUDGET ALLOCATION - MONTE CARLO RESULTS")
    print("="*60)
    
    meta = summary['analysis_metadata']
    print(f"Analysis completed: {meta['timestamp']}")
    print(f"Monte Carlo draws: {meta['n_draws']} (success rate: {meta['success_rate']:.1%})")
    print(f"Runtime: {meta['runtime_seconds']:.1f} seconds")
    
    timeline = summary['timeline']
    print(f"Timeline: {timeline['start_year']}-{timeline['end_year']} ({timeline['n_years']} years)")
    
    # Print scenario distribution
    print("\nSCENARIO DISTRIBUTION:")
    print("-" * 25)
    if 'scenario_statistics' in summary:
        for scenario, stats in summary['scenario_statistics'].items():
            temp_target = "1.5°C" if scenario == "1p5C" else "2.0°C"
            print(f"{temp_target} scenario: {stats['count']} draws ({stats['proportion']:.1%})")
    
    print("\nBUDGET ALLOCATION RESULTS:")
    print("-" * 30)
    
    for sector in ['industry', 'petrochem']:
        stats = summary['budget_statistics'][sector]
        print(f"\n{sector.upper()} SECTOR (All Scenarios):")
        print(f"  Budget range: {stats['p05']:.2e} - {stats['p95']:.2e} tCO2 (90% CI)")
        print(f"  Median budget: {stats['median']:.2e} tCO2")
        print(f"  Coefficient of variation: {stats['cv']:.3f}")
    
    # Print scenario-specific budget ranges
    if 'scenario_statistics' in summary:
        print("\nSCENARIO-SPECIFIC BUDGET RANGES:")
        print("-" * 35)
        for scenario, stats in summary['scenario_statistics'].items():
            temp_target = "1.5°C" if scenario == "1p5C" else "2.0°C"
            print(f"\n{temp_target} SCENARIO:")
            for sector in ['industry', 'petrochem']:
                sector_stats = stats[f'{sector}_budget_stats']
                print(f"  {sector.capitalize()}: {sector_stats['median']:.2e} tCO2 (median)")
    
    print("\nVALIDATION RESULTS:")
    print("-" * 20)
    val = summary['validation']
    print(f"Industry pathways valid: {val['industry_valid_count']}/{val['total_draws']} ({val['industry_valid_count']/val['total_draws']:.1%})")
    print(f"Petrochemical pathways valid: {val['petrochem_valid_count']}/{val['total_draws']} ({val['petrochem_valid_count']/val['total_draws']:.1%})")
    print(f"Mean budget errors: Industry {val['mean_industry_error']:.3f}%, Petrochemical {val['mean_petrochem_error']:.3f}%")
    
    print("\n" + "="*60)


def run_single_scenario_analysis(config: Dict[str, Any], years: np.ndarray, 
                               scenario: str, output_dir: Path, start_time: datetime) -> Dict[str, Any]:
    """
    Run Monte Carlo analysis for a single scenario.
    
    Args:
        config: Configuration dictionary
        years: Years array for timeline
        scenario: Scenario key ('1p5C' or '2p0C')
        output_dir: Output directory for this scenario
        start_time: Analysis start time
        
    Returns:
        Summary dictionary for this scenario
    """
    # Create scenario-specific sampler
    scenario_config = config.copy()
    
    # Initialize components
    sampler = Sampler(scenario_config)
    pathway_calc = PathwayCalculator(years, config.get('solver', {}))
    budget_allocator = BudgetAllocation(config=config)
    
    # Extract configuration parameters
    n_draws = config['n_draws']
    industry_fraction = config['industry_fraction']
    petrochem_fraction = config['petrochem_fraction']
    progress_interval = config.get('logging', {}).get('progress_interval', 500)
    
    logger.info(f"Running {n_draws} draws for {scenario} scenario")
    
    # Sample parameters for this specific scenario
    scenarios = [scenario] * n_draws  # All draws use the same scenario
    global_budgets = sampler.sample_global_budgets(scenarios)
    responsibility_shares, capability_shares, equality_shares = sampler.sample_allocation_factors(n_draws)
    weights = sampler.sample_weights(n_draws)
    curve_types = sampler.sample_curve_types(n_draws)
    
    samples = {
        'scenarios': scenarios,
        'global_budgets': global_budgets,
        'responsibility_shares': responsibility_shares,
        'capability_shares': capability_shares,
        'equality_shares': equality_shares,
        'weights': weights,
        'curve_types': curve_types
    }
    
    # Calculate budgets for all draws
    industry_budgets, petrochem_budgets = budget_allocator.calculate_sector_budgets_batch(
        global_budgets, responsibility_shares, capability_shares, 
        equality_shares, weights, industry_fraction, petrochem_fraction
    )
    
    # Get base emissions
    industry_base_emission = budget_allocator.get_base_emissions('industry')
    petrochem_base_emission = budget_allocator.get_base_emissions('petrochem')
    
    # Get 2023 baseline emissions for reduction rate calculations
    baseline_2023 = config.get('baseline_2023', {})
    industry_baseline_2023 = baseline_2023.get('industry', config['base_emissions']['industry'])
    petrochem_baseline_2023 = baseline_2023.get('petrochem', config['base_emissions']['petrochem'])
    
    # Initialize pathway storage
    industry_paths = np.zeros((n_draws, len(years)))
    petrochem_paths = np.zeros((n_draws, len(years)))
    
    # Storage for reduction rate calculations
    industry_reduction_rates = np.zeros(n_draws)
    petrochem_reduction_rates = np.zeros(n_draws)
    
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
                curve_type=curve_types[i]
            )
            industry_paths[i, :] = industry_path
            
            # Petrochemical pathway
            petrochem_path = pathway_calc.build_path(
                start_emission=petrochem_base_emission,
                budget=petrochem_budgets[i],
                curve_type=curve_types[i]
            )
            petrochem_paths[i, :] = petrochem_path
            
            # Calculate emission reduction rates for 2035 vs 2023
            industry_reduction_rates[i] = pathway_calc.calculate_emission_reduction_rate(
                industry_path, industry_baseline_2023, 2035
            )
            petrochem_reduction_rates[i] = pathway_calc.calculate_emission_reduction_rate(
                petrochem_path, petrochem_baseline_2023, 2035
            )
            
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
            logger.error(f"Failed to generate pathway for draw {i}: {e}")
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
    
    # Save outputs
    save_scenario_outputs(
        output_dir, years, industry_paths, petrochem_paths,
        industry_quantiles, petrochem_quantiles, quantiles,
        samples, {'industry': industry_budgets, 'petrochem': petrochem_budgets},
        validation_results, config, scenario
    )
    
    # Create summary
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    scenario_label = "1.5°C" if scenario == "1p5C" else "2.0°C"
    
    summary = {
        'analysis_metadata': {
            'timestamp': timestamp,
            'scenario': scenario_label,
            'n_draws': n_draws,
            'failed_draws': failed_draws,
            'success_rate': (n_draws - failed_draws) / n_draws,
            'runtime_seconds': (datetime.now() - start_time).total_seconds()
        },
        'timeline': {
            'start_year': int(years[0]),
            'end_year': int(years[-1]),
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
        'emission_reduction_rates_2035': {
            'industry': {
                'mean': float(np.mean(industry_reduction_rates)),
                'median': float(np.median(industry_reduction_rates)),
                'std': float(np.std(industry_reduction_rates)),
                'p05': float(np.percentile(industry_reduction_rates, 5)),
                'p25': float(np.percentile(industry_reduction_rates, 25)),
                'p75': float(np.percentile(industry_reduction_rates, 75)),
                'p95': float(np.percentile(industry_reduction_rates, 95)),
                'baseline_2023': float(industry_baseline_2023)
            },
            'petrochem': {
                'mean': float(np.mean(petrochem_reduction_rates)),
                'median': float(np.median(petrochem_reduction_rates)),
                'std': float(np.std(petrochem_reduction_rates)),
                'p05': float(np.percentile(petrochem_reduction_rates, 5)),
                'p25': float(np.percentile(petrochem_reduction_rates, 25)),
                'p75': float(np.percentile(petrochem_reduction_rates, 75)),
                'p95': float(np.percentile(petrochem_reduction_rates, 95)),
                'baseline_2023': float(petrochem_baseline_2023)
            }
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
    with open(output_dir / f'summary_{scenario}_{timestamp}.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    return summary


def save_scenario_outputs(output_dir: Path, years: np.ndarray, 
                        industry_paths: np.ndarray, petrochem_paths: np.ndarray,
                        industry_quantiles: np.ndarray, petrochem_quantiles: np.ndarray,
                        quantiles: list, samples: Dict[str, Any], budgets: Dict[str, np.ndarray],
                        validation_results: list, config: Dict[str, Any], scenario: str) -> None:
    """Save outputs for a single scenario."""
    
    # Save raw pathways if requested
    if config['outputs'].get('save_raw_paths', False):
        logger.info("Saving raw pathway data...")
        
        industry_df = pd.DataFrame(industry_paths, columns=[f'year_{year}' for year in years])
        industry_df.to_csv(output_dir / 'industry_emission_paths.csv', index=False)
        
        petrochem_df = pd.DataFrame(petrochem_paths, columns=[f'year_{year}' for year in years])
        petrochem_df.to_csv(output_dir / 'petrochem_emission_paths.csv', index=False)
    
    # Save quantile data
    logger.info("Saving quantile data...")
    quantile_cols = [f'p{int(q*100):02d}' for q in quantiles]
    
    industry_quantile_df = pd.DataFrame(industry_quantiles.T, columns=quantile_cols)
    industry_quantile_df['year'] = years
    industry_quantile_df.to_csv(output_dir / 'industry_fan_quantiles.csv', index=False)
    
    petrochem_quantile_df = pd.DataFrame(petrochem_quantiles.T, columns=quantile_cols)
    petrochem_quantile_df['year'] = years
    petrochem_quantile_df.to_csv(output_dir / 'petrochem_fan_quantiles.csv', index=False)
    
    # Create visualizations if requested
    if config['outputs'].get('create_plots', True):
        logger.info("Creating visualizations...")
        
        # Fan charts with blue theme
        create_fan_charts(
            years=years,
            industry_quantiles=industry_quantiles,
            petrochem_quantiles=petrochem_quantiles,
            quantile_levels=quantiles,
            output_dir=output_dir,
            config=config,
            scenario=scenario  # Pass scenario for blue theming
        )
        
        # Uncertainty plots with blue theme
        save_uncertainty_plots(
            samples=samples,
            budgets=budgets,
            output_dir=output_dir,
            scenario=scenario  # Pass scenario for blue theming
        )


def create_combined_summary(all_summaries: Dict[str, Dict[str, Any]], 
                          output_dir: Path, start_time: datetime) -> None:
    """Create a combined summary for all scenarios."""
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    combined_summary = {
        'analysis_metadata': {
            'timestamp': timestamp,
            'analysis_type': 'separate_scenarios',
            'scenarios_processed': list(all_summaries.keys()),
            'total_runtime_seconds': (datetime.now() - start_time).total_seconds()
        },
        'scenario_summaries': all_summaries
    }
    
    with open(output_dir / f'combined_summary_{timestamp}.json', 'w') as f:
        json.dump(combined_summary, f, indent=2, default=str)
    
    # Print combined results
    print_separate_scenario_results(all_summaries)


def run_single_method_analysis(config: Dict[str, Any], years: np.ndarray, 
                              climate_scenario: str, curve_type: str, output_dir: Path, 
                              start_time: datetime) -> Dict[str, Any]:
    """
    Run Monte Carlo analysis for a single climate scenario and decay method combination.
    
    Args:
        config: Configuration dictionary
        years: Years array for timeline
        climate_scenario: Climate scenario key ('1p5C' or '2p0C')
        curve_type: Decay curve type ('exp', 'log', 's_curve')
        output_dir: Output directory for this combination
        start_time: Analysis start time
        
    Returns:
        Summary dictionary for this combination
    """
    # Create scenario-specific sampler
    scenario_config = config.copy()
    
    # Initialize components
    sampler = Sampler(scenario_config)
    pathway_calc = PathwayCalculator(years, config.get('solver', {}))
    budget_allocator = BudgetAllocation(config=config)
    
    # Extract configuration parameters
    n_draws = config['n_draws']
    industry_fraction = config['industry_fraction']
    petrochem_fraction = config['petrochem_fraction']
    progress_interval = config.get('logging', {}).get('progress_interval', 500)
    
    logger.info(f"Running {n_draws} draws for {climate_scenario} scenario with {curve_type} decay")
    
    # Sample parameters for this specific combination
    scenarios = [climate_scenario] * n_draws  # All draws use the same climate scenario
    curve_types = [curve_type] * n_draws      # All draws use the same curve type
    global_budgets = sampler.sample_global_budgets(scenarios)
    responsibility_shares, capability_shares, equality_shares = sampler.sample_allocation_factors(n_draws)
    weights = sampler.sample_weights(n_draws)
    
    samples = {
        'scenarios': scenarios,
        'global_budgets': global_budgets,
        'responsibility_shares': responsibility_shares,
        'capability_shares': capability_shares,
        'equality_shares': equality_shares,
        'weights': weights,
        'curve_types': curve_types
    }
    
    # Calculate budgets for all draws
    industry_budgets, petrochem_budgets = budget_allocator.calculate_sector_budgets_batch(
        global_budgets, responsibility_shares, capability_shares, 
        equality_shares, weights, industry_fraction, petrochem_fraction
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
            # Industry pathway - all use the same curve type
            industry_path = pathway_calc.build_path(
                start_emission=industry_base_emission,
                budget=industry_budgets[i],
                curve_type=curve_type
            )
            industry_paths[i, :] = industry_path
            
            # Petrochemical pathway - all use the same curve type  
            petrochem_path = pathway_calc.build_path(
                start_emission=petrochem_base_emission,
                budget=petrochem_budgets[i],
                curve_type=curve_type
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
            logger.error(f"Failed to generate pathway for draw {i}: {e}")
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
    
    # Save outputs
    save_scenario_outputs(
        output_dir, years, industry_paths, petrochem_paths,
        industry_quantiles, petrochem_quantiles, quantiles,
        samples, {'industry': industry_budgets, 'petrochem': petrochem_budgets},
        validation_results, config, climate_scenario
    )
    
    # Create summary
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    climate_label = "1.5°C" if climate_scenario == "1p5C" else "2.0°C"
    
    summary = {
        'analysis_metadata': {
            'timestamp': timestamp,
            'climate_scenario': climate_label,
            'curve_type': curve_type,
            'combination': f"{climate_label}_{curve_type}",
            'n_draws': n_draws,
            'failed_draws': failed_draws,
            'success_rate': (n_draws - failed_draws) / n_draws,
            'runtime_seconds': (datetime.now() - start_time).total_seconds()
        },
        'timeline': {
            'start_year': int(years[0]),
            'end_year': int(years[-1]),
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
    with open(output_dir / f'summary_{climate_scenario}_{curve_type}_{timestamp}.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    return summary


def create_methods_combined_summary(all_summaries: Dict[str, Dict[str, Any]], 
                                   output_dir: Path, start_time: datetime) -> None:
    """Create a combined summary for all climate scenario and method combinations."""
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    combined_summary = {
        'analysis_metadata': {
            'timestamp': timestamp,
            'analysis_type': 'separate_climate_and_methods',
            'combinations_processed': list(all_summaries.keys()),
            'total_runtime_seconds': (datetime.now() - start_time).total_seconds()
        },
        'combination_summaries': all_summaries
    }
    
    with open(output_dir / f'combined_methods_summary_{timestamp}.json', 'w') as f:
        json.dump(combined_summary, f, indent=2, default=str)
    
    # Print combined results
    print_separate_methods_results(all_summaries)


def print_separate_methods_results(all_summaries: Dict[str, Dict[str, Any]]) -> None:
    """Print results for separate climate scenario and method analysis."""
    
    print("\n" + "="*80)
    print("SEPARATE CLIMATE SCENARIO & DECAY METHOD - MONTE CARLO RESULTS")
    print("="*80)
    
    # Group by climate scenario for cleaner display
    climate_groups = {'1p5C': [], '2p0C': []}
    for combo_key, summary in all_summaries.items():
        if '1p5C' in combo_key:
            climate_groups['1p5C'].append((combo_key, summary))
        else:
            climate_groups['2p0C'].append((combo_key, summary))
    
    for climate_key, combinations in climate_groups.items():
        climate_label = "1.5°C" if climate_key == "1p5C" else "2.0°C"
        
        print(f"\n{climate_label} CLIMATE SCENARIO:")
        print("=" * 40)
        
        for combo_key, summary in combinations:
            curve_type = summary['analysis_metadata']['curve_type']
            curve_display = {
                'exp': 'Exponential',
                'log': 'Logarithmic', 
                's_curve': 'S-curve'
            }.get(curve_type, curve_type)
            
            print(f"\n  {curve_display} Decay Method:")
            print(f"  {'-' * 25}")
            
            meta = summary['analysis_metadata']
            print(f"    Monte Carlo draws: {meta['n_draws']} (success rate: {meta['success_rate']:.1%})")
            
            # Budget results
            for sector in ['industry', 'petrochem']:
                stats = summary['budget_statistics'][sector]
                print(f"    {sector.capitalize()}: {stats['median']:.2e} tCO2 (median)")
            
            # Validation results
            val = summary['validation']
            print(f"    Validation: Industry {val['industry_valid_count']}/{val['total_draws']} ({val['industry_valid_count']/val['total_draws']:.1%}), "
                  f"Petrochem {val['petrochem_valid_count']}/{val['total_draws']} ({val['petrochem_valid_count']/val['total_draws']:.1%})")
    
    print("\n" + "="*80)


def print_separate_scenario_results(all_summaries: Dict[str, Dict[str, Any]]) -> None:
    """Print results for separate scenario analysis."""
    
    print("\n" + "="*70)
    print("SEPARATE SCENARIO CARBON BUDGET ALLOCATION - MONTE CARLO RESULTS")
    print("="*70)
    
    for scenario_key, summary in all_summaries.items():
        scenario_label = "1.5°C" if scenario_key == "1p5C" else "2.0°C"
        
        print(f"\n{scenario_label} CLIMATE SCENARIO:")
        print("-" * 30)
        
        meta = summary['analysis_metadata']
        print(f"Monte Carlo draws: {meta['n_draws']} (success rate: {meta['success_rate']:.1%})")
        
        timeline = summary['timeline']
        print(f"Timeline: {timeline['start_year']}-{timeline['end_year']} ({timeline['n_years']} years)")
        
        print("\nBUDGET ALLOCATION RESULTS:")
        for sector in ['industry', 'petrochem']:
            stats = summary['budget_statistics'][sector]
            print(f"  {sector.capitalize()}: {stats['median']:.2e} tCO2 (median)")
            print(f"    Range: {stats['p05']:.2e} - {stats['p95']:.2e} tCO2 (90% CI)")
        
        print("\nVALIDATION RESULTS:")
        val = summary['validation']
        print(f"  Industry pathways valid: {val['industry_valid_count']}/{val['total_draws']} ({val['industry_valid_count']/val['total_draws']:.1%})")
        print(f"  Petrochemical pathways valid: {val['petrochem_valid_count']}/{val['total_draws']} ({val['petrochem_valid_count']/val['total_draws']:.1%})")
    
    print("\n" + "="*70)


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