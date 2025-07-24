# model_D_MC/mc_runner.py
# -----------------------------------------------------------
# Main Monte Carlo runner for Korean carbon budget allocation
# Implements full BKIR model with 5-variable uncertainty
# -----------------------------------------------------------
from __future__ import annotations

import sys
import json
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from datetime import datetime

from mc_sampler import load_cfg, Sampler
from lib.budgetCalculation import budgetAllocation
from lib.pathwayCalculation import pathwayCalculator
from viz import plot_fan_chart, create_summary_plots, create_uncertainty_analysis


def calculate_quantiles(pathways: np.ndarray) -> pd.DataFrame:
    """
    Calculate 5%, 25%, 50%, 75%, 95% quantiles for pathways.
    
    Parameters:
    pathways: Emission pathways array (n_draws, n_years)
    
    Returns:
    pd.DataFrame: Quantiles with years
    """
    percentiles = [5, 25, 50, 75, 95]
    quantiles = np.percentile(pathways, percentiles, axis=0)
    
    # Create years array (2024-2050)
    years = np.arange(2024, 2051)
    
    df = pd.DataFrame({
        'year': years,
        'p05': quantiles[0],
        'p25': quantiles[1], 
        'p50': quantiles[2],
        'p75': quantiles[3],
        'p95': quantiles[4]
    })
    
    return df


def calculate_summary_statistics(pathways: np.ndarray, budgets: np.ndarray, 
                                years: np.ndarray) -> dict:
    """
    Calculate comprehensive summary statistics.
    
    Parameters:
    pathways: Emission pathways (n_draws, n_years)
    budgets: Budget allocations (n_draws,)
    years: Years array
    
    Returns:
    dict: Summary statistics
    """
    # Calculate cumulative emissions
    cumulative_emissions = np.sum(pathways, axis=1)
    
    # Overshoot analysis
    overshoot = np.maximum(0, cumulative_emissions - budgets)
    overshoot_prob = (overshoot > 0).mean()
    
    # Budget exhaustion analysis
    pathway_calc = pathwayCalculator()
    exhaustion_years = []
    for i in range(len(pathways)):
        exhaustion_year = pathway_calc.calculate_exhaustion_year(pathways[i])
        exhaustion_years.append(exhaustion_year)
    
    exhaustion_years = np.array(exhaustion_years)
    
    return {
        'mean_budget': float(np.mean(budgets)),
        'p05_budget': float(np.percentile(budgets, 5)),
        'p95_budget': float(np.percentile(budgets, 95)),
        'overshoot_prob': float(overshoot_prob),
        'mean_overshoot': float(np.mean(overshoot)),
        'median_exhaust_year': float(np.median(exhaustion_years)),
        'cumulative_emissions': {
            'mean': float(np.mean(cumulative_emissions)),
            'p05': float(np.percentile(cumulative_emissions, 5)),
            'p50': float(np.percentile(cumulative_emissions, 50)),
            'p75': float(np.percentile(cumulative_emissions, 75)),
            'p95': float(np.percentile(cumulative_emissions, 95))
        }
    }


def main():
    """Main Monte Carlo simulation execution"""
    print("="*60)
    print("Korean Carbon Budget Allocation - Monte Carlo Simulation")
    print("="*60)
    
    # Load configuration
    cfg = load_cfg("mc_config.yaml")
    print(f"Configuration loaded: {cfg['n_draws']} draws, seed={cfg['seed']}")
    
    # Initialize components
    sampler = Sampler(cfg)
    budget_calc = budgetAllocation()
    pathway_calc = pathwayCalculator()
    
    # Sample uncertainty variables
    print("Sampling uncertainty variables...")
    global_budget, responsibility, capability, equality, weights = sampler.sample_all()
    
    # Calculate BKIR budgets
    print("Calculating BKIR budget allocations...")
    bkir_budgets = budget_calc.calculate_BKIR(
        global_budget, responsibility, capability, equality, weights
    )
    
    # Allocate to industry and petrochemical sectors
    industry_budgets, petrochem_budgets = budget_calc.allocate_industry_petrochem(
        bkir_budgets, cfg['industry_fraction'], cfg['petrochem_fraction']
    )
    
    # Get base emissions
    E0_industry = budget_calc.get_base_emissions('industry')
    E0_petrochem = budget_calc.get_base_emissions('petrochem')
    
    print(f"Base emissions - Industry: {E0_industry:.2e} tCO2, Petrochemical: {E0_petrochem:.2e} tCO2")
    
    # Calculate emission pathways
    print("Calculating emission pathways...")
    years = pathway_calc.get_years()
    n_draws = cfg['n_draws']
    
    industry_pathways = np.zeros((n_draws, len(years)))
    petrochem_pathways = np.zeros((n_draws, len(years)))
    
    for i in range(n_draws):
        # Industry pathways
        industry_pathways[i] = pathway_calc.calculate_pathway(
            E0_industry, industry_budgets[i], cfg['curve_type']
        )
        
        # Petrochemical pathways  
        petrochem_pathways[i] = pathway_calc.calculate_pathway(
            E0_petrochem, petrochem_budgets[i], cfg['curve_type']
        )
        
        if (i + 1) % 500 == 0:
            print(f"  Completed {i + 1}/{n_draws} pathways")
    
    # Create output directory
    output_dir = Path(cfg.get('output_dir', 'outputs'))
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate quantile data
    print("Generating quantile data...")
    industry_quantiles = calculate_quantiles(industry_pathways)
    petrochem_quantiles = calculate_quantiles(petrochem_pathways)
    
    # Save CSV outputs
    industry_csv = output_dir / f"industry_fan_quantiles_{timestamp}.csv"
    petrochem_csv = output_dir / f"petrochem_fan_quantiles_{timestamp}.csv"
    
    industry_quantiles.to_csv(industry_csv, index=False)
    petrochem_quantiles.to_csv(petrochem_csv, index=False)
    
    print(f"CSV outputs saved:")
    print(f"  Industry: {industry_csv}")
    print(f"  Petrochemical: {petrochem_csv}")
    
    # Calculate summary statistics
    print("Calculating summary statistics...")
    industry_stats = calculate_summary_statistics(industry_pathways, industry_budgets, years)
    petrochem_stats = calculate_summary_statistics(petrochem_pathways, petrochem_budgets, years)
    
    # Create comprehensive summary
    summary = {
        'timestamp': timestamp,
        'configuration': cfg,
        'model_parameters': {
            'base_emissions_industry_tCO2': E0_industry,
            'base_emissions_petrochem_tCO2': E0_petrochem,
            'simulation_years': [int(years[0]), int(years[-1])],
            'transition_year': pathway_calc.mid_year
        },
        'industry_sector': industry_stats,
        'petrochemical_sector': petrochem_stats,
        'uncertainty_factors': {
            'global_budget': {
                'mean': float(np.mean(global_budget)),
                'std': float(np.std(global_budget)),
                'range': [float(np.min(global_budget)), float(np.max(global_budget))]
            },
            'responsibility_factor': {
                'mean': float(np.mean(responsibility)),
                'std': float(np.std(responsibility))
            },
            'capability_factor': {
                'mean': float(np.mean(capability)),
                'std': float(np.std(capability))
            },
            'equality_factor': {
                'mean': float(np.mean(equality)),
                'std': float(np.std(equality))
            }
        }
    }
    
    # Save JSON summary
    summary_json = output_dir / f"summary_{timestamp}.json"
    with open(summary_json, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"JSON summary saved: {summary_json}")
    
    # Create visualizations
    print("Creating visualizations...")
    create_summary_plots(
        years, industry_pathways, petrochem_pathways,
        {'industry': industry_budgets, 'petrochem': petrochem_budgets},
        output_dir
    )
    
    create_uncertainty_analysis(
        {
            'global_budget': global_budget,
            'responsibility': responsibility, 
            'capability': capability,
            'equality': equality
        },
        weights, output_dir
    )
    
    # Print key results
    print("\n" + "="*60)
    print("KEY RESULTS SUMMARY")
    print("="*60)
    print(f"Industry Sector:")
    print(f"  Mean Budget: {industry_stats['mean_budget']:.2e} tCO2")
    print(f"  90% CI: [{industry_stats['p05_budget']:.2e}, {industry_stats['p95_budget']:.2e}] tCO2")
    print(f"  Overshoot Probability: {industry_stats['overshoot_prob']:.1%}")
    print(f"  Median Exhaustion Year: {industry_stats['median_exhaust_year']:.0f}")
    
    print(f"\nPetrochemical Sector:")
    print(f"  Mean Budget: {petrochem_stats['mean_budget']:.2e} tCO2")  
    print(f"  90% CI: [{petrochem_stats['p05_budget']:.2e}, {petrochem_stats['p95_budget']:.2e}] tCO2")
    print(f"  Overshoot Probability: {petrochem_stats['overshoot_prob']:.1%}")
    print(f"  Median Exhaustion Year: {petrochem_stats['median_exhaust_year']:.0f}")
    
    print(f"\nAll outputs saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()