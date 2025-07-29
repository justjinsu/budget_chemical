"""
Visualization Module for Carbon Budget Analysis

This module provides functions for creating fan charts, uncertainty plots,
and other visualizations for Monte Carlo carbon budget analysis results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Set default matplotlib style
plt.style.use('default')
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14
})


def create_fan_charts(years: np.ndarray, 
                     industry_quantiles: np.ndarray,
                     petrochem_quantiles: np.ndarray,
                     quantile_levels: List[float],
                     output_dir: Path,
                     config: Dict[str, Any],
                     scenario: Optional[str] = None) -> None:
    """
    Create fan charts for industry and petrochemical sectors.
    
    Args:
        years: Array of years
        industry_quantiles: Industry quantile data (n_quantiles, n_years)
        petrochem_quantiles: Petrochemical quantile data (n_quantiles, n_years)
        quantile_levels: List of quantile levels
        output_dir: Output directory for saving plots
        config: Configuration dictionary
    """
    logger.info("Creating fan charts...")
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Define blue color scheme for different quantile bands
    if scenario == '1p5C':
        # Darker blues for 1.5C scenario
        colors = ['#08306b', '#2171b5', '#6baed6', '#c6dbef', '#f0f9ff']
    elif scenario == '2p0C':
        # Lighter blues for 2.0C scenario
        colors = ['#2171b5', '#6baed6', '#c6dbef', '#f0f9ff', '#f7fbff']
    else:
        # Default blue scheme for mixed scenarios
        colors = ['#08519c', '#3182bd', '#6baed6', '#bdd7e7', '#eff3ff']
    alphas = [0.9, 0.7, 0.5, 0.7, 0.9]
    
    # Industry sector fan chart
    plot_fan_chart(ax1, years, industry_quantiles, quantile_levels, 
                   colors, alphas, "Industry Sector", config)
    
    # Petrochemical sector fan chart  
    plot_fan_chart(ax2, years, petrochem_quantiles, quantile_levels,
                   colors, alphas, "Petrochemical Sector", config)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'combined_fan_chart.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create individual sector charts
    create_individual_fan_chart(years, industry_quantiles, quantile_levels,
                               "Industry Sector", output_dir / 'industry_fan_chart.png', config, scenario)
    
    create_individual_fan_chart(years, petrochem_quantiles, quantile_levels,
                               "Petrochemical Sector", output_dir / 'petrochem_fan_chart.png', config, scenario)
    
    logger.info("Fan charts created successfully")


def plot_fan_chart(ax: plt.Axes, years: np.ndarray, quantiles: np.ndarray,
                  quantile_levels: List[float], colors: List[str], alphas: List[float],
                  title: str, config: Dict[str, Any]) -> None:
    """
    Plot a fan chart on given axes.
    
    Args:
        ax: Matplotlib axes object
        years: Array of years
        quantiles: Quantile data (n_quantiles, n_years)
        quantile_levels: List of quantile levels
        colors: List of colors for bands
        alphas: List of alpha values for bands
        title: Chart title
        config: Configuration dictionary
    """
    n_quantiles = len(quantile_levels)
    
    # Plot quantile bands
    for i in range(n_quantiles - 1):
        upper = quantiles[i, :]
        lower = quantiles[i + 1, :]
        
        # Fill area between quantiles
        ax.fill_between(years, upper, lower, 
                       color=colors[i], alpha=alphas[i],
                       label=f'P{int(quantile_levels[i]*100):02d}-P{int(quantile_levels[i+1]*100):02d}')
    
    # Plot median line
    median_idx = len(quantile_levels) // 2
    ax.plot(years, quantiles[median_idx, :], 'k-', linewidth=2, label='Median')
    
    # Mark important years
    end_year = config['timeline']['end_year']
    
    ax.axvline(x=end_year, color='red', linestyle='--', alpha=0.7, label=f'Zero Emission Target ({end_year})' )
    
    # Formatting
    ax.set_xlabel('Year')
    ax.set_ylabel('Emissions (tCO2/year)')
    ax.set_title(title, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)
    
    # Format y-axis in scientific notation
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # Set reasonable y-limits
    y_max = np.max(quantiles) * 1.1
    ax.set_ylim(0, y_max)


def create_individual_fan_chart(years: np.ndarray, quantiles: np.ndarray,
                               quantile_levels: List[float], title: str,
                               save_path: Path, config: Dict[str, Any],
                               scenario: Optional[str] = None) -> None:
    """
    Create individual fan chart for a single sector.
    
    Args:
        years: Array of years
        quantiles: Quantile data (n_quantiles, n_years)
        quantile_levels: List of quantile levels
        title: Chart title
        save_path: Path to save the plot
        config: Configuration dictionary
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Blue color scheme for individual charts
    if scenario == '1p5C':
        colors = ['#08306b', '#2171b5', '#6baed6', '#c6dbef'][:len(quantile_levels)-1]
    elif scenario == '2p0C':
        colors = ['#2171b5', '#6baed6', '#c6dbef', '#f0f9ff'][:len(quantile_levels)-1] 
    else:
        colors = ['#08519c', '#3182bd', '#6baed6', '#bdd7e7'][:len(quantile_levels)-1]
    alphas = [0.8, 0.6, 0.4, 0.2][:len(quantile_levels)-1]
    
    plot_fan_chart(ax, years, quantiles, quantile_levels, colors, alphas, title, config)
    
    # Add additional formatting for individual charts
    ax.set_title(f'{title} - Carbon Budget Pathways', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_uncertainty_plots(samples: Dict[str, Any], budgets: Dict[str, np.ndarray],
                          output_dir: Path, scenario: Optional[str] = None) -> None:
    """
    Create and save uncertainty analysis plots.
    
    Args:
        samples: Dictionary of sampled parameters
        budgets: Dictionary of sector budgets
        output_dir: Output directory for saving plots
    """
    logger.info("Creating uncertainty analysis plots...")
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Monte Carlo Uncertainty Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Global budget distribution
    ax = axes[0, 0]
    budget_color = '#2171b5' if scenario == '1p5C' else '#6baed6' if scenario == '2p0C' else '#3182bd'
    ax.hist(samples['global_budgets'], bins=50, alpha=0.7, color=budget_color, edgecolor='navy')
    ax.set_xlabel('Global Budget (tCO2)')
    ax.set_ylabel('Frequency')
    ax.set_title('Global Budget Distribution')
    ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Allocation factors
    ax = axes[0, 1]
    ax.hist(samples['responsibility_shares'], bins=30, alpha=0.6, label='Responsibility', color='#08306b')
    ax.hist(samples['capability_shares'], bins=30, alpha=0.6, label='Capability', color='#2171b5')
    ax.hist(samples['equality_shares'], bins=30, alpha=0.6, label='Equality', color='#6baed6')
    ax.set_xlabel('Share Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Allocation Factor Distributions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Weight distributions
    ax = axes[0, 2]
    weights = samples['weights']
    ax.hist(weights[:, 0], bins=30, alpha=0.6, label='Responsibility Weight', color='#08306b')
    ax.hist(weights[:, 1], bins=30, alpha=0.6, label='Capability Weight', color='#2171b5')
    ax.hist(weights[:, 2], bins=30, alpha=0.6, label='Equality Weight', color='#6baed6')
    ax.set_xlabel('Weight Value')
    ax.set_ylabel('Frequency')
    ax.set_title('User Weight Distributions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Curve type distribution
    ax = axes[1, 0]
    curve_types, counts = np.unique(samples['curve_types'], return_counts=True)
    colors = {'exp': '#08306b', 'log': '#2171b5', 's_curve': '#6baed6', 'linear': '#c6dbef'}
    bar_colors = [colors.get(ct, '#bdd7e7') for ct in curve_types]
    ax.bar(curve_types, counts, alpha=0.7, color=bar_colors)
    ax.set_xlabel('Curve Type')
    ax.set_ylabel('Frequency')
    ax.set_title('Curve Type Distribution')
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Scenario distribution or Budget correlation
    ax = axes[1, 2]
    if 'scenarios' in samples:
        # Plot scenario distribution
        scenario_counts = {}
        for scenario in ['1p5C', '2p0C']:
            scenario_counts[scenario] = np.sum(np.array(samples['scenarios']) == scenario)
        
        scenario_labels = ['1.5°C', '2.0°C']
        scenario_values = [scenario_counts['1p5C'], scenario_counts['2p0C']]
        colors = ['#2171b5', '#6baed6']
        
        ax.pie(scenario_values, labels=scenario_labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title('Climate Scenario Distribution')
    else:
        # Fallback to budget correlation
        ax.scatter(budgets['industry'], budgets['petrochem'], alpha=0.5, s=1)
        ax.set_xlabel('Industry Budget (tCO2)')
        ax.set_ylabel('Petrochemical Budget (tCO2)')
        ax.set_title('Sector Budget Correlation')
        ax.ticklabel_format(style='scientific', axis='both', scilimits=(0,0))
        ax.grid(True, alpha=0.3)
        
        # Calculate and display correlation coefficient
        corr = np.corrcoef(budgets['industry'], budgets['petrochem'])[0, 1]
        ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'uncertainty_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create summary statistics plot
    create_summary_analysis_plot(samples, budgets, output_dir)
    
    logger.info("Uncertainty analysis plots created successfully")


def create_summary_analysis_plot(samples: Dict[str, Any], budgets: Dict[str, np.ndarray],
                                output_dir: Path) -> None:
    """
    Create summary analysis visualization.
    
    Args:
        samples: Dictionary of sampled parameters
        budgets: Dictionary of sector budgets
        output_dir: Output directory for saving plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Budget Allocation Summary Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Budget distributions comparison
    ax = axes[0, 0]
    ax.hist(budgets['industry'], bins=40, alpha=0.6, label='Industry', color='#2171b5', density=True)
    ax.hist(budgets['petrochem'], bins=40, alpha=0.6, label='Petrochemical', color='#6baed6', density=True)
    ax.set_xlabel('Budget (tCO2)')
    ax.set_ylabel('Density')
    ax.set_title('Sector Budget Distributions')
    ax.legend()
    ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Budget ratio distribution
    ax = axes[0, 1]
    budget_ratio = budgets['petrochem'] / budgets['industry']
    # Check if values are too similar for histogram
    if np.max(budget_ratio) - np.min(budget_ratio) < 1e-6:
        # Display as text when all values are identical
        ax.text(0.5, 0.5, f'All ratios = {np.mean(budget_ratio):.3f}\\n(Fixed configuration)', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12,
                bbox=dict(boxstyle='round', facecolor='#c6dbef', alpha=0.8))
        ax.set_title('Budget Ratio Distribution')
    else:
        # Use few bins for similar values
        ax.hist(budget_ratio, bins=max(3, len(budget_ratio)//10), alpha=0.7, color='#6baed6', edgecolor='navy')
        ax.set_xlabel('Petrochemical/Industry Budget Ratio')
        ax.set_ylabel('Frequency')
        ax.set_title('Budget Ratio Distribution')
        ax.axvline(x=np.mean(budget_ratio), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(budget_ratio):.3f}')
        ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Sensitivity analysis - weights vs budgets
    ax = axes[1, 0]
    weights = samples['weights']
    scatter = ax.scatter(weights[:, 1], budgets['industry'], 
                        c=weights[:, 0], cmap='Blues', alpha=0.6, s=10)
    ax.set_xlabel('Capability Weight')
    ax.set_ylabel('Industry Budget (tCO2)')
    ax.set_title('Budget Sensitivity to Weights')
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Responsibility Weight')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Scenarios vs budgets  
    ax = axes[1, 1]
    if 'scenarios' in samples:
        scenario_colors = {'1p5C': '#08306b', '2p0C': '#6baed6'}
        for scenario_key in ['1p5C', '2p0C']:
            mask = np.array(samples['scenarios']) == scenario_key
            if np.any(mask):
                temp_label = "1.5°C" if scenario_key == "1p5C" else "2.0°C"
                ax.scatter(budgets['industry'][mask], budgets['petrochem'][mask], 
                          alpha=0.6, s=20, color=scenario_colors[scenario_key], label=temp_label)
        ax.set_xlabel('Industry Budget (tCO2)')
        ax.set_ylabel('Petrochemical Budget (tCO2)')
        ax.set_title('Climate Scenarios vs Budgets')
        ax.ticklabel_format(style='scientific', axis='both', scilimits=(0,0))
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        # Fallback to curve types if scenarios not available
        curve_colors = {'exp': '#08306b', 'log': '#2171b5', 's_curve': '#6baed6', 'linear': '#c6dbef'}
        for curve_type in ['exp', 'log', 's_curve', 'linear']:
            mask = np.array(samples['curve_types']) == curve_type
            if np.any(mask):
                ax.scatter(budgets['industry'][mask], budgets['petrochem'][mask], 
                          alpha=0.6, s=20, color=curve_colors[curve_type], label=curve_type)
        ax.set_xlabel('Industry Budget (tCO2)')
        ax.set_ylabel('Petrochemical Budget (tCO2)')
        ax.set_title('Curve Types vs Budgets')
        ax.ticklabel_format(style='scientific', axis='both', scilimits=(0,0))
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'summary_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()