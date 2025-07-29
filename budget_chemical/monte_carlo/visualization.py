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
                     config: Dict[str, Any]) -> None:
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
    
    # Define colors for different quantile bands
    colors = ['#1f77b4', '#aec7e8', '#ffbb78', '#aec7e8', '#1f77b4']
    alphas = [0.8, 0.6, 0.4, 0.6, 0.8]
    
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
                               "Industry Sector", output_dir / 'industry_fan_chart.png', config)
    
    create_individual_fan_chart(years, petrochem_quantiles, quantile_levels,
                               "Petrochemical Sector", output_dir / 'petrochem_fan_chart.png', config)
    
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
    mid_year = config['timeline']['mid_year']
    end_year = config['timeline']['end_year']
    
    ax.axvline(x=mid_year, color='red', linestyle='--', alpha=0.7, label=f'Mid-year ({mid_year})')
    ax.axvline(x=end_year, color='red', linestyle=':', alpha=0.7, label=f'Target year ({end_year})')
    
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
                               save_path: Path, config: Dict[str, Any]) -> None:
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
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592E83'][:len(quantile_levels)-1]
    alphas = [0.2, 0.3, 0.5, 0.3, 0.2][:len(quantile_levels)-1]
    
    plot_fan_chart(ax, years, quantiles, quantile_levels, colors, alphas, title, config)
    
    # Add additional formatting for individual charts
    ax.set_title(f'{title} - Carbon Budget Pathways', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_uncertainty_plots(samples: Dict[str, Any], budgets: Dict[str, np.ndarray],
                          output_dir: Path) -> None:
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
    ax.hist(samples['global_budgets'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax.set_xlabel('Global Budget (tCO2)')
    ax.set_ylabel('Frequency')
    ax.set_title('Global Budget Distribution')
    ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Allocation factors
    ax = axes[0, 1]
    ax.hist(samples['responsibility_shares'], bins=30, alpha=0.6, label='Responsibility', color='red')
    ax.hist(samples['capability_shares'], bins=30, alpha=0.6, label='Capability', color='blue')
    ax.hist(samples['equality_shares'], bins=30, alpha=0.6, label='Equality', color='green')
    ax.set_xlabel('Share Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Allocation Factor Distributions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Weight distributions
    ax = axes[0, 2]
    weights = samples['weights']
    ax.hist(weights[:, 0], bins=30, alpha=0.6, label='Responsibility Weight', color='red')
    ax.hist(weights[:, 1], bins=30, alpha=0.6, label='Capability Weight', color='blue')
    ax.hist(weights[:, 2], bins=30, alpha=0.6, label='Equality Weight', color='green')
    ax.set_xlabel('Weight Value')
    ax.set_ylabel('Frequency')
    ax.set_title('User Weight Distributions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Lambda (front-loading) distribution
    ax = axes[1, 0]
    ax.hist(samples['lambdas'], bins=30, alpha=0.7, color='orange', edgecolor='black')
    ax.set_xlabel('Lambda (Front-loading Parameter)')
    ax.set_ylabel('Frequency')
    ax.set_title('Front-loading Parameter Distribution')
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Curve type distribution
    ax = axes[1, 1]
    curve_types, counts = np.unique(samples['curve_types'], return_counts=True)
    ax.bar(curve_types, counts, alpha=0.7, color=['red', 'blue', 'green'][:len(curve_types)])
    ax.set_xlabel('Curve Type')
    ax.set_ylabel('Frequency')
    ax.set_title('Curve Type Distribution')
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Budget correlation
    ax = axes[1, 2]
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
    ax.hist(budgets['industry'], bins=40, alpha=0.6, label='Industry', color='blue', density=True)
    ax.hist(budgets['petrochem'], bins=40, alpha=0.6, label='Petrochemical', color='red', density=True)
    ax.set_xlabel('Budget (tCO2)')
    ax.set_ylabel('Density')
    ax.set_title('Sector Budget Distributions')
    ax.legend()
    ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Budget ratio distribution
    ax = axes[0, 1]
    budget_ratio = budgets['petrochem'] / budgets['industry']
    ax.hist(budget_ratio, bins=30, alpha=0.7, color='purple', edgecolor='black')
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
                        c=weights[:, 0], cmap='viridis', alpha=0.6, s=10)
    ax.set_xlabel('Capability Weight')
    ax.set_ylabel('Industry Budget (tCO2)')
    ax.set_title('Budget Sensitivity to Weights')
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Responsibility Weight')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Lambda vs budget relationship
    ax = axes[1, 1]
    ax.scatter(samples['lambdas'], budgets['petrochem'], alpha=0.5, s=10, color='orange')
    ax.set_xlabel('Lambda (Front-loading Parameter)')
    ax.set_ylabel('Petrochemical Budget (tCO2)')
    ax.set_title('Front-loading vs Budget')
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'summary_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()