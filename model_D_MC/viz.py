# model_D_MC/viz.py
# -----------------------------------------------------------
# Visualization module for fan charts and uncertainty plots
# -----------------------------------------------------------
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any


def plot_fan_chart(years: np.ndarray, pathways: np.ndarray, 
                   title: str = "Carbon Budget Fan Chart",
                   ylabel: str = "Emissions (tCO2)",
                   save_path: Path = None) -> plt.Figure:
    """
    Create fan chart showing percentile ranges over time.
    
    Parameters:
    years: Array of years
    pathways: Array of emission pathways (n_draws, n_years)
    title: Chart title
    ylabel: Y-axis label
    save_path: Optional path to save the figure
    
    Returns:
    plt.Figure: Fan chart figure
    """
    # Calculate percentiles
    percentiles = [5, 25, 50, 75, 95]
    quantiles = np.percentile(pathways, percentiles, axis=0)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Fan chart colors (from dark to light)
    colors = ['#2c3e50', '#34495e', '#5D6D7E', '#85929E', '#BDC3C7']
    alphas = [0.3, 0.4, 0.6, 0.4, 0.3]
    
    # Plot fan bands
    ax.fill_between(years, quantiles[0], quantiles[4], 
                   alpha=alphas[0], color=colors[0], label='5-95th percentile')
    ax.fill_between(years, quantiles[1], quantiles[3], 
                   alpha=alphas[1], color=colors[1], label='25-75th percentile')
    
    # Plot median line
    ax.plot(years, quantiles[2], color='red', linewidth=3, label='Median (50th percentile)')
    
    # Formatting
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    # Set y-axis to start from 0
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_summary_plots(years: np.ndarray, industry_pathways: np.ndarray, 
                        petrochem_pathways: np.ndarray, budgets: Dict[str, np.ndarray],
                        output_dir: Path) -> None:
    """
    Create comprehensive summary plots for both sectors.
    
    Parameters:
    years: Array of years
    industry_pathways: Industry emission pathways (n_draws, n_years)
    petrochem_pathways: Petrochemical emission pathways (n_draws, n_years)
    budgets: Dictionary containing budget arrays
    output_dir: Directory to save plots
    """
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # 1. Fan charts for both sectors
    fig_industry = plot_fan_chart(
        years, industry_pathways,
        title="Korean Industry Sector - Carbon Budget Fan Chart",
        ylabel="Emissions (tCO2/year)",
        save_path=output_dir / "industry_fan_chart.png"
    )
    
    fig_petrochem = plot_fan_chart(
        years, petrochem_pathways,
        title="Korean Petrochemical Sector - Carbon Budget Fan Chart", 
        ylabel="Emissions (tCO2/year)",
        save_path=output_dir / "petrochem_fan_chart.png"
    )
    
    # 2. Budget distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Carbon Budget Analysis Summary', fontsize=16, fontweight='bold')
    
    # Industry budget distribution
    ax1 = axes[0, 0]
    ax1.hist(budgets['industry'], bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_xlabel('Industry Budget (tCO2)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Industry Budget Distribution')
    ax1.axvline(np.mean(budgets['industry']), color='red', linestyle='--', 
                label=f'Mean: {np.mean(budgets["industry"]):.2e}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Petrochemical budget distribution
    ax2 = axes[0, 1]
    ax2.hist(budgets['petrochem'], bins=50, alpha=0.7, color='green', edgecolor='black')
    ax2.set_xlabel('Petrochemical Budget (tCO2)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Petrochemical Budget Distribution')
    ax2.axvline(np.mean(budgets['petrochem']), color='red', linestyle='--',
                label=f'Mean: {np.mean(budgets["petrochem"]):.2e}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Cumulative emissions comparison
    ax3 = axes[1, 0]
    industry_cumsum = np.cumsum(industry_pathways, axis=1)
    petrochem_cumsum = np.cumsum(petrochem_pathways, axis=1)
    
    # Plot median cumulative emissions
    ax3.plot(years, np.median(industry_cumsum, axis=0), 
             label='Industry (Median)', color='blue', linewidth=2)
    ax3.plot(years, np.median(petrochem_cumsum, axis=0), 
             label='Petrochemical (Median)', color='green', linewidth=2)
    
    # Add budget constraint lines
    ax3.axhline(np.median(budgets['industry']), color='blue', linestyle='--', alpha=0.7)
    ax3.axhline(np.median(budgets['petrochem']), color='green', linestyle='--', alpha=0.7)
    
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Cumulative Emissions (tCO2)')
    ax3.set_title('Cumulative Emissions vs Budget')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Overshoot probability analysis
    ax4 = axes[1, 1]
    industry_overshoot = np.maximum(0, industry_cumsum[:, -1] - budgets['industry'])
    petrochem_overshoot = np.maximum(0, petrochem_cumsum[:, -1] - budgets['petrochem'])
    
    overshoot_data = [
        industry_overshoot[industry_overshoot > 0],
        petrochem_overshoot[petrochem_overshoot > 0]
    ]
    
    bp = ax4.boxplot(overshoot_data, labels=['Industry', 'Petrochemical'], patch_artist=True)
    bp['boxes'][0].set_facecolor('blue')
    bp['boxes'][1].set_facecolor('green')
    
    ax4.set_ylabel('Budget Overshoot (tCO2)')
    ax4.set_title('Budget Overshoot Distribution')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "summary_analysis.png", dpi=300, bbox_inches='tight')
    
    # Close figures to save memory
    plt.close(fig_industry)
    plt.close(fig_petrochem)
    plt.close(fig)


def create_uncertainty_analysis(factors: Dict[str, np.ndarray], weights: np.ndarray,
                               output_dir: Path) -> None:
    """
    Create uncertainty analysis plots for the 5 variables.
    
    Parameters:
    factors: Dictionary of uncertainty factors
    weights: User weights array (n_draws, 3)
    output_dir: Directory to save plots
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Uncertainty Factor Analysis', fontsize=16, fontweight='bold')
    
    # Plot distributions of uncertainty factors
    ax1 = axes[0, 0]
    ax1.hist(factors['global_budget'], bins=50, alpha=0.7, color='purple', edgecolor='black')
    ax1.set_xlabel('Global Budget (tCO2)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Global Budget Distribution')
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    ax2.hist(factors['responsibility'], bins=50, alpha=0.7, color='red', edgecolor='black')
    ax2.set_xlabel('Responsibility Factor δr')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Responsibility Factor Distribution')
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[0, 2]
    ax3.hist(factors['capability'], bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax3.set_xlabel('Capability Factor δc')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Capability Factor Distribution')
    ax3.grid(True, alpha=0.3)
    
    ax4 = axes[1, 0]
    ax4.hist(factors['equality'], bins=50, alpha=0.7, color='green', edgecolor='black')
    ax4.set_xlabel('Equality Factor δe')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Equality Factor Distribution')
    ax4.grid(True, alpha=0.3)
    
    # Weight distributions
    ax5 = axes[1, 1]
    weight_df = pd.DataFrame(weights, columns=['Responsibility', 'Capability', 'Equality'])
    weight_df.boxplot(ax=ax5)
    ax5.set_ylabel('Weight Value')
    ax5.set_title('User Weight Distributions')
    ax5.grid(True, alpha=0.3)
    
    # Correlation matrix
    ax6 = axes[1, 2]
    corr_data = np.column_stack([
        factors['global_budget'], factors['responsibility'], 
        factors['capability'], factors['equality']
    ])
    corr_df = pd.DataFrame(corr_data, columns=['Global Budget', 'Responsibility', 'Capability', 'Equality'])
    corr_matrix = corr_df.corr()
    
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax6)
    ax6.set_title('Factor Correlation Matrix')
    
    plt.tight_layout()
    plt.savefig(output_dir / "uncertainty_analysis.png", dpi=300, bbox_inches='tight')
    plt.close(fig)