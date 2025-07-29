"""
Emission Pathway Comparison Analysis

This module compares three different emission pathways:
1. 1.5°C scenario industry paths (Monte Carlo results)
2. 2.0°C scenario industry paths (Monte Carlo results)
3. Korean government's linear plan (10% reduction by 2030, 80% reduction by 2050)

The analysis provides statistical comparisons, visualization, and policy insights.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Tuple, List
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

class PathwayComparison:
    """
    Class for comparing emission pathways across different scenarios and policies.
    """
    
    def __init__(self, outputs_dir: str = "outputs"):
        """
        Initialize the pathway comparison analyzer.
        
        Args:
            outputs_dir: Directory containing Monte Carlo outputs
        """
        self.outputs_dir = Path(outputs_dir)
        self.baseline_2023 = 185_000_000.0  # 185 MtCO2 - industry baseline
        
        # Timeline configuration
        self.years = np.arange(2024, 2051)  # 2024-2050
        self.n_years = len(self.years)
        
        # Storage for pathway data
        self.pathways = {}
        self.statistics = {}
        
        logger.info(f"PathwayComparison initialized with outputs from {self.outputs_dir}")
    
    def load_scenario_paths(self, scenario: str) -> pd.DataFrame:
        """
        Load emission paths for a specific scenario.
        
        Args:
            scenario: Either '1.5C' or '2.0C'
            
        Returns:
            DataFrame with emission paths (rows=draws, columns=years)
        """
        scenario_dir = self.outputs_dir / f"scenario_{scenario}"
        paths_file = scenario_dir / "industry_emission_paths.csv"
        
        if not paths_file.exists():
            raise FileNotFoundError(f"Emission paths file not found: {paths_file}")
        
        # Load the CSV file
        df = pd.read_csv(paths_file)
        
        # Convert column names from 'year_YYYY' to just the year
        year_columns = [col for col in df.columns if col.startswith('year_')]
        if len(year_columns) != self.n_years:
            logger.warning(f"Expected {self.n_years} years, found {len(year_columns)} in {scenario} data")
        
        logger.info(f"Loaded {len(df)} emission paths for {scenario} scenario")
        return df
    
    def create_korean_government_plan(self) -> np.ndarray:
        """
        Create Korean government's linear emission reduction plan.
        
        The plan targets:
        - 10% reduction by 2030 (compared to 2023 baseline)
        - 80% reduction by 2050 (compared to 2023 baseline)
        - Linear interpolation between these points
        
        Returns:
            Array of emissions for each year 2024-2050
        """
        # Define key years and reduction targets
        baseline_year = 2023
        target_2030 = 2030
        target_2050 = 2050
        
        reduction_2030 = 0.10  # 10% reduction by 2030
        reduction_2050 = 0.80  # 80% reduction by 2050
        
        # Calculate emission levels
        emission_2023 = self.baseline_2023
        emission_2030 = emission_2023 * (1 - reduction_2030)  # 90% of baseline
        emission_2050 = emission_2023 * (1 - reduction_2050)  # 20% of baseline
        
        # Create the linear pathway
        gov_pathway = np.zeros(self.n_years)
        
        for i, year in enumerate(self.years):
            if year <= target_2030:
                # Linear interpolation from 2023 to 2030
                progress = (year - baseline_year) / (target_2030 - baseline_year)
                gov_pathway[i] = emission_2023 + progress * (emission_2030 - emission_2023)
            else:
                # Linear interpolation from 2030 to 2050
                progress = (year - target_2030) / (target_2050 - target_2030)
                gov_pathway[i] = emission_2030 + progress * (emission_2050 - emission_2030)
        
        logger.info(f"Created Korean government plan: {emission_2023/1e6:.1f} -> {emission_2030/1e6:.1f} -> {emission_2050/1e6:.1f} MtCO2")
        return gov_pathway
    
    def load_all_pathways(self):
        """Load all pathway data for comparison."""
        logger.info("Loading all emission pathways...")
        
        # Load Monte Carlo scenario results
        try:
            self.pathways['1.5C'] = self.load_scenario_paths('1.5C')
            logger.info("✓ Loaded 1.5°C scenario paths")
        except FileNotFoundError as e:
            logger.error(f"Could not load 1.5°C scenario: {e}")
            self.pathways['1.5C'] = None
        
        try:
            self.pathways['2.0C'] = self.load_scenario_paths('2.0C')
            logger.info("✓ Loaded 2.0°C scenario paths")
        except FileNotFoundError as e:
            logger.error(f"Could not load 2.0°C scenario: {e}")
            self.pathways['2.0C'] = None
        
        # Create Korean government plan
        self.pathways['Korean_Gov'] = self.create_korean_government_plan()
        logger.info("✓ Created Korean government linear plan")
    
    def calculate_pathway_statistics(self) -> Dict[str, Any]:
        """
        Calculate key statistics for all pathways.
        
        Returns:
            Dictionary with statistics for each pathway
        """
        stats = {}
        
        for name, pathway_data in self.pathways.items():
            if pathway_data is None:
                continue
                
            if name == 'Korean_Gov':
                # Single pathway statistics
                pathway = pathway_data
                stats[name] = self._calculate_single_pathway_stats(pathway, name)
            else:
                # Monte Carlo pathway statistics
                stats[name] = self._calculate_mc_pathway_stats(pathway_data, name)
        
        self.statistics = stats
        return stats
    
    def _calculate_single_pathway_stats(self, pathway: np.ndarray, name: str) -> Dict[str, Any]:
        """Calculate statistics for a single pathway."""
        # Find 2035 emissions for reduction calculation
        idx_2035 = np.where(self.years == 2035)[0][0]
        emission_2035 = pathway[idx_2035]
        reduction_2035 = ((self.baseline_2023 - emission_2035) / self.baseline_2023) * 100
        
        # Calculate cumulative emissions
        cumulative = np.trapezoid(pathway, dx=1.0)
        
        return {
            'type': 'single_pathway',
            'n_pathways': 1,
            'emission_2024': pathway[0],
            'emission_2035': emission_2035,
            'emission_2050': pathway[-1],
            'reduction_2035_pct': reduction_2035,
            'reduction_2050_pct': ((self.baseline_2023 - pathway[-1]) / self.baseline_2023) * 100,
            'cumulative_emissions': cumulative,
            'pathway_shape': 'linear'
        }
    
    def _calculate_mc_pathway_stats(self, pathway_df: pd.DataFrame, name: str) -> Dict[str, Any]:
        """Calculate statistics for Monte Carlo pathways."""
        # Extract year columns
        year_cols = [col for col in pathway_df.columns if col.startswith('year_')]
        pathway_data = pathway_df[year_cols].values
        
        # Find 2035 emissions for reduction calculation
        year_2035_col = 'year_2035'
        if year_2035_col in pathway_df.columns:
            emission_2035_values = pathway_df[year_2035_col].values
            reduction_2035_values = ((self.baseline_2023 - emission_2035_values) / self.baseline_2023) * 100
        else:
            logger.warning(f"Could not find 2035 data for {name}")
            reduction_2035_values = np.array([0.0])
        
        # Calculate cumulative emissions for each pathway
        cumulative_emissions = np.array([np.trapezoid(path, dx=1.0) for path in pathway_data])
        
        return {
            'type': 'monte_carlo',
            'n_pathways': len(pathway_df),
            'emission_2024_stats': self._get_percentile_stats(pathway_data[:, 0]),
            'emission_2035_stats': self._get_percentile_stats(emission_2035_values) if year_2035_col in pathway_df.columns else None,
            'emission_2050_stats': self._get_percentile_stats(pathway_data[:, -1]),
            'reduction_2035_stats': self._get_percentile_stats(reduction_2035_values),
            'cumulative_emissions_stats': self._get_percentile_stats(cumulative_emissions),
            'pathway_quantiles': np.percentile(pathway_data, [5, 25, 50, 75, 95], axis=0)
        }
    
    def _get_percentile_stats(self, data: np.ndarray) -> Dict[str, float]:
        """Calculate percentile statistics for an array."""
        return {
            'mean': float(np.mean(data)),
            'median': float(np.median(data)),
            'std': float(np.std(data)),
            'p05': float(np.percentile(data, 5)),
            'p25': float(np.percentile(data, 25)),
            'p75': float(np.percentile(data, 75)),
            'p95': float(np.percentile(data, 95)),
            'min': float(np.min(data)),
            'max': float(np.max(data))
        }
    
    def create_comparison_visualization(self, save_path: str = None):
        """
        Create comprehensive visualization comparing all pathways.
        
        Args:
            save_path: Optional path to save the figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Industry Emission Pathway Comparison', fontsize=16, fontweight='bold')
        
        # Plot 1: Main pathway comparison
        ax1 = axes[0, 0]
        self._plot_pathway_comparison(ax1)
        
        # Plot 2: 2035 reduction rates comparison
        ax2 = axes[0, 1]
        self._plot_reduction_rates(ax2)
        
        # Plot 3: Cumulative emissions comparison
        ax3 = axes[1, 0]
        self._plot_cumulative_emissions(ax3)
        
        # Plot 4: Pathway characteristics table
        ax4 = axes[1, 1]
        self._plot_summary_table(ax4)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
            plt.close()  # Close the figure to free memory
        else:
            plt.show()
    
    def _plot_pathway_comparison(self, ax):
        """Plot the main pathway comparison."""
        colors = {'1.5C': '#1f77b4', '2.0C': '#ff7f0e', 'Korean_Gov': '#2ca02c'}
        
        for name, pathway_data in self.pathways.items():
            if pathway_data is None:
                continue
                
            if name == 'Korean_Gov':
                # Plot single pathway
                ax.plot(self.years, pathway_data / 1e6, 
                       color=colors[name], linewidth=3, label=f'{name} (Linear Plan)', linestyle='--')
            else:
                # Plot Monte Carlo pathways with uncertainty bands
                year_cols = [col for col in pathway_data.columns if col.startswith('year_')]
                pathway_values = pathway_data[year_cols].values
                
                # Calculate percentiles
                p05 = np.percentile(pathway_values, 5, axis=0)
                p25 = np.percentile(pathway_values, 25, axis=0)
                p50 = np.percentile(pathway_values, 50, axis=0)
                p75 = np.percentile(pathway_values, 75, axis=0)
                p95 = np.percentile(pathway_values, 95, axis=0)
                
                # Plot uncertainty bands
                ax.fill_between(self.years, p05/1e6, p95/1e6, alpha=0.2, color=colors[name])
                ax.fill_between(self.years, p25/1e6, p75/1e6, alpha=0.4, color=colors[name])
                ax.plot(self.years, p50/1e6, color=colors[name], linewidth=2, 
                       label=f'{name} Scenario (Median)')
        
        ax.set_xlabel('Year')
        ax.set_ylabel('Emissions (MtCO2/year)')
        ax.set_title('Industry Emission Pathways')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add key milestone years
        ax.axvline(x=2030, color='gray', linestyle=':', alpha=0.7, label='2030 Target')
        ax.axvline(x=2035, color='red', linestyle=':', alpha=0.7, label='2035 Analysis')
    
    def _plot_reduction_rates(self, ax):
        """Plot 2035 reduction rates comparison."""
        scenarios = []
        medians = []
        errors_low = []
        errors_high = []
        
        for name, stats in self.statistics.items():
            if name == 'Korean_Gov':
                scenarios.append('Korean Gov\n(Linear)')
                medians.append(stats['reduction_2035_pct'])
                errors_low.append(0)
                errors_high.append(0)
            else:
                scenarios.append(f'{name}\nScenario')
                reduction_stats = stats['reduction_2035_stats']
                medians.append(reduction_stats['median'])
                errors_low.append(reduction_stats['median'] - reduction_stats['p25'])
                errors_high.append(reduction_stats['p75'] - reduction_stats['median'])
        
        bars = ax.bar(scenarios, medians, 
                     yerr=[errors_low, errors_high],
                     capsize=5, alpha=0.7,
                     color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        
        ax.set_ylabel('Reduction Rate (%)')
        ax.set_title('2035 Emission Reduction\n(vs 2023 Baseline)')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, median in zip(bars, medians):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{median:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    def _plot_cumulative_emissions(self, ax):
        """Plot cumulative emissions comparison."""
        scenarios = []
        values = []
        errors_low = []
        errors_high = []
        
        for name, stats in self.statistics.items():
            if name == 'Korean_Gov':
                scenarios.append('Korean Gov\n(Linear)')
                values.append(stats['cumulative_emissions'] / 1e9)  # Convert to GtCO2
                errors_low.append(0)
                errors_high.append(0)
            else:
                scenarios.append(f'{name}\nScenario')
                cum_stats = stats['cumulative_emissions_stats']
                values.append(cum_stats['median'] / 1e9)
                errors_low.append((cum_stats['median'] - cum_stats['p25']) / 1e9)
                errors_high.append((cum_stats['p75'] - cum_stats['median']) / 1e9)
        
        bars = ax.bar(scenarios, values,
                     yerr=[errors_low, errors_high],
                     capsize=5, alpha=0.7,
                     color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        
        ax.set_ylabel('Cumulative Emissions (GtCO2)')
        ax.set_title('Total Emissions 2024-2050')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
    
    def _plot_summary_table(self, ax):
        """Plot summary statistics table."""
        ax.axis('off')
        
        # Prepare table data
        table_data = []
        headers = ['Pathway', '2035 Reduction', '2050 Reduction', 'Cumulative\n(GtCO2)']
        
        for name, stats in self.statistics.items():
            if name == 'Korean_Gov':
                row = [
                    'Korean Gov Plan',
                    f"{stats['reduction_2035_pct']:.1f}%",
                    f"{stats['reduction_2050_pct']:.1f}%",
                    f"{stats['cumulative_emissions']/1e9:.1f}"
                ]
            else:
                row = [
                    f'{name} Scenario',
                    f"{stats['reduction_2035_stats']['median']:.1f}%\n({stats['reduction_2035_stats']['p25']:.1f}-{stats['reduction_2035_stats']['p75']:.1f}%)",
                    "100%\n(by design)",
                    f"{stats['cumulative_emissions_stats']['median']/1e9:.1f}\n({stats['cumulative_emissions_stats']['p25']/1e9:.1f}-{stats['cumulative_emissions_stats']['p75']/1e9:.1f})"
                ]
            table_data.append(row)
        
        # Create table
        table = ax.table(cellText=table_data,
                        colLabels=headers,
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.3, 0.25, 0.25, 0.2])
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style the table
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax.set_title('Pathway Comparison Summary', fontweight='bold', pad=20)
    
    def save_results(self, output_file: str = "pathway_comparison_results.json"):
        """
        Save analysis results to JSON file.
        
        Args:
            output_file: Name of output file
        """
        results = {
            'analysis_metadata': {
                'baseline_2023': self.baseline_2023,
                'analysis_years': self.years.tolist(),
                'pathways_analyzed': list(self.pathways.keys())
            },
            'pathway_statistics': self.statistics
        }
        
        output_path = self.outputs_dir / output_file
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_path}")
    
    def print_summary(self):
        """Print a summary of the comparison analysis."""
        print("\n" + "="*70)
        print("EMISSION PATHWAY COMPARISON ANALYSIS")
        print("="*70)
        print(f"Baseline (2023): {self.baseline_2023/1e6:.1f} MtCO2")
        print(f"Analysis period: {self.years[0]}-{self.years[-1]}")
        
        for name, stats in self.statistics.items():
            print(f"\n{name.upper()} PATHWAY:")
            print("-" * 30)
            
            if stats['type'] == 'single_pathway':
                print(f"  2035 emissions: {stats['emission_2035']/1e6:.1f} MtCO2")
                print(f"  2035 reduction: {stats['reduction_2035_pct']:.1f}%")
                print(f"  2050 reduction: {stats['reduction_2050_pct']:.1f}%")
                print(f"  Cumulative: {stats['cumulative_emissions']/1e9:.2f} GtCO2")
            else:
                red_stats = stats['reduction_2035_stats']
                cum_stats = stats['cumulative_emissions_stats']
                print(f"  Pathways: {stats['n_pathways']} Monte Carlo draws")
                print(f"  2035 reduction: {red_stats['median']:.1f}% ({red_stats['p25']:.1f}-{red_stats['p75']:.1f}%)")
                print(f"  Cumulative: {cum_stats['median']/1e9:.2f} GtCO2 ({cum_stats['p25']/1e9:.2f}-{cum_stats['p75']/1e9:.2f})")
        
        print("\n" + "="*70)


def main():
    """Main function to run the pathway comparison analysis."""
    # Initialize analyzer
    analyzer = PathwayComparison()
    
    # Load all pathway data
    analyzer.load_all_pathways()
    
    # Calculate statistics
    analyzer.calculate_pathway_statistics()
    
    # Print summary
    analyzer.print_summary()
    
    # Create visualization
    output_viz = analyzer.outputs_dir / "pathway_comparison_analysis.png"
    analyzer.create_comparison_visualization(save_path=output_viz)
    
    # Save detailed results
    analyzer.save_results()
    
    logger.info("Pathway comparison analysis completed successfully!")


if __name__ == "__main__":
    main()