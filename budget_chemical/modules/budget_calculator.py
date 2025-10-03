"""
Module 1: Korean Carbon Budget Calculator

This module calculates Korea's carbon budget independently from pathway generation.
It uses the BKIR (Budget Korea Industrial/Petrochemical) allocation formula based on
three equity principles: Responsibility, Capability, and Equality.

Class-based design for clean separation between backend logic and frontend usage.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from scipy.stats import triang, norm
import logging
import json
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class KoreaBudgetCalculator:
    """
    Calculate Korea's carbon budget using BKIR allocation formula.

    This class handles:
    1. Monte Carlo sampling of uncertain parameters
    2. Budget allocation using BKIR formula
    3. Sector-level budget distribution
    4. Statistical analysis and uncertainty quantification

    The BKIR formula:
        Korea_Budget = Global_Budget × (w_r × δ_r + w_c × δ_c + w_e × δ_e)

    Where:
        - w_r, w_c, w_e: User-defined weights for responsibility, capability, equality
        - δ_r, δ_c, δ_e: Korea's shares of global responsibility, capability, equality
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the budget calculator.

        Args:
            config: Configuration dictionary containing:
                - global_budget: Global carbon budget scenarios (1.5°C, 2.0°C)
                - uncertainty: Allocation factor distributions
                - user_weights: Weights for BKIR formula
                - industry_fraction: Share to industry sector
                - petrochem_fraction: Share to petrochemical sector
                - seed: Random seed for reproducibility
        """
        self.config = config
        self.seed = config.get('seed', 123)
        self.rng = np.random.RandomState(self.seed)

        # Extract configuration sections
        self.global_budget_cfg = config['global_budget']
        self.weights_cfg = config['user_weights']
        self.uncertainty_cfg = config['uncertainty']
        self.industry_fraction = config['industry_fraction']
        self.petrochem_fraction = config['petrochem_fraction']

        # Setup probability distributions
        self._setup_distributions()

        # Storage for results
        self.results = None

        logger.info(f"KoreaBudgetCalculator initialized with seed {self.seed}")
        logger.info(f"Allocation factors: R={self.uncertainty_cfg['responsibility']['mid']:.4f}, "
                   f"C={self.uncertainty_cfg['capability']['mu']:.4f}, "
                   f"E={self.uncertainty_cfg['equality']['mu']:.4f}")

    def _setup_distributions(self) -> None:
        """Setup probability distributions for Monte Carlo sampling."""

        # Global budget distributions for each scenario
        self.scenario_dists = {}
        for scenario in ['1p5C', '2p0C']:
            gb = self.global_budget_cfg[scenario]
            low = float(gb['low'])
            mid = float(gb['mid'])
            high = float(gb['high'])

            # Triangular distribution: loc=low, scale=high-low, c=(mode-low)/(high-low)
            self.scenario_dists[scenario] = triang(
                loc=low,
                scale=high - low,
                c=(mid - low) / (high - low)
            )

        # Responsibility distribution (triangular)
        resp = self.uncertainty_cfg['responsibility']
        self.resp_dist = triang(
            loc=resp['low'],
            scale=resp['high'] - resp['low'],
            c=(resp['mid'] - resp['low']) / (resp['high'] - resp['low'])
        )

        # Capability distribution (normal)
        cap = self.uncertainty_cfg['capability']
        self.cap_dist = norm(
            loc=cap['mu'],
            scale=cap['mu'] * cap['sd_pct']
        )

        # Equality distribution (normal)
        eq = self.uncertainty_cfg['equality']
        self.eq_dist = norm(
            loc=eq['mu'],
            scale=eq['mu'] * eq['sd_pct']
        )

        logger.debug("Probability distributions initialized for budget calculation")

    def calculate_single_budget(
        self,
        global_budget: float,
        responsibility_share: float,
        capability_share: float,
        equality_share: float,
        weights: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate Korea's budget for a single parameter set using BKIR formula.

        Args:
            global_budget: Global carbon budget (tCO2)
            responsibility_share: Korea's responsibility factor (δ_r)
            capability_share: Korea's capability factor (δ_c)
            equality_share: Korea's equality factor (δ_e)
            weights: User weights [w_r, w_c, w_e]

        Returns:
            Dictionary with korea_total, industry, and petrochem budgets
        """
        # Extract weights
        w_r, w_c, w_e = weights

        # BKIR formula: weighted sum of allocation factors
        weighted_factor = (
            w_r * responsibility_share +
            w_c * capability_share +
            w_e * equality_share
        )

        # Calculate Korea's total budget
        korea_total = global_budget * weighted_factor

        # Allocate to sectors
        industry_budget = korea_total * self.industry_fraction
        petrochem_budget = industry_budget * self.petrochem_fraction

        return {
            'korea_total': korea_total,
            'industry': industry_budget,
            'petrochem': petrochem_budget,
            'weighted_factor': weighted_factor
        }

    def run_monte_carlo(
        self,
        n_draws: int,
        scenario_mode: str = 'mixed'
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation to quantify budget uncertainty.

        Args:
            n_draws: Number of Monte Carlo draws
            scenario_mode: 'mixed' for random, 'separate' for individual scenarios

        Returns:
            Dictionary containing:
                - samples: All sampled parameters
                - budgets: Calculated budgets for korea, industry, petrochem
                - statistics: Summary statistics
        """
        logger.info(f"Running Monte Carlo simulation with {n_draws} draws (mode: {scenario_mode})")

        # Sample climate scenarios
        if scenario_mode == 'mixed':
            scenarios = self.rng.choice(['1p5C', '2p0C'], size=n_draws).tolist()
        else:
            # For separate mode, this will be called once per scenario
            raise ValueError("Use run_scenario_analysis() for separate scenario mode")

        # Sample global budgets based on scenarios
        global_budgets = np.zeros(n_draws)
        for i, scenario in enumerate(scenarios):
            global_budgets[i] = self.scenario_dists[scenario].rvs(random_state=self.rng)

        # Sample allocation factors
        responsibility_shares = self.resp_dist.rvs(size=n_draws, random_state=self.rng)
        capability_shares = np.abs(self.cap_dist.rvs(size=n_draws, random_state=self.rng))
        equality_shares = np.abs(self.eq_dist.rvs(size=n_draws, random_state=self.rng))

        # Sample weights with noise
        weights = self._sample_weights(n_draws)

        # Calculate budgets for all draws (vectorized)
        korea_budgets, industry_budgets, petrochem_budgets = self._calculate_budgets_batch(
            global_budgets, responsibility_shares, capability_shares,
            equality_shares, weights
        )

        # Store results
        self.results = {
            'samples': {
                'scenarios': scenarios,
                'global_budgets': global_budgets,
                'responsibility_shares': responsibility_shares,
                'capability_shares': capability_shares,
                'equality_shares': equality_shares,
                'weights': weights
            },
            'budgets': {
                'korea_total': korea_budgets,
                'industry': industry_budgets,
                'petrochem': petrochem_budgets
            },
            'statistics': self._calculate_statistics(
                korea_budgets, industry_budgets, petrochem_budgets, scenarios
            ),
            'metadata': {
                'n_draws': n_draws,
                'scenario_mode': scenario_mode,
                'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
                'seed': self.seed
            }
        }

        logger.info(f"Monte Carlo completed: Korea budget range {korea_budgets.min():.2e} - {korea_budgets.max():.2e} tCO2")

        return self.results

    def run_scenario_analysis(
        self,
        n_draws: int,
        scenario: str
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo for a single climate scenario.

        Args:
            n_draws: Number of Monte Carlo draws
            scenario: '1p5C' or '2p0C'

        Returns:
            Results dictionary for single scenario
        """
        logger.info(f"Running {scenario} scenario analysis with {n_draws} draws")

        # Sample from single scenario
        scenarios = [scenario] * n_draws
        global_budgets = self.scenario_dists[scenario].rvs(size=n_draws, random_state=self.rng)

        # Sample allocation factors
        responsibility_shares = self.resp_dist.rvs(size=n_draws, random_state=self.rng)
        capability_shares = np.abs(self.cap_dist.rvs(size=n_draws, random_state=self.rng))
        equality_shares = np.abs(self.eq_dist.rvs(size=n_draws, random_state=self.rng))

        # Sample weights
        weights = self._sample_weights(n_draws)

        # Calculate budgets
        korea_budgets, industry_budgets, petrochem_budgets = self._calculate_budgets_batch(
            global_budgets, responsibility_shares, capability_shares,
            equality_shares, weights
        )

        # Store results
        results = {
            'samples': {
                'scenarios': scenarios,
                'global_budgets': global_budgets,
                'responsibility_shares': responsibility_shares,
                'capability_shares': capability_shares,
                'equality_shares': equality_shares,
                'weights': weights
            },
            'budgets': {
                'korea_total': korea_budgets,
                'industry': industry_budgets,
                'petrochem': petrochem_budgets
            },
            'statistics': self._calculate_statistics(
                korea_budgets, industry_budgets, petrochem_budgets, scenarios
            ),
            'metadata': {
                'n_draws': n_draws,
                'scenario': scenario,
                'scenario_mode': 'separate',
                'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
                'seed': self.seed
            }
        }

        logger.info(f"{scenario} analysis completed")

        return results

    def _sample_weights(self, n: int) -> np.ndarray:
        """Sample user weights with Gaussian noise."""
        base_weights = np.array([
            self.weights_cfg['responsibility'],
            self.weights_cfg['capability'],
            self.weights_cfg['equality']
        ])

        noise_std = 0.025  # ±2.5 percentage points
        weights = np.zeros((n, 3))

        for i in range(n):
            noisy_weights = base_weights + self.rng.normal(0, noise_std, 3)
            noisy_weights = np.maximum(noisy_weights, 0.01)  # Ensure positive
            noisy_weights = noisy_weights / np.sum(noisy_weights)  # Normalize
            weights[i] = noisy_weights

        return weights

    def _calculate_budgets_batch(
        self,
        global_budgets: np.ndarray,
        responsibility_shares: np.ndarray,
        capability_shares: np.ndarray,
        equality_shares: np.ndarray,
        weights: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate budgets for all draws (vectorized)."""

        # BKIR formula (vectorized)
        weighted_factors = (
            weights[:, 0] * responsibility_shares +
            weights[:, 1] * capability_shares +
            weights[:, 2] * equality_shares
        )

        korea_budgets = global_budgets * weighted_factors
        industry_budgets = korea_budgets * self.industry_fraction
        petrochem_budgets = industry_budgets * self.petrochem_fraction

        return korea_budgets, industry_budgets, petrochem_budgets

    def _calculate_statistics(
        self,
        korea_budgets: np.ndarray,
        industry_budgets: np.ndarray,
        petrochem_budgets: np.ndarray,
        scenarios: List[str]
    ) -> Dict[str, Any]:
        """Calculate summary statistics for budgets."""

        stats = {
            'korea_total': self._get_budget_stats(korea_budgets),
            'industry': self._get_budget_stats(industry_budgets),
            'petrochem': self._get_budget_stats(petrochem_budgets)
        }

        # Scenario-specific statistics
        scenario_stats = {}
        for scenario in ['1p5C', '2p0C']:
            mask = np.array(scenarios) == scenario
            count = np.sum(mask)
            if count > 0:
                scenario_stats[scenario] = {
                    'count': int(count),
                    'proportion': float(count / len(scenarios)),
                    'korea_total': self._get_budget_stats(korea_budgets[mask]),
                    'industry': self._get_budget_stats(industry_budgets[mask]),
                    'petrochem': self._get_budget_stats(petrochem_budgets[mask])
                }

        stats['by_scenario'] = scenario_stats

        return stats

    def _get_budget_stats(self, budgets: np.ndarray) -> Dict[str, float]:
        """Calculate statistical measures for budget array."""
        return {
            'mean': float(np.mean(budgets)),
            'median': float(np.median(budgets)),
            'std': float(np.std(budgets)),
            'min': float(np.min(budgets)),
            'max': float(np.max(budgets)),
            'p05': float(np.percentile(budgets, 5)),
            'p25': float(np.percentile(budgets, 25)),
            'p50': float(np.percentile(budgets, 50)),
            'p75': float(np.percentile(budgets, 75)),
            'p95': float(np.percentile(budgets, 95)),
            'cv': float(np.std(budgets) / np.mean(budgets)) if np.mean(budgets) > 0 else 0
        }

    def export_results(
        self,
        output_dir: Path,
        save_raw: bool = True
    ) -> None:
        """
        Export budget calculation results.

        Args:
            output_dir: Directory to save results
            save_raw: Whether to save raw Monte Carlo samples
        """
        if self.results is None:
            raise ValueError("No results to export. Run Monte Carlo first.")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = self.results['metadata']['timestamp']

        # Save statistics as JSON
        stats_file = output_dir / f'budget_statistics_{timestamp}.json'
        with open(stats_file, 'w') as f:
            json.dump({
                'statistics': self.results['statistics'],
                'metadata': self.results['metadata'],
                'config': {
                    'weights': self.weights_cfg,
                    'allocation_factors': {
                        'responsibility': self.uncertainty_cfg['responsibility']['mid'],
                        'capability': self.uncertainty_cfg['capability']['mu'],
                        'equality': self.uncertainty_cfg['equality']['mu']
                    },
                    'sectors': {
                        'industry_fraction': self.industry_fraction,
                        'petrochem_fraction': self.petrochem_fraction
                    }
                }
            }, f, indent=2, default=str)

        logger.info(f"Statistics saved to {stats_file}")

        # Save raw data if requested
        if save_raw:
            # Budget distributions
            budgets_df = pd.DataFrame({
                'korea_total': self.results['budgets']['korea_total'],
                'industry': self.results['budgets']['industry'],
                'petrochem': self.results['budgets']['petrochem'],
                'scenario': self.results['samples']['scenarios']
            })
            budgets_file = output_dir / f'budget_samples_{timestamp}.csv'
            budgets_df.to_csv(budgets_file, index=False)
            logger.info(f"Budget samples saved to {budgets_file}")

            # Allocation factors
            factors_df = pd.DataFrame({
                'global_budget': self.results['samples']['global_budgets'],
                'responsibility': self.results['samples']['responsibility_shares'],
                'capability': self.results['samples']['capability_shares'],
                'equality': self.results['samples']['equality_shares'],
                'weight_r': self.results['samples']['weights'][:, 0],
                'weight_c': self.results['samples']['weights'][:, 1],
                'weight_e': self.results['samples']['weights'][:, 2],
                'scenario': self.results['samples']['scenarios']
            })
            factors_file = output_dir / f'allocation_factors_{timestamp}.csv'
            factors_df.to_csv(factors_file, index=False)
            logger.info(f"Allocation factors saved to {factors_file}")

        logger.info(f"Results exported to {output_dir}")

    def get_summary(self) -> str:
        """
        Get human-readable summary of results.

        Returns:
            Formatted summary string
        """
        if self.results is None:
            return "No results available. Run Monte Carlo first."

        stats = self.results['statistics']
        meta = self.results['metadata']

        summary = []
        summary.append("=" * 60)
        summary.append("KOREAN CARBON BUDGET CALCULATION RESULTS")
        summary.append("=" * 60)
        summary.append(f"Analysis Date: {meta['timestamp']}")
        summary.append(f"Monte Carlo Draws: {meta['n_draws']}")
        summary.append(f"Scenario Mode: {meta['scenario_mode']}")
        summary.append("")

        # Overall statistics
        summary.append("KOREA TOTAL BUDGET:")
        summary.append(f"  Mean: {stats['korea_total']['mean']:.2e} tCO2")
        summary.append(f"  Median: {stats['korea_total']['median']:.2e} tCO2")
        summary.append(f"  90% CI: {stats['korea_total']['p05']:.2e} - {stats['korea_total']['p95']:.2e} tCO2")
        summary.append("")

        summary.append("INDUSTRY SECTOR BUDGET:")
        summary.append(f"  Mean: {stats['industry']['mean']:.2e} tCO2")
        summary.append(f"  Median: {stats['industry']['median']:.2e} tCO2")
        summary.append(f"  90% CI: {stats['industry']['p05']:.2e} - {stats['industry']['p95']:.2e} tCO2")
        summary.append("")

        summary.append("PETROCHEMICAL SECTOR BUDGET:")
        summary.append(f"  Mean: {stats['petrochem']['mean']:.2e} tCO2")
        summary.append(f"  Median: {stats['petrochem']['median']:.2e} tCO2")
        summary.append(f"  90% CI: {stats['petrochem']['p05']:.2e} - {stats['petrochem']['p95']:.2e} tCO2")
        summary.append("")

        # Scenario-specific if available
        if 'by_scenario' in stats and stats['by_scenario']:
            summary.append("SCENARIO-SPECIFIC RESULTS:")
            for scenario, scenario_stats in stats['by_scenario'].items():
                temp = "1.5°C" if scenario == "1p5C" else "2.0°C"
                summary.append(f"\n  {temp} Scenario ({scenario_stats['count']} draws, {scenario_stats['proportion']:.1%}):")
                summary.append(f"    Korea Total: {scenario_stats['korea_total']['median']:.2e} tCO2")
                summary.append(f"    Industry: {scenario_stats['industry']['median']:.2e} tCO2")
                summary.append(f"    Petrochem: {scenario_stats['petrochem']['median']:.2e} tCO2")

        summary.append("")
        summary.append("=" * 60)

        return "\n".join(summary)
