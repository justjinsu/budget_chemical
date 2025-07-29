"""
Monte Carlo Metrics and Statistical Analysis

This module provides functions for calculating quantiles, summary statistics,
and performance metrics for the Monte Carlo carbon budget analysis.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


def compute_fan_quantiles(pathways: np.ndarray, quantiles: List[float]) -> np.ndarray:
    """
    Compute quantiles across pathway ensemble for fan chart visualization.
    
    Args:
        pathways: Array of emission pathways (n_draws, n_years)
        quantiles: List of quantile levels (e.g., [0.05, 0.25, 0.5, 0.75, 0.95])
        
    Returns:
        Array of quantiles (n_quantiles, n_years)
    """
    if pathways.size == 0:
        raise ValueError("Empty pathways array provided")
    
    n_draws, n_years = pathways.shape
    n_quantiles = len(quantiles)
    
    logger.debug(f"Computing quantiles for {n_draws} pathways over {n_years} years")
    
    # Compute quantiles for each year
    quantile_data = np.zeros((n_quantiles, n_years))
    
    for i, q in enumerate(quantiles):
        quantile_data[i, :] = np.percentile(pathways, q * 100, axis=0)
    
    # Validate results
    for i in range(n_years):
        year_quantiles = quantile_data[:, i]
        if not np.all(year_quantiles[:-1] <= year_quantiles[1:]):
            logger.warning(f"Non-monotonic quantiles detected at year index {i}")
    
    logger.debug(f"Successfully computed {n_quantiles} quantiles")
    return quantile_data


def calculate_summary_stats(pathways: np.ndarray, years: np.ndarray) -> Dict[str, Any]:
    """
    Calculate comprehensive summary statistics for emission pathways.
    
    Args:
        pathways: Array of emission pathways (n_draws, n_years)
        years: Array of years corresponding to pathway columns
        
    Returns:
        Dictionary with summary statistics
    """
    n_draws, n_years = pathways.shape
    
    # Basic pathway statistics
    mean_pathway = np.mean(pathways, axis=0)
    median_pathway = np.median(pathways, axis=0)
    std_pathway = np.std(pathways, axis=0)
    
    # Initial and final emissions
    initial_emissions = pathways[:, 0]  # 2024 emissions
    final_emissions = pathways[:, -1]   # 2050 emissions
    
    # Cumulative emissions (budget utilization)
    cumulative_emissions = np.trapz(pathways, dx=1, axis=1)
    
    # Reduction rates (from start to end)
    reduction_rates = np.zeros(n_draws)
    for i in range(n_draws):
        if initial_emissions[i] > 0:
            reduction_rates[i] = ((initial_emissions[i] - final_emissions[i]) / 
                                 initial_emissions[i] * 100)
        else:
            reduction_rates[i] = 0
    
    # Peak emission years (if any pathway increases before decreasing)
    peak_years = []
    for i in range(n_draws):
        pathway = pathways[i, :]
        peak_idx = np.argmax(pathway)
        peak_years.append(years[peak_idx])
    peak_years = np.array(peak_years)
    
    # Zero-crossing years (when emissions reach near-zero)
    zero_threshold = 1.0  # tCO2/year threshold for "zero" emissions
    zero_years = []
    for i in range(n_draws):
        pathway = pathways[i, :]
        zero_indices = np.where(pathway <= zero_threshold)[0]
        if len(zero_indices) > 0:
            zero_years.append(years[zero_indices[0]])
        else:
            zero_years.append(years[-1])  # If never reaches zero, use final year
    zero_years = np.array(zero_years)
    
    # Mid-point emissions (2035)
    mid_year = 2035
    if mid_year in years:
        mid_idx = np.where(years == mid_year)[0][0]
        mid_emissions = pathways[:, mid_idx]
    else:
        # Interpolate if 2035 not exactly in years array
        mid_emissions = np.interp(mid_year, years, mean_pathway) * np.ones(n_draws)
    
    # Pathway variability metrics
    cv_over_time = std_pathway / (mean_pathway + 1e-6)  # Coefficient of variation
    max_variability_year = years[np.argmax(cv_over_time)]
    
    # Compile statistics
    stats = {
        # Basic statistics
        'n_draws': int(n_draws),
        'n_years': int(n_years),
        'timeline': [int(years[0]), int(years[-1])],
        
        # Emissions statistics
        'initial_emissions': {
            'mean': float(np.mean(initial_emissions)),
            'median': float(np.median(initial_emissions)),
            'std': float(np.std(initial_emissions)),
            'p05': float(np.percentile(initial_emissions, 5)),
            'p95': float(np.percentile(initial_emissions, 95))
        },
        
        'final_emissions': {
            'mean': float(np.mean(final_emissions)),
            'median': float(np.median(final_emissions)),
            'std': float(np.std(final_emissions)),
            'p05': float(np.percentile(final_emissions, 5)),
            'p95': float(np.percentile(final_emissions, 95))
        },
        
        'mid_emissions': {
            'mean': float(np.mean(mid_emissions)),
            'median': float(np.median(mid_emissions)),
            'std': float(np.std(mid_emissions)),
            'p05': float(np.percentile(mid_emissions, 5)),
            'p95': float(np.percentile(mid_emissions, 95))
        },
        
        # Cumulative statistics
        'cumulative_emissions': {
            'mean': float(np.mean(cumulative_emissions)),
            'median': float(np.median(cumulative_emissions)),
            'std': float(np.std(cumulative_emissions)),
            'p05': float(np.percentile(cumulative_emissions, 5)),
            'p95': float(np.percentile(cumulative_emissions, 95))
        },
        
        # Reduction statistics
        'reduction_rates': {
            'mean': float(np.mean(reduction_rates)),
            'median': float(np.median(reduction_rates)),
            'std': float(np.std(reduction_rates)),
            'p05': float(np.percentile(reduction_rates, 5)),
            'p95': float(np.percentile(reduction_rates, 95))
        },
        
        # Timeline statistics
        'peak_years': {
            'mean': float(np.mean(peak_years)),
            'median': float(np.median(peak_years)),
            'most_common': int(mode_of_array(peak_years))
        },
        
        'zero_crossing_years': {
            'mean': float(np.mean(zero_years)),
            'median': float(np.median(zero_years)),
            'p05': float(np.percentile(zero_years, 5)),
            'p95': float(np.percentile(zero_years, 95))
        },
        
        # Variability metrics
        'pathway_variability': {
            'mean_cv': float(np.mean(cv_over_time)),
            'max_cv': float(np.max(cv_over_time)),
            'max_variability_year': int(max_variability_year)
        },
        
        # Pathway shape statistics
        'pathway_characteristics': analyze_pathway_shapes(pathways, years)
    }
    
    logger.debug("Summary statistics calculated successfully")
    return stats


def analyze_pathway_shapes(pathways: np.ndarray, years: np.ndarray) -> Dict[str, Any]:
    """
    Analyze the shape characteristics of emission pathways.
    
    Args:
        pathways: Array of emission pathways (n_draws, n_years)
        years: Array of years
        
    Returns:
        Dictionary with shape analysis results
    """
    n_draws, n_years = pathways.shape
    
    # Classify pathway shapes
    monotonic_decreasing = 0
    early_peak = 0
    late_peak = 0
    oscillating = 0
    
    # Front-loading vs back-loading analysis
    mid_idx = len(years) // 2
    front_loaded = 0
    back_loaded = 0
    
    # Curvature analysis
    convex_paths = 0
    concave_paths = 0
    
    for i in range(n_draws):
        pathway = pathways[i, :]
        
        # Check monotonicity
        if np.all(np.diff(pathway) <= 0):
            monotonic_decreasing += 1
        
        # Find peak position
        peak_idx = np.argmax(pathway)
        if peak_idx == 0:
            monotonic_decreasing += 1  # Peak at start
        elif peak_idx < n_years // 3:
            early_peak += 1
        elif peak_idx > 2 * n_years // 3:
            late_peak += 1
        
        # Check for oscillations (more than 2 direction changes)
        direction_changes = np.sum(np.diff(np.sign(np.diff(pathway))) != 0)
        if direction_changes > 2:
            oscillating += 1
        
        # Front vs back loading
        front_area = np.trapz(pathway[:mid_idx], dx=1)
        back_area = np.trapz(pathway[mid_idx:], dx=1)
        total_area = front_area + back_area
        
        if total_area > 0:
            front_fraction = front_area / total_area
            if front_fraction > 0.6:
                front_loaded += 1
            elif front_fraction < 0.4:
                back_loaded += 1
        
        # Curvature analysis (second derivative)
        if n_years >= 3:
            second_deriv = np.diff(pathway, 2)
            avg_curvature = np.mean(second_deriv)
            if avg_curvature > 0.1:
                convex_paths += 1
            elif avg_curvature < -0.1:
                concave_paths += 1
    
    return {
        'monotonic_decreasing_fraction': monotonic_decreasing / n_draws,
        'early_peak_fraction': early_peak / n_draws,
        'late_peak_fraction': late_peak / n_draws,
        'oscillating_fraction': oscillating / n_draws,
        'front_loaded_fraction': front_loaded / n_draws,
        'back_loaded_fraction': back_loaded / n_draws,
        'convex_fraction': convex_paths / n_draws,
        'concave_fraction': concave_paths / n_draws
    }


def mode_of_array(arr: np.ndarray) -> float:
    """
    Calculate the mode (most frequent value) of an array.
    
    Args:
        arr: Input array
        
    Returns:
        Mode value
    """
    values, counts = np.unique(arr, return_counts=True)
    mode_idx = np.argmax(counts)
    return values[mode_idx]


def calculate_budget_utilization(pathways: np.ndarray, budgets: np.ndarray) -> Dict[str, float]:
    """
    Calculate budget utilization statistics.
    
    Args:
        pathways: Array of emission pathways (n_draws, n_years)
        budgets: Array of allocated budgets (n_draws,)
        
    Returns:
        Dictionary with utilization statistics
    """
    # Calculate cumulative emissions
    cumulative_emissions = np.trapz(pathways, dx=1, axis=1)
    
    # Calculate utilization rates
    utilization_rates = cumulative_emissions / budgets
    
    # Calculate overshoot
    overshoot = np.maximum(0, cumulative_emissions - budgets)
    overshoot_fraction = overshoot / budgets
    
    # Budget efficiency (how close to budget without overshooting)
    undershoot = np.maximum(0, budgets - cumulative_emissions)
    undershoot_fraction = undershoot / budgets
    
    utilization_stats = {
        'mean_utilization': float(np.mean(utilization_rates)),
        'median_utilization': float(np.median(utilization_rates)),
        'utilization_std': float(np.std(utilization_rates)),
        'overshoot_probability': float(np.mean(overshoot > 0)),
        'mean_overshoot_fraction': float(np.mean(overshoot_fraction)),
        'mean_undershoot_fraction': float(np.mean(undershoot_fraction)),
        'efficient_pathways_fraction': float(np.mean((utilization_rates >= 0.95) & (utilization_rates <= 1.05))),
        'utilization_range': [float(np.min(utilization_rates)), float(np.max(utilization_rates))]
    }
    
    return utilization_stats