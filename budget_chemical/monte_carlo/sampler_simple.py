"""
Simplified Monte Carlo Sampler for Carbon Budget Allocation

This module provides sampling functionality for uncertain parameters in the
simplified carbon budget allocation model (no lambda, only linear decay).
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from scipy.stats import triang, norm
import logging

logger = logging.getLogger(__name__)


class Sampler:
    """
    Simplified Monte Carlo sampler for carbon budget allocation parameters.
    
    Handles sampling of:
    - Global carbon budgets (triangular distribution)
    - Allocation factors (responsibility, capability, equality)
    - User weights with noise
    - Curve types (simplified to only linear)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize sampler with configuration parameters.
        
        Args:
            config: Configuration dictionary containing sampling parameters
        """
        self.config = config
        self.rng = np.random.RandomState(config.get('seed', 123))
        
        # Extract configuration sections
        self.global_budget_cfg = config['global_budget']
        self.weights_cfg = config['user_weights']
        self.uncertainty_cfg = config['uncertainty']
        self.curve_types = config['curve_types']
        
        # Setup distributions
        self._setup_distributions()
        
        logger.info(f"Sampler initialized with seed {config.get('seed', 123)}")
    
    def _setup_distributions(self) -> None:
        """Setup probability distributions for sampling."""
        
        # Global budget triangular distribution  
        gb = self.global_budget_cfg
        # For triangular: loc=low, scale=high-low, c=(mode-low)/(high-low)
        self.global_budget_dist = triang(
            loc=gb['low'],
            scale=gb['high'] - gb['low'],
            c=(gb['mid'] - gb['low']) / (gb['high'] - gb['low'])
        )
        
        # Responsibility triangular distribution
        resp = self.uncertainty_cfg['responsibility']
        self.resp_dist = triang(
            loc=resp['low'],
            scale=resp['high'] - resp['low'],
            c=(resp['mid'] - resp['low']) / (resp['high'] - resp['low'])
        )
        
        # Capability normal distribution
        cap = self.uncertainty_cfg['capability']
        self.cap_dist = norm(
            loc=cap['mu'],
            scale=cap['mu'] * cap['sd_pct']
        )
        
        # Equality normal distribution
        eq = self.uncertainty_cfg['equality']
        self.eq_dist = norm(
            loc=eq['mu'],
            scale=eq['mu'] * eq['sd_pct']
        )
        
        logger.debug("All probability distributions initialized")
    
    def sample_global_budgets(self, n: int) -> np.ndarray:
        """
        Sample global carbon budgets from triangular distribution.
        
        Args:
            n: Number of samples to generate
            
        Returns:
            Array of global budget samples (tCO2)
        """
        return self.global_budget_dist.rvs(size=n, random_state=self.rng)
    
    def sample_allocation_factors(self, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample allocation factors from their respective distributions.
        
        Args:
            n: Number of samples to generate
            
        Returns:
            Tuple of (responsibility_shares, capability_shares, equality_shares)
        """
        responsibility = self.resp_dist.rvs(size=n, random_state=self.rng)
        
        # Ensure capability shares are positive
        capability = np.abs(self.cap_dist.rvs(size=n, random_state=self.rng))
        
        # Ensure equality shares are positive  
        equality = np.abs(self.eq_dist.rvs(size=n, random_state=self.rng))
        
        return responsibility, capability, equality
    
    def sample_weights(self, n: int) -> np.ndarray:
        """
        Sample user weights with Gaussian noise.
        
        Args:
            n: Number of samples to generate
            
        Returns:
            Array of weight vectors (n x 3) for [responsibility, capability, equality]
        """
        base_weights = np.array([
            self.weights_cfg['responsibility'],
            self.weights_cfg['capability'], 
            self.weights_cfg['equality']
        ])
        
        # Add Gaussian noise (±2.5 percentage points standard deviation)
        noise_std = 0.025
        weights = np.zeros((n, 3))
        
        for i in range(n):
            # Add noise to each weight
            noisy_weights = base_weights + self.rng.normal(0, noise_std, 3)
            
            # Ensure all weights are positive
            noisy_weights = np.maximum(noisy_weights, 0.01)
            
            # Normalize to sum to 1.0
            noisy_weights = noisy_weights / np.sum(noisy_weights)
            
            weights[i] = noisy_weights
        
        return weights
    
    def sample_curve_types(self, n: int) -> List[str]:
        """
        Sample curve types randomly from available options.
        
        Args:
            n: Number of samples to generate
            
        Returns:
            List of curve type strings
        """
        # Randomly sample from available curve types
        return self.rng.choice(self.curve_types, size=n).tolist()
    
    def sample_all(self, n: int) -> Dict[str, Any]:
        """
        Sample all parameters for Monte Carlo analysis.
        
        Args:
            n: Number of Monte Carlo draws
            
        Returns:
            Dictionary containing all sampled parameters:
            - global_budgets: Global carbon budgets
            - responsibility_shares: Responsibility allocation factors
            - capability_shares: Capability allocation factors  
            - equality_shares: Equality allocation factors
            - weights: User weight vectors
            - curve_types: Curve type selections
        """
        logger.info(f"Sampling {n} Monte Carlo draws")
        
        # Sample all parameters
        global_budgets = self.sample_global_budgets(n)
        responsibility_shares, capability_shares, equality_shares = self.sample_allocation_factors(n)
        weights = self.sample_weights(n)
        curve_types = self.sample_curve_types(n)
        
        samples = {
            'global_budgets': global_budgets,
            'responsibility_shares': responsibility_shares,
            'capability_shares': capability_shares,
            'equality_shares': equality_shares,
            'weights': weights,
            'curve_types': curve_types
        }
        
        # Log sampling summary
        logger.info(f"Global budget range: {global_budgets.min():.2e} - {global_budgets.max():.2e} tCO2")
        logger.info(f"Responsibility shares: {responsibility_shares.min():.4f} - {responsibility_shares.max():.4f}")
        logger.info(f"Capability shares: {capability_shares.min():.4f} - {capability_shares.max():.4f}")
        logger.info(f"Equality shares: {equality_shares.min():.4f} - {equality_shares.max():.4f}")
        
        return samples