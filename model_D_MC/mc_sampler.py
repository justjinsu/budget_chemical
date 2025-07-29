"""
Monte Carlo Sampler for Carbon Budget Allocation

This module provides sampling functionality for uncertain parameters in the
carbon budget allocation model, including global budgets, allocation factors,
weights, and pathway parameters.
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from scipy.stats import triang, norm, beta
import logging

logger = logging.getLogger(__name__)


class Sampler:
    """
    Monte Carlo sampler for carbon budget allocation parameters.
    
    Handles sampling of:
    - Global carbon budgets (triangular distribution)
    - Allocation factors (responsibility, capability, equality)
    - User weights with noise
    - Front-loading parameters (lambda)
    - Curve types
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
        self.lambda_cfg = config['lambda_dist']
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
        
        # Lambda (front-loading) beta distribution
        lam = self.lambda_cfg
        self.lambda_dist = beta(a=lam['a'], b=lam['b'])
        
        logger.debug("All probability distributions initialized")
    
    def sample_global_budget(self, n: int) -> np.ndarray:
        """
        Sample global carbon budgets.
        
        Args:
            n: Number of samples
            
        Returns:
            Array of global budget samples (tCO2)
        """
        return self.global_budget_dist.rvs(size=n, random_state=self.rng)
    
    def sample_allocation_factors(self, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample allocation factors for BKIR formula.
        
        Args:
            n: Number of samples
            
        Returns:
            Tuple of (responsibility_shares, capability_shares, equality_shares)
        """
        Sr = self.resp_dist.rvs(size=n, random_state=self.rng)
        Sc = np.maximum(0, self.cap_dist.rvs(size=n, random_state=self.rng))
        Se = np.maximum(0, self.eq_dist.rvs(size=n, random_state=self.rng))
        
        return Sr, Sc, Se
    
    def sample_weights(self, n: int) -> np.ndarray:
        """
        Sample user weights with noise and renormalization.
        
        Adds ±2.5 percentage-point noise to base weights, clips to ≥0,
        and renormalizes to sum to 1.0.
        
        Args:
            n: Number of samples
            
        Returns:
            Array of shape (n, 3) with weights [wr, wc, we]
        """
        base_weights = np.array([
            self.weights_cfg['responsibility'],
            self.weights_cfg['capability'], 
            self.weights_cfg['equality']
        ])
        
        # Add ±2.5 percentage-point noise
        noise = self.rng.uniform(-0.025, 0.025, size=(n, 3))
        noisy_weights = base_weights[None, :] + noise
        
        # Clip to non-negative
        noisy_weights = np.maximum(0, noisy_weights)
        
        # Renormalize to sum to 1.0
        weight_sums = noisy_weights.sum(axis=1, keepdims=True)
        weight_sums = np.where(weight_sums > 0, weight_sums, 1.0)  # Avoid division by zero
        normalized_weights = noisy_weights / weight_sums
        
        return normalized_weights
    
    def sample_lambdas(self, n: int) -> np.ndarray:
        """
        Sample front-loading parameters (λ).
        
        Args:
            n: Number of samples
            
        Returns:
            Array of lambda values in [0, 1]
        """
        return self.lambda_dist.rvs(size=n, random_state=self.rng)
    
    def sample_curve_types(self, n: int) -> List[str]:
        """
        Sample curve types for emission pathways.
        
        Args:
            n: Number of samples
            
        Returns:
            List of curve type strings
        """
        return self.rng.choice(self.curve_types, size=n).tolist()
    
    def sample_all(self, n: int) -> Dict[str, Any]:
        """
        Sample all uncertain parameters for Monte Carlo analysis.
        
        Args:
            n: Number of Monte Carlo draws
            
        Returns:
            Dictionary containing all sampled parameters:
            - global_budgets: Global carbon budgets (tCO2)
            - responsibility_shares: Historical responsibility factors
            - capability_shares: Economic capability factors  
            - equality_shares: Population equality factors
            - weights: User weights with noise (n, 3)
            - lambdas: Front-loading parameters
            - curve_types: Emission pathway curve types
        """
        logger.info(f"Sampling {n} Monte Carlo draws")
        
        # Sample all parameters
        global_budgets = self.sample_global_budget(n)
        Sr, Sc, Se = self.sample_allocation_factors(n)
        weights = self.sample_weights(n)
        lambdas = self.sample_lambdas(n)
        curve_types = self.sample_curve_types(n)
        
        samples = {
            'global_budgets': global_budgets,
            'responsibility_shares': Sr,
            'capability_shares': Sc,
            'equality_shares': Se,
            'weights': weights,
            'lambdas': lambdas,
            'curve_types': curve_types
        }
        
        # Log sample statistics
        logger.info(f"Global budget range: {global_budgets.min():.2e} - {global_budgets.max():.2e} tCO2")
        logger.info(f"Responsibility shares: {Sr.min():.4f} - {Sr.max():.4f}")
        logger.info(f"Capability shares: {Sc.min():.4f} - {Sc.max():.4f}")
        logger.info(f"Equality shares: {Se.min():.4f} - {Se.max():.4f}")
        logger.info(f"Lambda range: {lambdas.min():.3f} - {lambdas.max():.3f}")
        
        return samples