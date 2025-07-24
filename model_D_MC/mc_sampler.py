# model_D_MC/mc_sampler.py
# -----------------------------------------------------------
# Monte-Carlo Sampler for Carbon Budget Allocation Model
# Implements 5-variable uncertainty sampling with exponential decay
# -----------------------------------------------------------
from __future__ import annotations
from pathlib import Path
import numpy as np
import yaml
from typing import Dict, Any, Tuple


def load_cfg(path: str | Path = "mc_config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    # If relative path, make it relative to this script's directory
    if not Path(path).is_absolute():
        script_dir = Path(__file__).parent
        path = script_dir / path
    
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class Sampler:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.get("seed", 42))
        
        # Base user weights
        self.base_w = np.array([
            cfg["user_weights"]["responsibility"],
            cfg["user_weights"]["capability"], 
            cfg["user_weights"]["equality"]
        ], dtype=float)

    def _draw_global_budget(self, n: int) -> np.ndarray:
        """Sample global carbon budget from triangular distribution"""
        low = self.cfg["global_budget"]["low"]
        mid = self.cfg["global_budget"]["mid"] 
        high = self.cfg["global_budget"]["high"]
        return self.rng.triangular(low, mid, high, n)

    def _draw_responsibility_factor(self, n: int) -> np.ndarray:
        """Sample responsibility factor δr from triangular distribution"""
        unc = self.cfg["uncertainty"]["responsibility"]
        return self.rng.triangular(unc["low"], unc["mid"], unc["high"], n)

    def _draw_capability_factor(self, n: int) -> np.ndarray:
        """Sample capability factor δc from normal distribution"""
        unc = self.cfg["uncertainty"]["capability"]
        mu = unc["mu"]
        sigma = mu * unc["sd_pct"]  # 5% of mean
        return self.rng.normal(mu, sigma, n)

    def _draw_equality_factor(self, n: int) -> np.ndarray:
        """Sample equality factor δe from normal distribution"""
        unc = self.cfg["uncertainty"]["equality"]
        mu = unc["mu"]
        sigma = mu * unc["sd_pct"]  # 3% of mean
        return self.rng.normal(mu, sigma, n)

    def _draw_user_weights(self, n: int) -> np.ndarray:
        """Sample user weights with ±2.5 p-p clipping and re-normalization"""
        # Add ±2.5 percentage point noise
        eps = self.rng.normal(0.0, 0.025, (n, 3))
        w = self.base_w + eps
        
        # Clip to ensure non-negative
        w = np.clip(w, 0.0, None)
        
        # Re-normalize to sum to 1
        w = w / w.sum(axis=1, keepdims=True)
        
        return w

    def sample_all(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample all 5 uncertainty variables
        
        Returns:
            Tuple of (global_budget, responsibility_factor, capability_factor, 
                     equality_factor, user_weights)
        """
        n = self.cfg["n_draws"]
        
        global_budget = self._draw_global_budget(n)
        responsibility = self._draw_responsibility_factor(n)
        capability = self._draw_capability_factor(n)
        equality = self._draw_equality_factor(n)
        weights = self._draw_user_weights(n)
        
        return global_budget, responsibility, capability, equality, weights