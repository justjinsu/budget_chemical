# model_D_MC/mc_metrics.py
# -----------------------------------------------------------
# Monte-Carlo 결과 요약 유틸
# -----------------------------------------------------------
import numpy as np
from typing import Sequence, Tuple


# 팬 차트용 백분위 행렬 -------------------------------------
def fan_quantiles(
    emissions: np.ndarray,
    quantiles: Sequence[float] = (0.05, 0.25, 0.5, 0.75, 0.95),
) -> np.ndarray:
    """
    emissions : (n_draws, n_years)
    return    : (len(q), n_years)
    """
    return np.quantile(emissions, quantiles, axis=0)


# 예산 초과 확률 ---------------------------------------------
def overshoot_probability(
    cumulative: np.ndarray, budgets: np.ndarray
) -> float:
    """P(cumulative > budget)"""
    return float((cumulative > budgets).mean())


# 예산 p05–p95 범위 ------------------------------------------
def budget_range(budgets: np.ndarray) -> Tuple[float, float]:
    return tuple(np.quantile(budgets, [0.05, 0.95]))