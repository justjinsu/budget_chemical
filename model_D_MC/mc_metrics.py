# model_D_MC/mc_metrics.py
# -----------------------------------------------------------
# Monte-Carlo 결과 요약 유틸
# -----------------------------------------------------------
import numpy as np
from typing import Sequence, Tuple


# ---------- 1) 팬 차트 백분위 -------------------------------
def fan_quantiles(
    emissions: np.ndarray,
    quantiles: Sequence[float] = (0.05, 0.25, 0.5, 0.75, 0.95),
) -> np.ndarray:
    """
    emissions : (n_draws, n_years)
    return    : (len(q), n_years)  백분위 행렬
    """
    return np.quantile(emissions, quantiles, axis=0)


# ---------- 2) 예산 초과 위험 -------------------------------
def overshoot_probability(
    cum_emissions: np.ndarray, budgets: np.ndarray
) -> float:
    """
    cum_emissions : (n_draws,)  – 각 시뮬레이션 최종 누적
    budgets       : (n_draws,)  – 시뮬레이션별 예산
    return        : 초과 비율 (0~1)
    """
    return float((cum_emissions > budgets).mean())


# ---------- 3) p05-p95 범위 -----------------------------------
def budget_range(budgets: np.ndarray) -> Tuple[float, float]:
    """Return (p05, p95)"""
    return tuple(np.quantile(budgets, [0.05, 0.95]))