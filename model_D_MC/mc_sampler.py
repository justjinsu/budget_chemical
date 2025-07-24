# model_D_MC/mc_sampler.py
# -----------------------------------------------------------
# Monte-Carlo 샘플러 ― 불확실 변수 3개:
#   1) 전 지구 잔여탄소예산  B_glob  (삼각분포)
#   2) 형평성 가중치         w_r,w_c,w_e (Dirichlet±2.5 p-p)
#   3) 한국 GDP 비중         S_gdp (정규 ±5 %)
# -----------------------------------------------------------
from __future__ import annotations
from pathlib import Path
import numpy as np
import yaml
from typing import Tuple, Dict, Any


# -----------------------------------------------------------
# YAML 로드
# -----------------------------------------------------------
def load_cfg(path: str | Path) -> Dict[str, Any]:
    """YAML 설정파일 읽기"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# -----------------------------------------------------------
# 샘플러 클래스
# -----------------------------------------------------------
class Sampler:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.get("seed", 42))
        # Pre-extract 상수
        self.base_weights = np.array(
            [
                cfg["user_weights"]["responsibility"],
                cfg["user_weights"]["capability"],
                cfg["user_weights"]["equality"],
            ],
            dtype=float,
        )

    # ---------- 1) 글로벌 예산 ---------------------------------
    def _draw_global_budget(self, n: int) -> np.ndarray:
        a = self.cfg["global_budget"]["low"]
        m = self.cfg["global_budget"]["mid"]
        b = self.cfg["global_budget"]["high"]
        return self.rng.triangular(a, m, b, n)

    # ---------- 2) 형평 가중치 ----------------------------------
    def _draw_weights(self, n: int) -> np.ndarray:
        # ±2.5 p-p 오차를 정규로 더해준 뒤 0-컷 & 정규화
        eps = self.rng.normal(0.0, 0.025, (n, 3))
        w = np.clip(self.base_weights + eps, 0.0, None)
        w /= w.sum(axis=1, keepdims=True)
        return w  # (n,3)

    # ---------- 3) GDP 비중 -------------------------------------
    def _draw_gdp_share(self, n: int) -> np.ndarray:
        g0 = self.cfg["korea_gdp_share"][2024]
        g1 = self.cfg["korea_gdp_share"][2050]
        mu = 0.5 * (g0 + g1)
        sigma = 0.05 * mu
        return self.rng.normal(mu, sigma, n)

    # ---------- 샘플링 전체 --------------------------------------
    def sample_all(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """return (global_budget, weight_matrix, gdp_share)"""
        n = self.cfg["n_draws"]
        gb_vec = self._draw_global_budget(n)
        w_mat = self._draw_weights(n)
        gdp_vec = self._draw_gdp_share(n)
        return gb_vec, w_mat, gdp_vec