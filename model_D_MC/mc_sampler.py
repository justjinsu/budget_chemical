# model_D_MC/mc_sampler.py
# -----------------------------------------------------------
# Monte-Carlo 샘플러 (불확실 변수 3개)
#   1) 전 지구 잔여 탄소예산  Bglob  : 삼각분포(저·중·고)
#   2) 형평 가중치 w_r,w_c,w_e       : Dirichlet, 사용자 값 ±2.5 % 흔들림
#   3) 한국 GDP 비중 S_gdp           : 정규, ±5 % 흔들림
# -----------------------------------------------------------
from __future__ import annotations
from pathlib import Path
import numpy as np
import yaml
from typing import Dict, Any, Tuple


# ---------- YAML 설정 로드 -----------------------------------
def load_cfg(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------- 샘플러 클래스 ------------------------------------
class Sampler:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.get("seed", 42))

        # 사용자 입력 가중치 (합계 1.0)
        self.base_w = np.array(
            [
                cfg["user_weights"]["responsibility"],
                cfg["user_weights"]["capability"],
                cfg["user_weights"]["equality"],
            ],
            dtype=float,
        )

    # 1) 전 지구 잔여 탄소예산
    def _draw_global_budget(self, n: int) -> np.ndarray:
        a = self.cfg["global_budget"]["low"]
        m = self.cfg["global_budget"]["mid"]
        b = self.cfg["global_budget"]["high"]
        return self.rng.triangular(a, m, b, n)

    # 2) 형평 가중치
    def _draw_weights(self, n: int) -> np.ndarray:
        eps = self.rng.normal(0.0, 0.025, (n, 3))  # ±2.5 p-p
        w = np.clip(self.base_w + eps, 0.0, None)
        w /= w.sum(axis=1, keepdims=True)
        return w  # shape (n,3)

    # 3) 한국 GDP 비중
    def _draw_gdp_share(self, n: int) -> np.ndarray:
        g0 = self.cfg["korea_gdp_share"][2024]
        g1 = self.cfg["korea_gdp_share"][2050]
        mu = 0.5 * (g0 + g1)
        sigma = 0.05 * mu
        return self.rng.normal(mu, sigma, n)

    # 통합 샘플
    def sample_all(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """return (global_budget_vec, weight_matrix, gdp_share_vec)"""
        n = self.cfg["n_draws"]
        gb = self._draw_global_budget(n)
        w  = self._draw_weights(n)
        gdp = self._draw_gdp_share(n)
        return gb, w, gdp