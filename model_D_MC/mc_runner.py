import annotations
import numpy as np
from pathlib import Path
from mc_sampler import load_cfg, Sampler
from budgetCalculation import budgetAllocation
from pathwayCalculation import pathwayCalculator

IDX_R, IDX_C, IDX_E = 0, 1, 2

def main(cfg_path: str | Path):
    cfg       = load_cfg(cfg_path)
    sampler   = Sampler(cfg)
    gb_vec, w_mat, gdp_vec = sampler.sample_all()
    n = cfg['n_draws']
    alloc = budgetAllocation()     # 원본 Model‑D 클래스
    pc    = pathwayCalculator()    # 원본 경로 클래스 (아래 새 함수 추가됨)

    overshoot = []
    for i in range(n):
        shares = {
            'responsibility': alloc.get_responsibility_share(),
            'capability'    : gdp_vec[i],
            'equality'      : alloc.get_population_share(),
        }
        fair   = (w_mat[i, IDX_R]*shares['responsibility'] +
                  w_mat[i, IDX_C]*shares['capability']     +
                  w_mat[i, IDX_E]*shares['equality'])
        kr_budget   = gb_vec[i] * fair
        ind_budget  = kr_budget * cfg['industry_fraction']
        petro_budget= ind_budget * cfg['petrochem_fraction']

        path = pc.two_segment_linear(
                start_year=2024, mid_year=2035, end_year=2050,
                start_emissions=pc.get_base_emissions('petrochem'),
                budget=petro_budget)
        overshoot.append(max(0, path.sum() - petro_budget))

    overshoot = np.array(overshoot)
    print("P(overshoot):", (overshoot>0).mean()*100, "%")

if __name__ == '__main__':
    import sys; main(sys.argv[1] if len(sys.argv)>1 else 'mc_config.yaml')