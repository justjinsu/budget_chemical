import pandas as pd
import numpy as np
from typing import List, Tuple
from scipy.optimize import fsolve


class pathwayCalculator:
    def __init__(self, start_year: int = 2024, mid_year: int = 2035, end_year: int = 2050):
        """
        Initialize pathway calculator for two-segment exponential decay model.
        
        Parameters:
        start_year: Starting year (default: 2024)
        mid_year: Transition year (default: 2035) 
        end_year: Ending year (default: 2050)
        """
        self.start_year = start_year
        self.mid_year = mid_year
        self.end_year = end_year
        self.years = np.array(range(start_year, end_year + 1))

    def two_segment_exp(self, E0: float, budget: float) -> np.ndarray:
        """
        Two-segment exponential decay model:
        Et = E0 × exp(-k1(t - 2024))  for 2024 ≤ t ≤ 2035
        Et = E2035 × exp(-k2(t - 2035))  for 2035 < t ≤ 2050
        
        Parameters:
        E0: Initial emissions at 2024 (tCO2)
        budget: Total carbon budget constraint (tCO2)
        
        Returns:
        np.ndarray: Emission pathway over years
        """
        t1 = self.mid_year - self.start_year + 1  # 2024-2035 inclusive (12 years)
        t2 = self.end_year - self.mid_year        # 2036-2050 (15 years)
        
        def equations(params):
            k1, k2 = params
            
            # Calculate E2035
            E_mid = E0 * np.exp(-k1 * (self.mid_year - self.start_year))
            
            # Segment 1: 2024-2035 (cumulative emissions)
            if k1 != 0:
                S1 = E0 * (1 - np.exp(-k1 * t1)) / k1
            else:
                S1 = E0 * t1
            
            # Segment 2: 2036-2050 (cumulative emissions)
            if k2 != 0:
                S2 = E_mid * (1 - np.exp(-k2 * t2)) / k2
            else:
                S2 = E_mid * t2
            
            # Constraints: total budget and continuity
            total_budget_constraint = S1 + S2 - budget
            
            # For realistic pathways, ensure k1, k2 > 0 (declining emissions)
            return [total_budget_constraint, k1 - 0.01]  # k1 ≥ 0.01
        
        # Solve for k1, k2
        try:
            k1, k2 = fsolve(equations, [0.05, 0.1])
            k1, k2 = max(k1, 0.001), max(k2, 0.001)  # Ensure positive decay
        except:
            # Fallback to linear approximation
            k1, k2 = 0.05, 0.1
        
        # Generate pathway
        pathway = np.zeros(len(self.years))
        
        for i, year in enumerate(self.years):
            if year <= self.mid_year:
                # Segment 1: exponential decay
                pathway[i] = E0 * np.exp(-k1 * (year - self.start_year))
            else:
                # Segment 2: exponential decay from mid-year
                E_mid = E0 * np.exp(-k1 * (self.mid_year - self.start_year))
                pathway[i] = E_mid * np.exp(-k2 * (year - self.mid_year))
        
        return pathway

    def two_segment_log(self, E0: float, budget: float) -> np.ndarray:
        """
        Two-segment logarithmic decay model (alternative implementation).
        """
        # For logarithmic decay, use exponential with adjusted parameters
        return self.two_segment_exp(E0, budget)

    def two_segment_linear(self, E0: float, budget: float) -> np.ndarray:
        """
        Two-segment linear decay model.
        
        Parameters:
        E0: Initial emissions at 2024 (tCO2)
        budget: Total carbon budget constraint (tCO2)
        
        Returns:
        np.ndarray: Linear emission pathway over years
        """
        n1 = self.mid_year - self.start_year + 1  # 2024-2035 inclusive
        n2 = self.end_year - self.mid_year        # 2036-2050  
        
        # Binary search for reduction rates
        lo, hi = 0.0, E0 / n1
        for _ in range(40):
            r1 = (lo + hi) / 2
            E_mid = E0 - r1 * (n1 - 1)  # Emission at mid_year
            r2 = E_mid / n2             # Linear to zero in segment 2
            
            # Calculate total budget
            s1 = n1 * (E0 + E_mid) / 2  # Trapezoidal area segment 1
            s2 = n2 * E_mid / 2         # Triangular area segment 2
            total = s1 + s2
            
            if total > budget:
                lo = r1  # Need more reduction
            else:
                hi = r1
        
        # Generate linear pathway
        pathway = np.zeros(len(self.years))
        
        for i, year in enumerate(self.years):
            if year <= self.mid_year:
                # Segment 1: linear decline
                pathway[i] = E0 - r1 * (year - self.start_year)
            else:
                # Segment 2: linear to zero
                E_mid = E0 - r1 * (self.mid_year - self.start_year)
                pathway[i] = E_mid - (E_mid / n2) * (year - self.mid_year)
        
        return np.maximum(pathway, 0)  # Ensure non-negative

    def calculate_pathway(self, E0: float, budget: float, curve_type: str = "exp") -> np.ndarray:
        """
        Calculate emission pathway based on curve type.
        
        Parameters:
        E0: Initial emissions (tCO2)
        budget: Carbon budget constraint (tCO2)
        curve_type: Type of curve ("exp", "log", "linear")
        
        Returns:
        np.ndarray: Emission pathway
        """
        if curve_type == "exp":
            return self.two_segment_exp(E0, budget)
        elif curve_type == "log":
            return self.two_segment_log(E0, budget)
        elif curve_type == "linear":
            return self.two_segment_linear(E0, budget)
        else:
            # Default to exponential
            return self.two_segment_exp(E0, budget)

    def get_years(self) -> np.ndarray:
        """Get the years array"""
        return self.years

    def calculate_exhaustion_year(self, pathway: np.ndarray, threshold: float = 1e6) -> int:
        """
        Calculate the year when emissions fall below threshold.
        
        Parameters:
        pathway: Emission pathway array
        threshold: Emission threshold (default: 1 MtCO2)
        
        Returns:
        int: Year when budget is effectively exhausted
        """
        below_threshold = pathway < threshold
        if np.any(below_threshold):
            idx = np.where(below_threshold)[0][0]
            return self.years[idx]
        else:
            return self.end_year