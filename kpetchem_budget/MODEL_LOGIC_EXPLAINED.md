# K-PetChem Carbon Budget Model: Complete Logic Explanation

## 🎯 **Fundamental Concept**

The Korean Petrochemical Carbon Budget Model answers a critical question: **"What is Korea's fair share of the global carbon budget for petrochemicals?"**

### **Core Logic Flow**:
```
Global Carbon Budget → Korea's Fair Share → Korean Petrochemical Budget → Emission Pathways → Feasibility Analysis
```

---

## 📊 **The Budget Range Problem**

As demonstrated by our model results, the answer varies **dramatically** depending on which fairness principle you use:

### **Korean Budget Allocations (Mt CO₂e, 2023-2050)**:

| Scenario   | Population | GDP    | Historical | IEA Sector |
|------------|------------|--------|------------|------------|
| 1.5C-67%   | **1,016**  | 2,772  | 2,156      | **180**    |
| 1.5C-50%   | **1,294**  | 3,528  | 2,744      | **180**    |
| 1.7C-50%   | **2,125**  | 5,796  | 4,508      | **180**    |
| 2.0C-67%   | **2,864**  | 7,812  | 6,076      | **180**    |

### **Range Analysis**:
- **Minimum**: 180 Mt CO₂e (IEA Sector)
- **Maximum**: 7,812 Mt CO₂e (GDP, 2.0°C)
- **Range**: ±3,816 Mt CO₂e
- **Ratio**: **43.4x difference** between strictest and most generous!

---

## 🇰🇷 **Korea's Global Shares - The Foundation**

The model uses Korea's actual shares of different global indicators:

| Allocation Principle | Korea's Share | Rationale |
|---------------------|---------------|-----------|
| **Population** | **0.66%** | Equal per-capita emissions rights |
| **GDP** | **1.80%** | Emissions proportional to economic capacity |
| **Historical GHG** | **1.40%** | Responsibility for past emissions |
| **Production** | **3.00%** | Sector-specific activity share |

### **Why These Differ**:
- **Population**: Korea has relatively high emissions per capita
- **GDP**: Korea is economically developed (higher emissions capacity)
- **Historical**: Korea industrialized later (lower historical responsibility)
- **Production**: Korea is a major petrochemical producer (higher activity share)

---

## 🧮 **Allocation Standards Explained**

### **1. Population-Based Allocation**
```
Korean Budget = Global Budget × (Korean Population / World Population)
              = Global Budget × 0.66%
```
**Philosophy**: Everyone has equal right to emit
**Result**: Most restrictive for developed countries like Korea

### **2. GDP-Based Allocation**
```
Korean Budget = Global Budget × (Korean GDP / World GDP)
              = Global Budget × 1.80%
```
**Philosophy**: Richer countries can afford more abatement
**Result**: Higher budgets for economically developed countries

### **3. Historical GHG Allocation**
```
Korean Budget = Global Budget × (Korean Historical Emissions / World Historical)
              = Global Budget × 1.40%
```
**Philosophy**: Countries responsible for climate problem should do more
**Result**: Moderate burden for Korea (late industrializer)

### **4. IEA Sector Allocation**
```
Korean Budget = min(IEA Sector Budget, Global Budget) × Korean Production Share
              = min(6 Gt, Global Budget) × 3.00%
```
**Philosophy**: Sector-specific technical pathway
**Result**: Fixed budget regardless of global scenario (most restrictive)

---

## 🛤️ **From Budget to Pathways**

Once allocated, the budget constrains emission pathways (2023-2050):

### **Example: 1.5°C-50% Scenario**

**Starting Point**: 50 Mt CO₂e/year (2023 Korean petrochemical emissions)

| Method | Budget | 2035 Target | 2050 Target | Feasibility |
|--------|--------|-------------|-------------|-------------|
| Population | 1,294 Mt | 25 Mt/year | 0 Mt/year | **Challenging** |
| GDP | 3,528 Mt | 35 Mt/year | 0 Mt/year | **Moderate** |
| Historical | 2,744 Mt | 30 Mt/year | 0 Mt/year | **Moderate** |
| IEA Sector | 180 Mt | 5 Mt/year | 0 Mt/year | **Extremely Difficult** |

### **Pathway Families Tested**:
1. **Linear Decline**: Straight line to zero
2. **Constant Rate**: Fixed % reduction per year
3. **Logistic Decline**: S-curve (slow start, rapid middle, slow end)
4. **IEA Proxy**: Three-phase trajectory following IEA roadmap

---

## 🎲 **Monte Carlo Uncertainty Expansion**

The model expands beyond deterministic calculations to explore uncertainty:

### **768 Deterministic Cases**:
- **4** Budget scenarios (1.5C-67%, 1.5C-50%, 1.7C-50%, 2.0C-67%)
- **4** Allocation rules (Population, GDP, Historical, IEA)
- **4** Start years (2023, 2025, 2030, 2035)
- **3** Net-zero years (2045, 2050, 2055)
- **4** Pathway families (Linear, Constant, Logistic, IEA)

### **100 Monte Carlo Samples per Case** = **76,800 Total Simulations**:
- **Global budget uncertainty**: ±10% (normal distribution)
- **Korean production share**: Log-normal uncertainty
- **Pathway parameters**: Triangular distributions for curve shapes

### **Output**: 
- **Percentile ribbons** (5th-95th percentile uncertainty bands)
- **Composite visualization** showing full range of possibilities
- **Milestone uncertainty** (2035, 2050 emission distributions)

---

## 🎯 **Policy Implications**

### **The Fairness Debate**:
The 43x difference in budget allocations reflects real debates in international climate policy:

- **Developing countries** favor **population-based** allocation (equal per capita)
- **Developed countries** often prefer **capability-based** allocation (GDP)
- **Climate activists** emphasize **historical responsibility**
- **Industry** prefers **technical sectoral** approaches

### **Korean Petrochemical Strategy**:
The model quantifies what each approach means for Korea:

1. **Conservative Planning**: Use population-based allocation (strictest)
2. **Moderate Planning**: Use historical GHG allocation (middle ground)
3. **Ambitious Planning**: Use GDP-based allocation (most generous)
4. **Technical Planning**: Use IEA sector pathway (technically grounded)

---

## 🔍 **Model Validation**

### **Data Sources**:
- **Global budgets**: IPCC carbon budget estimates
- **Korean shares**: World Bank, IEA, national statistics
- **IEA pathways**: IEA Net Zero Roadmap for chemicals
- **Baseline emissions**: Korean petrochemical industry data

### **Assumptions**:
- Linear relationship between global and sectoral budgets
- Korean shares remain constant (2023-2050)
- No carbon leakage or trade effects
- Direct + indirect emissions only (Scope 1+2)

### **Limitations**:
- No explicit technology pathways
- No economic feedback effects
- No policy interaction modeling
- Limited to CO₂ (no CH₄, N₂O)

---

## 📈 **Practical Use Cases**

### **For Policymakers**:
- Compare fairness principles quantitatively
- Assess domestic policy ambition levels
- Prepare for international negotiations
- Set science-based targets

### **For Industry**:
- Strategic planning under uncertainty
- Investment decision support
- Technology pathway assessment
- Risk management

### **For Researchers**:
- Scenario analysis framework
- Uncertainty quantification
- Policy impact assessment
- International comparison

---

## 🚀 **Key Innovations**

1. **Multi-principle allocation**: Compares all major fairness approaches
2. **Full uncertainty quantification**: 76,800 Monte Carlo simulations
3. **Integrated pathway analysis**: From global budgets to emission trajectories
4. **Interactive visualization**: Real-time exploration of assumptions
5. **Policy-relevant metrics**: 2035/2050 milestones, budget utilization

---

## 💡 **Bottom Line**

The K-PetChem model reveals that **"Korea's fair share"** isn't a single number—it's a **range spanning two orders of magnitude** depending on your fairness philosophy.

This range (180-7,812 Mt CO₂e) represents the **real policy space** for Korean petrochemical decarbonization. The model helps navigate this space systematically, quantifying trade-offs and uncertainties that are usually debated qualitatively.

**The power isn't in finding "the right answer"—it's in understanding the full spectrum of possibilities.**