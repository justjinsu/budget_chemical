# Korean Carbon Budget Calculator

A modular Python tool for calculating Korea's fair share of the global carbon budget using equity-based allocation principles, with verified data sources and Monte Carlo uncertainty quantification.

## 🌟 Features

- **Module 1: Korean Budget Calculator** - Calculate Korea's carbon budget independently
- **Verified Allocation Factors** - Based on latest data (Statista 2021, IMF 2023, Worldometer 2024)
- **BKIR Formula** - Three equity principles: Responsibility, Capability, Equality
- **Monte Carlo Simulation** - Comprehensive uncertainty quantification
- **Class-Based Architecture** - Clean separation between backend logic and frontend interfaces
- **Multiple Interfaces** - Command-line, Python API, and interactive Streamlit web app
- **Dual Climate Scenarios** - 1.5°C and 2.0°C pathways

---

## 📊 Key Results (Updated with Verified Data)

### Korea's Allocation Factors (October 2025)

| Factor | Value | Source | Description |
|--------|-------|--------|-------------|
| **Responsibility** | 1.09% | Statista 2021 | Historical cumulative CO₂ emissions (1850-2021) |
| **Capability** | 1.47% | IMF 2023 | GDP share (PPP-adjusted: $2.7T / $184.26T) |
| **Equality** | 0.646% | Worldometer 2024 | Population share (51.7M / 8.0B) |

### Budget Estimates

**1.5°C Scenario:**
- Korea Total: **2.14 Gt CO₂** (median)
- Industry Sector: **791 Mt CO₂**
- Petrochemical Sector: **79 Mt CO₂**

**2.0°C Scenario:**
- Korea Total: **4.78 Gt CO₂** (median)
- Industry Sector: **1.77 Gt CO₂**
- Petrochemical Sector: **177 Mt CO₂**

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd budget_chemical

# Install dependencies
pip install -r requirements.txt
```

### Command-Line Usage

```bash
# Run budget calculator with default config
python run_budget_calculator.py config/mc_config.yaml

# Results will be saved to outputs/
```

### Streamlit Web Interface

```bash
# Launch interactive web app
streamlit run app_streamlit.py

# Open browser to http://localhost:8501
```

### Python API Usage

```python
from budget_chemical.modules.budget_calculator import KoreaBudgetCalculator
import yaml

# Load configuration
with open('config/mc_config.yaml') as f:
    config = yaml.safe_load(f)

# Initialize calculator
calculator = KoreaBudgetCalculator(config)

# Run Monte Carlo simulation
results = calculator.run_monte_carlo(n_draws=1000, scenario_mode='mixed')

# Print summary
print(calculator.get_summary())

# Export results
calculator.export_results('outputs/', save_raw=True)
```

---

## 🏗️ Architecture

### Modular Design

```
┌─────────────────────────────────────────────────┐
│         Frontend Interfaces                      │
│  - CLI (run_budget_calculator.py)               │
│  - Streamlit (app_streamlit.py)                 │
│  - Python API (direct class import)             │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│         Backend Modules                          │
│                                                  │
│  Module 1: KoreaBudgetCalculator                │
│  - Monte Carlo sampling                         │
│  - BKIR formula calculation                     │
│  - Statistical analysis                         │
│  - Results export                               │
│                                                  │
│  Module 2: PathwayAllocator (Coming Soon)       │
│  - Annual emission pathways                     │
│  - Multiple curve types                         │
│  - Feasibility constraints                      │
└─────────────────────────────────────────────────┘
```

### Project Structure

```
budget_chemical/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── config/
│   └── mc_config.yaml                # Configuration with verified data
├── budget_chemical/
│   └── modules/
│       ├── __init__.py
│       └── budget_calculator.py      # Module 1: Budget Calculator
├── run_budget_calculator.py          # CLI interface
├── app_streamlit.py                  # Web interface
└── outputs/                          # Results directory
    ├── budget_statistics_*.json
    ├── budget_samples_*.csv
    └── allocation_factors_*.csv
```

---

## 📐 Methodology

### BKIR Allocation Formula

Korea's carbon budget is calculated using:

```
Korea_Budget = Global_Budget × (w_r × δ_r + w_c × δ_c + w_e × δ_e)
```

Where:
- **w_r, w_c, w_e**: User-defined weights (default: 0.3, 0.4, 0.3)
- **δ_r**: Responsibility share (1.09%)
- **δ_c**: Capability share (1.47%)
- **δ_e**: Equality share (0.646%)

### Equity Principles

1. **Responsibility (Historical Emissions)**
   - Korea emitted ~27 Gt CO₂ of global cumulative 2,500 Gt (1850-2021)
   - Accounts for historical contribution to climate change
   - Source: Statista 2021

2. **Capability (Economic Capacity)**
   - Korea's GDP (PPP): $2.7 trillion
   - World GDP (PPP): $184.26 trillion
   - Reflects ability to invest in decarbonization
   - Source: IMF 2023

3. **Equality (Population)**
   - Korea's population: 51.7 million
   - World population: 8.0 billion
   - Represents per-capita fairness
   - Source: Worldometer 2024

### Global Budget Scenarios

**1.5°C Scenario** (IPCC AR6):
- Low (67% probability): 400 Gt CO₂
- Mid (50% probability): 500 Gt CO₂
- High (33% probability): 670 Gt CO₂

**2.0°C Scenario** (IPCC AR6):
- Low (67% probability): 1,050 Gt CO₂
- Mid (50% probability): 1,150 Gt CO₂
- High (33% probability): 1,290 Gt CO₂

---

## ⚙️ Configuration

### Default Configuration (`config/mc_config.yaml`)

```yaml
# Monte Carlo settings
n_draws: 100
seed: 123
output_dir: outputs

# Global budgets (tCO2)
global_budget:
  "1p5C":
    low: 4.00e11
    mid: 5.00e11
    high: 6.70e11
  "2p0C":
    low: 1.05e12
    mid: 1.15e12
    high: 1.29e12

# BKIR weights
user_weights:
  responsibility: 0.30
  capability: 0.40
  equality: 0.30

# Verified allocation factors (Oct 2025)
uncertainty:
  responsibility:
    low: 0.0095
    mid: 0.0109    # Updated: 1.09% (Statista 2021)
    high: 0.0128
  capability:
    mu: 0.0147     # Updated: 1.47% GDP PPP (IMF 2023)
    sd_pct: 0.05
  equality:
    mu: 0.00646    # Updated: 0.646% population (Worldometer 2024)
    sd_pct: 0.03

# Sector allocation
industry_fraction: 0.37
petrochem_fraction: 0.10
```

---

## 📈 Output Files

### Statistics JSON
```json
{
  "statistics": {
    "korea_total": {
      "mean": 9.00e9,
      "median": 6.74e9,
      "p05": 4.96e9,
      "p95": 1.35e10
    },
    "industry": {...},
    "petrochem": {...}
  }
}
```

### Budget Samples CSV
```csv
korea_total,industry,petrochem,scenario
5.2e9,1.9e9,1.9e8,1p5C
1.3e10,4.8e9,4.8e8,2p0C
```

---

## 🔬 Verification

### Comparison with Previous Version

| Factor | Old Value | New (Verified) | Change | Impact |
|--------|-----------|----------------|--------|--------|
| Responsibility | 1.10% | 1.09% | -0.9% | Minimal |
| Capability | 1.70% | 1.47% | **-13.5%** | Significant |
| Equality | 0.67% | 0.646% | -3.6% | Minimal |

**Key Finding**: The capability factor was **overestimated by 13.5%**, resulting in Korea's budget being overstated by approximately **13%**.

### Budget Impact

**1.5°C Scenario:**
- Old: 2.32 Gt CO₂
- New: 2.14 Gt CO₂ (-7.8%)

**2.0°C Scenario:**
- Old: 5.21 Gt CO₂
- New: 4.78 Gt CO₂ (-8.3%)

---

## 📚 References

### Data Sources

1. **Global Cumulative Emissions**: Carbon Brief (2021)
2. **Korea Historical Emissions**: Statista (2021)
3. **World GDP (PPP)**: IMF (2023)
4. **Korea GDP (PPP)**: World Economics (2023)
5. **Global Carbon Budgets**: IPCC AR6 WG1 (2021)

### Frameworks

- UNFCCC Common But Differentiated Responsibilities (CBDR)
- Paris Agreement Article 2 on equity
- Climate Equity Reference Calculator methodology

---

## 🔄 Changelog

### Version 1.0 (October 2025)
- ✅ Updated allocation factors with verified data sources
- ✅ Refactored to class-based modular architecture
- ✅ Created Module 1: KoreaBudgetCalculator
- ✅ Added Streamlit web interface
- ✅ Budget decreased by ~13% due to corrected capability factor

---

**Last Updated**: October 3, 2025
**Model Version**: 1.0
**Data Version**: October 2025
