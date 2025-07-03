# Korean Petrochemical Carbon Budget Toolkit v2.0

**Advanced Monte Carlo simulation engine for carbon budget allocation and emission pathway optimization (2023-2050)**

## Quickstart

```bash
# Install dependencies
pip install pandas numpy scipy matplotlib plotly streamlit pyarrow

# Launch interactive dashboard
streamlit run dashboard/app.py

# Run Monte Carlo simulation (14,400 scenarios)
python -m kpetchem_budget.simulator
```

## Key Features

- **🎯 2023-2050 Timeline**: Complete coverage from current baseline to net-zero targets
- **🎲 Monte Carlo Engine**: 14,400 simulations (144 deterministic cases × 100 MC samples)
- **📊 Four Allocation Methods**: Population, GDP, Historical GHG, IEA Sector pathway
- **🛤️ Four Pathway Families**: Linear-to-zero, Constant rate, Logistic decline, IEA proxy
- **⚡ High Performance**: <30s execution on 8 cores, <20MB Parquet storage
- **📈 Interactive Dashboard**: Real-time uncertainty visualization with milestone tracking

## Data Inputs

**Global Budget Data** (`globalbudget.csv`): Temperature scenarios (1.5°C-67%, 1.5°C-50%, 2.0°C-67%) with 2023-2050 cumulative budgets in Gt CO₂e.

**IEA Sector Allocation**: Fixed 6 Gt CO₂e for global chemicals industry (2023-2050) from IEA Net Zero Roadmap. Korean share applied via production percentage (~3%).

**Industry Baseline** (`sample_data/kpetchem_demo.csv`): Single 2023 row with production (90 Mt), direct emissions (50 Mt), and value added (38,000 billion KRW). Users can upload custom CSV starting from 2023.

## Architecture

```
kpetchem_budget/
├─ data_layer.py          # Cached data loading & Korean shares
├─ parameter_space.py     # 144-case grid + MC sampling  
├─ simulator.py          # Parallel execution engine
├─ datastore.py          # Parquet warehouse + queries
├─ pathway.py            # 2023-2050 trajectory generators
├─ dashboard/
│   ├─ app.py           # Streamlit interface
│   └─ components.py    # Reusable widgets
└─ tests/               # Comprehensive test suite
```

## Allocation Methods

1. **Population Share**: Korea's ~0.66% of global population × selected carbon budget
2. **GDP Share**: Korea's ~1.8% of global GDP × selected carbon budget  
3. **Historical GHG**: Korea's ~1.4% of cumulative emissions × selected carbon budget
4. **IEA Sector**: min(6 Gt, global budget) × Korea's ~3% petrochemical production share

## Pathway Generators

All pathways span **2023-2050** with milestone markers at **2035** and **2050**:

- **Linear-to-Zero**: Straight line decline to net-zero by target year
- **Constant Rate**: Fixed annual reduction percentage (user configurable)  
- **Logistic Decline**: S-curve with steepest reduction around 2035 midpoint
- **IEA Proxy**: Three-phase trajectory matching IEA chemicals roadmap

Budget validation uses trapezoidal integration. `BudgetOverflowError` raised if cumulative emissions exceed allocation.

## Dashboard Interface

**Sidebar Controls**: Baseline emissions (50 Mt default), allocation rule, budget scenario, net-zero year (2045/2050/2055), reduction rate, CSV upload, Monte Carlo toggle.

**Main Display**: KPI cards show remaining budget and milestone values (2035/2050 emissions + cumulative). Composite dark-theme chart overlays all four pathways with circular milestone markers. Optional Monte Carlo uncertainty cloud displays 10-90% percentile bands.

**Data Tabs**: Pathway analysis with comparison metrics, detailed year-by-year tables, and downloads (CSV pathways, summary stats, configuration).

## Performance Targets

- ✅ **14,400 simulations** complete in <30 seconds on 8 threads
- ✅ **Parquet storage** <20 MB with Snappy compression  
- ✅ **Streamlit dashboard** cold-load <2.5 seconds
- ✅ **Memory efficiency** via chunked processing and categorical dtypes

## Known Limitations

**Scope Coverage**: Direct and indirect emissions only; excludes Scope 3 upstream/downstream impacts and non-CO₂ GHGs (N₂O, CH₄).

**Technology Assumptions**: No explicit modeling of carbon capture, renewable feedstocks, or hydrogen integration pathways.

**Economic Feedbacks**: Carbon pricing effects and production volume responses not incorporated.

**Uncertainty Sources**: Limited to budget uncertainty (±10%), Korean production share error (±0.5%), and pathway shape parameters. Does not include policy or technology disruption scenarios.

---

*Version 2.0.0 • Updated timeline 2023-2050 • Enhanced Monte Carlo engine • Composite pathway visualization*