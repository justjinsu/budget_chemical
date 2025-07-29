# Carbon Budget and Industrial Emission Pathway Modeling

## Overview
This repository provides a comprehensive framework for **carbon budget allocation** and **emission reduction pathway modeling** with a focus on industrial and petrochemical sectors. The model integrates data from the World Bank and EDGAR databases, applies sophisticated budget allocation methods using the BKIR (Budget Korea Industrial/Petrochemical) formula, and calculates various emission reduction pathways with uncertainty quantification through Monte Carlo analysis.

## Key Features

### 🎯 Budget Allocation Methods
- **BKIR Formula Implementation**: Advanced budget allocation using responsibility, capability, and equality factors
- **Multi-Factor Allocation**: Distributes carbon budgets based on GDP, population, historical emissions, or custom indicators
- **Sectoral Breakdown**: Specialized allocation for industrial (37%) and petrochemical (10% of industrial) sectors
- **Inverted Share Options**: Prioritizes countries/sectors with lower emissions or economic capacity

### 📊 Emission Pathway Calculation
- **Linear to Zero**: Gradual reduction to zero emissions by target year
- **Linear Reduction**: Customizable linear reduction rates with flexible target years
- **Spline Pathway**: Smooth, curved reduction approach with mid-point targets (2030)
- **Fixed Annual Reduction**: Consistent percentage-based annual reductions

### 🔬 Monte Carlo Uncertainty Analysis
- **Probabilistic Modeling**: 3000-draw Monte Carlo simulations for uncertainty quantification
- **Parameter Uncertainty**: Configurable uncertainty distributions for all key parameters
- **Fan Chart Visualization**: Statistical visualization of emission pathway uncertainties
- **Quantile Analysis**: P10, P25, P50, P75, P90 quantile outputs for risk assessment

### 🌐 Data Integration
- **World Bank API**: Real-time fetching of GDP and population indicators
- **EDGAR Database**: Comprehensive greenhouse gas emission data processing (CO2, CH4, N2O, F-gases)
- **Multi-Source Support**: Handles 7+ different EDGAR datasets with automatic cleaning
- **Flexible Data Pipeline**: Configurable data extraction and processing workflows

## Project Structure

```
budget_chemical/
├── SimpleCarbonBudget.py      # Single-country budget analysis
├── MultipleCarbonBudget.py    # Multi-country comparative analysis
├── setup.py                   # Package configuration
├── LICENSE                    # GNU GPL v3.0 license
├── data/                      # Input datasets
│   ├── EDGAR_*.xlsx          # Emission datasets (CO2, CH4, N2O, F-gases)
│   └── globalbudget.csv      # Global carbon budget scenarios
├── lib/                       # Core library modules
│   ├── budgetCalculation.py  # BKIR formula and allocation logic
│   ├── dataAPI.py            # Data fetching and processing
│   ├── pathwayCalculation.py # Emission pathway algorithms
│   ├── utils.py              # Utility functions
│   └── process.py            # Data processing workflows
├── budget_chemical/          # Main package
│   ├── core/                 # Core functionality
│   │   ├── budget_calculation.py
│   │   ├── data_api.py
│   │   ├── pathway_calculation.py
│   │   └── utils.py
│   ├── monte_carlo/          # Monte Carlo framework
│   │   ├── runner.py         # Main Monte Carlo execution
│   │   ├── sampler.py
│   │   ├── metrics.py
│   │   ├── pathway_calculator.py
│   │   └── visualization.py
│   └── scripts/              # Standalone scripts
├── config/                   # Configuration files
│   └── mc_config.yaml
├── examples/                 # Debug and example scripts
├── tests/                    # Test files
│   └── pathwayCalculation.py # MC-specific pathway calculations
└── outputs/                  # Generated results and visualizations
    ├── *_fan_chart.png       # Uncertainty fan charts
    ├── *_quantiles_*.csv     # Quantile analysis results
    └── summary_*.json        # Aggregated analysis summaries
```

## Installation

### Prerequisites
- Python 3.7 or higher
- Internet connection for World Bank API access

### Setup Instructions
```bash
# Clone the repository
git clone https://github.com/PLANiT-Institute/carbonbudget.git
cd budget_chemical

# Install the package and dependencies
pip install -e .

# Or install dependencies manually
pip install pandas numpy openpyxl requests scipy matplotlib xlrd pyyaml
```

## Usage Guide

### 1. Basic Carbon Budget Analysis

#### Single Country Analysis
```bash
python SimpleCarbonBudget.py
```
**Configuration Parameters:**
- `countries`: Target countries (default: ['KOR'])
- `temp_values`: Temperature scenarios (1.5°C, 1.7°C, 2.0°C)
- `approach_values`: Allocation method ('NY.GDP.MKTP.PP.KD', 'SP.POP.TOTL', 'CO2')
- `start_year`, `mid_year`, `end_year`: Pathway timeline (2023-2050)

#### Multi-Country Comparative Analysis
```bash
python MultipleCarbonBudget.py
```
**Enhanced Features:**
- Multiple countries: ['KOR', 'JPN']
- Accumulated allocation methods
- Inverted GDP-based allocation
- Comprehensive pathway comparison

### 2. Monte Carlo Uncertainty Analysis

#### Configuration
Edit `model_D_MC/mc_config.yaml`:
```yaml
n_draws: 3000                    # Number of Monte Carlo draws
seed: 123                        # Random seed for reproducibility
curve_type: log                  # Distribution type for uncertainty

global_budget:                   # Global carbon budget scenarios (tCO2)
  low: 450000000000.0
  mid: 500000000000.0  
  high: 550000000000.0

user_weights:                    # BKIR formula weights
  responsibility: 0.3            # Historical responsibility factor
  capability: 0.4                # Economic capability factor  
  equality: 0.3                  # Per-capita equality factor

uncertainty:                     # Parameter uncertainty distributions
  responsibility:
    low: 0.009
    mid: 0.011
    high: 0.013
  capability:
    mu: 0.017
    sd_pct: 0.05
  equality:
    mu: 0.0067
    sd_pct: 0.03

industry_fraction: 0.37          # Industry sector allocation (37%)
petrochem_fraction: 0.10         # Petrochemical fraction of industry (10%)
```

#### Running Monte Carlo Analysis
```bash
python run_monte_carlo.py config/mc_config.yaml
```

**Outputs Generated:**
- `industry_fan_chart.png`: Industrial sector uncertainty fan chart
- `petrochem_fan_chart.png`: Petrochemical sector uncertainty fan chart
- `*_quantiles_*.csv`: Detailed quantile analysis (P10-P90)
- `summary_*.json`: Aggregated statistics and key metrics
- `uncertainty_analysis.png`: Comparative uncertainty visualization

## Core Components

### 🧮 Budget Calculation (`lib/budgetCalculation.py`)
**BKIR Formula Implementation:**
```
BKIR(j) = B_global(j) × (w_r × δ_r + w_c × δ_c + w_e × δ_e)
```
Where:
- `B_global(j)`: Global carbon budget for scenario j
- `w_r, w_c, w_e`: User-defined weights for responsibility, capability, equality
- `δ_r, δ_c, δ_e`: Normalized country factors

**Key Methods:**
- `calculate_BKIR()`: Core budget allocation using BKIR formula
- `allocate_industry_petrochem()`: Sectoral budget distribution
- `get_base_emissions()`: Historical emission baselines

### 📡 Data Management (`lib/dataAPI.py`)
**World Bank Integration:**
- `download_worldbank_data()`: Automated API data retrieval
- `cleanup_wbdata()`: Data standardization and quality control

**EDGAR Processing:**
- `clean_and_extract()`: Multi-format Excel file processing
- Handles 7 different EDGAR datasets with varying structures
- Automatic unit conversion and data validation

### 🛤️ Pathway Calculation (`lib/pathwayCalculation.py`)
**Algorithm Implementations:**
- `linear_to_zero()`: Linear reduction to zero emissions
- `linear_pathway()`: Flexible linear reduction with custom targets
- `spline_pathway()`: Smooth curved pathways using scipy interpolation
- `fixed_reduction_pathway()`: Percentage-based annual reductions

### 🎲 Monte Carlo Framework (`model_D_MC/`)
**Statistical Sampling:**
- Triangular distributions for bounded parameters
- Normal distributions for capability factors
- Log-normal distributions for budget scenarios
- Correlation handling between parameters

**Performance Metrics:**
- Emission reduction rates by pathway type
- Budget utilization efficiency
- Sectoral allocation optimization
- Uncertainty decomposition analysis

## Configuration Options

### Data Sources
```python
# EDGAR Dataset Configuration
file_paths = [
    'data/EDGAR_AR5_GHG_1970_2022.xlsx',    # All GHGs (AR5 GWP)
    'data/EDGAR_CH4_1970_2022.xlsx',        # Methane emissions
    'data/EDGAR_CO2bio_1970_2022.xlsx',     # Biogenic CO2
    'data/EDGAR_F-gases_1990_2022.xlsx',    # Fluorinated gases
    'data/EDGAR_N2O_1970_2022.xlsx',        # Nitrous oxide
    'data/IEA_EDGAR_CO2_1970_2022.xlsx'     # Fossil CO2 (IEA-based)
]

# World Bank Indicators
indicators = [
    'SP.POP.TOTL',           # Total population
    'NY.GDP.MKTP.PP.KD'      # GDP (PPP, constant 2017 international $)
]
```

### Filtering Criteria
```python
# Analysis Parameters
countries = ['KOR', 'JPN', 'CHN', 'USA']     # Target countries
temp_values = [1.5, 1.7, 2.0]                # Temperature scenarios (°C)
probability_values = [0.5, 0.67, 0.83]       # Probability levels
approach_values = ['NY.GDP.MKTP.PP.KD']       # Allocation method
period_values = [2022]                        # Base year for allocation
```

## Output Formats

### CSV Files
- **Budget Allocations**: Country-wise carbon budget distributions
- **Emission Pathways**: Annual emission projections by pathway type
- **Quantile Analysis**: Statistical uncertainty ranges (P10-P90)
- **Comparative Analysis**: Multi-country/sector comparisons

### Visualizations
- **Fan Charts**: Uncertainty visualization with percentile bands
- **Time Series**: Emission pathway trajectories
- **Bar Charts**: Budget allocation comparisons
- **Statistical Plots**: Distribution analysis and sensitivity testing

### JSON Summaries
```json
{
  "analysis_date": "2025-07-24",
  "config": {
    "n_draws": 3000,
    "seed": 123,
    "sectors": ["industry", "petrochem"]
  },
  "results": {
    "industry": {
      "mean_budget": 1.85e10,
      "p50_budget": 1.82e10,
      "uncertainty_range": [1.65e10, 2.05e10]
    },
    "petrochem": {
      "mean_budget": 1.85e9,
      "p50_budget": 1.82e9,
      "reduction_needed": 0.63
    }
  }
}
```

## Advanced Features

### 🔄 Sub-National Analysis
The framework supports sub-national and sectoral analysis by:
1. Modifying budget values to reflect national totals
2. Applying sector-specific emission factors
3. Customizing pathway parameters for different industries
4. Scaling analysis to regional or company levels

### 🎯 Custom Allocation Methods
Users can implement custom allocation approaches by:
```python
# Example: Innovation-based allocation
def innovation_allocation(data_df, innovation_index):
    # Custom allocation logic based on innovation capacity
    shares = calculate_innovation_shares(innovation_index)
    return apply_allocation(shares, global_budget)
```

### 📈 Real-time Data Integration
The system supports real-time updates through:
- Automated World Bank API polling
- EDGAR database version checking
- Dynamic parameter adjustment
- Continuous model recalibration

## Performance Considerations

### Computational Efficiency
- **Monte Carlo**: ~30 seconds for 3000 draws
- **Data Processing**: ~5-10 seconds for full EDGAR dataset
- **Visualization**: ~2-3 seconds per chart generation
- **Memory Usage**: ~200MB for typical analysis

### Scaling Recommendations
- **Large Countries**: Use parallel processing for >10 countries
- **High-Resolution Pathways**: Consider yearly vs. monthly timesteps
- **Uncertainty Analysis**: Balance draw count vs. computation time

## Troubleshooting

### Common Issues
1. **World Bank API Timeout**: Check internet connection, retry with exponential backoff
2. **EDGAR File Format**: Verify Excel file structure, check skip parameters
3. **Memory Errors**: Reduce Monte Carlo draws or optimize data loading
4. **Visualization Errors**: Install matplotlib backend, check output directory permissions

### Error Handling
The framework includes comprehensive error handling for:
- Missing data files
- API connectivity issues
- Invalid parameter combinations
- Numerical instabilities in pathway calculations

## Contributing

### Development Setup
```bash
# Fork the repository
git clone https://github.com/YOUR_USERNAME/carbonbudget.git
cd carbonbudget

# Create development environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
pip install pytest black flake8  # Development tools
```

### Testing
```bash
# Run unit tests
python -m pytest tests/

# Run integration tests
python -m pytest tests/integration/

# Check code formatting
black lib/ model_D_MC/
flake8 lib/ model_D_MC/
```

### Contribution Guidelines
1. **Fork & Branch**: Create feature branches from `main`
2. **Code Style**: Follow PEP 8, use Black formatter
3. **Documentation**: Update docstrings and README for new features
4. **Testing**: Add unit tests for new functionality
5. **Pull Request**: Provide detailed description of changes

## License
This project is licensed under the **GNU General Public License v3.0** (GPL-3.0). This ensures the software remains free and open-source, with any derivative works also being licensed under GPL-3.0.

### License Summary
- ✅ **Freedom to Use**: Run the program for any purpose
- ✅ **Freedom to Study**: Access and modify source code
- ✅ **Freedom to Share**: Distribute copies to help others
- ✅ **Freedom to Improve**: Distribute modified versions
- ⚠️ **Copyleft Requirement**: Derivative works must use GPL-3.0

## Citation
If you use this software in your research, please cite:
```bibtex
@software{carbonbudget2024,
  title={Carbon Budget and Industrial Emission Pathway Modeling Framework},
  author={Hong, Sanghyun and PLANiT Institute},
  year={2024},
  url={https://github.com/PLANiT-Institute/carbonbudget},
  license={GPL-3.0}
}
```

## Contact & Support
- **Primary Contact**: [sanghyun@planit.institute](mailto:sanghyun@planit.institute)
- **Institution**: PLANiT Institute
- **Documentation**: [Project Wiki](https://github.com/PLANiT-Institute/carbonbudget/wiki)
- **Issues**: [GitHub Issues](https://github.com/PLANiT-Institute/carbonbudget/issues)
- **Discussions**: [GitHub Discussions](https://github.com/PLANiT-Institute/carbonbudget/discussions)
