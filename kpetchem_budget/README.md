# Korean Petrochemical Carbon Budget Allocation

This package allocates global carbon budgets to the Korean petrochemical sector using multiple allocation criteria and generates emission pathways for 2035-2050.

## Quickstart

```bash
pip install streamlit pandas numpy matplotlib scipy
streamlit run app.py
```

## Data Inputs

The package uses three main data sources:

1. **Global Budget Data** (`globalbudget.csv`): Contains global carbon budgets by temperature target, probability, and approach. Should be placed in the parent `data/` directory.

2. **IEA Sector Budget**: Hard-coded at 6 Gt CO₂e total for the global chemicals sector (2023-2050), as per IEA Net Zero Roadmap.

3. **Demo Industry Data** (`sample_data/kpetchem_demo.csv`): Five-year Korean petrochemical production and emissions data (2019-2023). Users can upload custom CSV files with columns: `year`, `production_Mt`, `direct_CO2_Mt`.

## Allocation Methods

The package supports four allocation criteria:

- **Population**: Korea's share of global population (~0.66%)
- **GDP**: Korea's share of global GDP (~1.8%)  
- **Historical GHG**: Korea's share of historical greenhouse gas emissions (~1.4%)
- **IEA Sector**: Uses minimum of IEA sector budget (6 Gt) and user-selected global budget, multiplied by Korea's production share (~3%)

## Emission Pathways

Three pathway generators create annual trajectories (2035-2050):

- **Linear to Zero**: Emissions decline linearly to zero by 2050
- **Constant Rate**: Fixed percentage reduction year-over-year
- **IEA Proxy**: Scaled version of IEA global chemicals pathway

All pathways validate that cumulative emissions ≤ allocated budget using trapezoidal integration.

## Swapping the 50 Mt Default

The baseline emissions can be modified in two ways:

1. **Via Streamlit UI**: Use the sidebar number input "Current annual emissions (Mt CO₂e)"
2. **Via Code**: Initialize `BudgetAllocator(baseline_emissions=YOUR_VALUE)`

## Interface Features

The Streamlit interface provides:

- **KPI Cards**: Remaining budget, overshoot year, peak-to-zero reduction percentage
- **Interactive Charts**: Multi-line pathway visualization with optional historical bars
- **Data Tables**: Detailed pathway data with summary statistics
- **Downloads**: Export pathways as CSV or charts as PNG

## Known Limitations

- **Scope 3 Emissions**: Only direct and indirect emissions are considered; upstream/downstream emissions are excluded
- **N₂O Emissions**: Only CO₂ emissions are modeled; nitrous oxide and other GHGs are not included
- **Technology Transitions**: No explicit modeling of carbon capture, renewable feedstocks, or other decarbonization technologies
- **Economic Feedbacks**: No consideration of carbon pricing or economic impacts on production volumes

## Testing

Run the test suite with:

```bash
pytest tests/
```

The package includes comprehensive tests for allocation logic, pathway generation, and Streamlit integration.