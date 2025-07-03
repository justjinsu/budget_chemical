# Carbon Budget and Emission Pathway Modeling

## Overview
This repository provides a framework for **carbon budget allocation** and **emission reduction pathway modeling**. The model integrates **data from the World Bank and EDGAR**, applies **budget allocation methods**, and calculates **various emission reduction pathways**.

## Features
- **Budget Allocation**: Distributes carbon budgets based on GDP, population, or emissions.
- **Emission Pathways**:
  - **Linear to Zero**: Reduces emissions to zero over time.
  - **Linear Reduction**: Reduces emissions linearly with a target year.
  - **Spline Pathway**: Uses a flexible reduction approach with a midpoint target.
  - **Fixed Reduction**: Reduces emissions by a fixed percentage annually.
- **World Bank API Integration**: Fetches economic and population indicators for allocation.
- **EDGAR Data Processing**: Extracts emission data from multiple sources.
- **Sub-National & Sectoral Adaptability**: Allows users to modify pathways for specific sectors or regions.

## Installation
To use this repository, clone it and install the required dependencies:

```sh
# Clone the repository
git clone https://github.com/PLANiT-Institute/carbonbudget.git
cd carbon-budget-model

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running the Model
To execute the script and compute budget allocations and pathways:

```sh
python SimpleCarbonBudget.py
python MultipleCarbonBudget.py
```

### Configuration
The model reads from multiple data sources:
- **World Bank API** (`dataAPI.py`): Fetches GDP and population data.
- **EDGAR Data** (`data/IEA_EDGAR_CO2_1970_2022.xlsx` and others).
- **Global Budget File** (`data/globalbudget.csv`).

## Key Components

### `budgetCalculation.py`
- **`budgetAllocation`**: Allocates budgets based on economic, population, or emission data.
- **Inverted Share Allocation**: Allows prioritization of countries with lower emissions or GDP.

### `dataAPI.py`
- **Fetches World Bank Data**: Extracts country-level indicators.
- **Processes EDGAR Data**: Cleans and formats emission datasets.

### `pathwayCalculation.py`
- **Computes Reduction Pathways**:
  - Linear
  - Spline
  - Fixed Reduction

### `SimpleCarbonBudget.py`
- **Calculates single-country carbon budgets and pathways**.
- **Filters data based on user-defined parameters**.

### `MultipleCarbonBudget.py`
- **Expands budget allocation to multiple countries**.
- **Applies different allocation and reduction methods**.

## Example Output
Upon execution, the results are saved in the project directory:
- **Allocated Budget Data**
- **Emission Pathway Data**
- **Filtered Budget Outputs**

## Contributing
We welcome contributions to improve this repository! Please follow these steps:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-new`).
3. Commit your changes (`git commit -m "Added new feature"`).
4. Push to your fork (`git push origin feature-new`).
5. Submit a pull request.

## License
This project is licensed under the **GNU General Public License v3.0**.

## Contact
For inquiries or collaboration, please contact: **[sanghyun@planit.institute](mailto:sanghyun@planit.institute)**
