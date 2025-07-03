import pandas as pd
import lib.dataAPI as dataAPI
import lib.budgetCalculation as budgetCalculation
import lib.utils as utils
import lib.pathwayCalculation as pathwayCalculation

import importlib
importlib.reload(dataAPI)
importlib.reload(budgetCalculation)
importlib.reload(pathwayCalculation)

if __name__ == "__main__":
    """
    USER INPUT PARAMETERS
    This section defines the input parameters for the script, such as the start and end years, the indicators 
    to fetch from the World Bank API, file paths for EDGAR data, and parameters for budget allocation and 
    emissions pathway generation.

    - first_year and last_year define the time range for data extraction.
    - indicators specify the variables we are interested in (population and GDP).
    - file_paths contains the list of paths for the EDGAR datasets.
    - budget_path specifies the file path for the global budget data.
    - keywords are the indicators that will be used for budget allocation.
    - countries, temp_values, probability_values, approach_values, and period_values are filtering criteria 
      used later for budget filtering.
    - start_year, mid_year, and end_year define the timeline for calculating emission pathways.
    """

    first_year = 2020
    last_year = 2022
    years = list(range(first_year, last_year))
    indicators = ['SP.POP.TOTL', 'NY.GDP.MKTP.PP.KD']

    file_paths = ['data/EDGAR_AR5_GHG_1970_2022.xlsx',
                  'data/EDGAR_CH4_1970_2022.xlsx',
                  'data/EDGAR_CO2bio_1970_2022.xlsx',
                  'data/EDGAR_F-gases_1990_2022.xlsx',
                  'data/EDGAR_F-gases_AR5g_1990_2022.xlsx',
                  'data/EDGAR_N2O_1970_2022.xlsx',
                  'data/IEA_EDGAR_CO2_1970_2022.xlsx']

    budget_path = 'data/globalbudget.csv'

    # define the keywords to go through the allocation process
    keywords = ['SP.POP.TOTL', 'NY.GDP.MKTP.PP.KD', 'CO2']

    # Define the filter criteria
    countries = ['KOR', 'JPN']
    temp_values = [1.5, 1.7]
    probability_values = [0.5]
    approach_values = ['NY.GDP.MKTP.PP.KD']
    period_values = [2022]

    # Define start year of the annual pathway
    start_year = 2023
    mid_year = 2030
    end_year = 2050

    reduction_rate = 0.25

    """
    Data import from World Bank API & EDGAR file sets & clean up
    This section handles the data import, cleaning, and preparation processes.

    - World Bank data is fetched using the `download_worldbank_data` function for each indicator in the defined years.
    - The `cleanup_wbdata` function processes the World Bank data, ensuring it is in a usable format.
    - EDGAR data is imported from several Excel files, with the first few rows skipped for proper formatting.
    - The EDGAR data is concatenated into a single DataFrame.
    - The World Bank and EDGAR datasets are merged together into one combined DataFrame.
    """

    wbdata_lt = [dataAPI.download_worldbank_data(indicator, first_year, last_year) for indicator in indicators]
    wbdata_df = dataAPI.cleanup_wbdata(wbdata_lt)

    # Apply cleaning and extraction to each file
    edgar_lt = [dataAPI.clean_and_extract(file, skip = 8) for file in file_paths]
    edgar_df = pd.concat(edgar_lt)

    # merge datasets
    data_df = pd.concat([wbdata_df, edgar_df])
    budget = pd.read_csv(budget_path)

    """
    Budget calculation
    This section performs budget allocation using the data and budget information.

    - The `budgetAllocation` class is instantiated with the merged dataset and the budget data.
    - Various budget allocation processes are applied using different indicators (population, GDP, and CO2 emissions).
    - Both normal and inverted budget allocation methods are applied.
    - Results from different allocation methods are concatenated into one DataFrame for further processing.
    """
    # Create an instance of BudgetAllocation
    Ibudget = budgetCalculation.budgetAllocation(data_df, budget)

    normal_allocation = pd.concat([Ibudget.allocate_by_keyword(keyword, year = last_year) for keyword in keywords])
    accum_allocation = pd.concat([Ibudget.allocate_by_keyword(keyword, year = years) for keyword in keywords])

    # invert the share
    # share_inverted = 1 / share
    # share = share_inverted / share_inverted.sum()
    gdpinv_allocation = Ibudget.allocate_by_keyword(keyword='NY.GDP.MKTP.PP.KD', year = last_year, invert = True)
    gdpacminv_allocation = Ibudget.allocate_by_keyword(keyword='NY.GDP.MKTP.PP.KD', year=years, invert=True)

    budget_df = pd.concat([normal_allocation, accum_allocation, gdpinv_allocation, gdpacminv_allocation])

    """
    Filtering the budget
    This section filters the calculated budget data based on specific criteria such as countries, 
    temperature targets, probability values, and the selected approach (e.g., GDP).

    - The `filter_budget_df` function applies these filters and returns the filtered DataFrame.
    """

    # Use the filter function
    # select the budget and country
    filtered_budget_df = utils.filter_budget_df(budget_df, countries, temp_values, probability_values, approach_values, period_values)
    filtered_budget_df.reset_index(inplace=True, drop=True)

    """
    Create annual pathway
    In this section, various emission reduction pathways are calculated based on the filtered budget data.

    - Current CO2 emissions for the selected countries and last year are extracted and adjusted to match units (GtC).
    - Pathways are calculated using different methods (linear to zero, linear reduction, and spline pathways).
    - Each pathway calculation is applied to the filtered budget data, creating different emissions reduction pathways for the target countries.
    - A fixed reduction rate pathway (e.g., 2% reduction per year) is also calculated.
    """

    # current year's emission
    current_co2 = data_df[(data_df['country_code'].isin(countries)) &
                          (data_df['indicator'] == 'CO2') &
                          (data_df['year'] == last_year)]

    # estimate the annual pathway
    # use the index to match the approach to each pathway
    current_co2.loc[:, 'value'] = current_co2.loc[:, 'value'] / 3.664 / 1e6 # change the value to GtC (match the value), and then match the unit (Kt -> Gt)

    Ipathway = pathwayCalculation.pathwayCalculator(start_year, mid_year, end_year)

    lineartozeropathway_df = filtered_budget_df.apply(
        lambda row: Ipathway.linear_to_zero(
            emission_current=current_co2[current_co2['country_code'] == row['country_code']].value.iloc[0],
            allocation=row['allocation']
        ), axis=1)

    linearpathway_df = filtered_budget_df.apply(
        lambda row: Ipathway.linear_pathway(
            emission_current=current_co2[current_co2['country_code'] == row['country_code']].value.iloc[0],
            allocation=row['allocation']
        ), axis=1)


    midlinearpathway_df = filtered_budget_df.apply(
        lambda row: Ipathway.spline_pathway(
            emission_current=current_co2[current_co2['country_code'] == row['country_code']].value.iloc[0],
            allocation=row['allocation']
        ), axis = 1)

    fixed_reduction_df = filtered_budget_df.apply(
        lambda row: Ipathway.fixed_reduction_pathway(
            emission_current=current_co2[current_co2['country_code'] == row['country_code']].value.iloc[0],
            reduction_rate=reduction_rate  # 2% reduction per year
        ), axis=1)

    """
    
    If a user wants to create sub-national level carbon budget pathway, change the budget value to the national budget value.
    And do the rest based on the user's specific sectoral or sub-national data

    """

