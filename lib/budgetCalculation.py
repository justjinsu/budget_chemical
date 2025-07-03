import pandas as pd
import numpy as np


class budgetAllocation:
    def __init__(self, data_df, budget):
        """
        Initialize the budgetAllocation class with the data and budget.

        Parameters:
        data_df (pd.DataFrame): DataFrame containing country-level data with 'indicator', 'year', 'country_code', and 'value'.
        budget (pd.DataFrame): DataFrame containing 'budget', 'temp', and 'probability' to allocate based on countries' share.
        """
        self.data_df = data_df  # Data containing country-level statistics for different indicators and years
        self.budget = budget  # Budget DataFrame containing 'budget', 'temp', and 'probability' for allocation

    def allocate_by_keyword(self, keyword, year, invert=False):
        """
        Allocate budget to countries based on their share of a specified indicator.

        Parameters:
        keyword (str): The indicator keyword to filter the data for allocation.
        year (int or list of ints): The year or range of years to consider for the allocation. Can be a single year or multiple years.
        invert (bool): If True, inverts the share to favor smaller values, meaning countries with lower values get more budget allocation. Defaults to False.

        Returns:
        pd.DataFrame: A DataFrame containing the allocated budget for each country with additional metadata.
        """

        # Handle when year is a list or tuple (multiple years)
        if isinstance(year, (list, tuple)):
            # Filter data by the keyword and years, then sum the values over the specified years for each country
            temp_df = self.data_df[(self.data_df['indicator'] == keyword) & (self.data_df['year'].isin(year))].copy()
            temp_df = temp_df.groupby('country_code', as_index=False)['value'].sum()  # Sum values over the years
            period = f"{min(year)}-{max(year)}"  # Create a period string like "2000-2010"

        # Handle when year is a single integer (one year)
        elif isinstance(year, int):
            # Filter data by the keyword and year
            temp_df = self.data_df[(self.data_df['indicator'] == keyword) & (self.data_df['year'] == year)].copy()
            period = year  # Set the period as the single year

        # Raise error if year is of invalid type
        else:
            raise ValueError("Invalid type for 'year'. Must be int or list of years.")

        # Convert 'value' column to NumPy array and handle NaN values by replacing them with 0
        value_array = temp_df['value'].to_numpy()
        value_array = np.nan_to_num(value_array, nan=0.0)  # Replace NaNs with zeros

        # Calculate each country's share by dividing their value by the total sum
        share = value_array / value_array.sum()

        # If invert is True, invert the shares so countries with smaller values get larger allocation
        if invert:
            share_inverted = 1 / share
            share = share_inverted / share_inverted.sum()  # Normalize the inverted shares

        # Extract the country codes from the data
        country_code = temp_df['country_code'].to_numpy()

        # Initialize a list to store allocations for each budget scenario
        allocation_lt = []

        # Iterate over each row of the budget DataFrame and allocate the budget based on calculated shares
        for _, row in self.budget.iterrows():
            b = row['budget']  # Current budget to allocate
            allocation = share * b  # Allocate the budget according to the share

            # Create a DataFrame for the current allocation with relevant details
            allocation_df = pd.DataFrame({'temp': row['temp'],
                                          'probability': row['probability'],
                                          'country_code': country_code,
                                          'share': share,
                                          'allocation': allocation})

            allocation_lt.append(allocation_df)  # Append allocation to the list

        # Concatenate all allocations into a single DataFrame
        allocation_df = pd.concat(allocation_lt)

        # Insert additional metadata about the allocation process into the DataFrame
        allocation_df.insert(loc=2, column='approach', value=keyword)  # Add the keyword/indicator used
        allocation_df.insert(loc=3, column='period', value=period)  # Add the time period for the data
        allocation_df.insert(loc=4, column='invert', value=invert)  # Indicate if the shares were inverted

        return allocation_df  # Return the final DataFrame with the allocated budget
