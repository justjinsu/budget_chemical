import pandas as pd
import numpy as np


class pathwayCalculator:
    def __init__(self, start_year, mid_year=None, end_year=None, emission_current=None, allocation=None, reduction_rate=None):
        """
        Initialize the pathwayCalculator class with the start, mid, and end years.

        :param start_year: The year from which the pathway calculation starts.
        :param mid_year: The midpoint year for spline pathway calculations.
        :param end_year: The year by which emissions are targeted to reach a certain value.
        """
        self.start_year = start_year
        self.mid_year = mid_year
        self.end_year = end_year
        self.emission_current = emission_current  # Optional emission current value
        self.allocation = allocation
        self.reduction_rate = reduction_rate

    def linear_to_zero(self, emission_current=None, allocation=None):
        """
        Calculate the year when the allocated value (budget) will reach zero, using a linear pathway.

        :param emission_current: The current CO2 emission value.
        :param allocation: The total CO2 emission budget allocated for reduction.

        :return: A DataFrame showing the year-by-year linear decline in emissions until zero.
        """
        emission_current = emission_current if emission_current is not None else self.emission_current
        allocation = allocation if allocation is not None else self.allocation

        # Calculate the number of years required for emissions to reach zero based on current emission and allocation
        years_until_zero = 2 * allocation / emission_current + 1  # +1 adjusts for rounding to the nearest whole year

        # Determine the year when emissions will reach zero
        year_zero = self.start_year - 1 + int(round(years_until_zero))

        # Generate the range of years from the start year to the zero-emission year
        years = np.array(list(range(self.start_year - 1, year_zero + 1)), dtype='int')

        # Linearly interpolate emissions between the current emission and zero over the years
        emissions = np.linspace(emission_current, 0, len(years))

        # Create a DataFrame showing the year and corresponding emission values
        df = pd.DataFrame({
            'Year': years,
            'Emission': emissions
        })

        # Return only the rows from the start year onwards
        return df[df['Year'] >= self.start_year]

    def linear_pathway(self, emission_current=None, allocation=None):
        """
        Create a linear pathway for emissions reduction from the current emission to a target emission by end year.

        :param emission_current: The current CO2 emission value.
        :param allocation: The total CO2 emission budget allocated for reduction.

        :return: A DataFrame showing the year-by-year linear decline in emissions.
        """

        emission_current = emission_current if emission_current is not None else self.emission_current
        allocation = allocation if allocation is not None else self.allocation
        # Define the range of years from start to end
        years = range(self.start_year, self.end_year + 1)
        num_years = len(years)

        # Calculate the end value to ensure total emission matches the allocation over the period
        end_value = (allocation - emission_current * (num_years - 1) / 2) * 2 / num_years

        # Interpolate emissions linearly between current emissions and the calculated end value
        values = np.linspace(emission_current, end_value, num_years)

        # Create a DataFrame with the emission pathway over the years
        df = pd.DataFrame({'Year': years, 'Emission': values})

        # Adjust values so that the total sum of emissions matches the allocated budget
        adjustment_factor = allocation / df['Emission'].sum()
        df['Emission'] *= adjustment_factor

        return df[df['Year'] >= self.start_year]

    def spline_pathway(self, emission_current=None, allocation=None):
        """
        Create a spline pathway for emissions reduction, with a flexible midpoint value for smoother reduction.

        :param emission_current: The current CO2 emission value.
        :param allocation: The total CO2 emission budget allocated for reduction.

        :return: A DataFrame showing the year-by-year emission reductions using spline interpolation.
        """

        emission_current = emission_current if emission_current is not None else self.emission_current
        allocation = allocation if allocation is not None else self.allocation

        # Calculate the number of years for the first half and second half of the period
        fy = self.mid_year - (self.start_year - 1)
        ly = self.end_year - self.mid_year

        # Total value for distribution calculation and initial value (iv)
        al = allocation + emission_current / 2
        iv = emission_current

        # Calculate the mid-year value (mv) based on the allocation and emission current
        mv = ((al * 2) - iv * fy) / (fy + ly)

        years = range(self.start_year - 1, self.end_year + 1)
        values = []

        # Determine slopes based on whether the mid-year value is positive or adjusted to zero
        if mv > 0:
            # Slope for the first half of the period (from start year to mid year)
            slope_fy = (mv - iv) / fy
            # Slope for the second half of the period (from mid year to end year)
            slope_ly = -mv / ly
        else:
            # If the mid-year value would be negative, set it to zero and adjust slopes accordingly
            mv = 0
            slope_fy = -iv / fy
            fy_sum = (iv + mv) / 2 * (fy + 1)  # Calculate the triangular area for the first half
            gap = allocation - fy_sum + iv  # Calculate the remaining value for the second half
            lv = gap * 2 / (ly + 1)  # Calculate the end value to match the budget in the second half
            slope_ly = lv / ly  # Slope for the second half

        # Loop through the years and calculate emissions based on the slope for each year
        for year in years:
            if year < self.mid_year:
                value = iv + slope_fy * (year - (self.start_year - 1))
            else:
                value = mv + slope_ly * (year - self.mid_year)
            values.append(value)

        # Create a DataFrame with the emission pathway using spline interpolation
        df = pd.DataFrame({'Year': years, 'Emission': values})

        return df[df['Year'] >= self.start_year]

    def fixed_reduction_pathway(self, emission_current=None, reduction_rate=None):
        """
        Create a fixed reduction pathway where emissions reduce by a fixed percentage each year.

        :param emission_current: The current CO2 emission value.
        :param reduction_rate: The fixed percentage reduction rate per year (e.g., 0.02 for 2%).

        :return: A DataFrame showing the year-by-year emissions reduction with a fixed rate.
        """
        emission_current = emission_current if emission_current is not None else self.emission_current
        reduction_rate = reduction_rate if reduction_rate is not None else self.reduction_rate
        years = range(self.start_year, self.end_year + 1)
        emissions = [emission_current]

        for year in years[1:]:
            next_emission = emissions[-1] * (1 - reduction_rate)  # Apply the fixed reduction rate
            emissions.append(next_emission)

        df = pd.DataFrame({'Year': years, 'Emission': emissions})
        return df
