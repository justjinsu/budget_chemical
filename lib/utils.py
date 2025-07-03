import pycountry

def convert_iso2_to_iso3(country_code):
    """
    Converts an ISO 3166-1 alpha-2 country code to ISO 3166-1 alpha-3.

    :param country_code: ISO 3166-1 alpha-2 code (e.g., 'US').
    :return: ISO 3166-1 alpha-3 code (e.g., 'USA'), or None if not found.
    """
    try:
        country = pycountry.countries.get(alpha_2=country_code)
        return country.alpha_3
    except AttributeError:
        return None  # Return None if the country code is invalid


def filter_budget_df(budget_df, countries, temp_values, probability_values, approach_values, period_values):
    """
    Filters the budget dataframe based on specified criteria.

    Parameters:
    budget_df (pd.DataFrame): The dataframe containing budget data.
    countries (list): List of country codes to filter by (e.g., ['KOR', 'JPN']).
    temp_values (list): List of temperature values to filter by (e.g., [1.5, 1.7]).
    probability_values (list): List of probability values to filter by (e.g., [0.5]).
    approach_values (list): List of approach values to filter by (e.g., ['SP.POP.TOTL']).
    period_values (list): List of period values to filter by (e.g., [2022]).

    Returns:
    pd.DataFrame: The filtered dataframe.
    """
    filtered_df = budget_df[
        (budget_df['country_code'].isin(countries)) &
        (budget_df['temp'].isin(temp_values)) &
        (budget_df['probability'].isin(probability_values)) &
        (budget_df['approach'].isin(approach_values)) &
        (budget_df['period'].isin(period_values))
    ]
    return filtered_df

