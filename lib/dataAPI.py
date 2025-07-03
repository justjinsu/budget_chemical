import requests
import pandas as pd
import openpyxl

import pandas as pd
import requests
import warnings
import pycountry


def download_worldbank_data(indicator, start_year, end_year):
    """
    Fetches data from the World Bank API for the given indicators and time range.

    :param countries: A string or list of country codes (e.g., "US" or ["US", "IN", "CN"]).
    :param indicator: A World Bank indicator code (e.g., 'NY.GDP.MKTP.CD').
    :param start_year: The start year for the data.
    :param end_year: The end year for the data.
    :return: A pandas DataFrame containing the fetched data.
    """
    countries = 'all'
    # Ensure countries are in a semicolon-separated string
    if isinstance(countries, list):
        countries = ';'.join(countries)
    elif isinstance(countries, str):
        countries = countries.replace(',', ';')
    else:
        raise ValueError("Countries parameter must be a string or a list of country codes.")

    base_url = "https://api.worldbank.org/v2/country/{}/indicator/{}"
    params = {
        "date": f"{start_year}:{end_year}",
        "format": "json",
        "per_page": 1000  # Adjust if needed; maximum per_page allowed by API
    }

    # Construct the URL for the current indicator
    url = base_url.format(countries, indicator)

    # Initial request to get total pages
    response = requests.get(url, params=params)

    if response.status_code != 200:
        raise RuntimeError(f"Failed to fetch data for indicator {indicator}: HTTP {response.status_code}")

    try:
        data = response.json()
    except ValueError:
        raise RuntimeError(f"Invalid JSON response for indicator {indicator}")

    if not isinstance(data, list) or len(data) < 2:
        raise RuntimeError(f"No data found for indicator {indicator}")

    # Metadata is in data[0], actual data in data[1]
    metadata, records = data
    total_pages = metadata.get('pages', 1)

    # Collect data from all pages
    all_records = records.copy()
    for page in range(2, total_pages + 1):
        params['page'] = page
        response = requests.get(url, params=params)
        if response.status_code != 200:
            print(f"Failed to fetch page {page} for indicator {indicator}: HTTP {response.status_code}")
            break
        try:
            page_data = response.json()
        except ValueError:
            print(f"Invalid JSON response on page {page} for indicator {indicator}")
            break
        if len(page_data) < 2:
            print(f"No data on page {page} for indicator {indicator}")
            break
        all_records.extend(page_data[1])

    # Create DataFrame from records
    temp_df = pd.DataFrame(all_records)

    # Select relevant columns
    if not {'country', 'date', 'value'}.issubset(temp_df.columns):
        raise RuntimeError(f"Missing expected columns in data for indicator {indicator}")

    temp_df = temp_df[['country', 'date', 'value']]

    # Normalize country info (split into country code and country name)
    temp_df['country_code_alpha2'] = temp_df['country'].apply(lambda x: x['id'])  # Extract country code (2-digit)
    temp_df['country_name'] = temp_df['country'].apply(lambda x: x['value'])  # Extract country name

    # Add indicator column
    temp_df['indicator'] = indicator
    temp_df = temp_df[['country_code_alpha2', 'country_name', 'indicator', 'date', 'value']]
    temp_df.rename(columns={'date': 'year'}, inplace=True)
    temp_df['year'] = temp_df['year'].astype(int)

    # Forward-fill NaN values for each country_code_alpha2
    temp_df['value'] = temp_df.groupby('country_code_alpha2')['value'].ffill()

    # Drop rows where NaN values still exist
    na_countries = temp_df[temp_df['value'].isna()]['country_code_alpha2'].unique()
    temp_df = temp_df.dropna(subset=['value'])

    if len(na_countries) > 0:
        warnings.warn(f"Dropping countries with NaN values: {', '.join(na_countries)}.")


    return temp_df

def cleanup_wbdata(wbdata_lt):
    wbdata_df = pd.concat(wbdata_lt)

    # Use apply with lambda function to handle country code conversion from 2 digits (alpha 2) to 3 diguts (alpha 3)
    wbdata_df['country_code'] = wbdata_df['country_code_alpha2'].apply(
        lambda c: pycountry.countries.get(alpha_2=c).alpha_3 if pycountry.countries.get(alpha_2=c) else None)
    wbdata_df = wbdata_df[wbdata_df['country_code'].notna()].drop(columns=['country_code_alpha2'])

    # Reorder columns to have 'country_code' first
    columns = ['country_code'] + [col for col in wbdata_df.columns if col != 'country_code']
    wbdata_df = wbdata_df[columns]

    return wbdata_df


def clean_and_extract(file_path, skip = 8):
    # Skip initial metadata rows and set the first valid row as the header
    df_clean = pd.read_excel(file_path, sheet_name='TOTALS BY COUNTRY', skiprows=skip, header = 1)

    # Drop any columns that are completely empty (likely irrelevant metadata)
    df_clean = df_clean.dropna(how='all', axis=1).iloc[:, 2:]

    # Rename the first two columns for clarity (assuming these are country codes and names)
    df_clean.rename(columns={'Country_code_A3': 'country_code', 'Name': 'country_name'}, inplace=True)

    # Use apply to rename columns that start with "Y_" to the year as an integer
    df_clean.columns = df_clean.columns.to_series().apply(lambda col: int(col[2:]) if col.startswith('Y_') else col)


    # Convert the DataFrame from wide to long format
    df_long = pd.melt(df_clean, id_vars=['country_code', 'country_name', 'Substance'], var_name='year', value_name='value')

    # Insert the 'indicator' column between 'country_name' and 'year'
    df_long.rename(columns = {'Substance': 'indicator'}, inplace=True)
    df_long['year'] = df_long['year'].astype(int)

    return df_long