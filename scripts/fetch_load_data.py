"""Utilities to download power production time series from
the energy-charts API and save them as CSV files.

This module creates per-country, per-year CSV files under
`data/raw/energy_charts` and handles common JSON shapes
returned by the API.
"""

import re
import requests
import geometry as geom
import pandas as pd
from pathlib import Path
import time
import pytz
import numpy as np
from typing import List, Tuple, Union

EU27: List[str] = geom.EU27

DICT_COUNTRIES: dict[str, str] = {
    'all': 'all',
    'EU27': 'eu'
}

def find_repo_root(start_path: Path, max_up: int = 6) -> Path:
    """Find repository root by searching upward for README.md or .git"""
    p = start_path.resolve()
    for _ in range(max_up):
        if (p / 'README.md').exists() or (p / '.git').exists():
            return p
        if p.parent == p:
            break
        p = p.parent
    return start_path.resolve()

repo_root = find_repo_root(Path(__file__).parent)

def create_links(countries: List[str], years: List[int]) -> List[Tuple[str, str]]:
    """Create (country, link) pairs for the requested years.

    Parameters
    - countries: list of ISO2 country codes (upper-case)
    - years: list of years to download
    """
    pattern = re.compile(r'^[A-Z]{2}$')
    links: List[Tuple[str, str]] = []
    for country in countries:
        if country in ['EL', 'GB', 'MT']:
            continue  # not available in energy-charts API
        if not pattern.match(country):
            raise ValueError(f"Invalid country code: {country}")
        for year in years:
            start_date = f"{year}-01-01"
            end_date = f"{year}-12-31"
            link = (
                f"https://api.energy-charts.info/public_power?country="
                f"{country.lower()}&start={start_date}&end={end_date}"
            )
            links.append((country, link))

    return links

def json_to_dataframe(json_data: dict) -> pd.DataFrame:
    """Convert JSON data from the energy-charts API to a pandas DataFrame.

    The API returns `unix_seconds` and a list of `production_types` with
    their `data` arrays. The function localizes timestamps to UTC and
    shifts them by one hour to align with CET/CEST as used by the source.
    """
    tz = pytz.FixedOffset(60)
    timestamps = pd.to_datetime(json_data["unix_seconds"], unit="s", utc = True).tz_convert(tz)
    df = pd.DataFrame(index=timestamps)
    df['Timestamp'] = df.index
    
    for prod_type in json_data.get("production_types", []):
        name = prod_type.get("name")
        data = prod_type.get("data")
        df[name] = data
        
    return df

def download_data(link: str, outpath: Union[Path, str]) -> None:
    """Download JSON from `link` and save as CSV into `outpath`.

    Filenames include the country code and year to avoid overwriting
    when downloading multiple years for the same country.
    """
    outpath = Path(outpath)
    outpath.mkdir(parents=True, exist_ok=True)
    
    try:
        response = requests.get(link)
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error occurred while downloading {link}: {e}")
        return None
    
    # Try reading structured JSON and converting to DataFrame.
    try:
        json_data = response.json()
        data = json_to_dataframe(json_data)
    except ValueError:
        print("Falling back to json_normalize")
        data = pd.json_normalize(response.json())

    country = link.split("country=")[1].split("&")[0]
    m = re.search(r"start=(\d{4})", link)
    year = m.group(1) if m else ""
    filename = f"{country}_{year}.csv" if year else f"{country}.csv"

    data.to_csv(outpath / filename, index=False)

def check_dir_for_data(link: str, outpath: Union[Path, str]) -> bool:
    """Check if the CSV file for the given link already exists in outpath.

    Returns True if the file exists, False otherwise.
    """
    outpath = Path(outpath)
    country = link.split("country=")[1].split("&")[0]
    m = re.search(r"start=(\d{4})", link)
    year = m.group(1) if m else ""
    filename = f"{country}_{year}.csv" if year else f"{country}.csv"
    file_path = outpath / filename
    return file_path.exists()

def combine_all(countries: List[str], years: List[int], outpath: Union[Path, str]) -> None:
    """Combine all downloaded CSV files into a pandas data frame with 3 columns.
    First column: timestamp, second column: country, third column: value [Load].
    Load data come from the orignal per-country, per-year CSV files, column name is
    also 'Load'. The combined data frame uses multi-index with timestamp and country.
    The combined data frame is saved as single CSV file for all countries and years in outpath.
    Missing data is filled with NaN.

    Parameters
    - countries: list of ISO2 country codes (upper-case)
    - years: list of years to combine
    - outpath: directory where the CSV files are stored
    """
    outpath = Path(outpath)
    combined_df = pd.DataFrame()
    for country in countries:
        country_dfs = []
        for year in years:
            filename = f"{country.lower()}_{year}.csv"
            file_path = outpath / filename
            if file_path.exists():
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                # Resample to 'h' frequency to ensure consistent timestamps
                df = df.resample('h').mean()
                if 'Load' in df.columns:
                    df_country = df[['Load']].copy()
                    df_country['Country'] = country
                    country_dfs.append(df_country)
        if country_dfs:
            country_combined = pd.concat(country_dfs)
            combined_df = pd.concat([combined_df, country_combined])
    combined_df.reset_index(inplace=True)
    combined_df.rename(columns={'index': 'Timestamp'}, inplace=True)
    combined_df.set_index(['Timestamp', 'Country'], inplace=True)
    combined_df.columns = ['Load']
    combined_filename = outpath / "combined_load_data.csv"
    combined_df.to_csv(combined_filename)

def find_missing_ranges(df: pd.DataFrame, n_missing: int = 6) -> List[Tuple]:
    """Find consecutive missing values in the 'Load' column if there are more than n_missing in a row.
    
    Parameters
    - df: DataFrame with 'Load' column
    - n_missing: minimum number of consecutive missing values to report
    
    Returns
    - List of tuples (start_timestamp, end_timestamp) for missing ranges
    """
    missing_ranges = []
    in_missing = False
    start_missing = None
    missing_count = 0

    for idx, row in df.iterrows():
        if pd.isna(row['Load']):
            if not in_missing:
                in_missing = True
                start_missing = idx
            missing_count += 1
        else:
            if in_missing:
                if missing_count >= n_missing:
                    missing_ranges.append((start_missing, idx))
                in_missing = False
                missing_count = 0

    # Check if we ended while still in a missing range
    if in_missing and missing_count >= n_missing:
        missing_ranges.append((start_missing, df.index[-1]))

    return missing_ranges


def process_load_data(load_df: pd.DataFrame) -> pd.DataFrame:
    """Process combined load data with UTC conversion and missing value handling.
    
    Takes load data from combine_all (in long format with CET timestamps), converts to UTC,
    handles special cases for UK and IE, fills remaining missing values with time interpolation.
    
    Parameters
    - load_df: DataFrame from combine_all with structure:
        - Index: 'Timestamp' (in CET), 'Country' (multi-index)
        - Columns: 'Load'
    
    Returns
    - DataFrame with columns: 'Timestamp', 'Country', 'Load', 'Missing'
      - Timestamp: UTC timezone, from 2015-01-01 00:00:00+00:00 to 2024-12-31 23:00:00+00:00
      - Country: ISO2 country codes
      - Load: Hourly electricity load (no missing values except where originally all missing)
      - Missing: Boolean indicating if the Load value was originally missing
    """
    # Reset index to work with long format
    load_df = load_df.reset_index()
    
    # Convert from CET (GMT+1) to UTC
    cet_tz = pytz.FixedOffset(60)  # CET is GMT+1
    load_df['Timestamp'] = pd.to_datetime(load_df['Timestamp'])
    load_df['Timestamp'] = load_df['Timestamp'].dt.tz_convert(cet_tz)
    load_df['Timestamp'] = load_df['Timestamp'].dt.tz_convert('UTC')
    
    # Create complete time range in UTC
    time_2015_to_2024 = pd.date_range(
        start='2015-01-01', 
        end='2024-12-31 23:00', 
        freq='h', 
        tz='UTC'
    )
    
    # Process each country
    df_result = pd.DataFrame()
    unique_countries = load_df['Country'].unique()
    
    for country in unique_countries:
        newdf = pd.DataFrame()
        newdf.index = time_2015_to_2024
        newdf.index.name = 'Timestamp'
        newdf['Country'] = country
        
        # Get country-specific load data
        country_load_data = load_df[load_df['Country'] == country].copy()
        country_load_data = country_load_data.drop('Country', axis=1)
        country_load_data['Missing'] = country_load_data['Load'].isna()
        country_load_data = country_load_data.set_index('Timestamp')
        
        # Merge with the complete time range
        newdf = newdf.merge(country_load_data, left_index=True, right_index=True, how='left')
        
        # Special handling for UK
        if country == 'UK':
            # Scale UK data for specific time periods based on historical data
            newdf.loc[(newdf.index >= '2021-06-14 08:00:00+00:00') & 
                      (newdf.index <= '2022-06-14 07:00:00+00:00'), 'Load'] = \
                np.array(newdf.loc[(newdf.index >= '2020-06-14 08:00:00+00:00') & 
                                   (newdf.index <= '2021-06-14 07:00:00+00:00'), 'Load'].values) * 0.975
            
            newdf.loc[(newdf.index >= '2022-06-14 08:00:00+00:00') & 
                      (newdf.index <= '2023-06-14 07:00:00+00:00'), 'Load'] = \
                np.array(newdf.loc[(newdf.index >= '2020-06-14 08:00:00+00:00') & 
                                   (newdf.index <= '2021-06-14 07:00:00+00:00'), 'Load'].values) * 0.95
            
            newdf.loc[(newdf.index >= '2023-06-14 08:00:00+00:00') & 
                      (newdf.index <= '2024-02-28 23:00:00+00:00'), 'Load'] = \
                np.array(newdf.loc[(newdf.index >= '2020-06-14 08:00:00+00:00') & 
                                   (newdf.index <= '2021-02-28 23:00:00+00:00'), 'Load'].values) * 0.91
            
            newdf.loc[(newdf.index >= '2024-02-29 00:00:00+00:00') & 
                      (newdf.index <= '2024-02-29 23:00:00+00:00'), 'Load'] = \
                np.array(newdf.loc[(newdf.index >= '2024-02-28 00:00:00+00:00') & 
                                   (newdf.index <= '2024-02-28 23:00:00+00:00'), 'Load'].values) * 1.0
            
            newdf.loc[(newdf.index >= '2024-03-01 00:00:00+00:00') & 
                      (newdf.index <= '2024-06-14 07:00:00+00:00'), 'Load'] = \
                np.array(newdf.loc[(newdf.index >= '2021-03-01 00:00:00+00:00') & 
                                   (newdf.index <= '2021-06-14 07:00:00+00:00'), 'Load'].values) * 0.91
            
            newdf.loc[(newdf.index >= '2024-06-14 08:00:00+00:00') & 
                      (newdf.index <= '2024-12-31 23:00:00+00:00'), 'Load'] = \
                np.array(newdf.loc[(newdf.index >= '2020-06-14 08:00:00+00:00') & 
                                   (newdf.index <= '2020-12-31 23:00:00+00:00'), 'Load'].values) * 0.91
            
            newdf['Load'] = newdf['Load'].interpolate(method='time')
            df_result = pd.concat([df_result, newdf])
            continue
        
        # If all values are missing for a country, skip special processing
        if newdf['Load'].isna().all():
            df_result = pd.concat([df_result, newdf])
            continue
        
        # Special handling.
        miss = find_missing_ranges(newdf, 6)
        if len(miss) > 0:
            for start, end in miss:
                leng = (end - start).total_seconds() / 3600
                print(f"Missing range for {country} from {start} to {end}, length: {leng} hours")
                previous_day_start = start - pd.DateOffset(days=1)
                previous_day_end = end - pd.DateOffset(days=1)
                
                feasible = previous_day_end in time_2015_to_2024 and previous_day_start in time_2015_to_2024
                # Try filling from previous day
                if newdf.loc[previous_day_start:previous_day_end, 'Load'].isna().sum() <= 1 and feasible:
                    if len(newdf.loc[start:end, 'Load']) != len(newdf.loc[previous_day_start:previous_day_end, 'Load']):
                        print(f"Length of current missing data: {len(newdf.loc[start:end, 'Load'])} hours")
                        print(f"Length of previous day data: {len(newdf.loc[previous_day_start:previous_day_end, 'Load'])} hours")
                    newdf.loc[start:end, 'Load'] = newdf.loc[previous_day_start:previous_day_end, 'Load'].values
                    print(f"Filled missing range for {country} from {start} to {end} using previous day's data")
                    continue
                
                # Try filling from next day
                next_day_start = start + pd.DateOffset(days=1)
                next_day_end = end + pd.DateOffset(days=1)
                
                feasible = next_day_end in time_2015_to_2024 and next_day_start in time_2015_to_2024
                if newdf.loc[next_day_start:next_day_end, 'Load'].isna().sum() <= 1 and feasible:
                    if len(newdf.loc[start:end, 'Load']) != len(newdf.loc[next_day_start:next_day_end, 'Load']):
                        print(f"Length of current missing data: {len(newdf.loc[start:end, 'Load'])} hours")
                        print(f"Length of next day data: {len(newdf.loc[next_day_start:next_day_end, 'Load'])} hours")
                    newdf.loc[start:end, 'Load'] = newdf.loc[next_day_start:next_day_end, 'Load'].values
                    print(f"Filled missing range for {country} from {start} to {end} using next day's data")
                    continue
                
                # Try filling from same time last year
                previous_year_start = start - pd.DateOffset(years=1)
                previous_year_end = end - pd.DateOffset(years=1)
                prev_year = newdf.loc[previous_year_start:previous_year_end]
                
                feasible = previous_year_end in time_2015_to_2024 and previous_year_start in time_2015_to_2024
                if find_missing_ranges(prev_year, 6) == [] and feasible:
                    a = len(newdf.loc[start:end, 'Load'])
                    b = len(newdf.loc[previous_year_start:previous_year_end, 'Load'])
                    
                    if a != b:
                        # Handling leap year mismatch
                        if a > b:
                            # Current missing data is longer (filling Feb 29)
                            newdf.loc[start:end, 'Load'] = np.concatenate([
                                newdf.loc[previous_year_start:previous_year_end, 'Load'].values,
                                np.full((a - b,), newdf.loc[previous_year_end, 'Load'])
                            ])
                        else:
                            # Previous year data is longer
                            newdf.loc[start:end, 'Load'] = newdf.loc[
                                previous_year_start:previous_year_end - pd.DateOffset(hours=(b - a)), 'Load'
                            ].values
                    else:
                        newdf.loc[start:end, 'Load'] = newdf.loc[previous_year_start:previous_year_end, 'Load'].values
                    
                    print(f"Filled missing range for {country} from {start} to {end} using previous year's data")
                    continue
                
                # Try filling from same time next year
                next_year_start = start + pd.DateOffset(years=1)
                next_year_end = end + pd.DateOffset(years=1)
                next_year = newdf.loc[next_year_start:next_year_end]
                
                feasible = next_year_end in time_2015_to_2024 and next_year_start in time_2015_to_2024
                if find_missing_ranges(next_year, 6) == [] and feasible:
                    a = len(newdf.loc[start:end, 'Load'])
                    b = len(newdf.loc[next_year_start:next_year_end, 'Load'])
                    
                    if a != b:
                        # Handling leap year mismatch
                        if a > b:
                            # Current missing data is longer (filling Feb 29)
                            newdf.loc[start:end, 'Load'] = np.concatenate([
                                newdf.loc[next_year_start:next_year_end, 'Load'].values,
                                np.full((a - b,), newdf.loc[next_year_end, 'Load'])
                            ])
                        else:
                            # Previous year data is longer
                            newdf.loc[start:end, 'Load'] = newdf.loc[
                                next_year_start:next_year_end - pd.DateOffset(hours=(b - a)), 'Load'
                            ].values
                    else:
                        newdf.loc[start:end, 'Load'] = newdf.loc[next_year_start:next_year_end, 'Load'].values
                    
                    print(f"Filled missing range for {country} from {start} to {end} using previous year's data")
                    continue
                
                # If not possible, give a warning
                print(f"⚠ Could not fill missing range for {country} from {start} to {end}")
            
            df_result = pd.concat([df_result, newdf])
            continue
        
        # Fill missing values with linear time interpolation for all other countries
        newdf['Load'] = newdf['Load'].interpolate(method='time')
        df_result = pd.concat([df_result, newdf])
    
    # Reset index and prepare final output
    countries = df_result['Country'].unique()
    if len(countries) != 28:
        print(f"Warning: Expected 28 countries, found {len(countries)} countries.")
    # Interpolate again to ensure no missing values remain after all processing
    for country in countries:
        mask = df_result['Country'] == country
        df_result.loc[mask, 'Load'] = df_result.loc[mask, 'Load'].interpolate(method='time')
        # Final check for any remaining missing values
        if df_result.loc[mask, 'Load'].isna().any():
            # Interpolate with nearest if any missing values remain
            df_result.loc[mask, 'Load'] = df_result.loc[mask, 'Load'].interpolate(method='nearest')
            print(f"Warning: Used nearest interpolation for remaining missing values in {country}.")
    
    # At the end we are summing up all countries per timestamp and create "EU" entries
    eu_df = df_result.groupby('Timestamp')['Load'].sum().reset_index()
    eu_df['Country'] = 'EU'
    eu_df['Missing'] = False  # EU total should not have missing values after interpolation
    df_result = pd.concat([df_result, eu_df], ignore_index=True)
    
    df_result = df_result.reset_index()
    df_result = df_result[['Timestamp', 'Country', 'Load', 'Missing']]
    df_result = df_result.sort_values(['Timestamp', 'Country']).reset_index(drop=True)
    
    return df_result


def main():
    force_donwload = input("Force download all data? (y/n): ").strip().lower() == 'y'
    countries = EU27
    years = list(range(2015, 2025))
    links = create_links(countries, years)
    default_outpath = repo_root / 'data' / 'raw' / 'energy_charts'
    i = 0
    for country, link in links:
        if check_dir_for_data(link, default_outpath) and not force_donwload:
            year = link.split("start=")[1].split("-")[0]
            #print(f"Data for {country} and {year} already exists. Skipping download.")
            i += 1
            continue
        try:
            download_data(link, default_outpath)
        except Exception as exc:  # keep going if one file fails
            print(f"Failed to download {country} ({link}): {exc}")
            time.sleep(2)
        finally:
            time.sleep(2)  # To avoid overwhelming the server
    if i == len(links):
        print("All data files already exist. No downloads were performed.")
    
    combine_files = input("Combine all yearly files into single country files? (y/n): ").strip().lower() == 'y'
    if combine_files:
        print(default_outpath)
        combine_all(countries, years, default_outpath)
        print("Combined files have been created.")
        combined_df = pd.read_csv(default_outpath / "combined_load_data.csv", index_col=0, parse_dates=True)
        processed_df = process_load_data(combined_df)
        processed_df.to_csv(default_outpath / "processed_load_data.csv", index=False)
        print("Processed load data has been saved.")

if __name__ == "__main__":
    main()