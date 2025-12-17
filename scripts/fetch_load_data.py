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
    timestamps = pd.to_datetime(json_data["unix_seconds"], unit="s")
    df = pd.DataFrame(index=timestamps)
    df.index = df.index.tz_localize("UTC")
    df.index = df.index + pd.Timedelta(hours=1)

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

def main():
    force_donwload = input("Force download all data? (y/n): ").strip().lower() == 'y'
    countries = EU27
    years = list(range(2015, 2025))
    links = create_links(countries, years)
    default_outpath = repo_root / 'data' / 'raw' / 'energy_charts'
    i = 0
    for country, link in links:
        if check_dir_for_data(link, default_outpath) and not force_donwload:
            print(f"Data for {country} already exists. Skipping download.")
            i += 1
            continue:
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
        combine_all(countries, years, default_outpath)
        print("Combined files have been created.")
if __name__ == "__main__":
    main()