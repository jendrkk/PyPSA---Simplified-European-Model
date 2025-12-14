"""
Utility functions for filtering power network buses by geographical boundaries.

This module demonstrates practical integration of the geometry module
with the PyPSA power network data using the RawData container class.
"""

import sys
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon

# Add scripts and src to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from geometry import (
    download_country_shapes,
    download_nuts3_shapes,
    point_in_shape,
    join_shapes,
)

try:
    from pypsa_simplified.data_prep import prepare_osm_source, RawData
except ImportError:
    print("Warning: Could not import pypsa_simplified.data_prep")
    RawData = None


def load_buses(osm_dir: Optional[Path] = None) -> Union['RawData', pd.DataFrame]:
    """
    Load buses data using RawData container.
    
    Parameters
    ----------
    osm_dir : Path, optional
        Path to OSM directory. If None, uses default location.
        
    Returns
    -------
    RawData or pd.DataFrame
        RawData container with network data, or just buses DataFrame if RawData unavailable
        
    Examples
    --------
    >>> raw_data = load_buses()
    >>> buses = raw_data.data['buses']
    >>> print(f"Loaded {len(buses)} buses")
    """
    if osm_dir is None:
        osm_dir = Path(__file__).parent.parent / "data" / "raw" / "OSM Prebuilt Electricity Network"
    
    if RawData is not None:
        try:
            data_dict = prepare_osm_source(osm_dir)
            return RawData(data_dict)
        except Exception as e:
            print(f"Warning: Could not load RawData: {e}")
            # Fallback to direct CSV loading
            buses_path = osm_dir / "buses.csv"
            return pd.read_csv(buses_path)
    else:
        # Fallback to direct CSV loading
        buses_path = osm_dir / "buses.csv"
        return pd.read_csv(buses_path)


def filter_buses_by_countries(
    buses: pd.DataFrame,
    countries: List[str],
    use_geometry: bool = True
) -> pd.DataFrame:
    """
    Filter buses to only include those in specified countries.
    
    Parameters
    ----------
    buses : pd.DataFrame
        Buses dataframe with 'x', 'y', 'country' columns
    countries : list of str
        ISO 2-letter country codes
    use_geometry : bool
        If True, uses geometry module for accurate filtering.
        If False, uses only the 'country' column (faster but may be inaccurate).
        
    Returns
    -------
    pd.DataFrame
        Filtered buses
        
    Examples
    --------
    >>> buses = load_buses()
    >>> buses_de_pl = filter_buses_by_countries(buses, ['DE', 'PL'])
    >>> print(f"Found {len(buses_de_pl)} buses in Germany and Poland")
    """
    if use_geometry:
        # Use geometry module for accurate filtering
        country_shapes = download_country_shapes(countries)
        combined_shape = join_shapes(country_shapes)
        
        # Check each bus
        mask = buses.apply(
            lambda row: point_in_shape(row['y'], row['x'], combined_shape),
            axis=1
        )
        return buses[mask].copy()
    else:
        # Simple filtering by country column
        return buses[buses['country'].isin(countries)].copy()


def filter_buses_by_nuts3(
    buses: pd.DataFrame,
    nuts3_ids: Optional[List[str]] = None,
    countries: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Filter buses to only include those in specified NUTS-3 regions.
    
    Parameters
    ----------
    buses : pd.DataFrame
        Buses dataframe with 'x', 'y' columns
    nuts3_ids : list of str, optional
        Specific NUTS-3 region IDs (e.g., ['DE300', 'PL911'])
    countries : list of str, optional
        Get all NUTS-3 regions for these countries
        
    Returns
    -------
    pd.DataFrame
        Filtered buses with additional 'nuts3_id' column
        
    Examples
    --------
    >>> # Get buses in Berlin region
    >>> buses = load_buses()
    >>> buses_berlin = filter_buses_by_nuts3(buses, nuts3_ids=['DE300'])
    >>> 
    >>> # Get all buses in German NUTS-3 regions
    >>> buses_de_nuts3 = filter_buses_by_nuts3(buses, countries=['DE'])
    """
    # Download NUTS-3 shapes
    if nuts3_ids is not None:
        # Need to determine which countries to download
        # Extract country codes from NUTS-3 IDs (first 2 letters)
        country_codes = list(set([nid[:2] for nid in nuts3_ids]))
        nuts3_shapes = download_nuts3_shapes(country_codes)
        nuts3_shapes = nuts3_shapes[nuts3_shapes['NUTS_ID'].isin(nuts3_ids)]
    elif countries is not None:
        nuts3_shapes = download_nuts3_shapes(countries)
    else:
        raise ValueError("Must provide either nuts3_ids or countries")
    
    if len(nuts3_shapes) == 0:
        return pd.DataFrame(columns=list(buses.columns) + ['nuts3_id'])
    
    # Find which NUTS-3 region each bus belongs to
    results = []
    
    for bus_idx, bus_row in buses.iterrows():
        bus_lat, bus_lon = bus_row['y'], bus_row['x']
        
        # Check each NUTS-3 region
        for nuts_idx, nuts_row in nuts3_shapes.iterrows():
            if point_in_shape(bus_lat, bus_lon, nuts_row['geometry']):
                bus_data = bus_row.to_dict()
                bus_data['nuts3_id'] = nuts_row.get('NUTS_ID', nuts_idx)
                bus_data['nuts3_name'] = nuts_row.get('NAME_LATN', 'Unknown')
                results.append(bus_data)
                break  # Bus found in this region, move to next bus
    
    if len(results) == 0:
        return pd.DataFrame(columns=list(buses.columns) + ['nuts3_id', 'nuts3_name'])
    
    return pd.DataFrame(results)


def filter_buses_by_boundary(
    buses: pd.DataFrame,
    boundary: Union[Polygon, MultiPolygon, gpd.GeoDataFrame],
    return_with_flags: bool = False
) -> pd.DataFrame:
    """
    Filter buses to only include those within a custom boundary.
    
    Parameters
    ----------
    buses : pd.DataFrame
        Buses dataframe with 'x', 'y' columns
    boundary : Polygon, MultiPolygon, or GeoDataFrame
        Custom boundary to filter by
    return_with_flags : bool
        If True, returns all buses with a boolean 'in_boundary' column
        
    Returns
    -------
    pd.DataFrame
        Filtered buses (or all buses with flag if return_with_flags=True)
        
    Examples
    --------
    >>> from geometry import download_country_shapes, buffer_shape
    >>> 
    >>> # Get buses within 50km of German border
    >>> germany = download_country_shapes(['DE'])
    >>> germany_buffer = buffer_shape(germany, distance_km=50)
    >>> buses = load_buses()
    >>> buses_near_germany = filter_buses_by_boundary(buses, germany_buffer)
    """
    buses = buses.copy()
    
    # Check each bus
    mask = buses.apply(
        lambda row: point_in_shape(row['y'], row['x'], boundary),
        axis=1
    )
    
    if return_with_flags:
        buses['in_boundary'] = mask
        return buses
    else:
        return buses[mask]


def assign_nuts3_to_buses(
    buses: pd.DataFrame,
    countries: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Assign NUTS-3 region IDs to all buses.
    
    This adds 'nuts3_id' and 'nuts3_name' columns to the buses dataframe.
    
    Parameters
    ----------
    buses : pd.DataFrame
        Buses dataframe with 'x', 'y', 'country' columns
    countries : list of str, optional
        Only process buses in these countries. If None, uses all unique countries in buses.
        
    Returns
    -------
    pd.DataFrame
        Buses with added 'nuts3_id' and 'nuts3_name' columns
        
    Examples
    --------
    >>> buses = load_buses()
    >>> buses_with_nuts3 = assign_nuts3_to_buses(buses, countries=['DE', 'PL'])
    >>> 
    >>> # Group buses by NUTS-3 region
    >>> buses_per_region = buses_with_nuts3.groupby('nuts3_id').size()
    >>> print(buses_per_region.head())
    """
    if countries is None:
        countries = buses['country'].unique().tolist()
    
    # Download all NUTS-3 regions for these countries
    nuts3_shapes = download_nuts3_shapes(countries)
    
    # Initialize new columns
    buses = buses.copy()
    buses['nuts3_id'] = None
    buses['nuts3_name'] = None
    
    # Assign NUTS-3 to each bus
    for bus_idx, bus_row in buses.iterrows():
        if bus_row['country'] not in countries:
            continue
            
        bus_lat, bus_lon = bus_row['y'], bus_row['x']
        
        # Check each NUTS-3 region
        for nuts_idx, nuts_row in nuts3_shapes.iterrows():
            if point_in_shape(bus_lat, bus_lon, nuts_row['geometry']):
                buses.at[bus_idx, 'nuts3_id'] = nuts_row.get('NUTS_ID', nuts_idx)
                buses.at[bus_idx, 'nuts3_name'] = nuts_row.get('NAME_LATN', 'Unknown')
                break
    
    return buses


def get_buses_statistics_by_country(buses: pd.DataFrame) -> pd.DataFrame:
    """
    Get summary statistics of buses by country.
    
    Parameters
    ----------
    buses : pd.DataFrame
        Buses dataframe with 'country' and 'voltage' columns
        
    Returns
    -------
    pd.DataFrame
        Statistics with columns: country, total_buses, avg_voltage, voltage_levels
        
    Examples
    --------
    >>> buses = load_buses()
    >>> stats = get_buses_statistics_by_country(buses)
    >>> print(stats)
    """
    stats = buses.groupby('country').agg({
        'bus_id': 'count',
        'voltage': ['mean', 'std', 'nunique']
    }).round(1)
    
    stats.columns = ['total_buses', 'avg_voltage', 'std_voltage', 'voltage_levels']
    stats = stats.sort_values('total_buses', ascending=False)
    
    return stats


if __name__ == "__main__":
    """Example usage"""
    print("Loading network data using RawData...")
    raw_data = load_buses()
    
    # Handle both RawData and DataFrame
    if isinstance(raw_data, pd.DataFrame):
        buses = raw_data
        print(f"Loaded buses DataFrame with {len(buses)} buses")
    else:
        buses = raw_data.data['buses']
        print(f"Loaded RawData container with {len(buses)} buses")
        print(f"Available datasets: {list(raw_data.data.keys())}")
    
    print("\nFiltering buses in Germany and Poland...")
    buses_de_pl = filter_buses_by_countries(buses, ['DE', 'PL'])
    print(f"Buses in DE+PL: {len(buses_de_pl)}")
    
    print("\nBuses statistics by country:")
    stats = get_buses_statistics_by_country(buses)
    print(stats.head(10))
    
    print("\nFiltering buses in German NUTS-3 regions (sample)...")
    # Just get a few buses for demonstration
    buses_sample = buses.head(1000)
    buses_de_sample = filter_buses_by_countries(buses_sample, ['DE'])
    
    if len(buses_de_sample) > 0:
        print(f"Assigning NUTS-3 to {len(buses_de_sample)} German buses...")
        buses_with_nuts3 = assign_nuts3_to_buses(buses_de_sample, countries=['DE'])
        
        print("\nBuses by NUTS-3 region (top 10):")
        nuts3_counts = buses_with_nuts3['nuts3_id'].value_counts().head(10)
        print(nuts3_counts)
    
    print("\nDone!")
