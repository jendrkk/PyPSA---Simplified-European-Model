"""
Renewable generation profiles for simplified European power system model.

This module provides functions to:
1. Convert weather data (wind speed, solar radiation) to capacity factors
2. Assign generators to NUTS-2 regions
3. Create time-varying p_max_pu profiles for wind and solar generators
4. Handle offshore wind with coastal region approximation

Based on PyPSA-EUR methodology but simplified for use without Atlite cutouts.

Key simplifications vs. PyPSA-EUR:
- Uses pre-computed NUTS-2 level weather data instead of ERA5 cutouts
- Simplified wind power curve (power law scaling)
- Simplified solar irradiance to power conversion
- Offshore wind uses coastal regions with enhancement factor
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants from PyPSA-EUR and literature
# Reference: https://pypsa-eur.readthedocs.io/

# Wind turbine parameters (simplified Vestas V112 3MW characteristics)
WIND_CUT_IN = 3.0          # m/s - minimum wind speed for generation
WIND_RATED = 12.0          # m/s - wind speed at rated power
WIND_CUT_OUT = 25.0        # m/s - maximum wind speed (turbine shuts down)
WIND_POWER_CURVE_EXPONENT = 3.0  # For power ~ v^3 in cubic region

# Offshore wind enhancement factor
# Offshore wind speeds are typically 15-25% higher than onshore
# and have higher capacity factors due to steadier winds
OFFSHORE_WIND_FACTOR = 1.25  # Multiplier for coastal wind speeds

# Solar PV parameters
# Simplified conversion from irradiance to capacity factor
# Based on typical European PV systems (fixed tilt, south-facing)
SOLAR_STC_IRRADIANCE = 1000  # W/m² - Standard Test Conditions irradiance
SOLAR_PERFORMANCE_RATIO = 0.85  # Typical system losses (cables, inverter, etc.)
SOLAR_TEMP_COEFFICIENT = -0.004  # Power decrease per °C above 25°C

# Default reference temperatures for solar (simplified)
SOLAR_REFERENCE_TEMP = 25.0  # °C


def wind_power_curve(wind_speed: Union[float, np.ndarray, pd.Series]) -> Union[float, np.ndarray, pd.Series]:
    """
    Calculate wind turbine capacity factor from wind speed.
    
    Uses a simplified piecewise power curve:
    - Below cut-in: 0
    - Cut-in to rated: Cubic power law
    - Rated to cut-out: 1.0 (rated power)
    - Above cut-out: 0 (turbine shutdown)
    
    This approximates typical modern wind turbines (e.g., Vestas V112).
    PyPSA-EUR uses detailed power curves from atlite; this is a simplification.
    
    Parameters
    ----------
    wind_speed : float, np.ndarray, or pd.Series
        Wind speed at hub height (typically 100m) in m/s
        
    Returns
    -------
    Same type as input
        Capacity factor between 0 and 1
        
    References
    ----------
    - PyPSA-EUR: scripts/build_renewable_profiles.py
    - Atlite wind power conversion: https://atlite.readthedocs.io/
    
    Examples
    --------
    >>> wind_power_curve(5.0)  # Below rated
    0.072...
    >>> wind_power_curve(12.0)  # At rated
    1.0
    >>> wind_power_curve(30.0)  # Above cut-out
    0.0
    """
    ws = np.asarray(wind_speed)
    cf = np.zeros_like(ws, dtype=float)
    
    # Region 1: Below cut-in - no power
    # (default zeros already set)
    
    # Region 2: Cut-in to rated - cubic power law
    # P/P_rated = ((v - v_cut_in) / (v_rated - v_cut_in))^3
    mask_cubic = (ws >= WIND_CUT_IN) & (ws < WIND_RATED)
    cf[mask_cubic] = ((ws[mask_cubic] - WIND_CUT_IN) / (WIND_RATED - WIND_CUT_IN)) ** WIND_POWER_CURVE_EXPONENT
    
    # Region 3: Rated to cut-out - full power
    mask_rated = (ws >= WIND_RATED) & (ws <= WIND_CUT_OUT)
    cf[mask_rated] = 1.0
    
    # Region 4: Above cut-out - turbine shutdown
    mask_cutout = ws > WIND_CUT_OUT
    cf[mask_cutout] = 0.0
    
    # Return same type as input
    if isinstance(wind_speed, pd.Series):
        return pd.Series(cf, index=wind_speed.index)
    elif isinstance(wind_speed, np.ndarray):
        return cf
    else:
        return float(cf)


def solar_capacity_factor(
    irradiance: Union[float, np.ndarray, pd.Series],
    temperature: Optional[Union[float, np.ndarray, pd.Series]] = None
) -> Union[float, np.ndarray, pd.Series]:
    """
    Calculate solar PV capacity factor from irradiance.
    
    Simplified conversion from solar irradiance (W/m²) to capacity factor.
    Optionally includes temperature correction (PV efficiency decreases with heat).
    
    Parameters
    ----------
    irradiance : float, np.ndarray, or pd.Series
        Global horizontal irradiance (GHI) in W/m²
        Note: Our data uses "rad" which is surface solar radiation downward
    temperature : float, np.ndarray, or pd.Series, optional
        Ambient temperature in °C. If None, assumes 25°C (no correction)
        
    Returns
    -------
    Same type as input
        Capacity factor between 0 and 1
        
    Notes
    -----
    This is simplified vs. PyPSA-EUR which uses:
    - Detailed panel models (orientation, tilt)
    - Diffuse/direct irradiance components
    - Cell temperature models
    
    For a simplified model, we use:
    CF = (GHI / GHI_STC) * performance_ratio * temp_correction
    
    References
    ----------
    - PyPSA-EUR: scripts/build_renewable_profiles.py (pv method)
    - Typical European PV performance: 10-15% capacity factor average
    """
    irr = np.asarray(irradiance)
    
    # Basic conversion: linear scaling with STC
    cf = (irr / SOLAR_STC_IRRADIANCE) * SOLAR_PERFORMANCE_RATIO
    
    # Temperature correction (optional)
    if temperature is not None:
        temp = np.asarray(temperature)
        # Efficiency decreases ~0.4% per °C above 25°C
        temp_factor = 1.0 + SOLAR_TEMP_COEFFICIENT * (temp - SOLAR_REFERENCE_TEMP)
        cf = cf * np.clip(temp_factor, 0.5, 1.2)  # Limit extreme corrections
    
    # Clip to valid range
    cf = np.clip(cf, 0.0, 1.0)
    
    # Return same type as input
    if isinstance(irradiance, pd.Series):
        return pd.Series(cf, index=irradiance.index)
    elif isinstance(irradiance, np.ndarray):
        return cf
    else:
        return float(cf)


def get_nuts2_for_point(
    lat: float,
    lon: float,
    nuts2_shapes: gpd.GeoDataFrame,
    nuts_id_col: str = 'NUTS_ID'
) -> Optional[str]:
    """
    Find the NUTS-2 region containing a given point.
    
    Parameters
    ----------
    lat : float
        Latitude in degrees
    lon : float
        Longitude in degrees
    nuts2_shapes : gpd.GeoDataFrame
        GeoDataFrame with NUTS-2 region geometries
    nuts_id_col : str
        Column name containing NUTS-2 IDs
        
    Returns
    -------
    str or None
        NUTS-2 ID (e.g., 'DE11') or None if not found
    """
    point = Point(lon, lat)
    
    # Ensure CRS matches
    if nuts2_shapes.crs and nuts2_shapes.crs != 'EPSG:4326':
        nuts2_shapes = nuts2_shapes.to_crs('EPSG:4326')
    
    # Find containing region
    for idx, row in nuts2_shapes.iterrows():
        if row.geometry.contains(point):
            return row[nuts_id_col]
    
    # If no exact match, find nearest
    distances = nuts2_shapes.geometry.distance(point)
    nearest_idx = distances.idxmin()
    logger.debug(f"Point ({lat}, {lon}) not in any NUTS-2 region, using nearest: {nuts2_shapes.loc[nearest_idx, nuts_id_col]}")
    return nuts2_shapes.loc[nearest_idx, nuts_id_col]


def get_coastal_nuts2_regions(
    nuts2_shapes: gpd.GeoDataFrame,
    nuts_id_col: str = 'NUTS_ID',
    buffer_km: float = 50.0
) -> List[str]:
    """
    Identify NUTS-2 regions that are coastal (for offshore wind).
    
    A region is considered coastal if it intersects with a buffered coastline.
    For simplicity, we check if the region borders the sea by looking at
    its boundary intersection with the overall land boundary.
    
    Parameters
    ----------
    nuts2_shapes : gpd.GeoDataFrame
        GeoDataFrame with NUTS-2 region geometries
    nuts_id_col : str
        Column name containing NUTS-2 IDs
    buffer_km : float
        Buffer distance in km for coastal definition
        
    Returns
    -------
    List[str]
        List of NUTS-2 IDs that are coastal
        
    Notes
    -----
    This is a simplified approach. PyPSA-EUR uses actual shoreline data
    and offshore EEZ regions.
    """
    # Convert to projected CRS for distance calculations
    gdf = nuts2_shapes.to_crs('EPSG:3035')
    
    # Create union of all land
    all_land = gdf.union_all()
    
    # Get exterior boundary (coastline approximation)
    exterior = all_land.exterior if hasattr(all_land, 'exterior') else all_land.boundary
    
    # Buffer around boundary
    buffer_m = buffer_km * 1000
    coastal_zone = exterior.buffer(buffer_m)
    
    # Find regions intersecting coastal zone
    coastal_regions = []
    for idx, row in gdf.iterrows():
        if row.geometry.intersects(coastal_zone):
            coastal_regions.append(nuts2_shapes.loc[idx, nuts_id_col])
    
    logger.info(f"Identified {len(coastal_regions)} coastal NUTS-2 regions")
    return coastal_regions


def assign_generators_to_nuts2(
    generators: pd.DataFrame,
    nuts2_shapes: gpd.GeoDataFrame,
    lat_col: str = 'lat',
    lon_col: str = 'lon',
    nuts_id_col: str = 'NUTS_ID'
) -> pd.DataFrame:
    """
    Assign NUTS-2 region IDs to generators based on their location.
    
    Parameters
    ----------
    generators : pd.DataFrame
        Generator data with lat/lon columns
    nuts2_shapes : gpd.GeoDataFrame
        GeoDataFrame with NUTS-2 region geometries
    lat_col : str
        Name of latitude column in generators
    lon_col : str
        Name of longitude column in generators
    nuts_id_col : str
        Column name containing NUTS-2 IDs in shapes
        
    Returns
    -------
    pd.DataFrame
        Original dataframe with added 'nuts2_id' column
    """
    logger.info(f"Assigning {len(generators)} generators to NUTS-2 regions...")
    
    # Create GeoDataFrame from generators
    gen_points = gpd.GeoDataFrame(
        generators,
        geometry=gpd.points_from_xy(generators[lon_col], generators[lat_col]),
        crs='EPSG:4326'
    )
    
    # Ensure shapes are in same CRS
    if nuts2_shapes.crs != 'EPSG:4326':
        nuts2_shapes = nuts2_shapes.to_crs('EPSG:4326')
    
    # Spatial join
    gen_with_nuts2 = gpd.sjoin(
        gen_points, 
        nuts2_shapes[[nuts_id_col, 'geometry']], 
        how='left', 
        predicate='within'
    )
    
    # Handle generators outside any region (use nearest)
    missing_mask = gen_with_nuts2[nuts_id_col].isna()
    if missing_mask.any():
        logger.warning(f"{missing_mask.sum()} generators outside NUTS-2 regions, assigning to nearest")
        missing_points = gen_with_nuts2[missing_mask].geometry
        for idx in gen_with_nuts2[missing_mask].index:
            point = gen_with_nuts2.loc[idx, 'geometry']
            distances = nuts2_shapes.geometry.distance(point)
            nearest_idx = distances.idxmin()
            gen_with_nuts2.loc[idx, nuts_id_col] = nuts2_shapes.loc[nearest_idx, nuts_id_col]
    
    # Add to original dataframe
    result = generators.copy()
    result['nuts2_id'] = gen_with_nuts2[nuts_id_col].values
    
    assigned_count = result['nuts2_id'].notna().sum()
    logger.info(f"Assigned {assigned_count}/{len(generators)} generators to NUTS-2 regions")
    
    return result


def create_wind_profiles(
    weather_data: pd.DataFrame,
    generators: pd.DataFrame,
    date_col: str = 'Date',
    is_offshore: bool = False,
    offshore_factor: float = OFFSHORE_WIND_FACTOR
) -> pd.DataFrame:
    """
    Create wind capacity factor time series for generators.
    
    Parameters
    ----------
    weather_data : pd.DataFrame
        Weather data with columns: Date, ws_<NUTS2_ID> for wind speed
    generators : pd.DataFrame
        Generator data with 'nuts2_id' column (from assign_generators_to_nuts2)
    date_col : str
        Name of date/time column in weather_data
    is_offshore : bool
        If True, apply offshore enhancement factor to wind speeds
    offshore_factor : float
        Multiplier for offshore wind speeds (default 1.25)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with index=timestamps, columns=generator names
        Values are capacity factors (0-1)
    """
    logger.info(f"Creating wind profiles for {len(generators)} generators (offshore={is_offshore})")
    
    # Get timestamps
    timestamps = pd.DatetimeIndex(weather_data[date_col])
    
    # Initialize result
    profiles = pd.DataFrame(index=timestamps)
    
    # Get wind speed columns
    ws_cols = [c for c in weather_data.columns if c.startswith('ws_')]
    available_nuts2 = [c.replace('ws_', '') for c in ws_cols]
    
    # Process each generator
    for idx, gen in generators.iterrows():
        gen_name = gen.get('Name', gen.get('name', f'gen_{idx}'))
        nuts2_id = gen.get('nuts2_id')
        
        if nuts2_id is None or pd.isna(nuts2_id):
            logger.warning(f"Generator {gen_name} has no NUTS-2 ID, skipping")
            continue
        
        # Find matching wind speed column
        ws_col = f'ws_{nuts2_id}'
        if ws_col not in weather_data.columns:
            # Try to find closest NUTS-2 region
            logger.debug(f"No wind data for {nuts2_id}, using zeros")
            profiles[gen_name] = 0.0
            continue
        
        # Get wind speeds
        wind_speed = weather_data[ws_col].values
        
        # Apply offshore factor if needed
        if is_offshore:
            wind_speed = wind_speed * offshore_factor
        
        # Convert to capacity factor
        cf = wind_power_curve(wind_speed)
        profiles[gen_name] = cf
    
    logger.info(f"Created wind profiles: shape {profiles.shape}, mean CF: {profiles.mean().mean():.3f}")
    return profiles


def create_solar_profiles(
    weather_data: pd.DataFrame,
    generators: pd.DataFrame,
    date_col: str = 'Date'
) -> pd.DataFrame:
    """
    Create solar PV capacity factor time series for generators.
    
    Parameters
    ----------
    weather_data : pd.DataFrame
        Weather data with columns: Date, rad_<NUTS2_ID> for solar radiation
    generators : pd.DataFrame
        Generator data with 'nuts2_id' column (from assign_generators_to_nuts2)
    date_col : str
        Name of date/time column in weather_data
        
    Returns
    -------
    pd.DataFrame
        DataFrame with index=timestamps, columns=generator names
        Values are capacity factors (0-1)
    """
    logger.info(f"Creating solar profiles for {len(generators)} generators")
    
    # Get timestamps
    timestamps = pd.DatetimeIndex(weather_data[date_col])
    
    # Initialize result
    profiles = pd.DataFrame(index=timestamps)
    
    # Get solar radiation columns
    rad_cols = [c for c in weather_data.columns if c.startswith('rad_')]
    available_nuts2 = [c.replace('rad_', '') for c in rad_cols]
    
    # Process each generator
    for idx, gen in generators.iterrows():
        gen_name = gen.get('Name', gen.get('name', f'gen_{idx}'))
        nuts2_id = gen.get('nuts2_id')
        
        if nuts2_id is None or pd.isna(nuts2_id):
            logger.warning(f"Generator {gen_name} has no NUTS-2 ID, skipping")
            continue
        
        # Find matching radiation column
        rad_col = f'rad_{nuts2_id}'
        if rad_col not in weather_data.columns:
            logger.debug(f"No solar data for {nuts2_id}, using zeros")
            profiles[gen_name] = 0.0
            continue
        
        # Get irradiance values
        irradiance = weather_data[rad_col].values
        
        # Convert to capacity factor
        cf = solar_capacity_factor(irradiance)
        profiles[gen_name] = cf
    
    logger.info(f"Created solar profiles: shape {profiles.shape}, mean CF: {profiles.mean().mean():.3f}")
    return profiles


def prepare_renewable_generators(
    generators_raw: pd.DataFrame,
    weather_data: pd.DataFrame,
    nuts2_shapes: gpd.GeoDataFrame,
    fueltype_col: str = 'Fueltype',
    technology_col: str = 'Technology',
    capacity_col: str = 'Capacity',
    lat_col: str = 'lat',
    lon_col: str = 'lon',
    date_col: str = 'Date'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Prepare renewable generators with capacity factor profiles.
    
    This is the main entry point for creating renewable generation profiles.
    It filters generators by type (wind/solar), assigns NUTS-2 regions,
    and creates time-varying capacity factors.
    
    Parameters
    ----------
    generators_raw : pd.DataFrame
        Raw generator data (e.g., from powerplants.csv)
    weather_data : pd.DataFrame
        Weather data with wind speed and solar radiation per NUTS-2 region
    nuts2_shapes : gpd.GeoDataFrame
        NUTS-2 region geometries
    fueltype_col : str
        Column name for fuel type
    technology_col : str
        Column name for technology type
    capacity_col : str
        Column name for capacity (MW)
    lat_col : str
        Column name for latitude
    lon_col : str
        Column name for longitude
    date_col : str
        Column name for timestamps in weather data
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (wind_generators, solar_generators, profiles_dict)
        - wind_generators: DataFrame with wind generator data + nuts2_id
        - solar_generators: DataFrame with solar generator data + nuts2_id  
        - profiles_dict: Dict with 'wind' and 'solar' profile DataFrames
        
    Examples
    --------
    >>> wind_gens, solar_gens, profiles = prepare_renewable_generators(
    ...     generators, weather, nuts2_shapes
    ... )
    >>> # profiles['wind'] has shape (timestamps, n_wind_generators)
    >>> # profiles['solar'] has shape (timestamps, n_solar_generators)
    """
    logger.info("Preparing renewable generators with capacity factor profiles")
    
    # Filter for wind and solar
    wind_mask = generators_raw[fueltype_col].str.lower() == 'wind'
    solar_mask = generators_raw[fueltype_col].str.lower() == 'solar'
    
    wind_raw = generators_raw[wind_mask].copy()
    solar_raw = generators_raw[solar_mask].copy()
    
    logger.info(f"Found {len(wind_raw)} wind and {len(solar_raw)} solar generators")
    
    # Filter out generators without coordinates
    wind_raw = wind_raw.dropna(subset=[lat_col, lon_col])
    solar_raw = solar_raw.dropna(subset=[lat_col, lon_col])
    
    logger.info(f"After filtering for coordinates: {len(wind_raw)} wind, {len(solar_raw)} solar")
    
    # Assign NUTS-2 regions
    if len(wind_raw) > 0:
        wind_gens = assign_generators_to_nuts2(wind_raw, nuts2_shapes, lat_col, lon_col)
    else:
        wind_gens = wind_raw
        
    if len(solar_raw) > 0:
        solar_gens = assign_generators_to_nuts2(solar_raw, nuts2_shapes, lat_col, lon_col)
    else:
        solar_gens = solar_raw
    
    # Identify offshore wind
    if technology_col in wind_gens.columns:
        offshore_mask = wind_gens[technology_col].str.lower().str.contains('offshore', na=False)
        onshore_mask = ~offshore_mask
        
        wind_onshore = wind_gens[onshore_mask].copy()
        wind_offshore = wind_gens[offshore_mask].copy()
        
        logger.info(f"Wind breakdown: {len(wind_onshore)} onshore, {len(wind_offshore)} offshore")
    else:
        wind_onshore = wind_gens
        wind_offshore = pd.DataFrame()
    
    # Create profiles
    profiles = {}
    
    # Onshore wind profiles
    if len(wind_onshore) > 0:
        profiles_onshore = create_wind_profiles(
            weather_data, wind_onshore, date_col, is_offshore=False
        )
    else:
        profiles_onshore = pd.DataFrame()
    
    # Offshore wind profiles (with enhancement factor)
    if len(wind_offshore) > 0:
        profiles_offshore = create_wind_profiles(
            weather_data, wind_offshore, date_col, is_offshore=True
        )
    else:
        profiles_offshore = pd.DataFrame()
    
    # Combine wind profiles
    profiles['wind'] = pd.concat([profiles_onshore, profiles_offshore], axis=1)
    profiles['wind_onshore'] = profiles_onshore
    profiles['wind_offshore'] = profiles_offshore
    
    # Solar profiles
    if len(solar_gens) > 0:
        profiles['solar'] = create_solar_profiles(weather_data, solar_gens, date_col)
    else:
        profiles['solar'] = pd.DataFrame()
    
    return wind_gens, solar_gens, profiles


def aggregate_profiles_to_buses(
    generators: pd.DataFrame,
    profiles: pd.DataFrame,
    bus_col: str = 'bus',
    capacity_col: str = 'Capacity',
    name_col: str = 'Name'
) -> pd.DataFrame:
    """
    Aggregate generator profiles to bus level using capacity-weighted average.
    
    When multiple generators are connected to the same bus, their profiles
    are combined using a capacity-weighted average.
    
    Parameters
    ----------
    generators : pd.DataFrame
        Generator data with bus assignments
    profiles : pd.DataFrame
        Time series profiles (columns = generator names)
    bus_col : str
        Column name for bus assignment
    capacity_col : str
        Column name for capacity
    name_col : str
        Column name for generator names
        
    Returns
    -------
    pd.DataFrame
        Aggregated profiles with columns = bus IDs
    """
    if profiles.empty:
        return profiles
    
    logger.info(f"Aggregating {len(profiles.columns)} generator profiles to buses")
    
    # Build mapping from generator name to bus and capacity
    gen_bus_map = generators.set_index(name_col)[bus_col].to_dict()
    gen_cap_map = generators.set_index(name_col)[capacity_col].to_dict()
    
    # Get unique buses
    buses = generators[bus_col].unique()
    
    # Aggregate
    aggregated = pd.DataFrame(index=profiles.index)
    
    for bus in buses:
        # Get generators at this bus
        bus_gens = [g for g in profiles.columns if gen_bus_map.get(g) == bus]
        
        if not bus_gens:
            continue
        
        # Capacity-weighted average
        capacities = np.array([gen_cap_map.get(g, 1.0) for g in bus_gens])
        weights = capacities / capacities.sum()
        
        bus_profile = (profiles[bus_gens].values * weights).sum(axis=1)
        aggregated[bus] = bus_profile
    
    logger.info(f"Aggregated to {len(aggregated.columns)} buses")
    return aggregated


if __name__ == "__main__":
    # Test wind power curve
    import matplotlib.pyplot as plt
    
    wind_speeds = np.linspace(0, 30, 300)
    capacity_factors = wind_power_curve(wind_speeds)
    
    plt.figure(figsize=(10, 6))
    plt.plot(wind_speeds, capacity_factors, 'b-', linewidth=2)
    plt.axvline(WIND_CUT_IN, color='g', linestyle='--', label=f'Cut-in ({WIND_CUT_IN} m/s)')
    plt.axvline(WIND_RATED, color='orange', linestyle='--', label=f'Rated ({WIND_RATED} m/s)')
    plt.axvline(WIND_CUT_OUT, color='r', linestyle='--', label=f'Cut-out ({WIND_CUT_OUT} m/s)')
    plt.xlabel('Wind Speed (m/s)')
    plt.ylabel('Capacity Factor')
    plt.title('Simplified Wind Power Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('wind_power_curve.png', dpi=150)
    plt.show()
    
    # Test solar capacity factor
    irradiances = np.linspace(0, 1200, 200)
    solar_cfs = solar_capacity_factor(irradiances)
    
    plt.figure(figsize=(10, 6))
    plt.plot(irradiances, solar_cfs, 'orange', linewidth=2)
    plt.axvline(SOLAR_STC_IRRADIANCE, color='r', linestyle='--', label=f'STC ({SOLAR_STC_IRRADIANCE} W/m²)')
    plt.xlabel('Irradiance (W/m²)')
    plt.ylabel('Capacity Factor')
    plt.title('Simplified Solar PV Capacity Factor')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('solar_capacity_factor.png', dpi=150)
    plt.show()
