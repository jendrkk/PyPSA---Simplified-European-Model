"""
Consolidated Generator Module for PyPSA Network
================================================

This module provides a single-function interface to add all generators 
(renewable and conventional) to a PyPSA network with renewable capacity 
factor profiles derived from weather data.

Usage:
------
    from scripts.add_generators import add_all_generators
    
    # Add generators to network with one function call
    n = add_all_generators(
        network=n,
        powerplants_path="data/processed/powerplants.csv",
        weather_path="data/processed/weather_processed.csv.gz",
        nuts2_shapes_path="data/processed/nuts2_shapes.gpkg",
        aggregate_renewables=True
    )

Based on PyPSA-EUR methodology for generator modeling.

Author: European Energy Policy Course
Date: 2025
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
import logging
from typing import Optional, Union, List, Dict, Tuple
import sys

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project paths
try:
    from pypsa_simplified.network import (
        prepare_generator_data,
        add_generators_with_profiles,
        add_renewable_generators_aggregated,
        FUEL_TO_CARRIER,
        COUNTRY_NAME_TO_ISO,
        DEFAULT_MARGINAL_COSTS,
        DEFAULT_EFFICIENCIES
    )
except ImportError:
    # Add src to path if not already there
    src_path = Path(__file__).parent.parent / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    from pypsa_simplified.network import (
        prepare_generator_data,
        add_generators_with_profiles,
        add_renewable_generators_aggregated,
        FUEL_TO_CARRIER,
        COUNTRY_NAME_TO_ISO,
        DEFAULT_MARGINAL_COSTS,
        DEFAULT_EFFICIENCIES
    )

try:
    from renewable_profiles import (
        wind_power_curve,
        solar_capacity_factor,
        assign_generators_to_nuts2,
        WIND_CUT_IN,
        WIND_RATED,
        WIND_CUT_OUT,
        OFFSHORE_WIND_FACTOR
    )
except ImportError:
    scripts_path = Path(__file__).parent
    if str(scripts_path) not in sys.path:
        sys.path.insert(0, str(scripts_path))
    from renewable_profiles import (
        wind_power_curve,
        solar_capacity_factor,
        assign_generators_to_nuts2,
        WIND_CUT_IN,
        WIND_RATED,
        WIND_CUT_OUT,
        OFFSHORE_WIND_FACTOR
    )


# =============================================================================
# CONFIGURATION
# =============================================================================

# Default paths relative to project root
DEFAULT_PATHS = {
    'powerplants': 'data/raw/powerplants.csv',
    'weather': 'data/processed/weather_processed.csv.gz',
    'nuts2_shapes': 'data/cache/geometry/all_nuts2.parquet',
}

# Carrier classification
RENEWABLE_CARRIERS = {'wind', 'solar'}
CONVENTIONAL_CARRIERS = {
    'gas', 'hard coal', 'lignite', 'nuclear', 'oil', 
    'biomass', 'biogas', 'hydro', 'waste', 'geothermal', 'other'
}


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def add_all_generators(
    network,
    powerplants_path: Optional[Union[str, Path]] = None,
    weather_path: Optional[Union[str, Path]] = None,
    nuts2_shapes_path: Optional[Union[str, Path]] = None,
    project_root: Optional[Union[str, Path]] = None,
    aggregate_renewables: bool = True,
    include_conventional: bool = True,
    carriers: Optional[List[str]] = None,
    offshore_factor: float = OFFSHORE_WIND_FACTOR,
    verbose: bool = True
) -> 'pypsa.Network':
    """
    Add all generators to a PyPSA network with renewable profiles.
    
    This function:
    1. Loads power plant data
    2. Loads weather data and NUTS-2 shapes
    3. Creates wind/solar capacity factor profiles
    4. Adds renewable generators (aggregated or individual)
    5. Adds conventional generators
    
    Parameters
    ----------
    network : pypsa.Network
        PyPSA network with buses and snapshots already set
    powerplants_path : str or Path, optional
        Path to power plants CSV file
    weather_path : str or Path, optional
        Path to processed weather data (CSV or CSV.GZ)
    nuts2_shapes_path : str or Path, optional
        Path to NUTS-2 shapes GeoPackage
    project_root : str or Path, optional
        Project root directory (for resolving relative paths)
    aggregate_renewables : bool, default True
        If True, aggregate renewable generators to bus level
        If False, add individual generators
    include_conventional : bool, default True
        Whether to add conventional generators
    carriers : list, optional
        List of carriers to include. If None, include all.
    offshore_factor : float, default 1.25
        Enhancement factor for offshore wind
    verbose : bool, default True
        Print progress information
        
    Returns
    -------
    pypsa.Network
        Network with generators added
        
    Example
    -------
    >>> import pypsa
    >>> n = pypsa.Network()
    >>> # ... set up buses, loads, snapshots ...
    >>> n = add_all_generators(n, aggregate_renewables=True)
    >>> print(f"Added {len(n.generators)} generators")
    """
    import pypsa
    
    if verbose:
        logger.info("="*60)
        logger.info("ADDING GENERATORS TO NETWORK")
        logger.info("="*60)
    
    # Resolve paths
    if project_root is None:
        project_root = Path(__file__).parent.parent
    project_root = Path(project_root)
    
    powerplants_path = Path(powerplants_path) if powerplants_path else project_root / DEFAULT_PATHS['powerplants']
    weather_path = Path(weather_path) if weather_path else project_root / DEFAULT_PATHS['weather']
    nuts2_shapes_path = Path(nuts2_shapes_path) if nuts2_shapes_path else project_root / DEFAULT_PATHS['nuts2_shapes']
    
    # Validate inputs
    if len(network.buses) == 0:
        raise ValueError("Network must have buses before adding generators")
    if len(network.snapshots) == 0:
        raise ValueError("Network must have snapshots before adding generators")
    
    # Get countries from network
    network_countries = list(network.buses['country'].unique())
    if verbose:
        logger.info(f"Network countries: {network_countries}")
    
    # =================================================================
    # STEP 1: Load power plant data
    # =================================================================
    if verbose:
        logger.info("\n[1/5] Loading power plant data...")
    
    generators = pd.read_csv(powerplants_path)
    if verbose:
        logger.info(f"  Loaded {len(generators)} power plants ({generators['Capacity'].sum()/1000:.1f} GW)")
    
    # Prepare generators
    gens_prepared = prepare_generator_data(
        generators_raw=generators,
        network_buses=network.buses,
        countries=network_countries
    )
    
    if len(gens_prepared) == 0:
        logger.warning("No generators found for network countries!")
        return network
    
    # Filter by carrier if specified
    if carriers is not None:
        gens_prepared = gens_prepared[gens_prepared['carrier'].isin(carriers)].copy()
        if verbose:
            logger.info(f"  Filtered to {len(gens_prepared)} generators for carriers: {carriers}")
    
    # =================================================================
    # STEP 2: Separate renewable and conventional
    # =================================================================
    if verbose:
        logger.info("\n[2/5] Separating generator types...")
    
    wind_gens = gens_prepared[gens_prepared['carrier'] == 'wind'].copy()
    solar_gens = gens_prepared[gens_prepared['carrier'] == 'solar'].copy()
    
    conventional_carriers = [c for c in gens_prepared['carrier'].unique() 
                           if c not in RENEWABLE_CARRIERS]
    conv_gens = gens_prepared[gens_prepared['carrier'].isin(conventional_carriers)].copy()
    
    if verbose:
        logger.info(f"  Wind:          {len(wind_gens):>6} ({wind_gens['p_nom'].sum()/1000:.1f} GW)")
        logger.info(f"  Solar:         {len(solar_gens):>6} ({solar_gens['p_nom'].sum()/1000:.1f} GW)")
        logger.info(f"  Conventional:  {len(conv_gens):>6} ({conv_gens['p_nom'].sum()/1000:.1f} GW)")
    
    # =================================================================
    # STEP 3: Load weather data and create profiles
    # =================================================================
    if verbose:
        logger.info("\n[3/5] Creating renewable profiles from weather data...")
    
    # Load NUTS-2 shapes
    if nuts2_shapes_path.exists():
        # Use read_parquet for .parquet files (faster), otherwise read_file
        if nuts2_shapes_path.suffix == '.parquet':
            nuts2_shapes = gpd.read_parquet(nuts2_shapes_path)
        else:
            nuts2_shapes = gpd.read_file(nuts2_shapes_path)
        if verbose:
            logger.info(f"  Loaded {len(nuts2_shapes)} NUTS-2 regions")
    else:
        logger.warning(f"NUTS-2 shapes not found at {nuts2_shapes_path}")
        nuts2_shapes = None
    
    # Load weather data
    # Detect timestamp column name (could be 'Date' or 'timestamp')
    weather = pd.read_csv(weather_path, nrows=1)
    timestamp_col = 'Date' if 'Date' in weather.columns else 'timestamp'
    
    weather = pd.read_csv(weather_path, parse_dates=[timestamp_col])
    weather = weather.set_index(timestamp_col)
    if verbose:
        logger.info(f"  Loaded weather data: {len(weather)} timesteps")
    
    # Get wind speed and radiation columns
    ws_cols = [c for c in weather.columns if c.startswith('ws_')]
    rad_cols = [c for c in weather.columns if c.startswith('rad_') or c.startswith('ssrd_')]
    
    if verbose:
        logger.info(f"  Wind speed columns: {len(ws_cols)}")
        logger.info(f"  Radiation columns: {len(rad_cols)}")
    
    # Assign NUTS-2 regions to generators
    wind_profiles = None
    solar_profiles = None
    
    if nuts2_shapes is not None and len(wind_gens) > 0:
        wind_gens = assign_generators_to_nuts2(wind_gens, nuts2_shapes)
    if nuts2_shapes is not None and len(solar_gens) > 0:
        solar_gens = assign_generators_to_nuts2(solar_gens, nuts2_shapes)
    
    # Create wind profiles
    if len(wind_gens) > 0 and len(ws_cols) > 0:
        if verbose:
            logger.info("  Creating wind capacity factor profiles...")
        
        wind_profiles = _create_wind_profiles(
            wind_gens, weather, ws_cols, offshore_factor, verbose
        )
    
    # Create solar profiles
    if len(solar_gens) > 0 and len(rad_cols) > 0:
        if verbose:
            logger.info("  Creating solar capacity factor profiles...")
        
        solar_profiles = _create_solar_profiles(
            solar_gens, weather, rad_cols, verbose
        )
    
    # Filter to network snapshots
    snapshots = network.snapshots
    if wind_profiles is not None:
        common_idx = wind_profiles.index.intersection(snapshots)
        wind_profiles = wind_profiles.loc[common_idx]
    if solar_profiles is not None:
        common_idx = solar_profiles.index.intersection(snapshots)
        solar_profiles = solar_profiles.loc[common_idx]
    
    # =================================================================
    # STEP 4: Add renewable generators
    # =================================================================
    if verbose:
        logger.info("\n[4/5] Adding renewable generators...")
    
    if aggregate_renewables:
        # Aggregate to bus level
        if wind_profiles is not None:
            add_renewable_generators_aggregated(
                network, wind_gens, wind_profiles, 'wind'
            )
        if solar_profiles is not None:
            add_renewable_generators_aggregated(
                network, solar_gens, solar_profiles, 'solar'
            )
    else:
        # Add individual generators
        if wind_profiles is not None:
            add_generators_with_profiles(network, wind_gens, wind_profiles)
        if solar_profiles is not None:
            add_generators_with_profiles(network, solar_gens, solar_profiles)
    
    renewable_count = len(network.generators[network.generators['carrier'].isin(RENEWABLE_CARRIERS)])
    if verbose:
        logger.info(f"  Renewable generators added: {renewable_count}")
    
    # =================================================================
    # STEP 5: Add conventional generators
    # =================================================================
    if include_conventional and len(conv_gens) > 0:
        if verbose:
            logger.info("\n[5/5] Adding conventional generators...")
        
        add_generators_with_profiles(network, conv_gens, profiles=None)
        
        conv_count = len(network.generators) - renewable_count
        if verbose:
            logger.info(f"  Conventional generators added: {conv_count}")
    
    # =================================================================
    # SUMMARY
    # =================================================================
    if verbose:
        logger.info("\n" + "="*60)
        logger.info("GENERATOR ADDITION COMPLETE")
        logger.info("="*60)
        
        gen_summary = network.generators.groupby('carrier').agg({
            'p_nom': ['count', 'sum']
        }).round(2)
        gen_summary.columns = ['count', 'capacity_gw']
        gen_summary['capacity_gw'] /= 1000
        
        logger.info(f"\nTotal generators: {len(network.generators)}")
        logger.info(f"Total capacity: {network.generators['p_nom'].sum()/1000:.1f} GW")
        logger.info(f"\nBy carrier:\n{gen_summary}")
    
    return network


def _create_wind_profiles(
    wind_gens: pd.DataFrame,
    weather: pd.DataFrame,
    ws_cols: List[str],
    offshore_factor: float,
    verbose: bool = True
) -> pd.DataFrame:
    """Create wind capacity factor profiles for all generators."""
    
    profiles = pd.DataFrame(index=weather.index)
    
    # Check if generators have NUTS-2 assignment
    if 'nuts2' not in wind_gens.columns:
        logger.warning("Wind generators missing NUTS-2 assignment")
        return None
    
    # Get unique NUTS-2 regions
    nuts2_regions = wind_gens['nuts2'].dropna().unique()
    
    # Create capacity factors for each NUTS-2 region
    for nuts2_id in nuts2_regions:
        ws_col = f'ws_{nuts2_id}'
        if ws_col not in weather.columns:
            continue
        
        # Get wind speed and apply power curve
        wind_speed = weather[ws_col].values
        cf = wind_power_curve(wind_speed)
        
        # Get generators in this NUTS-2 region
        region_gens = wind_gens[wind_gens['nuts2'] == nuts2_id]
        
        for idx, gen in region_gens.iterrows():
            gen_idx = gen.name if hasattr(gen, 'name') else idx
            
            # Apply offshore factor if applicable
            gen_cf = cf.copy()
            if 'Offshore' in str(gen.get('Technology', '')) or \
               'offshore' in str(gen.get('Technology', '')).lower():
                gen_cf = np.minimum(gen_cf * offshore_factor, 1.0)
            
            profiles[gen_idx] = gen_cf
    
    if verbose:
        logger.info(f"  Created wind profiles: {profiles.shape}")
    
    return profiles


def _create_solar_profiles(
    solar_gens: pd.DataFrame,
    weather: pd.DataFrame,
    rad_cols: List[str],
    verbose: bool = True
) -> pd.DataFrame:
    """Create solar capacity factor profiles for all generators."""
    
    profiles = pd.DataFrame(index=weather.index)
    
    # Check if generators have NUTS-2 assignment
    if 'nuts2' not in solar_gens.columns:
        logger.warning("Solar generators missing NUTS-2 assignment")
        return None
    
    # Get unique NUTS-2 regions
    nuts2_regions = solar_gens['nuts2'].dropna().unique()
    
    # Create capacity factors for each NUTS-2 region
    for nuts2_id in nuts2_regions:
        # Try different column naming conventions
        rad_col = None
        for prefix in ['rad_', 'ssrd_']:
            candidate = f'{prefix}{nuts2_id}'
            if candidate in weather.columns:
                rad_col = candidate
                break
        
        if rad_col is None:
            continue
        
        # Get irradiance and apply solar capacity factor model
        irradiance = weather[rad_col].values
        cf = solar_capacity_factor(irradiance)
        
        # Assign to all generators in this NUTS-2 region
        region_gens = solar_gens[solar_gens['nuts2'] == nuts2_id]
        
        for idx, gen in region_gens.iterrows():
            gen_idx = gen.name if hasattr(gen, 'name') else idx
            profiles[gen_idx] = cf
    
    if verbose:
        logger.info(f"  Created solar profiles: {profiles.shape}")
    
    return profiles


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_generator_summary(network) -> pd.DataFrame:
    """Get a summary of generators in the network by carrier."""
    
    summary = network.generators.groupby('carrier').agg({
        'p_nom': ['count', 'sum'],
        'marginal_cost': 'mean',
        'efficiency': 'mean'
    }).round(2)
    
    summary.columns = ['count', 'capacity_gw', 'avg_marginal_cost', 'avg_efficiency']
    summary['capacity_gw'] /= 1000
    
    return summary.sort_values('capacity_gw', ascending=False)


def validate_generators(network) -> Dict:
    """Validate generator configuration in the network."""
    
    issues = []
    
    # Check for generators
    if len(network.generators) == 0:
        issues.append("No generators in network")
    
    # Check capacity
    total_capacity = network.generators['p_nom'].sum() / 1000  # GW
    if total_capacity < 100:
        issues.append(f"Low total capacity: {total_capacity:.1f} GW")
    
    # Check for renewable profiles
    if len(network.generators_t.p_max_pu.columns) == 0:
        issues.append("No time-varying profiles for generators")
    
    # Check renewables have profiles
    renewable_gens = network.generators[
        network.generators['carrier'].isin(RENEWABLE_CARRIERS)
    ].index
    profile_gens = network.generators_t.p_max_pu.columns
    missing_profiles = set(renewable_gens) - set(profile_gens)
    
    if len(missing_profiles) > 0:
        issues.append(f"{len(missing_profiles)} renewable generators missing profiles")
    
    # Check capacity margin
    if len(network.loads_t.p_set.columns) > 0:
        peak_demand = network.loads_t.p_set.sum(axis=1).max() / 1000  # GW
        margin = (total_capacity / peak_demand - 1) * 100
        if margin < 20:
            issues.append(f"Low capacity margin: {margin:.1f}%")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'n_generators': len(network.generators),
        'total_capacity_gw': total_capacity,
        'n_profiles': len(network.generators_t.p_max_pu.columns)
    }


# =============================================================================
# CLI INTERFACE
# =============================================================================

if __name__ == "__main__":
    """Command-line interface for testing."""
    
    import argparse
    import pypsa
    
    parser = argparse.ArgumentParser(description="Add generators to a PyPSA network")
    parser.add_argument('network_path', help="Path to network NetCDF file")
    parser.add_argument('--output', '-o', help="Output path for modified network")
    parser.add_argument('--aggregate', action='store_true', default=True,
                       help="Aggregate renewable generators to bus level")
    parser.add_argument('--verbose', '-v', action='store_true', default=True,
                       help="Print progress information")
    
    args = parser.parse_args()
    
    # Load network
    print(f"Loading network from {args.network_path}...")
    n = pypsa.Network(args.network_path)
    
    # Add generators
    n = add_all_generators(
        n,
        aggregate_renewables=args.aggregate,
        verbose=args.verbose
    )
    
    # Save if output specified
    if args.output:
        print(f"Saving network to {args.output}...")
        n.export_to_netcdf(args.output)
    
    # Validate
    validation = validate_generators(n)
    print(f"\nValidation: {'PASS' if validation['valid'] else 'FAIL'}")
    if not validation['valid']:
        for issue in validation['issues']:
            print(f"  - {issue}")
    
    print("\nDone!")
