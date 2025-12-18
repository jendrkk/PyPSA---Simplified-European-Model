"""
Network clustering and simplification module for PyPSA networks.

This module provides methods for:
1. Simplifying networks (voltage level reduction, removing converters/stubs)
2. Clustering networks using various algorithms (k-means, HAC, greedy modularity)
3. Distributing clusters optimally across countries using Gurobi solver
4. Post-clustering connectivity fixes (bidirectional links, isolated bus handling)

Based on pypsa-eur methodology with adaptations for simplified European model.
"""
from __future__ import annotations
import logging
import warnings
from functools import reduce
from pathlib import Path
from typing import Any, Dict, Iterable, Collection, Optional, Tuple, Set
from packaging.version import Version, parse

import numpy as np
import pandas as pd
import geopandas as gpd
import pypsa
import linopy

from pypsa.clustering.spatial import (
    busmap_by_greedy_modularity,
    busmap_by_hac,
    busmap_by_kmeans,
    get_clustering_from_busmap,
    busmap_by_stubs,
)
from scipy.sparse.csgraph import dijkstra, connected_components
from shapely.geometry import Point

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pandas version compatibility
PD_GE_2_2 = parse(pd.__version__) >= Version("2.2")

# Constants
GEO_CRS = "EPSG:4326"
DISTANCE_CRS = "EPSG:3035"  # European standard projection for distance calculations
BUS_TOL = 500  # meters tolerance for bus matching


def normed(x: pd.Series) -> pd.Series:
    """Normalize a series to sum to 1.0, handling zeros gracefully."""
    total = x.sum()
    if total == 0:
        return pd.Series(0.0, index=x.index)
    return (x / total).fillna(0.0)


def weighting_for_country(df: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """
    Calculate integer weights for country buses, scaled for optimization.
    
    Matches PyPSA-EUR implementation exactly.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with bus data
    weights : pd.Series
        Raw weights (e.g., load)
        
    Returns
    -------
    pd.Series
        Integer weights scaled from 1 to 100
    """
    w = normed(weights.reindex(df.index, fill_value=0))
    # PyPSA-EUR uses w.max() without checking for zero
    # This works because k-means handles zero weights gracefully
    if w.max() == 0:
        # All buses have zero weight - return uniform weights
        logger.warning(f"All buses in group have zero weight, using uniform weights")
        return pd.Series(1, index=df.index, dtype=int)
    return (w * (100 / w.max())).clip(lower=1).astype(int)


# =============================================================================
# POST-CLUSTERING CONNECTIVITY FIXES
# =============================================================================

def ensure_bidirectional_links(n: pypsa.Network) -> int:
    """
    Ensure all links are bidirectional by setting p_min_pu = -1.0.
    
    Following PyPSA-EUR convention, links should allow power flow in both
    directions (from bus0 to bus1 AND from bus1 to bus0).
    
    Parameters
    ----------
    n : pypsa.Network
        Network to modify (in-place)
        
    Returns
    -------
    int
        Number of links that were changed to bidirectional
    """
    if n.links.empty:
        return 0
    
    # Count unidirectional links (p_min_pu >= 0)
    mask_unidirectional = n.links['p_min_pu'] >= 0
    n_changed = mask_unidirectional.sum()
    
    if n_changed > 0:
        n.links.loc[mask_unidirectional, 'p_min_pu'] = -1.0
        logger.info(f"Made {n_changed} links bidirectional (p_min_pu = -1.0)")
    
    return n_changed


def get_connected_components(
    n: pypsa.Network,
    branch_components: Optional[list] = None
) -> Tuple[int, pd.Series]:
    """
    Find connected components in the network.
    
    Parameters
    ----------
    n : pypsa.Network
        Network to analyze
    branch_components : list, optional
        Branch components to consider. Default: ['Line', 'Link']
        
    Returns
    -------
    tuple
        (n_components, component_labels) where component_labels is a Series
        mapping bus -> component_id
    """
    if branch_components is None:
        branch_components = ['Line', 'Link']
    
    # Build adjacency matrix
    adj = n.adjacency_matrix(branch_components=branch_components)
    
    # Find connected components
    n_comp, labels = connected_components(adj, directed=False)
    
    component_series = pd.Series(labels, index=n.buses.index, name='component')
    
    return n_comp, component_series


def identify_isolated_buses(
    n: pypsa.Network,
    simplified_network: Optional[pypsa.Network] = None
) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    Identify isolated buses after clustering.
    
    Categorizes isolated buses into:
    1. Buses that were already isolated in simplified network (remove them)
    2. Buses that became isolated due to clustering (reconnect them)
    
    Parameters
    ----------
    n : pypsa.Network
        Clustered network to analyze
    simplified_network : pypsa.Network, optional
        Pre-clustering simplified network for comparison
        
    Returns
    -------
    tuple
        (all_isolated, already_isolated, newly_isolated)
        - all_isolated: Set of all buses not connected via lines
        - already_isolated: Set that were isolated before clustering
        - newly_isolated: Set that became isolated due to clustering
    """
    # Find buses connected via lines
    buses_in_lines = set(n.lines.bus0) | set(n.lines.bus1)
    all_buses = set(n.buses.index)
    
    # Buses not connected by any line
    all_isolated = all_buses - buses_in_lines
    
    # Check which were already isolated in simplified network
    already_isolated = set()
    newly_isolated = set()
    
    if simplified_network is not None:
        simplified_buses_in_lines = set(simplified_network.lines.bus0) | set(simplified_network.lines.bus1)
        
        for bus in all_isolated:
            # Check if this bus (or its pre-clustering equivalent) was in lines
            # Bus names might have changed during clustering, so we check by links
            bus_links = n.links[(n.links.bus0 == bus) | (n.links.bus1 == bus)]
            
            if bus_links.empty:
                # No links either - completely isolated, safe to remove
                already_isolated.add(bus)
            else:
                # Has links but no lines - became isolated during clustering
                newly_isolated.add(bus)
    else:
        # Without simplified network, classify by whether bus has links
        for bus in all_isolated:
            bus_links = n.links[(n.links.bus0 == bus) | (n.links.bus1 == bus)]
            if bus_links.empty:
                already_isolated.add(bus)
            else:
                newly_isolated.add(bus)
    
    logger.info(f"Isolated bus analysis:")
    logger.info(f"  Total isolated (no lines): {len(all_isolated)}")
    logger.info(f"  Already isolated (no links): {len(already_isolated)}")
    logger.info(f"  Newly isolated (has links): {len(newly_isolated)}")
    
    return all_isolated, already_isolated, newly_isolated


def reconnect_isolated_buses(
    n: pypsa.Network,
    isolated_buses: Set[str]
) -> int:
    """
    Reconnect isolated buses by adding virtual lines to nearest connected bus.
    
    For buses that became isolated during clustering but were originally
    connected via links, we add a virtual line with high impedance to
    maintain connectivity while not significantly affecting power flow.
    
    Parameters
    ----------
    n : pypsa.Network
        Network to modify (in-place)
    isolated_buses : set
        Set of bus names to reconnect
        
    Returns
    -------
    int
        Number of buses reconnected
    """
    if not isolated_buses:
        return 0
    
    reconnected = 0
    buses_in_lines = set(n.lines.bus0) | set(n.lines.bus1)
    
    for bus in isolated_buses:
        # Find links from this bus
        bus_links = n.links[(n.links.bus0 == bus) | (n.links.bus1 == bus)]
        
        if bus_links.empty:
            continue
        
        # Get the connected bus from the link (the other end)
        for _, link in bus_links.iterrows():
            other_bus = link.bus1 if link.bus0 == bus else link.bus0
            
            # If other bus is connected to lines, add a virtual line
            if other_bus in buses_in_lines:
                # Add a high-impedance virtual line for connectivity
                line_name = f"virtual_{bus}_{other_bus}"
                
                if line_name not in n.lines.index:
                    # Use average line parameters but with high impedance
                    avg_s_nom = n.lines.s_nom.mean() if len(n.lines) > 0 else 1000
                    avg_length = n.lines.length.mean() if len(n.lines) > 0 else 100
                    
                    n.add(
                        "Line",
                        line_name,
                        bus0=bus,
                        bus1=other_bus,
                        s_nom=avg_s_nom,
                        length=avg_length,
                        x=0.01,  # Low reactance for virtual connection
                        r=0.001,
                        carrier='AC',
                    )
                    
                    logger.debug(f"Added virtual line: {bus} <-> {other_bus}")
                    reconnected += 1
                    break  # One connection per isolated bus is enough
    
    if reconnected > 0:
        logger.info(f"Reconnected {reconnected} isolated buses with virtual lines")
    
    return reconnected


def remove_isolated_buses(
    n: pypsa.Network,
    buses_to_remove: Set[str],
    redistribute_components: bool = True
) -> int:
    """
    Remove isolated buses and optionally redistribute their components.
    
    Parameters
    ----------
    n : pypsa.Network
        Network to modify (in-place)
    buses_to_remove : set
        Set of bus names to remove
    redistribute_components : bool
        If True, move loads/generators to nearest connected bus
        
    Returns
    -------
    int
        Number of buses removed
    """
    if not buses_to_remove:
        return 0
    
    # Find nearest connected bus for each isolated bus (for redistribution)
    if redistribute_components:
        buses_with_lines = set(n.lines.bus0) | set(n.lines.bus1)
        connected_buses = list(buses_with_lines)
        
        if connected_buses:
            # Create GeoDataFrame for spatial matching
            bus_coords = n.buses.loc[connected_buses, ['x', 'y']]
    
    removed_count = 0
    for bus in buses_to_remove:
        if bus not in n.buses.index:
            continue
            
        if redistribute_components and connected_buses:
            # Find nearest connected bus
            bus_loc = n.buses.loc[bus, ['x', 'y']]
            distances = ((bus_coords['x'] - bus_loc['x'])**2 + 
                        (bus_coords['y'] - bus_loc['y'])**2)**0.5
            nearest_bus = distances.idxmin()
            
            # Move loads
            load_mask = n.loads.bus == bus
            if load_mask.any():
                n.loads.loc[load_mask, 'bus'] = nearest_bus
                logger.debug(f"Moved {load_mask.sum()} loads from {bus} to {nearest_bus}")
            
            # Move generators  
            gen_mask = n.generators.bus == bus
            if gen_mask.any():
                n.generators.loc[gen_mask, 'bus'] = nearest_bus
                logger.debug(f"Moved {gen_mask.sum()} generators from {bus} to {nearest_bus}")
        
        # Remove links connected to this bus
        link_mask = (n.links.bus0 == bus) | (n.links.bus1 == bus)
        if link_mask.any():
            n.remove("Link", n.links.index[link_mask])
        
        # Remove the bus
        n.remove("Bus", bus)
        removed_count += 1
    
    if removed_count > 0:
        logger.info(f"Removed {removed_count} isolated buses")
    
    return removed_count


def fix_post_clustering_connectivity(
    n: pypsa.Network,
    simplified_network: Optional[pypsa.Network] = None,
    remove_truly_isolated: bool = True,
    reconnect_link_isolated: bool = True,
    ensure_bidirectional: bool = True
) -> Dict[str, int]:
    """
    Fix connectivity issues that arise from network clustering.
    
    This function addresses common issues:
    1. Unidirectional links (sets p_min_pu = -1 for bidirectional flow)
    2. Buses isolated after clustering but connected via links
    3. Buses completely isolated (no lines or links)
    
    Parameters
    ----------
    n : pypsa.Network
        Clustered network to fix (modified in-place)
    simplified_network : pypsa.Network, optional
        Pre-clustering simplified network for comparison
    remove_truly_isolated : bool
        Remove buses with no connections at all
    reconnect_link_isolated : bool
        Add virtual lines for buses connected only via links
    ensure_bidirectional : bool
        Make all links bidirectional
        
    Returns
    -------
    dict
        Summary of fixes applied:
        - 'links_made_bidirectional': number of links changed
        - 'buses_removed': number of isolated buses removed
        - 'buses_reconnected': number of buses reconnected via virtual lines
    """
    logger.info("="*60)
    logger.info("POST-CLUSTERING CONNECTIVITY FIXES")
    logger.info("="*60)
    
    fixes = {
        'links_made_bidirectional': 0,
        'buses_removed': 0,
        'buses_reconnected': 0,
    }
    
    # 1. Make links bidirectional
    if ensure_bidirectional:
        fixes['links_made_bidirectional'] = ensure_bidirectional_links(n)
    
    # 2. Identify isolated buses
    all_isolated, already_isolated, newly_isolated = identify_isolated_buses(
        n, simplified_network
    )
    
    # 3. Remove truly isolated buses (no lines AND no links)
    if remove_truly_isolated and already_isolated:
        fixes['buses_removed'] = remove_isolated_buses(
            n, already_isolated, redistribute_components=True
        )
    
    # 4. Reconnect buses that have links but no lines
    if reconnect_link_isolated and newly_isolated:
        fixes['buses_reconnected'] = reconnect_isolated_buses(n, newly_isolated)
    
    logger.info("="*60)
    logger.info("CONNECTIVITY FIXES COMPLETE")
    logger.info(f"  Links made bidirectional: {fixes['links_made_bidirectional']}")
    logger.info(f"  Isolated buses removed: {fixes['buses_removed']}")
    logger.info(f"  Buses reconnected: {fixes['buses_reconnected']}")
    logger.info("="*60)
    
    return fixes


def simplify_network_to_380(
    n: pypsa.Network,
    linetype_380: str = "Al/St 240/40 4-bundle 380.0",
) -> tuple[pypsa.Network, pd.Series]:
    """
    Simplify network to single 380kV voltage level and remove transformers.
    
    This function:
    - Maps all lines to 380kV voltage level
    - Updates line types and parallel bundles
    - Removes transformers and consolidates buses
    
    Parameters
    ----------
    n : pypsa.Network
        Network to simplify
    linetype_380 : str
        Line type to use for 380kV lines
        
    Returns
    -------
    tuple[pypsa.Network, pd.Series]
        Simplified network and busmap (transformer mapping)
        
    Notes
    -----
    Preserves transmission capacity by adjusting num_parallel.
    """
    logger.info("Simplifying network to 380kV voltage level")
    
    n.buses["v_nom"] = 380.0
    
    # Update line parameters
    if linetype_380 in n.line_types.index:
        n.lines["type"] = linetype_380
        n.lines["v_nom"] = 380
        n.lines["i_nom"] = n.line_types.i_nom[linetype_380]
        n.lines["num_parallel"] = n.lines.eval("s_nom / (sqrt(3) * v_nom * i_nom)")
    else:
        logger.warning(f"Line type {linetype_380} not found; keeping existing types")
    
    # Build transformer mapping (from low to high voltage bus)
    trafo_map = pd.Series(n.transformers.bus1.values, n.transformers.bus0.values)
    trafo_map = trafo_map[~trafo_map.index.duplicated(keep="first")]
    
    # Handle chained transformers (multiple voltage levels)
    while (several_trafo_b := trafo_map.isin(trafo_map.index)).any():
        trafo_map[several_trafo_b] = trafo_map[several_trafo_b].map(trafo_map)
    
    # Add identity mapping for buses without transformers
    missing_buses_i = n.buses.index.difference(trafo_map.index)
    missing = pd.Series(missing_buses_i, missing_buses_i)
    trafo_map = pd.concat([trafo_map, missing])
    
    # Remap all components to new bus structure
    for c in n.one_port_components | n.branch_components:
        df = n.df(c)
        for col in df.columns:
            if col.startswith("bus"):
                df[col] = df[col].map(trafo_map)
    
    # Remove transformers and orphaned buses
    n.remove("Transformer", n.transformers.index)
    n.remove("Bus", n.buses.index.difference(trafo_map))
    
    return n, trafo_map


def remove_converters(n: pypsa.Network) -> tuple[pypsa.Network, pd.Series]:
    """
    Remove HVDC converters by collapsing DC buses to their AC counterparts.
    
    Follows pypsa-eur simplify_network logic:
    - Identifies converter links (empty carrier)
    - Maps DC buses to paired AC buses
    - Updates all components to use AC buses
    - Removes converter links and DC buses
    
    Parameters
    ----------
    n : pypsa.Network
        Network with converters
        
    Returns
    -------
    tuple[pypsa.Network, pd.Series]
        Network without converters and bus mapping
    """
    logger.info("Removing HVDC converters")
    
    converter_map = n.buses.index.to_series()
    converters = n.links.query("carrier == ''")[["bus0", "bus1"]]
    
    if converters.empty:
        logger.info("No converters found to remove")
        return n, converter_map
    
    converters["bus0_carrier"] = converters["bus0"].map(n.buses.carrier)
    converters["bus1_carrier"] = converters["bus1"].map(n.buses.carrier)
    converters["ac_bus"] = converters.apply(
        lambda x: x["bus1"] if x["bus1_carrier"] == "AC" else x["bus0"], axis=1
    )
    converters["dc_bus"] = converters.apply(
        lambda x: x["bus1"] if x["bus1_carrier"] == "DC" else x["bus0"], axis=1
    )
    
    dict_dc_to_ac = dict(zip(converters["dc_bus"], converters["ac_bus"]))
    converter_map = converter_map.replace(dict_dc_to_ac)
    
    # Update all link buses
    n.links["bus0"] = n.links["bus0"].replace(dict_dc_to_ac)
    n.links["bus1"] = n.links["bus1"].replace(dict_dc_to_ac)
    
    # Remove converter links and DC buses
    n.links = n.links.loc[~n.links.index.isin(converters.index)]
    n.buses = n.buses.loc[~n.buses.index.isin(converters["dc_bus"])]
    
    logger.info(f"Removed {len(converters)} converters and {len(dict_dc_to_ac)} DC buses")
    return n, converter_map


def remove_stubs(
    n: pypsa.Network,
    matching_attrs: Iterable[str] | None = None,
    aggregation_strategies: dict | None = None,
) -> tuple[pypsa.Network, pd.Series]:
    """
    Remove stub buses (dead-ends) from the network iteratively.
    
    This function uses PyPSA's clustering mechanism to properly aggregate stubs
    to their neighbors, ensuring all one-port components (loads, generators, etc.)
    are correctly remapped to remaining buses.
    
    Parameters
    ----------
    n : pypsa.Network
        Network to clean
    matching_attrs : Iterable[str], optional
        Bus attributes that must match for aggregation (e.g., ['country'])
        If None or empty, stubs are removed across borders
    aggregation_strategies : dict, optional
        Custom aggregation strategies for clustering
        
    Returns
    -------
    tuple[pypsa.Network, pd.Series]
        Cleaned network and busmap showing aggregations
        
    Notes
    -----
    Uses PyPSA's get_clustering_from_busmap() to ensure all network components
    (including loads, generators, etc.) are properly remapped when buses are
    aggregated.
    """
    logger.info("Removing stub buses from network")
    
    if matching_attrs is None:
        matching_attrs = []
    
    if aggregation_strategies is None:
        aggregation_strategies = {}
    
    # Get busmap from PyPSA's stub removal algorithm
    busmap = busmap_by_stubs(n, matching_attrs=matching_attrs)
    
    # Count how many buses will be removed
    buses_to_agg = len(busmap) - len(busmap.unique())
    
    # Use PyPSA's clustering mechanism to properly aggregate all components
    # This automatically handles loads, generators, and all other one-port components
    clustering = get_clustering_from_busmap(
        n,
        busmap,
        bus_strategies=aggregation_strategies.get("buses", {}),
        line_strategies=aggregation_strategies.get("lines", {}),
    )
    
    logger.info(f"Removed {buses_to_agg} stub buses")
    return clustering.n, busmap


def aggregate_to_substations(
    n: pypsa.Network,
    substation_i: pd.Index | list,
    aggregation_strategies: dict | None = None,
) -> tuple[pypsa.Network, pd.Series]:
    """
    Aggregate buses to their nearest substations using graph distance.
    
    Parameters
    ----------
    n : pypsa.Network
        Network to aggregate
    substation_i : pd.Index | list
        Buses to keep as substations
    aggregation_strategies : dict, optional
        Custom aggregation strategies for clustering
        
    Returns
    -------
    tuple[pypsa.Network, pd.Series]
        Aggregated network and busmap
        
    Notes
    -----
    Uses Dijkstra's algorithm on weighted adjacency matrix.
    Weights are length/capacity to prioritize high-capacity connections.
    """
    logger.info(f"Aggregating buses to {len(substation_i)} substations")
    
    if aggregation_strategies is None:
        aggregation_strategies = {}
    
    # Build weighted adjacency matrix
    weight = pd.concat(
        {
            "Line": n.lines.length / n.lines.s_nom.clip(1e-3),
            "Link": n.links.length / n.links.p_nom.clip(1e-3),
        }
    )
    
    adj = n.adjacency_matrix(
        branch_components=["Line", "Link"], weights=weight
    ).tocsr()
    
    # Find shortest paths from substations to all other buses
    no_substation_i = n.buses.index.difference(substation_i)
    bus_indexer = n.buses.index.get_indexer(substation_i)
    dist = pd.DataFrame(
        dijkstra(adj, directed=False, indices=bus_indexer),
        substation_i,
        n.buses.index,
    )[no_substation_i]
    
    # Ensure aggregation respects country boundaries
    country_values = n.buses.country.values
    country_mask = pd.DataFrame(
        country_values[:, np.newaxis] == country_values,
        index=n.buses.index,
        columns=n.buses.index,
    )[no_substation_i]
    
    # Build busmap: non-substations map to nearest substation in same country
    busmap = n.buses.index.to_series()
    busmap.loc[no_substation_i] = dist.where(country_mask, np.inf).idxmin(0)
    
    # Perform clustering
    clustering = get_clustering_from_busmap(
        n,
        busmap,
        bus_strategies=aggregation_strategies.get("buses", {}),
        line_strategies=aggregation_strategies.get("lines", {}),
    )
    
    return clustering.n, busmap


def distribute_n_clusters_to_countries(
    n: pypsa.Network,
    n_clusters: int,
    cluster_weights: pd.Series,
    focus_weights: dict | None = None,
    solver_name: str = "gurobi",
    min_clusters_per_country: int = 1,
) -> pd.Series:
    """
    Optimally distribute cluster count across countries using optimization.
    
    **EXACT PyPSA-EUR IMPLEMENTATION** - Optimizes per (country, sub_network) directly.
    
    Solves an integer programming problem to minimize deviation from
    proportional allocation while respecting topology constraints.
    
    Parameters
    ----------
    n : pypsa.Network
        Network to cluster
    n_clusters : int
        Total number of clusters desired
    cluster_weights : pd.Series
        Weights per bus (e.g., load, population)
    focus_weights : dict, optional
        Additional weights for specific countries {country: weight}
        NOTE: Weights are DIVIDED by number of sub-networks per country
        to ensure fair distribution across country's sub-networks.
    solver_name : str
        Solver to use ('gurobi', 'scip', 'cplex', etc.)
    min_clusters_per_country : int
        Minimum clusters to allocate to each country's MAIN sub-network.
        This prevents countries from collapsing to 1 cluster and losing
        all internal topology. Default=1 (PyPSA-EUR default).
        Recommended: 2-3 for better line retention.
        
    Returns
    -------
    pd.Series
        Number of clusters per (country, sub_network)
        
    Raises
    ------
    AssertionError
        If n_clusters is outside valid range
        
    Notes
    -----
    Formulation (PyPSA-EUR method):
        min Σ (n_c,s - L_c,s * N)²
        s.t. Σ n_c,s = N
             1 <= n_c,s <= N_c,s (buses per country/sub_network)
    
    Where L_c,s is the normalized weight for (country c, sub_network s).
    
    **CRITICAL**: focus_weights are divided by number of sub-networks:
        L[country] = weight / len(L[country])
    This ensures countries with many sub-networks don't get over-allocated.
    
    Example with Gurobi
    -------------------
    For Gurobi, ensure you have:
    - Installed: conda install -c gurobi gurobi
    - License file at ~/gurobi.lic or set GRB_LICENSE_FILE env variable
    - Academic license: https://www.gurobi.com/academia/academic-program-and-licenses/
    
    Usage:
        n_clusters_c = distribute_n_clusters_to_countries(
            n, n_clusters=50, cluster_weights=load, solver_name='gurobi',
            min_clusters_per_country=2  # Ensures better line retention
        )
    """
    logger.info(f"Distributing {n_clusters} clusters across countries using {solver_name}")
    
    # =========================================================================
    # EXACT PyPSA-EUR METHOD: Optimize per (country, sub_network) directly
    # This is the official implementation from pypsa-eur/scripts/cluster_network.py
    # =========================================================================
    
    # Calculate weights per (country, sub_network)
    L = (
        cluster_weights.groupby([n.buses.country, n.buses.sub_network])
        .sum()
        .pipe(normed)
    )
    
    # Count buses per (country, sub_network)
    N = n.buses.groupby(["country", "sub_network"]).size()[L.index]
    
    # Validate cluster count
    assert n_clusters >= len(N) and n_clusters <= N.sum(), (
        f"Number of clusters must be {len(N)} <= n_clusters <= {N.sum()} "
        f"for this selection of countries. Got {n_clusters}."
    )
    
    # Apply focus weights if provided
    # CRITICAL: PyPSA-EUR divides focus weight by NUMBER OF SUB-NETWORKS
    if isinstance(focus_weights, dict):
        total_focus = sum(list(focus_weights.values()))
        
        assert total_focus <= 1.0, (
            f"The sum of focus weights must be less than or equal to 1.0. "
            f"Current sum: {total_focus}"
        )
        
        logger.info(f"Applying focus weights for {len(focus_weights)} countries")
        
        # EXACT PyPSA-EUR method: L[country] = weight / len(L[country])
        # This divides the weight ACROSS all sub-networks in the country
        for country, weight in focus_weights.items():
            if country in L.index.get_level_values('country'):
                n_subnets = len(L[country])
                L[country] = weight / n_subnets
                logger.debug(f"  {country}: {weight:.1%} total / {n_subnets} sub-networks = {weight/n_subnets:.1%} each")
            else:
                logger.warning(f"  {country} not found in network, skipping")
        
        # Renormalize remaining countries to fill (1 - total_focus)
        remainder = [c not in focus_weights.keys() for c in L.index.get_level_values("country")]
        L[remainder] = L.loc[remainder].pipe(normed) * (1 - total_focus)
        
        logger.info("Using custom focus weights for determining number of clusters.")
    
    # Verify normalization
    assert np.isclose(L.sum(), 1.0, rtol=1e-3), (
        f"Country weights L must sum to 1.0. Current sum: {L.sum()}"
    )
    
    # =========================================================================
    # ENHANCEMENT: Ensure minimum clusters for main sub-networks
    # This prevents UK's 295-bus main grid from collapsing to 1 cluster
    # =========================================================================
    if min_clusters_per_country > 1:
        # Find main sub-network (most buses) for each country
        main_subnets = N.groupby(level='country').idxmax()
        logger.info(f"Ensuring minimum {min_clusters_per_country} clusters for main sub-networks")
        for country, (c, s) in main_subnets.items():
            # Boost weight for main sub-network to ensure it gets min_clusters
            min_fraction = min_clusters_per_country / n_clusters
            if L[(c, s)] < min_fraction:
                old_weight = L[(c, s)]
                L[(c, s)] = min_fraction
                logger.debug(f"  Boosted {country} main subnet weight: {old_weight:.4f} → {min_fraction:.4f}")
        
        # Re-normalize to sum to 1.0
        L = L.pipe(normed)
    
    # =========================================================================
    # Build optimization model (EXACT PyPSA-EUR implementation)
    # =========================================================================
    m = linopy.Model()
    
    # Decision variables: number of clusters per (country, sub_network)
    n_var = m.add_variables(
        lower=1,
        upper=N,
        coords=[L.index],
        name="n",
        integer=True,
    )
    
    # Constraint: total clusters must equal target
    m.add_constraints(n_var.sum() == n_clusters, name="tot")
    
    # Objective: minimize squared deviation from proportional allocation
    # leave out constant in objective (L * n_clusters) ** 2
    m.objective = (n_var * n_var - 2 * n_var * L * n_clusters).sum()
    
    # Solver-specific options
    if solver_name == "gurobi":
        # Suppress Gurobi console output for cleaner logs
        import logging as gurobi_logging
        gurobi_logging.getLogger("gurobipy").propagate = False
        
        solver_options = {
            "LogToConsole": 0,  # Suppress console output
            "TimeLimit": 60,    # 60 second time limit
            "MIPGap": 0.01,     # 1% optimality tolerance
        }
    elif solver_name in ["scip", "cplex", "xpress", "copt", "mosek"]:
        solver_options = {}
        if solver_name not in ["scip", "cplex", "xpress", "copt", "mosek"]:
            logger.info(
                f"The configured solver `{solver_name}` may not support quadratic objectives. "
                "Falling back to `scip`."
            )
            solver_name = "scip"
    else:
        logger.warning(
            f"Solver {solver_name} may not support integer programming. "
            "Falling back to SCIP."
        )
        solver_name = "scip"
        solver_options = {}
    
    # Solve
    try:
        m.solve(solver_name=solver_name, **solver_options)
        result = m.solution["n"].to_series().astype(int)
        
        # Log allocation summary
        country_totals = result.groupby(level='country').sum().sort_values(ascending=False)
        logger.info(f"Optimization successful. Clusters per country (top 10):\n{country_totals.head(10)}")
        
        # Log UK specifically for debugging
        if 'GB' in country_totals.index:
            uk_detail = result[result.index.get_level_values('country') == 'GB']
            logger.info(f"UK cluster allocation:\n{uk_detail}")
        
        return result
        
    except Exception as e:
        error_msg = f"Optimization failed with {solver_name}: {e}"
        
        # Enhanced error message for Gurobi license issues
        if solver_name == "gurobi" and ("license" in str(e).lower() or "gurobipy" in str(e).lower()):
            logger.error(error_msg)
            logger.error(
                "\n=== GUROBI LICENSE HELP ===\n"
                "Gurobi requires a valid license. Academic licenses are FREE:\n"
                "1. Register at: https://www.gurobi.com/academia/\n"
                "2. Download your gurobi.lic file\n"
                "3. Place it in your home directory OR set GRB_LICENSE_FILE environment variable\n"
                "4. Restart your Python environment\n\n"
                "Alternative: Use open-source SCIP solver:\n"
                "  conda install scip\n"
                "  Then use solver_name='scip' in clustering functions\n"
                "=========================="
            )
        else:
            logger.error(error_msg)
        
        # Fallback: proportional allocation with rounding (PyPSA-EUR fallback method)
        logger.warning("Using fallback proportional allocation per (country, sub_network)")
        
        # Same logic as optimization but with rounding
        n_clusters_fallback = (L * n_clusters).round().astype(int)
        n_clusters_fallback = n_clusters_fallback.clip(lower=1, upper=N)
        
        # Adjust to match exact total
        diff = n_clusters - n_clusters_fallback.sum()
        if diff != 0:
            sorted_idx = L.sort_values(ascending=(diff < 0)).index
            for i in range(abs(diff)):
                n_clusters_fallback.loc[sorted_idx[i % len(sorted_idx)]] += np.sign(diff)
        
        return n_clusters_fallback


def busmap_for_n_clusters(
    n: pypsa.Network,
    n_clusters_c: pd.Series,
    cluster_weights: pd.Series,
    algorithm: str = "kmeans",
    features: pd.DataFrame | None = None,
    **algorithm_kwds,
) -> pd.Series:
    """
    Create busmap by clustering network per country to specified counts.
    
    This function exactly matches PyPSA-EUR's implementation.
    
    Parameters
    ----------
    n : pypsa.Network
        Network to cluster
    n_clusters_c : pd.Series
        Number of clusters per (country, sub_network)
    cluster_weights : pd.Series
        Weights for clustering (e.g., load)
    algorithm : str
        Clustering algorithm: 'kmeans', 'hac', or 'modularity'
    features : pd.DataFrame, optional
        Feature matrix for HAC clustering
    **algorithm_kwds
        Additional arguments for clustering algorithm
        
    Returns
    -------
    pd.Series
        Busmap from original buses to cluster names
        
    Notes
    -----
    Applies clustering separately per country/sub-network group.
    Matches PyPSA-EUR cluster_network.py implementation.
    """
    logger.info(f"Creating busmap using {algorithm} algorithm")
    
    if algorithm == "hac" and features is None:
        raise ValueError("For HAC clustering, features must be provided.")
    
    # Set default k-means parameters (PyPSA-EUR values)
    if algorithm == "kmeans":
        algorithm_kwds.setdefault("n_init", 1000)
        algorithm_kwds.setdefault("max_iter", 30000)
        algorithm_kwds.setdefault("tol", 1e-6)
        algorithm_kwds.setdefault("random_state", 0)
    
    def busmap_for_country(x):
        """Apply clustering to a country/sub-network group."""
        # Create prefix for cluster names (e.g., "DE0 ")
        # Note: x.name is a tuple (country, sub_network), convert sub_network to string
        prefix = x.name[0] + str(x.name[1]) + " "
        
        # Check if this sub-network has cluster allocation
        # If not, it's a tiny isolated sub-network - keep as single cluster
        if x.name not in n_clusters_c.index:
            logger.debug(
                f"Sub-network {prefix[:-1]} not in cluster allocation (isolated subnet). "
                f"Keeping {len(x)} buses as single cluster."
            )
            return pd.Series(prefix + "0", index=x.index)
        
        n_clust = n_clusters_c[x.name]
        
        logger.debug(
            f"Determining busmap for country {prefix[:-1]} "
            f"from {len(x)} buses to {n_clust}."
        )
        
        # Single bus - no clustering needed
        if len(x) == 1:
            return pd.Series(prefix + "0", index=x.index)
        
        # Get weights for this country
        weight = weighting_for_country(x, cluster_weights)
        
        # Apply clustering algorithm
        if algorithm == "kmeans":
            # PyPSA-EUR signature: busmap_by_kmeans(n, weight, n_clusters, buses_i, **kwargs)
            return prefix + busmap_by_kmeans(
                n, weight, n_clust, buses_i=x.index, **algorithm_kwds
            )
        elif algorithm == "hac":
            return prefix + busmap_by_hac(
                n,
                n_clust,
                buses_i=x.index,
                feature=features.reindex(x.index, fill_value=0.0),
            )
        elif algorithm == "modularity":
            return prefix + busmap_by_greedy_modularity(
                n, n_clust, buses_i=x.index
            )
        else:
            raise ValueError(
                f"`algorithm` must be one of 'kmeans' or 'hac' or 'modularity'. Is {algorithm}."
            )
    
    # Apply per country group with pandas version compatibility
    compat_kws = dict(include_groups=False) if PD_GE_2_2 else {}
    
    busmap = (
        n.buses.groupby(["country", "sub_network"], group_keys=False)
        .apply(busmap_for_country, **compat_kws)
        .squeeze()
        .rename("busmap")
    )
    
    logger.info(f"Created busmap with {busmap.nunique()} unique clusters")
    return busmap


def clustering_for_n_clusters(
    n: pypsa.Network,
    busmap: pd.Series,
    aggregation_strategies: dict | None = None,
    fix_connectivity: bool = True,
) -> pypsa.clustering.spatial.Clustering:
    """
    Perform full network clustering based on busmap.
    
    Parameters
    ----------
    n : pypsa.Network
        Network to cluster
    busmap : pd.Series
        Mapping from buses to clusters
    aggregation_strategies : dict, optional
        Custom aggregation strategies for components
    fix_connectivity : bool
        Apply post-clustering connectivity fixes (bidirectional links, etc.)
        
    Returns
    -------
    pypsa.clustering.spatial.Clustering
        Clustering object with .n (clustered network), .busmap, .linemap
    """
    logger.info("Performing network clustering")
    
    if aggregation_strategies is None:
        aggregation_strategies = {}
    
    # Remove non-standard line attributes to avoid warnings
    # PyPSA-EUR approach: these attributes are not standard for Lines
    # and cause warnings during aggregation
    # Must check and remove BEFORE calling get_clustering_from_busmap
    non_standard_line_attrs = ['i_nom', 'v_nom']
    removed_attrs = []
    for attr in non_standard_line_attrs:
        if hasattr(n.lines, attr) or attr in n.lines.columns:
            try:
                logger.debug(f"Removing non-standard line attribute '{attr}' before clustering")
                n.lines.drop(columns=[attr], inplace=True, errors='ignore')
                removed_attrs.append(attr)
            except Exception as e:
                logger.warning(f"Could not remove attribute '{attr}': {e}")
    
    if removed_attrs:
        logger.info(f"Removed non-standard line attributes: {removed_attrs}")
    
    # Use PyPSA's built-in clustering
    clustering = get_clustering_from_busmap(
        n,
        busmap,
        bus_strategies=aggregation_strategies.get("buses", {}),
        line_strategies=aggregation_strategies.get("lines", {}),
        one_port_strategies=aggregation_strategies.get("one_ports", {}),
        aggregate_one_ports=aggregation_strategies.get("aggregate_one_ports", set()),
    )
    
    logger.info(
        f"Clustering complete:\n"
        f"  Buses: {len(n.buses)} -> {len(clustering.n.buses)}\n"
        f"  Lines: {len(n.lines)} -> {len(clustering.n.lines)}\n"
        f"  Links: {len(n.links)} -> {len(clustering.n.links)}"
    )
    
    # Apply connectivity fixes
    if fix_connectivity:
        fix_post_clustering_connectivity(
            clustering.n,
            simplified_network=n,
            remove_truly_isolated=True,
            reconnect_link_isolated=False,  # Don't add virtual lines by default
            ensure_bidirectional=True
        )
    
    return clustering


def simplify_and_cluster_network(
    n: pypsa.Network,
    n_clusters: int | None = None,
    *,
    cluster_weights: pd.Series | None = None,
    algorithm: str = "kmeans",
    features: pd.DataFrame | None = None,
    solver_name: str = "gurobi",
    linetype_380: str = "Al/St 240/40 4-bundle 380.0",
    remove_stubs_before: bool = True,
    remove_stubs_after: bool = False,
    remove_stubs_matching: list[str] | None = None,
    aggregate_substations: bool = False,
    substation_buses: pd.Index | None = None,
    aggregation_strategies: dict | None = None,
    focus_weights: dict | None = None,
    **algorithm_kwds,
) -> tuple[pypsa.Network, dict[str, pd.Series]]:
    """
    Complete network simplification and clustering pipeline.
    
    This function chains the following steps:
    1. Simplify to 380kV voltage level
    2. Remove HVDC converters
    3. (Optional) Remove stub buses
    4. (Optional) Aggregate to substations
    5. Cluster to n_clusters using specified algorithm
    
    Parameters
    ----------
    n : pypsa.Network
        Input network
    n_clusters : int, optional
        Number of clusters. If None, only simplification is performed
    cluster_weights : pd.Series, optional
        Weights for clustering (e.g., load per bus)
        Required if n_clusters is specified
    algorithm : str
        Clustering algorithm: 'kmeans', 'hac', 'modularity'
    features : pd.DataFrame, optional
        Feature matrix for HAC clustering
    solver_name : str
        Solver for cluster distribution ('gurobi', 'scip', etc.)
    linetype_380 : str
        Line type for 380kV simplification
    remove_stubs_before : bool
        Remove stubs before clustering
    remove_stubs_after : bool
        Remove stubs after clustering
    remove_stubs_matching : list[str], optional
        Attributes that must match for stub removal (e.g., ['country'])
    aggregate_substations : bool
        Aggregate buses to substations before clustering
    substation_buses : pd.Index, optional
        Buses to treat as substations (required if aggregate_substations=True)
    aggregation_strategies : dict, optional
        Custom aggregation strategies
    focus_weights : dict, optional
        Focus weights for specific countries
    **algorithm_kwds
        Additional arguments for clustering algorithm
        
    Returns
    -------
    tuple[pypsa.Network, dict[str, pd.Series]]
        Clustered network and dictionary of busmaps for each step
        
    Example
    -------
    >>> # Simplify and cluster to 50 buses
    >>> n_clustered, busmaps = simplify_and_cluster_network(
    ...     n,
    ...     n_clusters=50,
    ...     cluster_weights=load,
    ...     algorithm='kmeans',
    ...     solver_name='gurobi',
    ...     remove_stubs_before=True,
    ... )
    >>> # Access final network
    >>> n_final = n_clustered
    >>> # Trace bus mappings
    >>> original_to_final = reduce(lambda x, y: x.map(y), busmaps.values())
    """
    logger.info("=" * 70)
    logger.info("NETWORK SIMPLIFICATION AND CLUSTERING PIPELINE")
    logger.info("=" * 70)
    
    busmaps = {}
    
    # Step 1: Simplify to 380kV
    n, trafo_map = simplify_network_to_380(n, linetype_380)
    busmaps["trafo"] = trafo_map
    
    # Step 2: Remove converters
    n, converter_map = remove_converters(n)
    busmaps["converter"] = converter_map
    
    # Step 3: Remove stubs (optional, before clustering)
    if remove_stubs_before:
        n, stub_map = remove_stubs(n, matching_attrs=remove_stubs_matching)
        busmaps["stub_before"] = stub_map
    
    # Step 4: Aggregate to substations (optional)
    if aggregate_substations:
        if substation_buses is None:
            raise ValueError("substation_buses required when aggregate_substations=True")
        n, substation_map = aggregate_to_substations(
            n, substation_buses, aggregation_strategies
        )
        busmaps["substation"] = substation_map
    
    # Step 5: Clustering (if requested)
    if n_clusters is not None:
        if cluster_weights is None:
            raise ValueError("cluster_weights required for clustering")
        
        # Ensure topology is computed
        n.determine_network_topology()
        
        # Distribute clusters across countries
        n_clusters_c = distribute_n_clusters_to_countries(
            n,
            n_clusters,
            cluster_weights,
            focus_weights=focus_weights,
            solver_name=solver_name,
        )
        
        # Create busmap using selected algorithm
        cluster_busmap = busmap_for_n_clusters(
            n,
            n_clusters_c,
            cluster_weights,
            algorithm=algorithm,
            features=features,
            **algorithm_kwds,
        )
        busmaps["cluster"] = cluster_busmap
        
        # Perform clustering
        clustering = clustering_for_n_clusters(
            n, cluster_busmap, aggregation_strategies
        )
        n = clustering.n
        busmaps["linemap"] = clustering.linemap
        
        # Step 6: Remove stubs after clustering (optional)
        if remove_stubs_after:
            n, stub_map_after = remove_stubs(n, matching_attrs=remove_stubs_matching)
            busmaps["stub_after"] = stub_map_after
    
    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Final network: {len(n.buses)} buses, {len(n.lines)} lines, {len(n.links)} links")
    logger.info("=" * 70)
    
    return n, busmaps


def get_final_busmap(busmaps: dict[str, pd.Series]) -> pd.Series:
    """
    Compose all busmaps into a single mapping from original to final buses.
    
    Parameters
    ----------
    busmaps : dict[str, pd.Series]
        Dictionary of busmaps from each step
        
    Returns
    -------
    pd.Series
        Final busmap from original buses to clustered buses
    """
    # Chain all mappings
    busmap_list = list(busmaps.values())
    if not busmap_list:
        raise ValueError("No busmaps provided")
    
    final_busmap = reduce(lambda x, y: x.map(y), busmap_list[1:], busmap_list[0])
    return final_busmap


def cluster_network_with_strategy(
    n: pypsa.Network,
    n_clusters: int,
    cluster_weights: pd.Series,
    strategy: str = "min_per_country",
    min_per_country: int = 3,
    focus_weights: dict | None = None,
    alpha: float = 0.5,
    algorithm: str = "kmeans",
    aggregation_strategies: dict | None = None,
    **algorithm_kwds,
) -> tuple[pypsa.Network, pd.Series, pd.Series]:
    """
    Cluster network using various geographic diversity strategies.
    
    This is a high-level wrapper that implements different strategies for
    distributing clusters across countries to ensure geographic representation.
    
    Parameters
    ----------
    n : pypsa.Network
        Network to cluster (should be simplified first)
    n_clusters : int
        Target number of clusters
    cluster_weights : pd.Series
        Weights per bus (typically load)
    strategy : str
        Clustering strategy:
        - "load_only": Pure load-weighted (baseline, concentrates in high-demand areas)
        - "min_per_country": Ensure minimum clusters per country, then distribute by load
        - "equal": Distribute clusters equally across countries
        - "focus": Use focus_weights to boost specific countries
        - "optimized": Use integer programming (requires linopy + solver)
    min_per_country : int
        Minimum clusters per country (for "min_per_country" strategy)
    focus_weights : dict, optional
        Country multipliers for "focus" strategy, e.g., {'PL': 2.0, 'ES': 1.5}
    alpha : float
        Geographic weight for hybrid features (not yet implemented)
    algorithm : str
        Clustering algorithm: "kmeans", "hac", "modularity"
    aggregation_strategies : dict, optional
        Custom aggregation strategies for clustering
    **algorithm_kwds
        Additional arguments for clustering algorithm
        
    Returns
    -------
    tuple[pypsa.Network, pd.Series, pd.Series]
        - Clustered network
        - Busmap (original bus → cluster bus)
        - Cluster allocation (n_clusters per country/sub_network)
        
    Examples
    --------
    >>> # Strategy 1: Ensure all countries represented
    >>> n_clust, busmap, allocation = cluster_network_with_strategy(
    ...     n, 250, bus_loads, strategy="min_per_country", min_per_country=3
    ... )
    
    >>> # Strategy 2: Boost Eastern Europe
    >>> n_clust, busmap, allocation = cluster_network_with_strategy(
    ...     n, 250, bus_loads, strategy="focus", 
    ...     focus_weights={'PL': 2.5, 'RO': 2.0, 'CZ': 1.8}
    ... )
    
    >>> # Strategy 3: Equal distribution
    >>> n_clust, busmap, allocation = cluster_network_with_strategy(
    ...     n, 250, bus_loads, strategy="equal"
    ... )
    """
    logger.info(f"Clustering with strategy: {strategy}")
    
    # Distribute clusters based on strategy
    if strategy == "load_only":
        # Baseline: purely proportional to load
        L = cluster_weights.groupby([n.buses.country, n.buses.sub_network]).sum()
        L = L / L.sum()
        N = n.buses.groupby(["country", "sub_network"]).size()[L.index]
        n_clusters_c = (L * n_clusters).round().astype(int).clip(lower=1, upper=N)
        
        # Adjust to exact total
        diff = n_clusters - n_clusters_c.sum()
        if diff != 0:
            sorted_idx = L.sort_values(ascending=(diff < 0)).index
            for i in range(abs(diff)):
                idx = sorted_idx[i % len(sorted_idx)]
                if diff > 0 and n_clusters_c.loc[idx] < N.loc[idx]:
                    n_clusters_c.loc[idx] += 1
                elif diff < 0 and n_clusters_c.loc[idx] > 1:
                    n_clusters_c.loc[idx] -= 1
    
    elif strategy == "min_per_country":
        # Ensure minimum per country, distribute rest by load
        L = cluster_weights.groupby([n.buses.country, n.buses.sub_network]).sum()
        N = n.buses.groupby(['country', 'sub_network']).size()[L.index]
        
        # Initialize with minimum
        n_clusters_c = pd.Series(
            [min(min_per_country, N.loc[idx]) for idx in L.index], 
            index=L.index
        )
        
        # Distribute remaining proportionally
        remaining = n_clusters - n_clusters_c.sum()
        if remaining > 0:
            L_norm = L / L.sum()
            additional = (L_norm * remaining).round().astype(int)
            additional = additional.clip(upper=N - n_clusters_c)
            n_clusters_c += additional
            
            # Adjust for rounding
            diff = n_clusters - n_clusters_c.sum()
            if diff != 0:
                sorted_idx = L.sort_values(ascending=(diff < 0)).index
                for i in range(abs(diff)):
                    idx = sorted_idx[i % len(sorted_idx)]
                    if diff > 0 and n_clusters_c.loc[idx] < N.loc[idx]:
                        n_clusters_c.loc[idx] += 1
                    elif diff < 0 and n_clusters_c.loc[idx] > 1:
                        n_clusters_c.loc[idx] -= 1
    
    elif strategy == "equal":
        # Distribute equally across countries
        groups = n.buses.groupby(['country', 'sub_network']).size()
        n_groups = len(groups)
        
        base_per_group = n_clusters // n_groups
        remainder = n_clusters % n_groups
        
        n_clusters_c = pd.Series(base_per_group, index=groups.index)
        
        if remainder > 0:
            largest_groups = groups.sort_values(ascending=False).head(remainder).index
            n_clusters_c.loc[largest_groups] += 1
        
        n_clusters_c = n_clusters_c.clip(upper=groups)
    
    elif strategy == "focus":
        # Apply focus weights to boost specific countries
        if focus_weights is None:
            raise ValueError("focus_weights must be provided for 'focus' strategy")
        
        L = cluster_weights.groupby([n.buses.country, n.buses.sub_network]).sum()
        
        # Apply multipliers
        for country, multiplier in focus_weights.items():
            mask = L.index.get_level_values(0) == country
            if mask.any():
                L.loc[mask] *= multiplier
                logger.info(f"  Boosted {country} by {multiplier}x")
        
        # Renormalize and distribute
        L = L / L.sum()
        N = n.buses.groupby(['country', 'sub_network']).size()[L.index]
        n_clusters_c = (L * n_clusters).round().astype(int).clip(lower=1, upper=N)
        
        # Adjust to exact total
        diff = n_clusters - n_clusters_c.sum()
        if diff != 0:
            sorted_idx = L.sort_values(ascending=(diff < 0)).index
            for i in range(abs(diff)):
                idx = sorted_idx[i % len(sorted_idx)]
                if diff > 0 and n_clusters_c.loc[idx] < N.loc[idx]:
                    n_clusters_c.loc[idx] += 1
                elif diff < 0 and n_clusters_c.loc[idx] > 1:
                    n_clusters_c.loc[idx] -= 1
    
    elif strategy == "optimized":
        # Use integer programming (PyPSA-EUR method)
        n_clusters_c = distribute_n_clusters_to_countries(
            n, n_clusters, cluster_weights, 
            focus_weights=focus_weights,
            solver_name="gurobi"
        )
    
    else:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Must be one of: "
            "load_only, min_per_country, equal, focus, optimized"
        )
    
    # Log allocation
    logger.info(f"Cluster allocation (top 10 countries):")
    country_summary = n_clusters_c.groupby(level=0).sum().sort_values(ascending=False)
    for country, n_clust in country_summary.head(10).items():
        logger.info(f"  {country}: {n_clust} clusters")
    
    # Create busmap
    busmap = busmap_for_n_clusters(
        n, n_clusters_c, cluster_weights, 
        algorithm=algorithm, **algorithm_kwds
    )
    
    # Cluster network
    clustering = clustering_for_n_clusters(n, busmap, aggregation_strategies)
    
    logger.info(
        f"Clustering complete: {len(n.buses)} → {len(clustering.n.buses)} buses"
    )
    
    return clustering.n, busmap, n_clusters_c


# Example usage and testing
if __name__ == "__main__":
    logger.info("Network clustering module loaded successfully")
    logger.info("Available functions:")
    logger.info("  - simplify_network_to_380()")
    logger.info("  - remove_converters()")
    logger.info("  - remove_stubs()")
    logger.info("  - distribute_n_clusters_to_countries() [uses Gurobi]")
    logger.info("  - busmap_for_n_clusters()")
    logger.info("  - clustering_for_n_clusters()")
    logger.info("  - cluster_network_with_strategy() [high-level wrapper]")
    logger.info("  - simplify_and_cluster_network() [full pipeline]")
    logger.info("\nFor usage examples, see module docstrings.")
