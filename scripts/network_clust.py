"""
Network clustering and simplification module for PyPSA networks.

This module provides methods for:
1. Simplifying networks (voltage level reduction, removing converters/stubs)
2. Clustering networks using various algorithms (k-means, HAC, greedy modularity)
3. Distributing clusters optimally across countries using Gurobi solver

Based on pypsa-eur methodology with adaptations for simplified European model.
"""
from __future__ import annotations
import logging
import warnings
from functools import reduce
from pathlib import Path
from typing import Any, Dict, Iterable, Collection

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
from scipy.sparse.csgraph import dijkstra
from shapely.geometry import Point

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    return (w * (100 / w.max())).clip(lower=1).astype(int)


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
) -> tuple[pypsa.Network, pd.Series]:
    """
    Remove stub buses (dead-ends) from the network iteratively.
    
    A stub is a bus with only one connection. These are removed sequentially
    until no more stubs exist.
    
    Parameters
    ----------
    n : pypsa.Network
        Network to clean
    matching_attrs : Iterable[str], optional
        Bus attributes that must match for aggregation (e.g., ['country'])
        If None or empty, stubs are removed across borders
        
    Returns
    -------
    tuple[pypsa.Network, pd.Series]
        Cleaned network and busmap showing aggregations
    """
    logger.info("Removing stub buses from network")
    
    if matching_attrs is None:
        matching_attrs = []
    
    busmap = busmap_by_stubs(n, matching_attrs=matching_attrs)
    
    # Remove clustered buses and branches
    buses_to_del = n.buses.index.difference(busmap)
    n.remove("Bus", buses_to_del)
    
    for c in n.branch_components:
        df = n.df(c)
        bus_cols = [col for col in df.columns if col.startswith("bus")]
        mask = df[bus_cols].isin(busmap).all(axis=1)
        to_remove = df[~mask].index
        if len(to_remove) > 0:
            n.remove(c, to_remove)
    
    logger.info(f"Removed {len(buses_to_del)} stub buses")
    return n, busmap


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
) -> pd.Series:
    """
    Optimally distribute cluster count across countries using Gurobi.
    
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
    solver_name : str
        Solver to use ('gurobi', 'scip', 'cplex', etc.)
        
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
    Formulation:
        min Σ (n_c - L_c * N)²
        s.t. Σ n_c = N
             1 <= n_c <= N_c (buses per country)
    
    Where L_c is the normalized weight for country c.
    
    Example with Gurobi
    -------------------
    For Gurobi, ensure you have:
    - Installed: conda install -c gurobi gurobi
    - License file at ~/gurobi.lic or set GRB_LICENSE_FILE env variable
    - Academic license: https://www.gurobi.com/academia/academic-program-and-licenses/
    
    Usage:
        n_clusters_c = distribute_n_clusters_to_countries(
            n, n_clusters=50, cluster_weights=load, solver_name='gurobi'
        )
    """
    logger.info(f"Distributing {n_clusters} clusters across countries using {solver_name}")
    
    # Calculate normalized weights per country and sub-network
    L = (
        cluster_weights.groupby([n.buses.country, n.buses.sub_network])
        .sum()
        .pipe(normed)
    )
    
    # Count available buses per group
    N = n.buses.groupby(["country", "sub_network"]).size()[L.index]
    
    # Validate cluster count
    assert n_clusters >= len(N) and n_clusters <= N.sum(), (
        f"Number of clusters must be {len(N)} <= n_clusters <= {N.sum()} "
        f"for this selection of countries."
    )
    
    # Apply focus weights if provided
    if isinstance(focus_weights, dict):
        logger.info(f"Applying focus weights: {focus_weights}")
        for country, weight in focus_weights.items():
            mask = L.index.get_level_values(0) == country
            if mask.any():
                L.loc[mask] *= weight
        L = L / L.sum()  # Renormalize
    
    # Verify normalization
    assert np.isclose(L.sum(), 1.0, rtol=1e-3), (
        f"Country weights L must sum to 1.0. Current sum: {L.sum()}"
    )
    
    # Build optimization model
    m = linopy.Model()
    
    # Decision variables: number of clusters per country/sub-network
    clusters = m.add_variables(
        lower=1,
        upper=N,
        coords=[L.index],
        name="n_clusters",
        integer=True,
    )
    
    # Constraint: total clusters must equal target
    m.add_constraints(clusters.sum() == n_clusters, name="total_clusters")
    
    # Objective: minimize squared deviation from proportional allocation
    # min Σ (n_c - L_c * N)²
    # Expand: Σ (n_c² - 2*n_c*L_c*N + (L_c*N)²)
    # Drop constant term (L_c*N)² for optimization
    m.objective = (clusters * clusters - 2 * clusters * L * n_clusters).sum()
    
    # Solver-specific options
    if solver_name == "gurobi":
        solver_options = {
            "LogToConsole": 0,  # Suppress console output
            "TimeLimit": 60,    # 60 second time limit
        }
    elif solver_name in ["scip", "cplex", "xpress", "copt", "mosek"]:
        solver_options = {}
    else:
        logger.warning(
            f"Solver {solver_name} may not support integer programming. "
            "Falling back with basic options."
        )
        solver_options = {}
    
    # Solve
    try:
        m.solve(solver_name=solver_name, **solver_options)
        result = m.solution["n_clusters"].to_series().astype(int)
        logger.info(f"Optimization successful. Cluster distribution:\n{result}")
        return result
    except Exception as e:
        logger.error(f"Optimization failed with {solver_name}: {e}")
        # Fallback: proportional allocation with rounding
        logger.warning("Using fallback proportional allocation")
        result = (L * n_clusters).round().astype(int)
        result = result.clip(lower=1, upper=N)
        # Adjust to match exact total
        diff = n_clusters - result.sum()
        if diff != 0:
            # Add/remove from largest groups
            sorted_idx = L.sort_values(ascending=(diff < 0)).index
            for i in range(abs(diff)):
                result.loc[sorted_idx[i % len(sorted_idx)]] += np.sign(diff)
        return result


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
    """
    logger.info(f"Creating busmap using {algorithm} algorithm")
    
    if algorithm == "hac" and features is None:
        raise ValueError("HAC algorithm requires features DataFrame")
    
    # Choose clustering function
    if algorithm == "kmeans":
        cluster_func = busmap_by_kmeans
        cluster_args = {"bus_weightings": cluster_weights, **algorithm_kwds}
    elif algorithm == "hac":
        cluster_func = busmap_by_hac
        cluster_args = {"feature": features, **algorithm_kwds}
    elif algorithm == "modularity":
        cluster_func = busmap_by_greedy_modularity
        cluster_args = algorithm_kwds
    else:
        raise ValueError(
            f"Unknown algorithm: {algorithm}. "
            "Choose from 'kmeans', 'hac', or 'modularity'"
        )
    
    def busmap_for_country(buses_df):
        """Apply clustering to a country/sub-network group."""
        country = buses_df.name[0] if isinstance(buses_df.name, tuple) else buses_df.name
        sub_net = buses_df.name[1] if isinstance(buses_df.name, tuple) else 0
        n_clusters = n_clusters_c.loc[(country, sub_net)]
        
        if n_clusters == len(buses_df):
            # No clustering needed
            return buses_df.index.to_series()
        
        # Build weighted clustering arguments
        if algorithm == "kmeans":
            weights_country = weighting_for_country(buses_df, cluster_weights)
            return cluster_func(
                n,
                bus_weightings=weights_country,
                n_clusters=n_clusters,
                buses_i=buses_df.index,
                **{k: v for k, v in cluster_args.items() if k != "bus_weightings"},
            )
        elif algorithm == "hac":
            features_country = features.reindex(buses_df.index, fill_value=0)
            return cluster_func(
                n,
                n_clusters=n_clusters,
                buses_i=buses_df.index,
                feature=features_country,
                **{k: v for k, v in cluster_args.items() if k != "feature"},
            )
        else:  # modularity
            return cluster_func(
                n,
                n_clusters=n_clusters,
                buses_i=buses_df.index,
                **cluster_args,
            )
    
    # Apply per country group
    busmap = (
        n.buses.groupby(["country", "sub_network"], group_keys=False)
        .apply(busmap_for_country, include_groups=False)
        .squeeze()
        .rename("busmap")
    )
    
    logger.info(f"Created busmap with {busmap.nunique()} unique clusters")
    return busmap


def clustering_for_n_clusters(
    n: pypsa.Network,
    busmap: pd.Series,
    aggregation_strategies: dict | None = None,
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
        
    Returns
    -------
    pypsa.clustering.spatial.Clustering
        Clustering object with .n (clustered network), .busmap, .linemap
    """
    logger.info("Performing network clustering")
    
    if aggregation_strategies is None:
        aggregation_strategies = {}
    
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


# Example usage and testing
if __name__ == "__main__":
    logger.info("Network clustering module loaded successfully")
    logger.info("Available functions:")
    logger.info("  - simplify_network_to_380()")
    logger.info("  - remove_converters()")
    logger.info("  - remove_stubs()")
    logger.info("  - distribute_n_clusters_to_countries() [uses Gurobi]")
    logger.info("  - busmap_for_n_clusters()")
    logger.info("  - simplify_and_cluster_network() [main pipeline]")
    logger.info("\nFor usage examples, see module docstrings.")
