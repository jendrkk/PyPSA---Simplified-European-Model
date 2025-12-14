"""
Network construction and optimization helpers using PyPSA.
"""
from __future__ import annotations
from typing import Any, Dict
from shapely.wkt import loads
import pandas as pd
import geopandas as gpd

import pypsa
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import data_prep as dp

def build_network_from_source(data_dict: dp.OSMData, options: Dict[str, Any] | None = None) -> pypsa.Network:
    """
    Build a `pypsa.Network` from a lightweight source dictionary.

    This function must be adapted to mirror the logic in `main.ipynb`.
    For now, we load CSVs if paths are provided in `data_dict`.
    """
    net = pypsa.Network()
    
    # Example: if data_dict contains paths for core components
    buses_csv = data_dict.get("buses_csv")
    lines_csv = data_dict.get("lines_csv")
    generators_csv = data_dict.get("generators_csv")
    loads_csv = data_dict.get("loads_csv")
    
    if options is not None:
        countries = options['countries']
    else:
        countries = set(buses_csv['country'])
    
    buses_csv = buses_csv[buses_csv['country'].isin(countries)]
    
    if buses_csv:
        net.import_components_from_csv(buses_csv, components=["Bus"])
    if lines_csv:
        net.import_components_from_csv(lines_csv, components=["Line"])
    if generators_csv:
        net.import_components_from_csv(generators_csv, components=["Generator"])
    if loads_csv:
        net.import_components_from_csv(loads_csv, components=["Load"])

    # Additional options (CRS, snapshots, carriers) can be applied here
    return net


def build_network_from_serialized_source(source_path: str, options: Dict[str, Any] | None = None) -> pypsa.Network:
    """
    Convenience helper: load a gzipped JSON source file and build a network.
    """
    data = dp.OSMdata(None)
    data = data.load(source_path)
    
    return build_network_from_source(data, options=options)


def optimize_network(network_obj: pypsa.Network, solver_opts: Dict[str, Any] | None = None) -> pypsa.Network:
    """
    Optimize the network using PyPSA built-in optimization.

    This uses `network_obj.optimize()` (PyPSA >= 0.26) or `lopf()` for older versions.
    """
    # Prefer `optimize` when available
    if hasattr(network_obj, "optimize"):
        network_obj.optimize(solver_options=solver_opts or {})
    else:
        # Fallback to LOPF
        network_obj.lopf(network_obj.snapshots, solver_options=solver_opts or {})
    return network_obj


def network_summary(network_obj: pypsa.Network) -> Dict[str, Any]:
    """Return a small summary dictionary for quick inspection."""
    return {
        "n_buses": len(network_obj.buses),
        "n_lines": len(network_obj.lines),
        "n_generators": len(network_obj.generators),
        "n_loads": len(network_obj.loads),
        "snapshots": list(map(str, getattr(network_obj, "snapshots", []))),
    }

def add_converters_transformers_to_network(n: pypsa.Network, converters_df: pd.DataFrame, transformers_df: pd.DataFrame, crs="EPSG:4326") -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Add converters (HVDC terminals) and transformers from the provided CSVs into the PyPSA network.
    converters_df should have bus0, bus1, voltage (v_nom), pnorm (MW) and geometry; transformers_df should have bus0, bus1, voltage_bus0, voltage_bus1, s_nom/pnorm and geometry."""
    # Converters become Links with empty carrier, matching PyPSA-EUR convention before simplification
    for idx, row in converters_df.iterrows():
        p_nom = float(row.get("pnorm", 1000))
        n.add(
            "Link",
            name=f"converter_{idx}",
            bus0=row["bus0"],
            bus1=row["bus1"],
            p_nom=p_nom,
            efficiency=1.0,
            carrier="",  # marks HVDC converters in PyPSA-EUR
            length=row.get("length", 0.0),
        )

    # Transformers use voltage info per side; fall back to default impedance if not provided
    for idx, row in transformers_df.iterrows():
        s_nom = float(row.get("s_nom", row.get("pnorm", 1000)))
        n.add(
            "Transformer",
            name=f"transformer_{idx}",
            bus0=row["bus0"],
            bus1=row["bus1"],
            s_nom=s_nom,
            x=row.get("x", 0.1),
            r=row.get("r", 0.01),
            tap_ratio=row.get("tap_ratio", 1.0),
            phase_shift=row.get("phase_shift", 0.0),
        )

    # Optionally keep geometries as GeoDataFrames for mapping/QA
    converters_geo = converters_df.copy()
    transformers_geo = transformers_df.copy()
    if "geometry" in converters_geo:
        converters_geo["geometry"] = converters_geo["geometry"].apply(lambda g: loads(g) if isinstance(g, str) else g)
        converters_geo = gpd.GeoDataFrame(converters_geo, geometry="geometry", crs=crs)
    if "geometry" in transformers_geo:
        transformers_geo["geometry"] = transformers_geo["geometry"].apply(lambda g: loads(g) if isinstance(g, str) else g)
        transformers_geo = gpd.GeoDataFrame(transformers_geo, geometry="geometry", crs=crs)
    return converters_geo, transformers_geo

# Example usage (commented to avoid accidental execution):
# converters_geo, transformers_geo = add_converters_transformers_to_network(n, converters, transformers)
# n, converter_map = remove_converters(n)

def remove_converters(n: pypsa.Network) -> tuple[pypsa.Network, Dict[str, str]]:
    """Collapse HVDC converters by mapping DC buses to their paired AC buses (PyPSA-EUR simplify_network logic)."""
    converter_map = n.buses.index.to_series()
    converters = n.links.query("carrier == ''")[ ["bus0", "bus1"] ]
    converters["bus0_carrier"] = converters["bus0"].map(n.buses.carrier)
    converters["bus1_carrier"] = converters["bus1"].map(n.buses.carrier)
    converters["ac_bus"] = converters.apply(lambda x: x["bus1"] if x["bus1_carrier"] == "AC" else x["bus0"], axis=1)
    converters["dc_bus"] = converters.apply(lambda x: x["bus1"] if x["bus1_carrier"] == "DC" else x["bus0"], axis=1)

    dict_dc_to_ac = dict(zip(converters["dc_bus"], converters["ac_bus"]))
    converter_map = converter_map.replace(dict_dc_to_ac)

    n.links["bus0"] = n.links["bus0"].replace(dict_dc_to_ac)
    n.links["bus1"] = n.links["bus1"].replace(dict_dc_to_ac)

    n.links = n.links.loc[~n.links.index.isin(converters.index)]
    n.buses = n.buses.loc[~n.buses.index.isin(converters["dc_bus"])]
    return n, converter_map


