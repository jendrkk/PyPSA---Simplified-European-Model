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

from pypsa_simplified import data_prep as dp

G_CARRIERS = {
    "coal": {
        "co2_emissions": 0.35,  # tonnes CO2/MWh thermal; ÷ efficiency for electric
        "nice_name": "Coal",
        "color": "#8B7355",
    },
    "gas": {
        "co2_emissions": 0.20,  # tonnes CO2/MWh thermal
        "nice_name": "Natural Gas",
        "color": "#FF6B6B",
    },
    "nuclear": {
        "co2_emissions": 0.0,  # Nuclear has negligible lifecycle emissions
        "nice_name": "Nuclear",
        "color": "#FFD700",
    },
    "wind": {
        "co2_emissions": 0.0,  # Renewables have zero operational emissions
        "nice_name": "Wind",
        "color": "#87CEEB",
    },
    "solar": {
        "co2_emissions": 0.0,  # Renewables have zero operational emissions
        "nice_name": "Solar",
        "color": "#FFA500",
    },
    "hydro": {
        "co2_emissions": 0.0,
        "nice_name": "Hydro",
        "color": "#4169E1",
    },
    "biomass": {
        "co2_emissions": 0.0,  # Often considered carbon-neutral in lifecycle assessments
        "nice_name": "Biomass",
        "color": "#228B22",
    },
}

T_CARRIERS = {
    "AC": {
        "co2_emissions": 0.0,
        "nice_name": "AC Transmission",
        "color": "#333333",
    },
    "DC": {
        "co2_emissions": 0.0,
        "nice_name": "DC Transmission (HVDC)",
        "color": "#666666",
    },
}

def build_network_from_source(n: pypsa.Network,data_dict: dp.OSMData, options: Dict[str, Any] | None = None) -> pypsa.Network:
    """
    Build a `pypsa.Network` from a lightweight source dictionary.

    This function must be adapted to mirror the logic in `main.ipynb`.
    For now, we load CSVs if paths are provided in `data_dict`.
    """
    net = pypsa.Network()
    data_dict = data_dict.data
    
    # Example: if data_dict contains paths for core components
    buses_csv = data_dict.get("buses")
    lines_csv = data_dict.get("lines")
    links_csv = data_dict.get("links")
    generators_csv = data_dict.get("generators")
    loads_csv = data_dict.get("loads")
    
    
    if options['generation_carriers'] is not None:
        generation_carriers = options['generation_carriers']
    else:
        generation_carriers = G_CARRIERS
    
    if options['transmission_carriers'] is not None:
        transmission_carriers = options['transmission_carriers']
    else:
        transmission_carriers = T_CARRIERS
        
    all_carriers = {**generation_carriers, **transmission_carriers}
    
    for carrier_name, carrier_attrs in all_carriers.items():
        if carrier_name not in n.carriers.index:
            n.add(
                "Carrier",
                carrier_name,
                co2_emissions=carrier_attrs.get("co2_emissions", 0.0),
                nice_name=carrier_attrs.get("nice_name", carrier_name),
                color=carrier_attrs.get("color", "#CCCCCC"),
            )
    
    if options['countries'] is not None:
        countries = options['countries']
    else:
        countries = set(buses_csv['country'])
    
    buses_csv = buses_csv[buses_csv['country'].isin(countries)]
    
    for idx, row in buses_csv.iterrows():
        voltage_kv = float(row['voltage'])  # Extract actual voltage from OSM data
        
        n.add(
            "Bus",
            name=row['bus_id'],
            x=float(row['x']),
            y=float(row['y']),
            country=row['country'],
            v_nom=voltage_kv,  # Use actual voltage level from OSM
            carrier="AC" if row['dc'] == 'f' else "DC",  # All buses are AC by default; DC buses introduced via converters only
            under_construction=False if row['under_construction'] == 'f' else True,
        )
    
    bus_ids_subset = set(buses_csv['bus_id'].values)

    lines_subset = lines_csv[
        (lines_csv['bus0_id'].isin(bus_ids_subset)) & 
        (lines_csv['bus1_id'].isin(bus_ids_subset))
    ].copy()

    print(f"Adding {len(lines_subset)} AC transmission lines")

    for idx, row in lines_subset.iterrows():
        try:
            # Extract real line parameters from CSV, with fallbacks
            r_val = float(row.get('r', 0.01)) if pd.notna(row.get('r')) and float(row.get('r', 0)) > 0 else 0.01
            x_val = float(row.get('x', 0.1)) if pd.notna(row.get('x')) and float(row.get('x', 0)) > 0 else 0.1
            s_nom_val = float(row.get('s_nom', 1000)) if pd.notna(row.get('s_nom')) else 1000
            length_val = float(row.get('length', 100)) if pd.notna(row.get('length')) else 100
            b_val = float(row.get('b', 100)) if pd.notna(row.get('b')) else 100
            circuits = int(row.get('circuits', 1)) if pd.notna(row.get('circuits')) else 1

            # If all above are default values, then:
            type = ""
            if (
                r_val == 0.01
                and x_val == 0.1
                and s_nom_val == 1000
                and length_val == 100
                and b_val == 100
                and circuits == 1
                and pd.notna(row.get("type"))
                and str(row.get("type")).strip() != ""
            ):
                type = row['type']

            n.add(
                "Line",
                type=type,
                name=row['line_id'],
                bus0=row['bus0_id'],
                bus1=row['bus1_id'],
                x=x_val,
                r=r_val,
                b=b_val,
                s_nom=s_nom_val,
                length=length_val,
                num_parallel=circuits,
                carrier='AC'            # Lines implicitly operate at AC; carrier not a parameter but important for context
            )
        except Exception as e:
            if idx < 3:  # Print first few errors only
                print(f"Error adding line {row['line_id']}: {e}")
    
    
    
    if buses_csv:
        net.import_components_from_csv(buses_csv, components=["Bus"])
    if lines_csv:
        net.import_components_from_csv(lines_csv, components=["Line"])
    if generators_csv:
        net.import_components_from_csv(generators_csv, components=["Generator"])
    if loads_csv:
        net.import_components_from_csv(loads_csv, components=["Load"])
    if links_csv:
        net.import_components_from_csv(links_csv, components=["Link"])

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


