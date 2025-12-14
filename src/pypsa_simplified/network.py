"""
Network construction and optimization helpers using PyPSA.
"""
from __future__ import annotations
from typing import Any, Dict
from shapely.wkt import loads
import pandas as pd
import geopandas as gpd
import numpy as np

import pypsa
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from pypsa_simplified import data_prep as dp

G_CARRIERS = {
    "hard coal": {
        "co2_emissions": 0.35,  # tonnes CO2/MWh thermal; ÷ efficiency for electric
        "nice_name": "Hard Coal",
        "color": "#8B7355",
    },
    "lignite": {
        "co2_emissions": 0.41,  # tonnes CO2/MWh thermal; ÷ efficiency for electric
        "nice_name": "Lignite",
        "color": "#A0522D",
    },
    "gas": {
        "co2_emissions": 0.20,  # tonnes CO2/MWh thermal
        "nice_name": "Natural Gas",
        "color": "#FF6B6B",
    },
    "oil": {
        "co2_emissions": 0.27,  # tonnes CO2/MWh thermal
        "nice_name": "Oil",
        "color": "#D2691E",
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
    "geothermal": {
        "co2_emissions": 0.0,
        "nice_name": "Geothermal",
        "color": "#FF4500",
    },
    "biogas": {
        "co2_emissions": 0.0,  # Often considered carbon-neutral in lifecycle assessments
        "nice_name": "Biogas",
        "color": "#32CD32",
    },
    "biomass": {
        "co2_emissions": 0.0,  # Often considered carbon-neutral in lifecycle assessments
        "nice_name": "Solid Biomass",
        "color": "#228B22",
    },
    "waste": {
        "co2_emissions": 0.0,
        "nice_name": "Waste to Energy",
        "color": "#708090",
    },
    "other": {
        "co2_emissions": 0.0,
        "nice_name": "Other",
        "color": "#A9A9A9",
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

def build_network(n: pypsa.Network, data_dict: dp.RawData, options: Dict[str, Any] | None = None) -> pypsa.Network:
    """
    Build a `pypsa.Network` from a lightweight source dictionary.

    This function must be adapted to mirror the logic in `main.ipynb`.
    For now, we load CSVs if paths are provided in `data_dict`.
    """
    net = pypsa.Network()
    data_dict = data_dict.data
    
    # Example: if data_dict contains paths for core components
    buses = data_dict.get("buses")
    lines = data_dict.get("lines")
    links = data_dict.get("links")
    generators = data_dict.get("generators")
    loads = data_dict.get("loads")
    transformers = data_dict.get("transformers", pd.DataFrame())
    converters = data_dict.get("converters", pd.DataFrame())
    
    if options is not None and options['snapshots'] is not None:
        net.set_snapshots(options['snapshots'])
    
    if options is not None and options['name'] is not None:
        net.name = options['name']
    
    if options is not None and options['generation_carriers'] is not None:
        generation_carriers = options['generation_carriers']
    else:
        generation_carriers = G_CARRIERS
    
    if options is not None and options['transmission_carriers'] is not None:
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
    
    if options is not None and options['countries'] is not None:
        countries = options['countries']
    else:
        countries = set(buses['country'])
    
    buses = buses[buses['country'].isin(countries)].copy()
    
    print(f"Adding {len(buses)} buses...")
    
    for idx, row in buses.iterrows():
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
    
    bus_ids_subset = set(buses['bus_id'].values)

    lines = lines[
        (lines['bus0'].isin(bus_ids_subset)) & 
        (lines['bus1'].isin(bus_ids_subset))
    ].copy()
    
    print(f"Adding {len(lines)} lines...")
    
    # Averages
    ave_r_val = np.mean(lines['r'].dropna()) if not lines['r'].dropna().empty else 0.01
    ave_x_val = np.mean(lines['x'].dropna()) if not lines['x'].dropna().empty else 0.1
    ave_s_nom = np.mean(lines['s_nom'].dropna()) if not lines['s_nom'].dropna().empty else 1000
    ave_length = np.mean(lines['length'].dropna()) if not lines['length'].dropna().empty else 100
    ave_b_val = np.mean(lines['b'].dropna()) if not lines['b'].dropna().empty else 100
    ave_circuits = np.mean(lines['circuits'].dropna()) if not lines['circuits'].dropna().empty else 1
    
    for idx, row in lines.iterrows():
        try:
            # Extract real line parameters from CSV, with fallbacks
            r_val = float(row.get('r', ave_r_val)) if pd.notna(row.get('r')) and float(row.get('r', 0)) > 0 else ave_r_val
            x_val = float(row.get('x', ave_x_val)) if pd.notna(row.get('x')) and float(row.get('x', 0)) > 0 else ave_x_val
            s_nom_val = float(row.get('s_nom', ave_s_nom)) if pd.notna(row.get('s_nom')) else ave_s_nom
            length_val = float(row.get('length', ave_length)) if pd.notna(row.get('length')) else ave_length
            b_val = float(row.get('b', ave_b_val)) if pd.notna(row.get('b')) else ave_b_val
            circuits = int(row.get('circuits', ave_circuits)) if pd.notna(row.get('circuits')) else ave_circuits

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
                bus0=row['bus0'],
                bus1=row['bus1'],
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
    
    transformers = transformers[
        (transformers['bus0'].isin(bus_ids_subset)) &
        (transformers['bus1'].isin(bus_ids_subset))
    ].copy()

    print(f"Adding {len(transformers)} transformers...")
    
    # Averages for missing data
    ave_s_nom = np.mean(transformers['s_nom'].dropna()) if not transformers['s_nom'].dropna().empty else 50000
    
    for idx, row in transformers.iterrows():
        try:
            # Extract transformer parameters
            s_nom_val = float(row.get('s_nom', ave_s_nom)) if pd.notna(row.get('s_nom')) else ave_s_nom
            
            type = ''
            if s_nom_val <= 0 or pd.isna(s_nom_val):
                type = 'Unknown'
            
            # Voltage levels for base impedance calculation (if needed)
            v_bus0 = float(row.get('voltage_bus0', 220))
            v_bus1 = float(row.get('voltage_bus1', 380))
            
            def get_trafo_impedance(v_in, v_out, s_nom):
                """
                Calculates per-unit series reactance (x) and resistance (r) for 
                European EHV transformers based on voltage level and nominal power.
                
                Parameters:
                -----------
                v_in, v_out : float
                    Voltages on primary and secondary side in kV.
                s_nom : float
                    Nominal power in MVA.
                    
                Returns:
                --------
                (x_pu, r_pu) : tuple of floats
                """
                # 1. Determine Voltage Class (use the highest voltage)
                v_max = max(v_in, v_out)
                
                # 2. Assign Standard Short Circuit Voltage (uk) -> x_pu
                # Based on typical ENTSO-E / IEC standard values for European Grid
                if v_max >= 380:      # 380/400 kV level
                    x_pu = 0.15       # Typical range: 14-16%
                    base_x_r = 60.0   # Very high efficiency
                elif v_max >= 220:    # 220 kV level
                    x_pu = 0.12       # Typical range: 12-14%
                    base_x_r = 50.0   # High efficiency
                else:                 # Fallback for unexpected lower voltages
                    x_pu = 0.10
                    base_x_r = 40.0

                # 3. Adjust X/R ratio based on size (S_nom)
                # Larger transformers are more efficient (higher X/R).
                # We apply a logarithmic scaling to account for your large range [500, 25000].
                # We saturate the scaling to avoid unrealistic values for aggregate transformers.
                
                # Scale factor: typically X/R increases with size. 
                # For a 5000 MVA aggregate, we assume it behaves like multiple large units in parallel,
                # so we keep the X/R of a large single unit (approx 60-80).
                
                if s_nom > 1000:
                    # For huge aggregates, we cap the efficiency at the limit of physical large units
                    x_r_ratio = min(base_x_r * 1.5, 90.0) 
                else:
                    # For smaller specific units, scale down slightly
                    x_r_ratio = base_x_r
                    
                # 4. Calculate Resistance
                # r is derived from x and the X/R ratio
                r_pu = x_pu / x_r_ratio
                
                # Safety clamp for numerical stability in solvers (avoid exactly 0)
                r_pu = max(r_pu, 1e-5)
                
                return x_pu, r_pu
            
            x_val, r_val = get_trafo_impedance(v_bus0, v_bus1, s_nom_val)
            
            n.add(
                "Transformer",
                name=row['transformer_id'],
                bus0=row['bus0'],
                bus1=row['bus1'],
                type=type,
                s_nom=s_nom_val,
                x=x_val,
                r=r_val,
                model="t",  # T-model for transformers
            )
        except Exception as e:
            if idx < 3:
                print(f"Error adding transformer {row.get('transformer_id', 'unknown')}: {e}")
    
    converters = converters.copy()
    
    print(f"Adding {len(converters)} converters...")
    
    # Average p_nom if missing
    ave_p_nom = np.mean(converters['p_nom'].dropna()) if not converters['p_nom'].dropna().empty else 500
    
    for idx, row in converters.iterrows():
        try:
            converter_id = row.get('converter_id', f"conv_{idx}")
            bus_ac = row.get('bus0')  # AC side
            bus_dc_ref = row.get('bus1')  # DC side bus reference
            
            # Only add converter if AC bus exists
            if bus_ac not in n.buses.index and bus_dc_ref not in n.buses.index:
                continue
            
            # Create DC bus if it doesn't exist
            dc_bus_id = f"{bus_dc_ref}-DC"
            if dc_bus_id not in n.buses.index:
                # Extract coordinates (roughly at same location as AC bus)
                try:
                    ac_bus = n.buses.loc[bus_ac]
                    x_coord = ac_bus['x']
                    y_coord = ac_bus['y']
                except:
                    x_coord, y_coord = 0, 0
                
                n.add(
                    "Bus",
                    name=dc_bus_id,
                    x=x_coord,
                    y=y_coord,
                    carrier="DC",
                    v_nom=float(row.get('voltage', 300)),  # DC voltage nominal
                )
            
            # Add converter as a Link
            p_nom_val = float(row.get('p_nom', ave_p_nom)) if pd.notna(row.get('p_nom')) else ave_p_nom
            
            n.add(
                "Link",
                name=converter_id,
                bus0=bus_ac,
                bus1=dc_bus_id,
                p_nom=p_nom_val,
                efficiency=0.99,  # HVDC converter loss
                carrier="",  # PyPSA-EUR convention: empty string for converters (bidirectional)
                under_construction=False if row.get('under_construction') == 'f' else True,
            )
        except Exception as e:
            if idx < 3:
                print(f"Error processing converter {row.get('converter_id', 'unknown')}: {e}")

    links = links[
        (links['bus0'].isin(bus_ids_subset)) & 
        (links['bus1'].isin(bus_ids_subset))
    ].copy()

    print(f"Adding {len(links)} links...")

    # Averages for missing data
    ave_p_nom = np.mean(links['p_nom'].dropna()) if not links['p_nom'].dropna().empty else 500
    for idx, row in links.iterrows():
        try:
            p_nom_val = float(row.get('p_nom', ave_p_nom)) if pd.notna(row.get('p_nom')) else ave_p_nom
            
            n.add(
                "Link",
                name=row['link_id'],
                bus0=row['bus0'],
                bus1=row['bus1'],
                p_nom=p_nom_val,
                efficiency=1.0,  # No loss for simplified AC links (can adjust for realism)
                carrier="AC",  # AC transmission
            )
        except Exception as e:
            if idx < 3:
                print(f"Error adding AC link {row.get('link_id', 'unknown')}: {e}")

    # Additional options (CRS, snapshots, carriers) can be applied here
    return net

def add_generators(n: pypsa.Network, RawData: dp.RawData) -> pypsa.Network:
    """Add generators from the provided DataFrame into the PyPSA network."""
    generators_df = RawData.get('generators')
        
    RawData.bus_coords()
        
    for idx, row in generators_df.iterrows():
        
        
        
        try:
            n.add(
                "Generator",
                name=row['Name'],
                bus=row['bus'],
                p_nom=float(row['p_nom']),
                carrier=row['carrier'],
                efficiency=float(row.get('efficiency', 0.4)),
                marginal_cost=float(row.get('marginal_cost', 50.0)),
            )
        except Exception as e:
            if idx < 3:
                print(f"Error adding generator {row.get('generator_id', 'unknown')}: {e}")
    return n


def build_network_from_serialized_source(n: pypsa.Network, source_path: str, options: Dict[str, Any] | None = None) -> pypsa.Network:
    """
    Convenience helper: load a gzipped JSON source file and build a network.
    """
    
    data = dp.RawData(None)
    data = data.load(source_path)
    
    return build_network(n, data, options=options)


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


