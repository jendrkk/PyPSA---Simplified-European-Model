"""
Network construction and optimization helpers using PyPSA.
"""
from __future__ import annotations
from typing import Any, Dict
from shapely.wkt import loads
import pandas as pd
import geopandas as gpd
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial

import pypsa
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from pypsa_simplified import data_prep as dp


def _safe_cpu_workers() -> int:
    """Choose a reasonable worker count without overcommitting cores."""
    try:
        # Keep one core free for the main process where possible.
        return max(1, cpu_count() - 1)
    except NotImplementedError:
        return 1


def _parallel_map(records: list[dict], func) -> list[dict]:
    """Thin wrapper over Pool.map to keep invocation concise."""
    if not records:
        return []
    with Pool(processes=_safe_cpu_workers()) as pool:
        return pool.map(func, records)


def _prepare_bus_add_kwargs(row: dict) -> dict:
    try:
        return {
            "ok": True,
            "name": row["bus_id"],
            "kwargs": {
                "name": row["bus_id"],
                "x": float(row["x"]),
                "y": float(row["y"]),
                "country": row["country"],
                "v_nom": float(row["voltage"]),
                "carrier": "AC" if row["dc"] == "f" else "DC",
                "under_construction": False if row["under_construction"] == "f" else True,
            },
        }
    except Exception as error:  # Preserve original behavior: log a few errors only in caller
        return {"ok": False, "name": row.get("bus_id"), "error": str(error)}


def _prepare_line_add_kwargs(row: dict, ave_r_val: float, ave_x_val: float, ave_s_nom: float, ave_length: float, ave_b_val: float, ave_circuits: float) -> dict:
    try:
        r_val = float(row.get("r", ave_r_val)) if pd.notna(row.get("r")) and float(row.get("r", 0)) > 0 else ave_r_val
        x_val = float(row.get("x", ave_x_val)) if pd.notna(row.get("x")) and float(row.get("x", 0)) > 0 else ave_x_val
        s_nom_val = float(row.get("s_nom", ave_s_nom)) if pd.notna(row.get("s_nom")) else ave_s_nom
        length_val = float(row.get("length", ave_length)) if pd.notna(row.get("length")) else ave_length
        b_val = float(row.get("b", ave_b_val)) if pd.notna(row.get("b")) else ave_b_val
        circuits = int(row.get("circuits", ave_circuits)) if pd.notna(row.get("circuits")) else ave_circuits

        type_val = ""
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
            type_val = row["type"]

        return {
            "ok": True,
            "name": row["line_id"],
            "kwargs": {
                "type": type_val,
                "name": row["line_id"],
                "bus0": row["bus0"],
                "bus1": row["bus1"],
                "x": x_val,
                "r": r_val,
                "b": b_val,
                "s_nom": s_nom_val,
                "length": length_val,
                "num_parallel": circuits,
                "carrier": "AC",
            },
        }
    except Exception as error:
        return {"ok": False, "name": row.get("line_id"), "error": str(error)}


def _compute_trafo_impedance(v_in: float, v_out: float, s_nom: float) -> tuple[float, float]:
    """Calculate per-unit series reactance (x) and resistance (r)."""
    v_max = max(v_in, v_out)
    if v_max >= 380:
        x_pu = 0.15
        base_x_r = 60.0
    elif v_max >= 220:
        x_pu = 0.12
        base_x_r = 50.0
    else:
        x_pu = 0.10
        base_x_r = 40.0

    if s_nom > 1000:
        x_r_ratio = min(base_x_r * 1.5, 90.0)
    else:
        x_r_ratio = base_x_r

    r_pu = x_pu / x_r_ratio
    r_pu = max(r_pu, 1e-5)
    return x_pu, r_pu


def _prepare_transformer_add_kwargs(row: dict, ave_s_nom: float) -> dict:
    try:
        s_nom_val = float(row.get("s_nom", ave_s_nom)) if pd.notna(row.get("s_nom")) else ave_s_nom
        type_val = ""
        if s_nom_val <= 0 or pd.isna(s_nom_val):
            type_val = "Unknown"

        v_bus0 = float(row.get("voltage_bus0", 220))
        v_bus1 = float(row.get("voltage_bus1", 380))
        x_val, r_val = _compute_trafo_impedance(v_bus0, v_bus1, s_nom_val)

        return {
            "ok": True,
            "name": row["transformer_id"],
            "kwargs": {
                "name": row["transformer_id"],
                "bus0": row["bus0"],
                "bus1": row["bus1"],
                "type": type_val,
                "s_nom": s_nom_val,
                "x": x_val,
                "r": r_val,
                "model": "t",
            },
        }
    except Exception as error:
        return {"ok": False, "name": row.get("transformer_id"), "error": str(error)}


def _prepare_link_add_kwargs(row: dict, ave_p_nom: float) -> dict:
    try:
        p_nom_val = float(row.get("p_nom", ave_p_nom)) if pd.notna(row.get("p_nom")) else ave_p_nom
        return {
            "ok": True,
            "name": row["link_id"],
            "kwargs": {
                "name": row["link_id"],
                "bus0": row["bus0"],
                "bus1": row["bus1"],
                "p_nom": p_nom_val,
                "efficiency": 1.0,
                "carrier": "AC",
            },
        }
    except Exception as error:
        return {"ok": False, "name": row.get("link_id"), "error": str(error)}


def _prepare_generator_add_kwargs(row: dict) -> dict:
    try:
        return {
            "ok": True,
            "name": row["Name"],
            "kwargs": {
                "name": row["Name"],
                "bus": row["bus"],
                "p_nom": float(row["p_nom"]),
                "carrier": row["carrier"],
                "efficiency": float(row.get("efficiency", 0.4)),
                "marginal_cost": float(row.get("marginal_cost", 50.0)),
            },
        }
    except Exception as error:
        return {"ok": False, "name": row.get("Name"), "error": str(error)}

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


def _prepare_load_series(
    record: tuple[str, pd.Series],
    snapshots: pd.DatetimeIndex,
    clip_non_negative: bool = False,
) -> dict:
    """Prepare a single load time series for insertion.

    Parameters
    record: (bus_id, series)
        Tuple with the bus identifier and its demand series.
    snapshots: pd.DatetimeIndex
        Target snapshots to align the series to.
    clip_non_negative: bool
        If True, negative values are clipped to zero.

    Returns
    dict: Mapping with fields: ok, bus, name, p_set (pd.Series) or error.

    Notes
    - Timezone-aware indices are converted to timezone-naive.
    - Series are reindexed to `snapshots`, filling missing with 0.
    """
    try:
        bus_id, ser = record
        if not isinstance(ser.index, pd.DatetimeIndex):
            raise TypeError(
                f"Demand series for bus {bus_id} must have DatetimeIndex."
            )

        if ser.index.tz is not None:
            ser.index = ser.index.tz_localize(None)

        ser = ser.sort_index()
        ser = ser.reindex(snapshots, fill_value=0.0)
        if clip_non_negative:
            ser = ser.clip(lower=0.0)

        return {
            "ok": True,
            "bus": str(bus_id),
            "name": f"load_{bus_id}",
            "p_set": ser.astype(float),
        }
    except Exception as exc:
        return {"ok": False, "bus": str(record[0]), "error": str(exc)}


def add_loads_from_series(
    n: pypsa.Network,
    demand_by_bus: Dict[str, pd.Series],
    *,
    extend_snapshots: bool = False,
    carrier: str = "AC",
    clip_non_negative: bool = False,
) -> pypsa.Network:
    """Add per-bus demand time series as Loads into the network.

    Parameters
    n: pypsa.Network
        Target network to modify.
    demand_by_bus: Dict[str, pd.Series]
        Mapping from bus_id to demand series (MW). The series must have a
        DatetimeIndex. If `n.snapshots` is not set, the union of all series
        indices is used. Otherwise, series are aligned to `n.snapshots`.
    extend_snapshots: bool (default: False)
        If True and `n.snapshots` is already set, extend it to the union of
        existing snapshots and all series indices; otherwise, align series to
        the current `n.snapshots`.
    carrier: str (default: "AC")
        Carrier to assign to all added loads.
    clip_non_negative: bool (default: True)
        If True, negative demand values are clipped to zero.

    Returns
    pypsa.Network: The modified network with new loads added.

    Approach
    - Determine target snapshots (existing, or union of series indices).
    - Preprocess each series in parallel (timezone drop, align, clip).
    - Add one Load per bus (name: "load_{bus_id}") and assign p_set.

    Edge cases handled
    - Missing buses are skipped with a warning.
    - Empty input returns immediately.
    - Differing indices across series are unified via (optional) union.
    """
    if not demand_by_bus:
        print("No demand series provided; skipping load creation.")
        return n

    existing = getattr(n, "snapshots", pd.DatetimeIndex([]))
    if existing is None or len(existing) == 0:
        # Use union of all series indices
        all_idx = []
        for ser in demand_by_bus.values():
            idx = ser.index
            if getattr(idx, "tz", None) is not None:
                idx = idx.tz_localize(None)
            all_idx.append(pd.DatetimeIndex(idx))
        target_snapshots = pd.DatetimeIndex(sorted(pd.Index.union_many(all_idx)))
        n.set_snapshots(target_snapshots)
    else:
        if extend_snapshots:
            all_idx = [pd.DatetimeIndex(existing)]
            for ser in demand_by_bus.values():
                idx = ser.index
                if getattr(idx, "tz", None) is not None:
                    idx = idx.tz_localize(None)
                all_idx.append(pd.DatetimeIndex(idx))
            target_snapshots = pd.DatetimeIndex(
                sorted(pd.Index.union_many(all_idx))
            )
            if not target_snapshots.equals(existing):
                n.set_snapshots(target_snapshots)
        else:
            target_snapshots = pd.DatetimeIndex(existing)

    print(f"Adding demand for {len(demand_by_bus)} buses...")

    records = list(demand_by_bus.items())
    prepare = partial(
        _prepare_load_series,
        snapshots=target_snapshots,
        clip_non_negative=clip_non_negative,
    )
    results = _parallel_map(records, prepare)

    missing_bus = 0
    errors = 0
    added = 0
    existing_buses = set(n.buses.index.astype(str))

    for res in results:
        if not res.get("ok", False):
            errors += 1
            if errors <= 5:
                print(
                    f"Load prep failed for bus {res.get('bus')}: {res.get('error')}"
                )
            continue

        bus = res["bus"]
        name = res["name"]
        if bus not in existing_buses:
            missing_bus += 1
            if missing_bus <= 5:
                print(f"Bus {bus} not in network; skipping load {name}.")
            continue

        try:
            n.add("Load", name=name, bus=bus, carrier=carrier)
            n.loads_t.p_set[name] = res["p_set"]
            added += 1
        except Exception as exc:
            errors += 1
            if errors <= 5:
                print(f"Failed adding load {name} on {bus}: {exc}")

    if missing_bus:
        print(f"Skipped {missing_bus} loads due to missing buses.")
    if errors:
        print(f"Encountered {errors} errors while preparing/adding loads.")
    print(f"Added {added} loads with time series.")

    return n


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
    
    bus_results = _parallel_map(buses.to_dict("records"), _prepare_bus_add_kwargs)
    bus_error_count = 0
    for bus_result in bus_results:
        if bus_result.get("ok"):
            n.add("Bus", **bus_result["kwargs"])
        else:
            if bus_error_count < 3:
                print(f"Error adding bus {bus_result.get('name')}: {bus_result.get('error')}")
            bus_error_count += 1
    
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
    
    line_prepare = partial(
        _prepare_line_add_kwargs,
        ave_r_val=ave_r_val,
        ave_x_val=ave_x_val,
        ave_s_nom=ave_s_nom,
        ave_length=ave_length,
        ave_b_val=ave_b_val,
        ave_circuits=ave_circuits,
    )
    line_results = _parallel_map(lines.to_dict("records"), line_prepare)
    line_error_count = 0
    for line_result in line_results:
        if line_result.get("ok"):
            n.add("Line", **line_result["kwargs"])
        else:
            if line_error_count < 3:
                print(f"Error adding line {line_result.get('name')}: {line_result.get('error')}")
            line_error_count += 1
    
    transformers = transformers[
        (transformers['bus0'].isin(bus_ids_subset)) &
        (transformers['bus1'].isin(bus_ids_subset))
    ].copy()

    print(f"Adding {len(transformers)} transformers...")
    
    # Averages for missing data
    ave_s_nom = np.mean(transformers['s_nom'].dropna()) if not transformers['s_nom'].dropna().empty else 50000
    
    transformer_prepare = partial(_prepare_transformer_add_kwargs, ave_s_nom=ave_s_nom)
    transformer_results = _parallel_map(transformers.to_dict("records"), transformer_prepare)
    transformer_error_count = 0
    for transformer_result in transformer_results:
        if transformer_result.get("ok"):
            n.add("Transformer", **transformer_result["kwargs"])
        else:
            if transformer_error_count < 3:
                print(
                    f"Error adding transformer {transformer_result.get('name', 'unknown')}: {transformer_result.get('error')}"
                )
            transformer_error_count += 1
    
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
    link_prepare = partial(_prepare_link_add_kwargs, ave_p_nom=ave_p_nom)
    link_results = _parallel_map(links.to_dict("records"), link_prepare)
    link_error_count = 0
    for link_result in link_results:
        if link_result.get("ok"):
            n.add("Link", **link_result["kwargs"])
        else:
            if link_error_count < 3:
                print(f"Error adding AC link {link_result.get('name', 'unknown')}: {link_result.get('error')}")
            link_error_count += 1

    # Additional options (CRS, snapshots, carriers) can be applied here
    return net

def add_generators(n: pypsa.Network, RawData: dp.RawData) -> pypsa.Network:
    """Add generators from the provided DataFrame into the PyPSA network."""
    generators_df = RawData.get('generators')
    
    RawData.bus_coords()

    generator_results = _parallel_map(generators_df.to_dict("records"), _prepare_generator_add_kwargs)
    generator_error_count = 0
    for generator_result in generator_results:
        if generator_result.get("ok"):
            n.add("Generator", **generator_result["kwargs"])
        else:
            if generator_error_count < 3:
                print(f"Error adding generator {generator_result.get('name', 'unknown')}: {generator_result.get('error')}")
            generator_error_count += 1
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


