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
import logging
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from pypsa_simplified import data_prep as dp


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src to path for data_prep import
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
src_path = repo_root / 'scripts'
if str(src_path) not in sys.path:
    sys.path.insert(1, str(src_path))

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
    """
    Prepare kwargs for adding a Link to the network.
    
    Links are bidirectional by default (p_min_pu=-1.0) following PyPSA-EUR convention.
    This allows power flow in both directions up to p_nom capacity.
    """
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
                "p_min_pu": -1.0,  # Bidirectional: power can flow bus1 -> bus0
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
    record: dict,
    snapshots: pd.DatetimeIndex,
    default_timestamps: pd.DatetimeIndex,
    clip_non_negative: bool = False,
) -> dict:
    """Prepare a single load time series for insertion.

    Parameters
    record: dict
        Dictionary with 'bus_id' (str) and 'demand_series' (list of floats).
    snapshots: pd.DatetimeIndex
        Target snapshots to align the series to.
    default_timestamps: pd.DatetimeIndex
        Default timestamps in UTC corresponding to the demand_series list.
    clip_non_negative: bool
        If True, negative values are clipped to zero.

    Returns
    dict: Mapping with fields: ok, bus, name, p_set (pd.Series) or error.

    Notes
    - demand_series list is converted to pd.Series with default_timestamps as index.
    - Series are reindexed to `snapshots`, filling missing with 0.
    - All timestamps remain in UTC.
    """
    try:
        bus_id = record['bus_id']
        demand_list = record['demand_series']
        
        # Accept both lists and numpy arrays
        if not isinstance(demand_list, (list, np.ndarray)):
            raise TypeError(
                f"Demand series for bus {bus_id} must be a list or array of numbers."
            )
        
        # Convert list/array to Series with default_timestamps as index (UTC)
        ser = pd.Series(demand_list, index=default_timestamps, dtype=float)
        
        # Reindex to target snapshots
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
        return {"ok": False, "bus": str(record.get('bus_id', 'unknown')), "error": str(exc)}


def add_loads_from_series(
    n: pypsa.Network,
    join: bool = True,
    if_float: bool = False,
    extend_snapshots: bool = False,
    carrier: str = "AC",
    clip_non_negative: bool = False,
) -> pypsa.Network:
    """Add per-bus demand time series as Loads into the network.

    Parameters
    n: pypsa.Network
        Target network to modify.
    join: bool (default: True)
        If True, reads 'voronoi_demand_join.gzip'; otherwise 'voronoi_demand.gzip'.
    extend_snapshots: bool (default: False)
        If True and `n.snapshots` is already set, extend it to include default_timestamps;
        otherwise, align series to the current `n.snapshots`.
    carrier: str (default: "AC")
        Carrier to assign to all added loads.
    clip_non_negative: bool (default: False)
        If True, negative demand values are clipped to zero.

    Returns
    pypsa.Network: The modified network with new loads added.

    Data Format
    - Reads parquet file with columns: 'bus_id' (str), 'demand_series' (list of floats).
    - Each demand_series list contains 87672 hourly values from 2015-01-01 to 2024-12-31 23:00:00 (UTC).
    - All timestamps remain in UTC throughout processing.

    Approach
    - Load demand data from processed parquet file.
    - Determine target snapshots (existing, or default_timestamps, or union).
    - Convert each demand list to pd.Series with UTC timestamps.
    - Add one Load per bus (name: "load_{bus_id}") and assign p_set.

    Edge cases handled
    - Missing buses are skipped with a warning.
    - Empty input returns immediately.
    """
    
    repo_root = find_repo_root(Path(__file__).parent)
    demand_dir = repo_root / 'data' / 'processed' / f"voronoi_demand{'_join' if join else ''}{'_f' if if_float else ''}.gzip"
    
    # Read demand data (columns: bus_id, demand_series)
    demand_df = pd.read_parquet(demand_dir, engine="pyarrow")
    
    # Default timestamps in UTC (87672 hourly values)
    default_timestamps = pd.date_range(
        start='2015-01-01', 
        end='2024-12-31 23:00:00', 
        freq='h',
        tz=None  # UTC, timezone-naive
    )
    
    # Determine target snapshots
    existing = getattr(n, "snapshots", pd.DatetimeIndex([]))
    if existing is None or len(existing) == 0:
        # No snapshots set: use default_timestamps
        target_snapshots = default_timestamps
        n.set_snapshots(target_snapshots)
    else:
        if extend_snapshots:
            # Extend to include both existing and default_timestamps
            all_idx = [pd.DatetimeIndex(existing), default_timestamps]
            target_snapshots = pd.DatetimeIndex(
                sorted(set(existing) | set(default_timestamps))
            )
            if not target_snapshots.equals(existing):
                n.set_snapshots(target_snapshots)
        else:
            # Use existing snapshots
            target_snapshots = pd.DatetimeIndex(existing)

    print(f"Adding demand for {len(demand_df)} buses...")

    # Prepare records as list of dicts
    records = demand_df.to_dict('records')
    
    prepare = partial(
        _prepare_load_series,
        snapshots=target_snapshots,
        default_timestamps=default_timestamps,
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
            bus_loads = np.array([l for l in res["p_set"]], dtype=float)
            logger.info(f"{bus_loads.dtype}")
            n.loads_t.p_set[name] = bus_loads
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
    data_dict = data_dict.data
    
    # Example: if data_dict contains paths for core components
    buses = data_dict.get("buses")
    
    lines = data_dict.get("lines")
    links = data_dict.get("links")
    transformers = data_dict.get("transformers", pd.DataFrame())
    converters = data_dict.get("converters", pd.DataFrame())
    
    if options is not None and options['snapshots'] is not None:
        n.set_snapshots(options['snapshots'])
    
    if options is not None and options['name'] is not None:
        n.name = options['name']
    
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
    return n

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


# =============================================================================
# RENEWABLE GENERATOR FUNCTIONS (with time-varying profiles)
# =============================================================================

# Fuel type to carrier mapping (align powerplants.csv with PyPSA carriers)
FUEL_TO_CARRIER = {
    'Wind': 'wind',
    'Solar': 'solar',
    'Hydro': 'hydro',
    'Natural Gas': 'gas',
    'Hard Coal': 'hard coal',
    'Lignite': 'lignite',
    'Nuclear': 'nuclear',
    'Oil': 'oil',
    'Bioenergy': 'biomass',
    'Biogas': 'biogas',
    'Solid Biomass': 'biomass',
    'Waste': 'waste',
    'Geothermal': 'geothermal',
    'Other': 'other',
}

# Country name to ISO code mapping (powerplants.csv uses full names)
COUNTRY_NAME_TO_ISO = {
    'Germany': 'DE', 'France': 'FR', 'Spain': 'ES', 'Italy': 'IT',
    'United Kingdom': 'GB', 'Poland': 'PL', 'Netherlands': 'NL',
    'Belgium': 'BE', 'Greece': 'GR', 'Portugal': 'PT', 'Sweden': 'SE',
    'Austria': 'AT', 'Finland': 'FI', 'Denmark': 'DK', 'Norway': 'NO',
    'Switzerland': 'CH', 'Ireland': 'IE', 'Romania': 'RO', 'Czechia': 'CZ',
    'Czech Republic': 'CZ', 'Hungary': 'HU', 'Slovakia': 'SK', 'Bulgaria': 'BG',
    'Croatia': 'HR', 'Slovenia': 'SI', 'Estonia': 'EE', 'Latvia': 'LV',
    'Lithuania': 'LT', 'Luxembourg': 'LU', 'Cyprus': 'CY', 'Malta': 'MT',
    'Ukraine': 'UA', 'Serbia': 'RS', 'Bosnia and Herzegovina': 'BA',
    'Montenegro': 'ME', 'North Macedonia': 'MK', 'Albania': 'AL',
    'Kosovo': 'XK', 'Moldova': 'MD', 'Belarus': 'BY', 'Turkey': 'TR',
    'Great Britain': 'GB', 'UK': 'GB',
}

# Reverse mapping (ISO to full name)
ISO_TO_COUNTRY_NAME = {v: k for k, v in COUNTRY_NAME_TO_ISO.items()}

# Default marginal costs (€/MWh) based on fuel type
# Can be refined with actual fuel prices
DEFAULT_MARGINAL_COSTS = {
    'wind': 0.0,        # No fuel cost for renewables
    'solar': 0.0,
    'hydro': 0.0,
    'gas': 50.0,        # Natural gas CCGT (fuel + variable O&M)
    'hard coal': 35.0,  # Hard coal (fuel + variable O&M)
    'lignite': 25.0,    # Lignite (low fuel cost, high emissions)
    'nuclear': 10.0,    # Nuclear (low variable cost)
    'oil': 80.0,        # Oil (expensive fuel)
    'biomass': 40.0,    # Biomass (fuel procurement)
    'waste': 20.0,      # Waste to energy
    'geothermal': 5.0,  # Geothermal (low variable cost)
    'other': 50.0,
}

# Default efficiencies by fuel type
DEFAULT_EFFICIENCIES = {
    'wind': 1.0,        # Renewable: efficiency = 1 (capacity factor in p_max_pu)
    'solar': 1.0,
    'hydro': 0.9,
    'gas': 0.55,        # Combined cycle gas turbine
    'hard coal': 0.40,
    'lignite': 0.38,
    'nuclear': 0.33,
    'oil': 0.35,
    'biomass': 0.35,
    'waste': 0.25,
    'geothermal': 0.20,
    'other': 0.35,
}


def _map_generators_to_buses(
    generators: pd.DataFrame,
    buses: pd.DataFrame,
    lat_col: str = 'lat',
    lon_col: str = 'lon'
) -> pd.DataFrame:
    """
    Map generators to the nearest bus based on geographic coordinates.
    
    Parameters
    ----------
    generators : pd.DataFrame
        Generator data with lat/lon columns
    buses : pd.DataFrame  
        Bus data with x (lon) and y (lat) columns
    lat_col : str
        Name of latitude column in generators
    lon_col : str
        Name of longitude column in generators
        
    Returns
    -------
    pd.DataFrame
        Generators with added 'bus' column
    """
    logger.info(f"Mapping {len(generators)} generators to {len(buses)} buses...")
    
    result = generators.copy()
    
    # Filter generators with valid coordinates
    valid_coords = result[[lat_col, lon_col]].notna().all(axis=1)
    result_valid = result[valid_coords].copy()
    result_invalid = result[~valid_coords].copy()
    
    if len(result_invalid) > 0:
        logger.warning(f"{len(result_invalid)} generators have missing coordinates")
    
    if len(result_valid) == 0:
        result['bus'] = None
        return result
    
    # Create arrays for distance calculation
    gen_coords = result_valid[[lat_col, lon_col]].values
    bus_coords = buses[['y', 'x']].values  # y=lat, x=lon
    bus_ids = buses.index.values
    
    # Find nearest bus for each generator using vectorized calculation
    def haversine_distance(lat1, lon1, lat2, lon2):
        """Calculate haversine distance in km."""
        R = 6371  # Earth radius in km
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c
    
    nearest_buses = []
    for gen_lat, gen_lon in gen_coords:
        distances = haversine_distance(gen_lat, gen_lon, bus_coords[:, 0], bus_coords[:, 1])
        nearest_idx = np.argmin(distances)
        nearest_buses.append(bus_ids[nearest_idx])
    
    result_valid['bus'] = nearest_buses
    
    # Combine back with invalid generators
    result = pd.concat([result_valid, result_invalid], ignore_index=True)
    
    assigned = result['bus'].notna().sum()
    logger.info(f"Assigned {assigned}/{len(generators)} generators to buses")
    
    return result


def prepare_generator_data(
    generators_raw: pd.DataFrame,
    network_buses: pd.DataFrame,
    capacity_col: str = 'Capacity',
    fueltype_col: str = 'Fueltype',
    technology_col: str = 'Technology',
    lat_col: str = 'lat',
    lon_col: str = 'lon',
    name_col: str = 'Name',
    efficiency_col: str = 'Efficiency',
    countries: list | None = None
) -> pd.DataFrame:
    """
    Prepare generator data for addition to PyPSA network.
    
    This function:
    1. Filters generators by country (if specified)
    2. Maps fuel types to carriers
    3. Assigns generators to nearest buses
    4. Sets default efficiencies and marginal costs
    5. Creates unique generator names
    
    Parameters
    ----------
    generators_raw : pd.DataFrame
        Raw generator data (e.g., from powerplants.csv)
    network_buses : pd.DataFrame
        Bus DataFrame from the PyPSA network (n.buses)
    capacity_col : str
        Column name for capacity (MW)
    fueltype_col : str
        Column name for fuel type
    technology_col : str
        Column name for technology
    lat_col : str
        Column name for latitude
    lon_col : str
        Column name for longitude
    name_col : str
        Column name for generator name
    efficiency_col : str
        Column name for efficiency
    countries : list, optional
        List of country codes to filter (e.g., ['DE', 'FR'])
        
    Returns
    -------
    pd.DataFrame
        Prepared generator data ready for network addition
    """
    logger.info(f"Preparing {len(generators_raw)} generators...")
    
    gens = generators_raw.copy()
    
    # Map country names to ISO codes for filtering
    if 'Country' in gens.columns:
        gens['country_iso'] = gens['Country'].map(COUNTRY_NAME_TO_ISO)
        # Keep original country name and add ISO code
        unmapped = gens[gens['country_iso'].isna()]['Country'].unique()
        if len(unmapped) > 0:
            logger.warning(f"Unmapped countries: {list(unmapped)[:10]}")
    
    # Filter by country if specified (using ISO codes)
    if countries is not None:
        if 'country_iso' in gens.columns:
            gens = gens[gens['country_iso'].isin(countries)].copy()
            logger.info(f"Filtered to {len(gens)} generators in {len(countries)} countries")
        elif 'Country' in gens.columns:
            # Try direct match if ISO mapping failed
            gens = gens[gens['Country'].isin(countries)].copy()
            logger.info(f"Filtered to {len(gens)} generators (direct country match)")
    
    # Filter out zero/negative capacity
    gens = gens[gens[capacity_col] > 0].copy()
    
    # Map fuel types to carriers
    gens['carrier'] = gens[fueltype_col].map(FUEL_TO_CARRIER).fillna('other')
    
    # Rename capacity column
    gens['p_nom'] = gens[capacity_col]
    
    # Set efficiency (use provided or default by carrier)
    if efficiency_col in gens.columns:
        # Use provided efficiency where available, otherwise default
        gens['efficiency'] = gens.apply(
            lambda row: row[efficiency_col] if pd.notna(row[efficiency_col]) and row[efficiency_col] > 0
            else DEFAULT_EFFICIENCIES.get(row['carrier'], 0.4),
            axis=1
        )
    else:
        gens['efficiency'] = gens['carrier'].map(DEFAULT_EFFICIENCIES).fillna(0.4)
    
    # Set marginal cost by carrier
    gens['marginal_cost'] = gens['carrier'].map(DEFAULT_MARGINAL_COSTS).fillna(50.0)
    
    # Map to buses
    gens = _map_generators_to_buses(gens, network_buses, lat_col, lon_col)
    
    # Filter out generators without bus assignment
    gens = gens[gens['bus'].notna()].copy()
    
    # Create unique names (handle duplicates)
    if name_col in gens.columns:
        # Add index suffix to ensure uniqueness
        gens['gen_name'] = gens[name_col].astype(str) + '_' + gens.index.astype(str)
    else:
        gens['gen_name'] = 'gen_' + gens.index.astype(str)
    
    # Add technology info for wind (onshore/offshore)
    if technology_col in gens.columns:
        gens['is_offshore'] = gens[technology_col].str.lower().str.contains('offshore', na=False)
    else:
        gens['is_offshore'] = False
    
    logger.info(f"Prepared {len(gens)} generators for network addition")
    
    return gens


def add_generators_with_profiles(
    n: pypsa.Network,
    generators: pd.DataFrame,
    profiles: dict | None = None,
    name_col: str = 'gen_name',
    bus_col: str = 'bus',
    capacity_col: str = 'p_nom',
    carrier_col: str = 'carrier',
    efficiency_col: str = 'efficiency',
    marginal_cost_col: str = 'marginal_cost'
) -> pypsa.Network:
    """
    Add generators to the network, optionally with time-varying profiles.
    
    This is the main function for adding generators with renewable profiles.
    For wind and solar generators, capacity factors (p_max_pu) are set from
    the provided profiles dictionary.
    
    Parameters
    ----------
    n : pypsa.Network
        Target network to modify
    generators : pd.DataFrame
        Prepared generator data (from prepare_generator_data)
    profiles : dict, optional
        Dictionary with 'wind' and 'solar' DataFrames containing capacity factors.
        Each DataFrame has timestamps as index and generator names as columns.
        If None, renewable generators are added with p_max_pu = 1.0
    name_col : str
        Column name for generator name (used as index)
    bus_col : str
        Column name for bus assignment
    capacity_col : str
        Column name for nominal capacity (MW)
    carrier_col : str
        Column name for carrier type
    efficiency_col : str
        Column name for efficiency
    marginal_cost_col : str
        Column name for marginal cost
        
    Returns
    -------
    pypsa.Network
        Network with added generators
        
    Notes
    -----
    - Wind and solar generators get p_max_pu time series from profiles
    - Conventional generators get p_max_pu = 1.0 (always available)
    - Missing profiles result in p_max_pu = 0.0 (unavailable)
    """
    logger.info(f"Adding {len(generators)} generators to network...")
    
    # Ensure snapshots are set
    if len(n.snapshots) == 0:
        logger.error("Network has no snapshots set. Please set snapshots before adding generators.")
        return n
    
    # Track statistics
    added_count = 0
    error_count = 0
    profile_count = 0
    
    for idx, gen in generators.iterrows():
        try:
            gen_name = str(gen[name_col])
            bus = str(gen[bus_col])
            p_nom = float(gen[capacity_col])
            carrier = str(gen[carrier_col])
            efficiency = float(gen.get(efficiency_col, 0.4))
            marginal_cost = float(gen.get(marginal_cost_col, 50.0))
            
            # Skip if bus doesn't exist in network
            if bus not in n.buses.index:
                logger.warning(f"Bus {bus} not in network, skipping generator {gen_name}")
                error_count += 1
                continue
            
            # Add generator
            n.add(
                "Generator",
                name=gen_name,
                bus=bus,
                p_nom=p_nom,
                carrier=carrier,
                efficiency=efficiency,
                marginal_cost=marginal_cost,
            )
            added_count += 1
            
            # Add time-varying profile for renewables
            if profiles is not None and carrier in ['wind', 'solar']:
                profile_df = profiles.get(carrier, pd.DataFrame())
                
                # Try to find matching profile column
                profile_col = None
                original_name = gen.get('Name', gen_name)
                
                # Check various possible column names
                for candidate in [gen_name, original_name, str(original_name)]:
                    if candidate in profile_df.columns:
                        profile_col = candidate
                        break
                
                if profile_col is not None:
                    # Align profile to network snapshots
                    profile_series = profile_df[profile_col]
                    profile_aligned = profile_series.reindex(n.snapshots).fillna(0.0)
                    n.generators_t.p_max_pu[gen_name] = profile_aligned.values
                    profile_count += 1
                else:
                    # No profile found - set to 0 (conservative)
                    logger.debug(f"No profile found for {gen_name}, setting p_max_pu=0")
                    n.generators_t.p_max_pu[gen_name] = 0.0
                    
        except Exception as e:
            error_count += 1
            if error_count <= 5:
                logger.error(f"Error adding generator {gen.get(name_col, idx)}: {e}")
    
    logger.info(f"Added {added_count} generators ({profile_count} with time-varying profiles)")
    if error_count > 0:
        logger.warning(f"{error_count} generators failed to add")
    
    return n


def add_renewable_generators_aggregated(
    n: pypsa.Network,
    generators: pd.DataFrame,
    profiles: dict,
    carrier: str,
    name_col: str = 'gen_name',
    bus_col: str = 'bus',
    capacity_col: str = 'p_nom',
    aggregate_to_bus: bool = True
) -> pypsa.Network:
    """
    Add renewable generators with bus-level aggregation.
    
    Instead of adding individual generators, this function aggregates
    capacity per bus and uses a capacity-weighted average profile.
    This reduces model size while preserving aggregate behavior.
    
    Parameters
    ----------
    n : pypsa.Network
        Target network
    generators : pd.DataFrame
        Generator data filtered for specific carrier
    profiles : dict
        Profiles dictionary with carrier key
    carrier : str
        Carrier type ('wind' or 'solar')
    name_col : str
        Column for generator names
    bus_col : str
        Column for bus assignments
    capacity_col : str
        Column for capacity
    aggregate_to_bus : bool
        If True, aggregate to bus level; if False, add individual generators
        
    Returns
    -------
    pypsa.Network
        Network with added generators
    """
    if not aggregate_to_bus:
        return add_generators_with_profiles(n, generators, profiles, name_col, bus_col, capacity_col)
    
    logger.info(f"Adding aggregated {carrier} generators...")
    
    profile_df = profiles.get(carrier, pd.DataFrame())
    if profile_df.empty:
        logger.warning(f"No profiles available for {carrier}")
        return n
    
    # Group by bus
    bus_groups = generators.groupby(bus_col)
    
    added = 0
    for bus, group in bus_groups:
        if bus not in n.buses.index:
            continue
        
        # Aggregate capacity
        total_capacity = group[capacity_col].sum()
        
        # Calculate capacity-weighted average profile
        capacities = group[capacity_col].values
        weights = capacities / capacities.sum()
        
        profile_cols = []
        for idx, gen in group.iterrows():
            gen_name = gen[name_col]
            original_name = gen.get('Name', gen_name)
            
            # Find matching profile column
            for candidate in [gen_name, original_name, str(original_name)]:
                if candidate in profile_df.columns:
                    profile_cols.append(profile_df[candidate])
                    break
            else:
                # No profile - assume zeros
                profile_cols.append(pd.Series(0.0, index=profile_df.index))
        
        if profile_cols:
            # Stack profiles and compute weighted average
            profiles_array = np.column_stack([p.values for p in profile_cols])
            weighted_profile = (profiles_array * weights).sum(axis=1)
        else:
            weighted_profile = np.zeros(len(profile_df))
        
        # Add aggregated generator
        agg_name = f"{carrier}_{bus}"
        n.add(
            "Generator",
            name=agg_name,
            bus=bus,
            p_nom=total_capacity,
            carrier=carrier,
            marginal_cost=DEFAULT_MARGINAL_COSTS.get(carrier, 0.0),
        )
        
        # Assign profile
        profile_aligned = pd.Series(weighted_profile, index=profile_df.index)
        profile_aligned = profile_aligned.reindex(n.snapshots).fillna(0.0)
        n.generators_t.p_max_pu[agg_name] = profile_aligned.values
        added += 1
    
    logger.info(f"Added {added} aggregated {carrier} generators")
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

def save_network(n: pypsa.Network, path: str) -> None:
    n.export_to_netcdf(path)
    logger.info(f"Network saved to {path}")

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


