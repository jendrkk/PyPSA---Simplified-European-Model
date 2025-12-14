"""
IO helpers for serializing sources and PyPSA networks.

Designed to keep payloads compact and reproducible.
"""
#from __future__ import annotations
import json
import os
import gzip
import pickle
from typing import Any, Dict
import pypsa


def serialize_network_source(output_path: str, data_dict: Dict[str, Any]) -> str:
    """
    Serialize source inputs needed to build a network.

    Attempts gzipped JSON first for portability; if the data contains
    non-JSON-serializable objects (e.g., pandas DataFrames), falls back
    to gzipped pickle (`.pkl.gz`). Returns the written file path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Try JSON gz
    try:
        with gzip.open(output_path, "wt", encoding="utf-8") as f:
            json.dump(data_dict, f, separators=(",", ":"))
        return output_path
    except (TypeError, OverflowError):
        # Fallback to pickle gzip. If user requested .json.gz, switch to .pkl.gz
        base, ext = os.path.splitext(output_path)
        if base.endswith(".json"):
            base = base[:-5]
        out = base + ".pkl.gz"
        with gzip.open(out, "wb") as f:
            pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        return out


def load_serialized_source(input_path: str) -> Dict[str, Any]:
    """Load previously serialized source inputs supporting gzipped JSON and gzipped pickle."""
    # Try JSON text mode first
    try:
        with gzip.open(input_path, "rt", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # Fallback to pickle binary
        with gzip.open(input_path, "rb") as f:
            return pickle.load(f)


def save_optimized_network(output_path: str, network_obj: pypsa.Network) -> str:
    """
    Save an optimized PyPSA network. If `output_path` ends with `.nc` it
    writes NetCDF, otherwise it writes NetCDF with `.nc` appended.
    Returns the written file path.
    """
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if output_path.endswith(".nc"):
        network_obj.export_to_netcdf(output_path)
    else:
        base, _ = os.path.splitext(output_path)
        output_path = base + ".nc"
        network_obj.export_to_netcdf(output_path)
    return output_path


def load_optimized_network(input_path: str) -> pypsa.Network:
    """
    Load a PyPSA network previously saved in NetCDF format.
    """
    net = pypsa.Network()
    net.import_from_netcdf(input_path)
    return net
