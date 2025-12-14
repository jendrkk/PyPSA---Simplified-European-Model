"""
IO helpers for serializing sources and PyPSA networks.

Designed to keep payloads compact and reproducible.
"""
from __future__ import annotations
import json
import os
import gzip
from typing import Any, Dict

import pypsa


def serialize_network_source(output_path: str, data_dict: Dict[str, Any]) -> str:
    """
    Serialize lightweight source inputs needed to build a network.

    Uses gzipped JSON for compactness.

    Returns the written file path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with gzip.open(output_path, "wt", encoding="utf-8") as f:
        json.dump(data_dict, f, separators=(",", ":"))
    return output_path


def load_serialized_source(input_path: str) -> Dict[str, Any]:
    """Load previously serialized source inputs (gzipped JSON)."""
    with gzip.open(input_path, "rt", encoding="utf-8") as f:
        return json.load(f)


def save_optimized_network(output_path: str, network_obj: pypsa.Network) -> str:
    """
    Save an optimized PyPSA network in a compact format.

    Preference order:
    - NetCDF (`.nc`) via `network_obj.export_to_netcdf` for compactness and fidelity.

    Returns the written file path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if output_path.endswith(".nc"):
        network_obj.export_to_netcdf(output_path)
    else:
        # Fallback: write NetCDF next to requested path.
        base, _ = os.path.splitext(output_path)
        output_path = base + ".nc"
        network_obj.export_to_netcdf(output_path)
    return output_path


def load_optimized_network(input_path: str) -> pypsa.Network:
    """
    Load a PyPSA network previously saved, supporting NetCDF.
    """
    net = pypsa.Network()
    net.import_from_netcdf(input_path)
    return net
