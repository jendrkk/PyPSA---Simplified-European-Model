"""
Network construction and optimization helpers using PyPSA.
"""
from __future__ import annotations
from typing import Any, Dict

import pypsa
from .io import load_serialized_source


def build_network_from_source(data_dict: Dict[str, Any], options: Dict[str, Any] | None = None) -> pypsa.Network:
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
    data = load_serialized_source(source_path)
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
