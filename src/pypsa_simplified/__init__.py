"""PyPSA Simplified European Model Package."""

__version__ = "0.2.0"

"""Top-level package for pypsa_simplified.

This module uses lazy imports so `import pypsa_simplified` is lightweight
and does not immediately require heavy optional dependencies (e.g. `pypsa`).
Accessing attributes will import the appropriate submodule on demand.
"""

_lazy_map = {
	# io
	"serialize_network_source": "pypsa_simplified.io",
	"load_serialized_source": "pypsa_simplified.io",
	"save_optimized_network": "pypsa_simplified.io",
	"load_optimized_network": "pypsa_simplified.io",
	# remote
	"SSHConfig": "pypsa_simplified.remote",
	"transfer_to_server": "pypsa_simplified.remote",
	"run_remote_job": "pypsa_simplified.remote",
	"fetch_from_server": "pypsa_simplified.remote",
	# network
	"build_network": "pypsa_simplified.network",
	"build_network_from_serialized_source": "pypsa_simplified.network",
	"optimize_network": "pypsa_simplified.network",
	"network_summary": "pypsa_simplified.network",
	# data prep
	"prepare_osm_source": "pypsa_simplified.data_prep",
	"prepare_generator_data": "pypsa_simplified.data_prep",
	# demand
	"prepare_demand_data": "pypsa_simplified.demand",
	"calculate_population_voronoi": "pypsa_simplified.demand",
}


def __getattr__(name: str):
	"""Lazy-load attribute `name` from the mapped submodule."""
	if name in _lazy_map:
		module_name = _lazy_map[name]
		import importlib

		mod = importlib.import_module(module_name)
		val = getattr(mod, name)
		globals()[name] = val
		return val
	raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__():
	return sorted(list(globals().keys()) + list(_lazy_map.keys()))


__all__ = list(_lazy_map.keys())