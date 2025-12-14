import os
from pypsa_simplified import (
    serialize_network_source,
    load_serialized_source,
    build_network_from_source,
    optimize_network,
    save_optimized_network,
    load_optimized_network,
    network_summary,
)


def test_io_roundtrip(tmp_path):
    src = {"buses_csv": "PyPSA/tmp/buses.csv"}
    spath = tmp_path / "src.json.gz"
    serialize_network_source(str(spath), src)
    loaded = load_serialized_source(str(spath))
    assert loaded == src


def test_network_build_summary():
    src = {"buses_csv": "PyPSA/tmp/buses.csv"}
    net = build_network_from_source(src, options=None)
    assert isinstance(network_summary(net), dict)


def test_network_save_load(tmp_path):
    src = {"buses_csv": "PyPSA/tmp/buses.csv"}
    net = build_network_from_source(src, options=None)
    out = tmp_path / "net.nc"
    save_optimized_network(str(out), net)
    net2 = load_optimized_network(str(out))
    assert network_summary(net2)["n_buses"] == network_summary(net)["n_buses"]
