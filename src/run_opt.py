from __future__ import annotations
import argparse
from pathlib import Path
from pypsa_simplified import (
    load_serialized_source,
    build_network_from_source,
    optimize_network,
    save_optimized_network,
)


def main():
    p = argparse.ArgumentParser("Run remote optimization")
    p.add_argument("--input", required=True, help="Path to source.json.gz on server")
    p.add_argument("--output", required=True, help="Output NetCDF path, e.g., net.nc")
    args = p.parse_args()

    src = load_serialized_source(args.input)
    net = build_network_from_source(src, options=None)
    net = optimize_network(net, solver_opts={"log": True})
    save_optimized_network(args.output, net)
    print("Wrote:", args.output)


if __name__ == "__main__":
    main()
