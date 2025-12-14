# Simplified PyPSA-EUR Pipeline (Refactor)

This repository has been refactored to move heavy logic into `src/pypsa_simplified` while keeping notebooks lightweight and explanatory.
## Module layout
- `src/pypsa_simplified/io.py`: Compact serialization of source inputs and NetCDF save/load for networks.
- `src/pypsa_simplified/network.py`: Build and optimize networks using PyPSA.
- `src/pypsa_simplified/remote.py`: SSH file transfer and remote execution helpers (password prompt supported).
- `src/pypsa_simplified/env_install.py`: Detect/install required packages on server or print exact conda commands.
- `src/pypsa_simplified/run_opt.py`: CLI for running optimization on the server.

## Notebook workflow
- `notebooks/01_data_preparation.ipynb`: Prepares `data/processed/source.json.gz`.
- `notebooks/main.ipynb`: Orchestrates local → server → local pipeline.

## Pipeline usage
1. Prepare inputs locally:
	- Open `notebooks/01_data_preparation.ipynb` and run all cells.
2. Optional local sanity check:
	- In `notebooks/main.ipynb`, run the "Local quick build" cell to inspect summary.
3. Transfer + run remotely:
	- In `notebooks/main.ipynb`, set your SSH config (host, user, port) and run the remote section.
	- On the server, `run_opt.py` reads `source.json.gz`, builds and optimizes the network, and writes `net.nc`.
4. Fetch results:
	- Use the fetch cell in `main.ipynb` to download `net.nc` locally and load for plotting/analysis.

## Server setup
If conda is available on the server:
```
python -m pypsa_simplified.env_install myenv
conda activate myenv
```
If conda is not available, run manually:
```
conda create -n myenv python=3.11 -y
conda activate myenv
conda install -c conda-forge pypsa pandas numpy matplotlib -y
```

## Slurm (optional)
You can adapt `remote.run_remote_job` to submit jobs via Slurm, e.g.:
```
sbatch --chdir <remote_dir> --wrap "python -m pypsa_simplified.run_opt --input source.json.gz --output net.nc"
```
Then poll for completion and fetch `net.nc`.

## Notes
- Behavior is preserved; heavy tasks moved into modules. Adjust `build_network_from_source` to mirror your existing CSV import logic precisely.
- Serialization keeps artifacts compact (gzipped JSON and NetCDF).
# PyPSA---Simplified-European-Model

A modular pipeline for building and optimizing simplified PyPSA (Python for Power System Analysis) networks, with a focus on European energy system modeling. Notebooks now remain thin orchestration layers; all heavy lifting lives in reusable modules under `src/pypsa_simplified`.

## Repository Structure

```
PyPSA---Simplified-European-Model/
├── src/
│   └── pypsa_simplified/        # Main Python package
│       ├── __init__.py
│       ├── core.py              # Legacy toy network helpers
│       ├── data_prep.py         # OSM loading, geometry handling, bus mapping
│       ├── network.py           # PyPSA network build/optimize/summarize
│       ├── io.py                # Serialization of sources and optimized nets
│       ├── remote.py            # SSH transfer + remote execution helpers
│       ├── remote_job.py        # CLI to run heavy optimization on a server
│       ├── env_install.py       # Conda dependency detection/installation hints
│       └── utils.py             # JSON and filesystem utilities
├── notebooks/                   # Jupyter notebooks for analysis
│   ├── 01_data_preparation.ipynb
│   └── main.ipynb               # High-level orchestration (refactored)
├── data/                        # Data directory (gitignored)
│   ├── raw/                     # Raw input data (not tracked)
│   └── processed/               # Processed data (not tracked)
├── tests/                       # Minimal sanity checks
├── requirements-dev.txt         # Development dependencies
└── README.md                    # This file
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

```bash
git clone https://github.com/jendrkk/PyPSA---Simplified-European-Model.git
cd PyPSA---Simplified-European-Model
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
# Optional: install pypsa manually if not pulled by requirements
pip install pypsa
```

### Running the refactored pipeline

1) **Local prep (small):**
```python
from pathlib import Path
from pypsa_simplified import prepare_osm_source, serialize_network_source

osm_dir = Path("data/raw/OSM Prebuilt Electricity Network")
source_pkg = prepare_osm_source(osm_dir)
artifact = serialize_network_source("data/processed/network_source.pkl.gz", source_pkg)
```

2) **Transfer + run remotely (heavy):**
```python
from pypsa_simplified import SSHConfig, transfer_to_server, run_remote_job, transfer_to_server_pw, run_remote_job_pw

ssh = SSHConfig(host="server", user="me", port=22, identity_file="~/.ssh/id_rsa")
transfer_to_server(artifact, "~/work/network_source.pkl.gz", ssh)
remote_cmd = "python -m pypsa_simplified.remote_job --source network_source.pkl.gz --output optimized.nc --snapshot-limit 168"
run_remote_job(ssh, remote_cmd, remote_workdir="~/work")
```

If you login with a password and don't have SSH keys, install `paramiko` and use the password helpers:

```python
pip install paramiko

ssh_pw = SSHConfig(host="server", user="me", port=22, password="YOUR_PASSWORD")
transfer_to_server_pw(artifact, "~/work/network_source.pkl.gz", ssh_pw)
run_remote_job_pw(ssh_pw, remote_cmd, remote_workdir="~/work")
```

3) **Fetch + analyze locally (light):**
```python
from pypsa_simplified import fetch_from_server, load_optimized_network, network_summary

fetch_from_server("~/work/optimized.nc", "data/processed/optimized.nc", ssh)
network = load_optimized_network("data/processed/optimized.nc")
print(network_summary(network))
```

### Running on Slurm (optional)

- Copy `network_source.pkl.gz` to the cluster scratch folder.
- Submit a batch job that calls the CLI: `python -m pypsa_simplified.remote_job --source network_source.pkl.gz --output optimized.nc --snapshot-limit 168`.
- Adjust the `snapshot-limit` or solver settings via environment variables in your Slurm script; wrap the command in `srun` if needed.

### Data layout

- **`data/raw/OSM Prebuilt Electricity Network/`**: OSM-derived CSVs (buses, lines, links, converters, transformers)
- **`data/processed/`**: Serialized artifacts for transfer (gitignored)

### Key APIs

- `prepare_osm_source(osm_dir, countries)` → dict with cleaned buses/lines/links
- `serialize_network_source(path, data)` / `load_serialized_source(path)`
- `build_network_from_source(data, NetworkOptions)` → PyPSA ``Network``
- `optimize_network(network, snapshot_slice)` → runs PyPSA optimization
- `save_optimized_network(path, network)` / `load_optimized_network(path)`
- `SSHConfig`, `transfer_to_server`, `run_remote_job`, `fetch_from_server`
- `env_install.ensure_packages()` → conda guidance for servers

### Server dependency helper

```bash
python -m pypsa_simplified.env_install
# or inside Python
from pypsa_simplified import ensure_packages, format_plan
print(format_plan(ensure_packages()))
```

If your server uses system Python and conda isn't available, use:

```bash
pip install pypsa pandas numpy matplotlib paramiko
```

### Tests

Minimal sanity checks live in `tests/` and can be run with `pytest` (PyPSA required for network test):

```bash
pytest tests/test_io_and_network.py
```
