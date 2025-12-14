# PyPSA---Simplified-European-Model

A simplified Python package for building and optimizing PyPSA (Python for Power System Analysis) networks, with a focus on European energy system modeling.

## Repository Structure

```
PyPSA---Simplified-European-Model/
├── src/
│   └── pypsa_simplified/        # Main Python package
│       ├── __init__.py          # Package initialization
│       ├── core.py              # Network building and data loading
│       ├── optimize.py          # Optimization functions
│       └── utils.py             # Utility functions
├── notebooks/                   # Jupyter notebooks for analysis
│   ├── 01_data_preparation.ipynb
│   ├── 02_run_optimization.ipynb
│   └── 03_analysis.ipynb
├── data/                        # Data directory (gitignored)
│   ├── raw/                     # Raw input data (not tracked)
│   └── processed/               # Processed data (not tracked)
├── requirements-dev.txt         # Development dependencies
└── README.md                    # This file
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/jendrkk/PyPSA---Simplified-European-Model.git
cd PyPSA---Simplified-European-Model
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install development requirements:
```bash
pip install -r requirements-dev.txt
```

4. (Optional) Install pulp for optimization:
```bash
pip install pulp
```

### Running the Notebooks

The notebooks are configured to import the package directly from the `src/` directory.

1. Start Jupyter:
```bash
jupyter notebook
```

2. Navigate to the `notebooks/` directory and open:
   - `01_data_preparation.ipynb` - Load data and build networks
   - `02_run_optimization.ipynb` - Run optimization on networks
   - `03_analysis.ipynb` - Analysis guide and tips

### Importing the Package

When working in notebooks or scripts within the repository:

```python
import sys
from pathlib import Path

# Add src directory to Python path
repo_root = Path().absolute()  # Adjust as needed
src_path = repo_root / "src"
sys.path.insert(0, str(src_path))

# Now you can import the package
from pypsa_simplified import build_network, optimize_network, load_csv
```

### Basic Usage

```python
from pypsa_simplified import build_network, optimize_network

# Define a simple network
nodes = ["Berlin", "Paris", "London"]
edges = [("Berlin", "Paris"), ("Paris", "London")]

# Build the network
network = build_network(nodes, edges)

# Run optimization
result = optimize_network(network)
print(f"Status: {result['status']}")
print(f"Objective: {result['objective_value']}")
```

## Working with Data

### Data Directory

Place your datasets in the `data/` directory:

- **`data/raw/`**: Store raw input files (CSV, JSON, etc.). This directory is gitignored and not tracked in version control.
- **`data/processed/`**: Store processed/cleaned data. Also gitignored.

### Example Data Organization

```
data/
├── raw/
│   ├── nodes.csv              # Node/bus data
│   ├── lines.csv              # Transmission line data
│   └── generators.csv         # Generator specifications
└── processed/
    └── network.json           # Preprocessed network
```

**Note**: Large data files should always be placed in `data/raw/` or `data/processed/` as these directories are gitignored.

## Package Modules

### core.py
- `build_network(nodes, edges)`: Build a network structure
- `load_csv(path)`: Load CSV data with error handling

### optimize.py
- `optimize_network(network, options=None)`: Optimize network configuration
  - Uses pulp if available, otherwise falls back to deterministic placeholder

### utils.py
- `ensure_dir(path)`: Create directory if it doesn't exist
- `read_json(path)`: Read JSON file
- `write_json(obj, path)`: Write object to JSON file

## Development

### Code Structure

The package follows a simple, modular structure:
- Core functionality in `src/pypsa_simplified/`
- Examples and workflows in `notebooks/`
- Data management through utility functions

### Adding New Features

1. Add new functions to appropriate module (`core.py`, `optimize.py`, `utils.py`)
2. Include docstrings with examples
3. Update notebooks to demonstrate new features
4. Update this README if adding major functionality

## Notes

- Notebooks are committed with outputs cleared to keep repository size small
- The `.gitignore` file excludes `data/raw/` and `data/processed/` directories
- The optimization module gracefully handles missing pulp dependency

## License

This project is provided as-is for educational and research purposes.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.
