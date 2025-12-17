# Network Clustering Quick Reference

## One-Line Commands

### Basic Simplification
```python
from scripts.network_clust import simplify_and_cluster_network
n_simple, maps = simplify_and_cluster_network(n, n_clusters=None, remove_stubs_before=True)
```

### K-Means Clustering (Fast)
```python
n_clust, maps = simplify_and_cluster_network(n, n_clusters=50, cluster_weights=load, algorithm='kmeans', solver_name='gurobi')
```

### HAC Clustering (Topology-Aware)
```python
n_clust, maps = simplify_and_cluster_network(n, n_clusters=50, cluster_weights=load, algorithm='hac', features=features, solver_name='gurobi')
```

### Modularity Clustering
```python
n_clust, maps = simplify_and_cluster_network(n, n_clusters=50, cluster_weights=load, algorithm='modularity', solver_name='gurobi')
```

## Common Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `n_clusters` | Target number of buses | `50`, `100`, `None` |
| `cluster_weights` | Weights per bus (e.g., load) | `load` Series |
| `algorithm` | Clustering method | `'kmeans'`, `'hac'`, `'modularity'` |
| `solver_name` | Optimization solver | `'gurobi'`, `'scip'` |
| `remove_stubs_before` | Remove dead-ends first | `True`, `False` |
| `remove_stubs_matching` | Attributes to match | `['country']` |
| `focus_weights` | Regional emphasis | `{'DE': 2.0}` |

## Gurobi Setup

### Academic License (Free)
```bash
# 1. Register at gurobi.com/academia
# 2. Download license to ~/gurobi.lic
# 3. Install
conda install -c gurobi gurobi
```

### Check License
```python
import gurobipy as gp
try:
    gp.Model()
    print("✓ Gurobi OK")
except:
    print("✗ Use solver_name='scip'")
```

## Algorithm Selection

| Algorithm | Speed | Quality | Best For |
|-----------|-------|---------|----------|
| K-Means | ⚡⚡⚡ | ⭐⭐ | Large networks, geographic |
| HAC | ⚡ | ⭐⭐⭐ | Topology-aware, features |
| Modularity | ⚡ | ⭐⭐ | Pure topology, communities |

## Common Workflows

### 1. Simple Clustering
```python
n_clust, maps = simplify_and_cluster_network(
    n, n_clusters=50, cluster_weights=load, algorithm='kmeans', solver_name='gurobi'
)
```

### 2. With Regional Focus
```python
n_clust, maps = simplify_and_cluster_network(
    n, n_clusters=100, cluster_weights=load, algorithm='kmeans',
    solver_name='gurobi', focus_weights={'DE': 2.0, 'FR': 1.5}
)
```

### 3. Topology-Aware
```python
# Create features from profiles
features = pd.concat([solar_profile, wind_profile], axis=1).T

n_clust, maps = simplify_and_cluster_network(
    n, n_clusters=50, cluster_weights=load, algorithm='hac',
    features=features, solver_name='gurobi'
)
```

### 4. Preserve Substations
```python
substations = n.buses.query("v_nom >= 220").index
n_clust, maps = simplify_and_cluster_network(
    n, n_clusters=50, cluster_weights=load, algorithm='kmeans',
    solver_name='gurobi', aggregate_substations=True, substation_buses=substations
)
```

## Get Final Busmap

```python
from functools import reduce
final_map = reduce(lambda x, y: x.map(y), maps.values())
final_map.to_csv("busmap_original_to_clustered.csv")
```

## Aggregate Demand

```python
# Original demand
demand_orig = pd.read_csv("demand_original.csv", index_col=0)

# Get busmap
final_map = reduce(lambda x, y: x.map(y), maps.values())

# Aggregate
demand_agg = {}
for col in demand_orig.columns:
    if col in final_map.index:
        new_bus = final_map[col]
        demand_agg.setdefault(new_bus, []).append(demand_orig[col])

demand_clustered = {bus: pd.concat(vals, axis=1).sum(axis=1) 
                   for bus, vals in demand_agg.items()}

# Add to network
from src.pypsa_simplified.network import add_loads_from_series
n_clust = add_loads_from_series(n_clust, demand_clustered)
```

## Save and Load

```python
# Save
n_clust.export_to_netcdf("network_50.nc")
for name, m in maps.items():
    m.to_csv(f"busmap_{name}.csv")

# Load
n = pypsa.Network("network_50.nc")
busmap = pd.read_csv("busmap_cluster.csv", index_col=0, squeeze=True)
```

## Troubleshooting

### "Gurobi license not found"
→ Use `solver_name='scip'` or set `GRB_LICENSE_FILE` env variable

### "HAC requires features"
→ Create feature matrix: `features = pd.DataFrame({...})`

### "n_clusters out of range"
→ Check: `len(n.buses.groupby(['country', 'sub_network'])) <= n_clusters <= len(n.buses)`

### Memory issues
→ Simplify first: `n_simple, _ = simplify_and_cluster_network(n, n_clusters=None)`

## Performance Tips

1. **Simplify before clustering** - Reduces network size first
2. **Use k-means for large networks** - Much faster than HAC
3. **Remove stubs early** - `remove_stubs_before=True`
4. **Limit HAC features** - Keep feature matrix small (<20 columns)
5. **Cache intermediate results** - Save after each major step

## Integration with PyPSA Workflow

```python
# 1. Build network
from src.pypsa_simplified.network import build_network
from src.pypsa_simplified.data_prep import RawData
data = RawData(None).load("data.json.gz")
n = build_network(pypsa.Network(), data)

# 2. Cluster
n_clust, maps = simplify_and_cluster_network(
    n, n_clusters=50, cluster_weights=load, algorithm='kmeans', solver_name='gurobi'
)

# 3. Add demand
from src.pypsa_simplified.network import add_loads_from_series
n_clust = add_loads_from_series(n_clust, demand_by_bus)

# 4. Add generators
from src.pypsa_simplified.network import add_generators
n_clust = add_generators(n_clust, RawData)

# 5. Optimize
n_clust.optimize(solver_name='gurobi')

# 6. Analyze
print(n_clust.objective)
n_clust.generators.p.plot()
```

## Documentation

- Full docs: `scripts/docs/NETWORK_CLUSTERING.md`
- Examples: `scripts/example_clustering.py`
- Source: `scripts/network_clust.py`

## Contact

Issues? Check the documentation or review pypsa-eur reference implementation.
