# Network Clustering Strategies for Geographic Diversity

## Overview

This document explains the clustering strategies implemented to ensure better geographic representation in the European network model.

## The Problem

**Default behavior**: Load-weighted k-means clustering concentrates all clusters in high-demand regions (Western Europe), leaving peripheral countries underrepresented.

**Impact**:
- Eastern Europe: Poland, Romania, Czech Republic get 1-2 clusters despite being large countries
- Scandinavia: Sweden, Norway, Finland appear mostly as DC link endpoints
- Iberia: Spain and Portugal undersampled despite significant renewable potential
- Result: Unrealistic model for European energy policy analysis

## Implemented Solutions

### Strategy 1: Per-Country Minimum Allocation

**Method**: Guarantee each country at least N clusters (default: 3), then distribute remaining by load.

**Use case**: Balanced approach ensuring all countries represented while respecting demand patterns.

**Implementation**:
```python
from scripts import network_clust as netclust

n_clustered, busmap, allocation = netclust.cluster_network_with_strategy(
    n, 
    n_clusters=250, 
    cluster_weights=bus_loads,
    strategy="min_per_country",
    min_per_country=3
)
```

**Advantages**:
- ✅ Every country guaranteed representation
- ✅ Still load-responsive for major consumers
- ✅ Good default for policy analysis

**Disadvantages**:
- ⚠️  May oversample very small countries
- ⚠️  Minimum threshold needs tuning

---

### Strategy 2: Hybrid Geographic-Load Weighting

**Method**: Use combined features `α * [x,y] + (1-α) * load` in k-means.

**Use case**: Intuitive balance between geography and economics via α parameter.

**Parameters**:
- α = 0.0: Pure load-weighting (baseline)
- α = 0.5: Equal balance
- α = 1.0: Pure geographic clustering

**Implementation**:
```python
# Currently requires manual feature creation in notebook
# See network_04A.ipynb cells for full implementation
```

**Advantages**:
- ✅ Intuitive control via single parameter
- ✅ Smooth trade-off between geography and load

**Disadvantages**:
- ⚠️  Need to normalize features properly
- ⚠️  Requires coordinate projection

---

### Strategy 3: Equal Geographic Distribution

**Method**: Distribute clusters equally across all countries, ignoring load.

**Use case**: When every country should have equal representation (e.g., sovereignty-based analysis).

**Implementation**:
```python
n_clustered, busmap, allocation = netclust.cluster_network_with_strategy(
    n, 
    n_clusters=250, 
    cluster_weights=bus_loads,
    strategy="equal"
)
```

**Advantages**:
- ✅ Perfect geographic balance
- ✅ Every country gets equal "voice"
- ✅ Good for political/policy analysis

**Disadvantages**:
- ⚠️  Oversamples low-demand regions
- ⚠️  May be inefficient for economic optimization

---

### Strategy 4: Focus Weights

**Method**: Manually boost cluster allocation for specific countries using multipliers.

**Use case**: Custom scenarios where you know which regions need more detail.

**Implementation**:
```python
focus_regions = {
    'PL': 2.5,   # Poland - Eastern European hub
    'RO': 2.5,   # Romania - Balkan integration
    'ES': 2.0,   # Spain - Solar potential
    'PT': 2.0,   # Portugal - Wind potential
    'SE': 1.5,   # Sweden - Hydro + wind
}

n_clustered, busmap, allocation = netclust.cluster_network_with_strategy(
    n, 
    n_clusters=250, 
    cluster_weights=bus_loads,
    strategy="focus",
    focus_weights=focus_regions
)
```

**Advantages**:
- ✅ Precise control over specific countries
- ✅ Flexible for research questions
- ✅ Can be combined with load weighting

**Disadvantages**:
- ⚠️  Requires domain knowledge
- ⚠️  Manual tuning needed

---

### Strategy 5: Optimization-Based (PyPSA-EUR Method)

**Method**: Solve integer programming problem to minimize deviation from proportional allocation.

**Mathematical formulation**:
```
minimize:   Σ (n_c - L_c * N)²
subject to: Σ n_c = N
            1 <= n_c <= N_c
```

Where:
- `n_c` = clusters for country c
- `L_c` = normalized load weight
- `N` = total target clusters
- `N_c` = available buses in country c

**Requirements**:
- `linopy` package
- Solver: Gurobi (academic license), SCIP, or CPLEX

**Implementation**:
```python
# With standard allocation
n_clustered, busmap, allocation = netclust.cluster_network_with_strategy(
    n, 
    n_clusters=250, 
    cluster_weights=bus_loads,
    strategy="optimized"
)

# With focus weights
n_clustered, busmap, allocation = netclust.cluster_network_with_strategy(
    n, 
    n_clusters=250, 
    cluster_weights=bus_loads,
    strategy="optimized",
    focus_weights={'PL': 0.15, 'ES': 0.12, 'RO': 0.08}
)
```

**Advantages**:
- ✅ Mathematically optimal
- ✅ Official PyPSA-EUR method
- ✅ Can include focus weights

**Disadvantages**:
- ⚠️  Requires commercial solver (Gurobi) or complex setup
- ⚠️  Slower than heuristic methods
- ⚠️  May not converge for very large problems

---

## Comparison Matrix

| Strategy | Geographic Coverage | Load Respect | Complexity | Best For |
|----------|---------------------|--------------|------------|----------|
| **Baseline (Load-Only)** | ⭐ Poor | ⭐⭐⭐⭐⭐ Perfect | ⭐ Simple | Economic dispatch only |
| **Min Per-Country** | ⭐⭐⭐⭐ Good | ⭐⭐⭐⭐ Good | ⭐⭐ Easy | General purpose |
| **Equal Distribution** | ⭐⭐⭐⭐⭐ Perfect | ⭐ Poor | ⭐⭐ Easy | Policy analysis |
| **Focus Weights** | ⭐⭐⭐⭐ Good | ⭐⭐⭐ Tunable | ⭐⭐⭐ Moderate | Custom scenarios |
| **Optimized** | ⭐⭐⭐⭐ Good | ⭐⭐⭐⭐ Good | ⭐⭐⭐⭐ Complex | Research projects |

---

## Verification Against PyPSA-EUR

All implementations have been verified against PyPSA-EUR's clustering methodology:

### ✅ Verified Components

1. **`distribute_n_clusters_to_countries()`**
   - Matches PyPSA-EUR's mathematical formulation
   - Uses linopy for optimization
   - Supports focus_weights parameter
   - Source: `pypsa-eur/scripts/cluster_network.py`

2. **`busmap_for_n_clusters()`**
   - Groups by (country, sub_network)
   - Uses PyPSA's `busmap_by_kmeans()` within groups
   - Proper prefix handling for cluster names
   - Source: `pypsa-eur/scripts/cluster_network.py`

3. **`clustering_for_n_clusters()`**
   - Uses `get_clustering_from_busmap()` API
   - Preserves load assignments (not aggregated)
   - Aggregates lines/links properly
   - Source: `pypsa-eur/scripts/cluster_network.py`

4. **Stub removal**
   - Uses `busmap_by_stubs()` from PyPSA
   - Applies clustering mechanism for aggregation
   - Handles loads correctly (fixed from previous version)
   - Source: `pypsa-eur/scripts/simplify_network.py`

### 🔧 Key Differences from PyPSA-EUR

1. **Additional strategies**: We added min_per_country, equal, and hybrid approaches
2. **Convenience wrapper**: `cluster_network_with_strategy()` function simplifies usage
3. **Fallback options**: Strategies work without Gurobi (heuristic allocation)

### 📖 References

- PyPSA-EUR Documentation: https://pypsa-eur.readthedocs.io/
- PyPSA Clustering API: https://pypsa.readthedocs.io/en/latest/api_reference.html#clustering
- PyPSA-EUR Source Code: https://github.com/PyPSA/pypsa-eur

---

## Recommendations

### For EU27 + NO + CH + UK Analysis

**Recommended**: Strategy 1 (Min Per-Country) with `min_per_country=3-5`

**Rationale**:
- Ensures all 30 countries represented
- Respects economic reality (high-demand areas get more detail)
- Works without commercial solvers
- Good balance for policy analysis

**Example**:
```python
# 250 clusters, minimum 3 per country
n_clustered, busmap, allocation = netclust.cluster_network_with_strategy(
    n, 250, bus_loads, strategy="min_per_country", min_per_country=3
)
```

### For Renewable Integration Studies

**Recommended**: Strategy 4 (Focus Weights) boosting Spain, Portugal, Scandinavia

**Rationale**:
- Highlights regions with high wind/solar potential
- Maintains detail in areas with existing infrastructure
- Flexible for different renewable scenarios

**Example**:
```python
renewable_focus = {
    'ES': 2.0,  # Solar potential
    'PT': 2.0,  # Atlantic wind
    'SE': 1.5,  # Hydro + wind
    'NO': 1.5,  # Hydro reserves
    'DK': 1.8,  # Offshore wind
}

n_clustered, busmap, allocation = netclust.cluster_network_with_strategy(
    n, 250, bus_loads, strategy="focus", focus_weights=renewable_focus
)
```

### For Academic Research

**Recommended**: Strategy 5 (Optimized) with PyPSA-EUR's official method

**Rationale**:
- Mathematically optimal
- Reproducible (matches PyPSA-EUR)
- Can include focus weights
- Suitable for publication

**Example**:
```python
# Requires: conda install -c gurobi gurobi
n_clustered, busmap, allocation = netclust.cluster_network_with_strategy(
    n, 250, bus_loads, strategy="optimized"
)
```

---

## Validation Checklist

After clustering, always verify:

✅ **Load conservation**: `n.loads_t.p_set.sum().sum()` identical before/after  
✅ **Country coverage**: All target countries in `n.buses.country.unique()`  
✅ **Cluster distribution**: No country with 0 clusters (unless intended)  
✅ **Network connectivity**: No orphaned buses or isolated sub-networks  
✅ **Link preservation**: DC links maintained with correct endpoints  

---

## Troubleshooting

### "Optimization failed" error

**Cause**: Gurobi not installed or no valid license

**Solutions**:
1. Use fallback: `strategy="min_per_country"` instead
2. Install open-source solver: `conda install scip`
3. Get academic license: https://www.gurobi.com/academia/

### Clusters still concentrated despite strategy

**Cause**: Sub-networks have very few buses

**Solutions**:
1. Check `n.buses.groupby(['country','sub_network']).size()`
2. Increase `min_per_country` value
3. Use `strategy="equal"` for maximum distribution

### Some countries missing after clustering

**Cause**: Country has fewer buses than `min_per_country`

**Solutions**:
1. Reduce `min_per_country` value
2. Check if country was removed during simplification
3. Verify country codes in `n.buses.country`

---

## Files

**Implementation**: `scripts/network_clust.py`  
**Examples**: `notebooks/network_04A.ipynb`  
**Documentation**: This file

---

## Citation

If using these methods in research, please cite:

1. **PyPSA-EUR**:
   ```
   Jonas Hörsch et al., PyPSA-Eur: An Open Optimisation Model of the European 
   Transmission System, Energy Strategy Reviews, 2018.
   ```

2. **PyPSA**:
   ```
   Tom Brown et al., PyPSA: Python for Power System Analysis, 
   Journal of Open Research Software, 2018.
   ```

---

**Last Updated**: December 2025  
**Author**: Based on PyPSA-EUR methodology with extensions for geographic diversity
