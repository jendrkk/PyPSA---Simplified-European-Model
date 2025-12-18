# Clustering Bug Fix Summary

## Problem
After clustering, UK (GB) and other countries had **0 internal lines** - a critical bug that destroyed network topology.

## Root Cause Analysis
Comparing our implementation with PyPSA-EUR's official `cluster_network.py`, we identified **THREE bugs**:

### Bug 1: Focus Weights Not Divided by Sub-Network Count (CRITICAL)

**PyPSA-EUR (correct):**
```python
for country, weight in focus_weights.items():
    L[country] = weight / len(L[country])  # Divides by number of sub-networks!
```

**Our code (buggy):**
```python
for country, weight in focus_weights.items():
    L_country[country] = weight  # No division!
```

**Impact:** Countries with many sub-networks (UK has 13) got their entire focus weight concentrated in one sub-network instead of distributed across all.

### Bug 2: Two-Stage Distribution Was Wrong Approach

Our "fix" created a two-stage process:
1. Optimize clusters per COUNTRY
2. Distribute within sub-networks

**But PyPSA-EUR intentionally optimizes per (country, sub_network) in ONE stage.** The two-stage approach broke the optimization model's ability to balance weights properly.

### Bug 3: No Minimum Cluster Guarantee

When a country has many sub-networks but gets few clusters (due to focus_weights giving other countries priority), its main sub-network could collapse to 1 cluster, losing all internal topology.

## The Fix

Rewrote `distribute_n_clusters_to_countries()` to **exactly match PyPSA-EUR**:

```python
def distribute_n_clusters_to_countries(
    n, n_clusters, cluster_weights, 
    focus_weights=None, solver_name="gurobi",
    min_clusters_per_country=1  # NEW: ensures minimum clusters
):
    # EXACT PyPSA-EUR: Optimize per (country, sub_network)
    L = (
        cluster_weights.groupby([n.buses.country, n.buses.sub_network])
        .sum()
        .pipe(normed)
    )
    N = n.buses.groupby(["country", "sub_network"]).size()[L.index]
    
    if isinstance(focus_weights, dict):
        for country, weight in focus_weights.items():
            # CRITICAL: Divide by number of sub-networks
            L[country] = weight / len(L[country])
        
        remainder = [c not in focus_weights.keys() for c in L.index.get_level_values("country")]
        L[remainder] = L.loc[remainder].pipe(normed) * (1 - total_focus)
    
    # ENHANCEMENT: Boost main sub-networks to ensure minimum clusters
    if min_clusters_per_country > 1:
        main_subnets = N.groupby(level='country').idxmax()
        for country, (c, s) in main_subnets.items():
            min_fraction = min_clusters_per_country / n_clusters
            if L[(c, s)] < min_fraction:
                L[(c, s)] = min_fraction
        L = L.pipe(normed)
    
    # Standard PyPSA-EUR optimization model
    m = linopy.Model()
    n_var = m.add_variables(lower=1, upper=N, coords=[L.index], name="n", integer=True)
    m.add_constraints(n_var.sum() == n_clusters, name="tot")
    m.objective = (n_var * n_var - 2 * n_var * L * n_clusters).sum()
    m.solve(solver_name=solver_name)
    return m.solution["n"].to_series().astype(int)
```

## Key Changes

1. **Single-stage optimization** per (country, sub_network) - matches PyPSA-EUR exactly
2. **Focus weights divided by sub-network count** - ensures fair distribution
3. **New `min_clusters_per_country` parameter** - prevents main grids from collapsing
4. **Proper fallback method** - uses same (country, sub_network) logic

## Recommended Settings

For good line retention:
- `n_clusters_target = 400` (or higher)
- `min_clusters_per_country = 2-3`
- Don't over-focus: keep `sum(focus_weights) <= 0.5`

```python
focus_weights = {
    'PL': 0.08, 'ES': 0.07, 'RO': 0.06, 'SE': 0.06,
    'PT': 0.04, 'GR': 0.04, 'GB': 0.05, 'NO': 0.04
}  # Total: 44%

n_clusters_c = netclust.distribute_n_clusters_to_countries(
    n, 400, bus_weights,
    focus_weights=focus_weights,
    solver_name='gurobi',
    min_clusters_per_country=3
)
```

## Files Modified

- **scripts/network_clust.py**: Complete rewrite of `distribute_n_clusters_to_countries()`
- **notebooks/network_04gurobi_clean.ipynb**: New clean notebook with proper workflow

## Verification

After fix, UK should have:
- Multiple clusters (not 1)
- Internal lines > 0
- Line retention > 10% (ideally 15-25%)

Run the clean notebook to verify the fix works.
