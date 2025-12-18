# Critical Clustering Bug: Internal Lines Being Lost

## Date: 2024-12-18

## User Report

User correctly identified that internal lines are being lost during clustering:
- **UK**: 409 internal lines → 0 internal lines (100% loss)
- **Germany**: 654 internal lines → 42 internal lines (93.6% loss)
- **Finland, Sweden, Norway, Portugal, Denmark**: Similar massive losses

## Root Cause Identified

The clustering algorithm had a **CRITICAL BUG** in how it distributes clusters:

### The Problem

1. **Optimization distributes to (country, sub_network) tuples**  
   - Example: UK has 13 sub-networks (main grid + 12 islands)
   - Each gets 1 cluster = 13 total clusters for UK
   
2. **Main grid collapses to single cluster**
   - UK sub-network '2': 295 buses → 1 cluster
   - ALL 409 internal lines collapse to 0 lines
   - Same issue for other countries with multiple sub-networks

### Why This Happened

```python
# OLD BUGGY CODE:
L = cluster_weights.groupby([n.buses.country, n.buses.sub_network]).sum().pipe(normed)
N = n.buses.groupby(["country", "sub_network"]).size()[L.index]
```

This treats each (country, sub_network) as an independent unit to optimize. Result:
- Tiny island sub-networks get clusters
- Main grid sub-networks get squeezed down to 1 cluster
- 295 buses → 1 cluster = all internal topology lost

## Fix Applied

### Strategy 1: Two-Stage Distribution

**Changed from:** Optimize per (country, sub_network)  
**Changed to:** 
1. First optimize per COUNTRY
2. Then distribute within sub-networks proportionally

```python
# NEW FIXED CODE:
# Stage 1: Optimize per country
L_country = cluster_weights.groupby(n.buses.country).sum().pipe(normed)
N_country = n.buses.groupby("country").size()[L_country.index]

# ... optimization ...

# Stage 2: Distribute to sub-networks
for country in n_clusters_per_country.index:
    if country_clusters < num_subnets:
        # Give all to LARGEST subnet (by bus count)
        largest_subnet = subnet_sizes.idxmax()
    else:
        # Distribute proportionally among significant subnets
        ...
```

### Strategy 2: Smart Sub-Network Allocation

- **If fewer clusters than sub-networks**: Allocate ALL to largest subnet
- **If enough clusters**: Distribute proportionally by load
- **Filter out**: Tiny isolated single-bus sub-networks

## Results

### Before Fix (OLD Algorithm)

| Country | Buses | Internal Lines (Original) | Internal Lines (Clustered) | Retention % |
|---------|-------|--------------------------|----------------------------|-------------|
| GB      | 314   | 409                      | 0                          | **0.0%**    |
| NO      | 217   | 249                      | 0                          | **0.0%**    |
| SE      | 132   | 171                      | 0                          | **0.0%**    |
| FI      | 84    | 100                      | 0                          | **0.0%**    |
| DK      | 34    | 30                       | 0                          | **0.0%**    |
| FR      | 785   | 1175                     | 35                         | **3.0%**    |
| ES      | 561   | 783                      | 12                         | **1.5%**    |
| DE      | 479   | 654                      | 42                         | **6.4%**    |

### After Fix (NEW Algorithm)

| Country | Buses | Internal Lines (Original) | Internal Lines (Clustered) | Retention % | **Improvement** |
|---------|-------|--------------------------|----------------------------|-------------|-----------------|
| PL      | 147   | 224                      | 54                         | 24.1%       | **+12.5%** ✅   |
| SE      | 132   | 171                      | 23                         | 13.5%       | **+13.5%** ✅   |
| RO      | 91    | 113                      | 27                         | 23.9%       | **+17.7%** ✅   |
| ES      | 561   | 783                      | 39                         | 5.0%        | **+3.5%** ✅    |
| DE      | 479   | 654                      | 42                         | 6.4%        | (same)          |
| **GB**  | 314   | 409                      | **0**                      | **0.0%**    | **STILL BAD** ⚠️|
| **NO**  | 217   | 249                      | **0**                      | **0.0%**    | **STILL BAD** ⚠️|
| **FI**  | 84    | 100                      | **0**                      | **0.0%**    | **STILL BAD** ⚠️|
| **DK**  | 34    | 30                       | **0**                      | **0.0%**    | **STILL BAD** ⚠️|

**Summary:**
- ✅ **Significant improvement** for PL, SE, RO (now have internal lines)
- ⚠️ **Still problematic** for GB, NO, FI, DK (still 0% retention)

## Remaining Problem

### Why Some Countries Still Have 0% Retention

These countries get only **1 total cluster**:
- GB: 314 buses → 1 cluster → 0 internal lines
- NO: 217 buses → 1 cluster → 0 internal lines  
- FI: 84 buses → 1 cluster → 0 internal lines
- DK: 34 buses → 1 cluster → 0 internal lines

### Root Cause: Aggressive Focus Weights

Current settings (`network_04gurobi.ipynb`):
```python
focus_opt = {
    'PL': 0.15,  # 15% of 200 = 30 clusters
    'ES': 0.12,  # 12% of 200 = 24 clusters
    'RO': 0.08,  # 8% of 200 = 16 clusters
    'SE': 0.08,  # 8% of 200 = 16 clusters
    'PT': 0.06,  # 6% of 200 = 12 clusters
    'GR': 0.06,  # 6% of 200 = 12 clusters
}
# Total: 55% allocated to 6 countries
# Remaining 45% (90 clusters) shared by 22 other countries
```

**Result:** Small peripheral countries get squeezed down to 1 cluster each.

## Solutions to Consider

### Option 1: Increase Total Clusters

**Current:** `n_clusters_target = 200`  
**Recommended:** `n_clusters_target = 300` or `400`

**Pros:**
- More clusters to distribute
- Small countries get more than 1 cluster
- Better topology preservation

**Cons:**
- Larger problem size (but still manageable: 3954 → 400 is 90% reduction)
- Slightly longer optimization times

### Option 2: Adjust Focus Weights

**Current:** 55% allocated to 6 countries  
**Recommended:** Lower focus weights or add minimum constraints

```python
focus_opt = {
    'PL': 0.12,  # Reduce from 15%
    'ES': 0.10,  # Reduce from 12%
    'RO': 0.06,  # Reduce from 8%
    'SE': 0.06,  # Reduce from 8%
    'PT': 0.05,  # Reduce from 6%
    'GR': 0.05,  # Reduce from 6%
}
# Total: 44% (leaves 56% for others)
```

**Pros:**
- More balanced distribution
- Small countries get fair share

**Cons:**
- May underrepresent your focus countries

### Option 3: Enforce Minimum Clusters Per Country

Add constraint: `min_clusters_per_country = max(2, int(np.sqrt(num_buses)))`

Example:
- GB (314 buses): minimum 17 clusters
- NO (217 buses): minimum 14 clusters
- DE (479 buses): minimum 21 clusters

**Pros:**
- Guarantees minimum topology preservation
- Scales with country size

**Cons:**
- Requires modifying optimization constraints
- May conflict with focus weights

### Option 4: Different Clustering Approach

Instead of load-weighted k-means, use:
- **Modularity-based clustering**: Preserves network topology better
- **Hierarchical clustering**: Can enforce minimum clusters per region

## Recommendation

**Immediate:** Use Option 1 + Option 2 together:

```python
# In network_04gurobi.ipynb, modify:

n_clusters_target = 400  # Increase from 200

focus_opt = {
    'PL': 0.10,  # Slightly lower focus
    'ES': 0.09,
    'RO': 0.06,
    'SE': 0.06,
    'PT': 0.05,
    'GR': 0.05,
}
# Total: 41% to focus countries
# Remaining 59% (236 clusters) for 22 other countries = ~10-11 per country average
```

**Expected Result:**
- GB: ~10-15 clusters (enough for internal topology)
- NO: ~8-10 clusters
- FI: ~5-7 clusters
- DK: ~4-5 clusters
- All countries retain 10-30% of internal lines

## Implementation Status

✅ **FIXED:** Two-stage distribution algorithm  
✅ **FIXED:** Smart sub-network allocation  
✅ **IMPROVED:** Poland, Sweden, Romania line retention  
⚠️ **REMAINING:** GB, NO, FI, DK still need more clusters  

**Action Required:** User should decide on total cluster count and focus weights.

##Files Modified

1. **scripts/network_clust.py**  
   - Function: `distribute_n_clusters_to_countries()`
   - Changed: Two-stage optimization (country first, then sub-networks)
   - Changed: Smart subnet allocation based on bus count

2. **scripts/network_clust.py**  
   - Function: `busmap_for_n_clusters()` → `busmap_for_country()`
   - Changed: Handle missing sub-network entries gracefully

3. **notebooks/network_04gurobi.ipynb**
   - Added: Comprehensive analysis cells showing line retention
   - Added: Comparison before/after fix
   - Added: This documentation

## Testing

Run the analysis cells in `network_04gurobi.ipynb` to verify current state:
- Cell analyzing internal lines before clustering
- Cell analyzing internal lines after clustering  
- Cell comparing retention rates

Current test results show fix works but requires parameter tuning.

## References

- PyPSA-EUR cluster_network.py: https://github.com/PyPSA/pypsa-eur/blob/master/scripts/cluster_network.py
- User report: "In unclustered version of network, there are 409 lines that start and end in GB..."
- Analysis document: `docs/UK_CLUSTERING_ANALYSIS.md`
