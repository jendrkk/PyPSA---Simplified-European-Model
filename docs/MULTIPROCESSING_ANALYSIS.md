# Multiprocessing Analysis for add_generators.py

## Executive Summary

**Key Findings:**
- ✅ **Wind profile generation**: ~350 NUTS-2 regions × ~10,000 timesteps = **High benefit**
- ✅ **Solar profile generation**: ~350 NUTS-2 regions × ~10,000 timesteps = **High benefit**
- ⚠️ **Generator-level loops**: Limited benefit due to small iteration counts and fast operations
- ❌ **Network modifications**: Not applicable (pypsa.Network is not thread-safe)

**Recommendation:** Implement multiprocessing for NUTS-2 region-level profile generation. Estimated speedup: **3-5x** on typical hardware.

---

## Detailed Analysis

### 1. Wind Profile Generation (`_create_wind_profiles()`)

**Current Implementation:**
```python
for nuts2_id in nuts2_regions:  # ~350 iterations
    ws_col = f'ws_{nuts2_id}'
    wind_speed = weather[ws_col].values  # ~87,000 timesteps (10 years hourly)
    cf = wind_power_curve(wind_speed)  # NumPy vectorized operation
    
    region_gens = wind_gens[wind_gens['nuts2'] == nuts2_id]  # ~27 generators per region avg
    for idx, gen in region_gens.iterrows():
        gen_cf = cf.copy()  # Fast array copy
        if is_offshore:
            gen_cf = np.minimum(gen_cf * 1.25, 1.0)
        profiles[gen_idx] = gen_cf
```

**Bottleneck Analysis:**
- **Outer loop (NUTS-2 regions)**: ~350 iterations × ~0.02s = **7 seconds**
  - Compute-bound: `wind_power_curve()` processes ~87,000 values
  - Highly parallelizable: Each region is independent
  - Data size: ~350 KB per region (87,000 × 4 bytes)
  
- **Inner loop (generators)**: ~27 iterations × ~0.0001s = **0.003 seconds**
  - Memory-bound: Array copies and assignment
  - Not worth parallelizing (overhead > benefit)

**Multiprocessing Benefits:**
- ✅ **CPU utilization**: Each region processes independently
- ✅ **Data locality**: Weather data can be shared (read-only)
- ✅ **Scalability**: Linear speedup with CPU cores (up to ~8 cores)
- ⚠️ **Overhead**: ~50ms startup + data serialization (~5MB total)

**Expected Speedup:**
- 4 cores: **3.5x** (7s → 2s)
- 8 cores: **6.0x** (7s → 1.2s)
- 16 cores: **7.5x** (7s → 0.9s, diminishing returns)

**Implementation Complexity:** Medium
- Need to partition NUTS-2 regions across processes
- Combine results into single DataFrame
- Handle edge cases (missing data, offshore factors)

---

### 2. Solar Profile Generation (`_create_solar_profiles()`)

**Current Implementation:**
```python
for nuts2_id in nuts2_regions:  # ~350 iterations
    rad_col = find_radiation_column(nuts2_id)  # Fast lookup
    irradiance = weather[rad_col].values  # ~87,000 timesteps
    cf = solar_capacity_factor(irradiance)  # NumPy vectorized operation
    
    region_gens = solar_gens[solar_gens['nuts2'] == nuts2_id]  # ~35 generators per region avg
    for idx, gen in region_gens.iterrows():
        profiles[gen_idx] = cf
```

**Bottleneck Analysis:**
- **Outer loop (NUTS-2 regions)**: ~350 iterations × ~0.015s = **5 seconds**
  - Compute-bound: `solar_capacity_factor()` processes ~87,000 values
  - Highly parallelizable: Each region is independent
  - Slightly faster than wind due to simpler calculation
  
- **Inner loop (generators)**: ~35 iterations × ~0.00005s = **0.002 seconds**
  - Memory-bound: Simple array assignment
  - Not worth parallelizing

**Multiprocessing Benefits:**
- ✅ **CPU utilization**: Similar to wind profiles
- ✅ **Simpler logic**: No offshore factors
- ✅ **Good speedup**: Similar to wind profiles

**Expected Speedup:**
- 4 cores: **3.5x** (5s → 1.4s)
- 8 cores: **6.0x** (5s → 0.8s)

**Implementation Complexity:** Medium (similar to wind)

---

### 3. Main Loop in `add_all_generators()`

**Current Implementation:**
```python
# NOT PARALLELIZABLE - modifies network object
if wind_profiles is not None:
    add_renewable_generators_aggregated(network, wind_gens, wind_profiles, 'wind')
if solar_profiles is not None:
    add_renewable_generators_aggregated(network, solar_gens, solar_profiles, 'solar')
add_generators_with_profiles(network, conv_gens, profiles=None)
```

**Analysis:**
- ❌ **Cannot parallelize**: `pypsa.Network` is not thread-safe
- ❌ **Sequential dependencies**: Each step modifies the network state
- ❌ **Already optimized**: Uses pandas/numpy operations internally

**Conclusion:** No multiprocessing applicable here.

---

## Profiling Results (Estimated)

### Current Performance (Single-Core)
```
Step                              Time     % Total
─────────────────────────────────────────────────
Load power plants                0.5s     3%
Load weather data                1.2s     7%
Load NUTS-2 shapes               0.3s     2%
Create wind profiles             7.0s     40%  ← BOTTLENECK
Create solar profiles            5.0s     29%  ← BOTTLENECK
Add renewable generators         2.5s     14%
Add conventional generators      0.8s     5%
─────────────────────────────────────────────────
TOTAL                           17.3s    100%
```

### With Multiprocessing (4 cores)
```
Step                              Time     % Total  Speedup
───────────────────────────────────────────────────────────
Load power plants                0.5s     6%       1.0x
Load weather data                1.2s     14%      1.0x
Load NUTS-2 shapes               0.3s     4%       1.0x
Create wind profiles             2.0s     24%      3.5x ✅
Create solar profiles            1.4s     17%      3.6x ✅
Add renewable generators         2.5s     30%      1.0x
Add conventional generators      0.8s     9%       1.0x
─────────────────────────────────────────────────────────
TOTAL                            8.7s    100%      2.0x ✅
```

**Overall Speedup: ~2x** (17.3s → 8.7s)

---

## Implementation Strategy

### Option 1: ProcessPoolExecutor (Recommended)

**Pros:**
- Built-in Python `concurrent.futures` module
- Simple API: `executor.map()`
- Automatic process management
- Good for I/O + CPU mixed workloads

**Cons:**
- Pickle overhead for large DataFrames
- Fixed process pool size

**Example:**
```python
from concurrent.futures import ProcessPoolExecutor
from functools import partial

def _process_wind_region(nuts2_id, weather, wind_gens, offshore_factor):
    """Process one NUTS-2 region (must be top-level function for pickling)."""
    ws_col = f'ws_{nuts2_id}'
    if ws_col not in weather.columns:
        return nuts2_id, None
    
    wind_speed = weather[ws_col].values
    cf = wind_power_curve(wind_speed)
    
    region_gens = wind_gens[wind_gens['nuts2'] == nuts2_id]
    region_profiles = {}
    
    for idx, gen in region_gens.iterrows():
        gen_idx = gen.name if hasattr(gen, 'name') else idx
        gen_cf = cf.copy()
        if is_offshore(gen):
            gen_cf = np.minimum(gen_cf * offshore_factor, 1.0)
        region_profiles[gen_idx] = gen_cf
    
    return nuts2_id, region_profiles

def _create_wind_profiles_parallel(wind_gens, weather, ws_cols, offshore_factor, 
                                   n_jobs=4, verbose=True):
    """Parallel version of _create_wind_profiles()."""
    nuts2_regions = wind_gens['nuts2'].dropna().unique()
    
    # Parallel processing
    process_func = partial(_process_wind_region, 
                          weather=weather, 
                          wind_gens=wind_gens, 
                          offshore_factor=offshore_factor)
    
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        results = list(executor.map(process_func, nuts2_regions))
    
    # Combine results
    profiles = pd.DataFrame(index=weather.index)
    for nuts2_id, region_profiles in results:
        if region_profiles is not None:
            for gen_idx, cf in region_profiles.items():
                profiles[gen_idx] = cf
    
    return profiles
```

### Option 2: Multiprocessing.Pool

**Pros:**
- More control over process management
- Can use shared memory with `multiprocessing.Manager()`
- Better for NumPy arrays

**Cons:**
- More complex API
- Manual cleanup required

### Option 3: Dask (Over-Engineering?)

**Pros:**
- Scales to distributed computing
- Lazy evaluation
- Great for very large datasets

**Cons:**
- Heavy dependency
- Overkill for this use case
- Steeper learning curve

**Recommendation:** **Option 1 (ProcessPoolExecutor)** for simplicity and maintenance.

---

## Implementation Checklist

### Phase 1: Add Multiprocessing Support (Optional)
- [ ] Add `n_jobs` parameter to `add_all_generators()`
- [ ] Create `_process_wind_region()` helper function
- [ ] Create `_process_solar_region()` helper function
- [ ] Implement `_create_wind_profiles_parallel()`
- [ ] Implement `_create_solar_profiles_parallel()`
- [ ] Add fallback to single-core if `n_jobs=1`
- [ ] Test with small dataset (1 week)
- [ ] Benchmark with full dataset (10 years)

### Phase 2: Optimization Details
- [ ] Use `chunksize` parameter for better load balancing
- [ ] Consider `initializer` for one-time setup (load weather once)
- [ ] Add progress bar with `tqdm` for long operations
- [ ] Handle exceptions gracefully in worker processes

### Phase 3: Testing
- [ ] Verify identical results between serial and parallel versions
- [ ] Test with different `n_jobs` values
- [ ] Measure actual speedup on target hardware
- [ ] Test edge cases (missing NUTS-2, offshore, no generators)

---

## Cost-Benefit Analysis

### Benefits
- ✅ **Speed**: 2-3x faster on typical hardware (4-8 cores)
- ✅ **Scalability**: Better performance for large networks
- ✅ **User experience**: Faster iterations during development

### Costs
- ⚠️ **Complexity**: +150 lines of code
- ⚠️ **Testing**: Need to verify parallel correctness
- ⚠️ **Debugging**: Harder to debug parallel code
- ⚠️ **Dependencies**: May need `tqdm` for progress bars
- ⚠️ **Platform**: Windows requires `if __name__ == '__main__'` guard

### Recommendation

**For Production Use:**
- ✅ **Implement** if network has >100 buses or >10 years of data
- ✅ **Implement** if this code runs frequently (e.g., batch processing)

**For Academic/Learning Project:**
- ⚠️ **Optional** - adds complexity without critical need
- ✅ **Good learning opportunity** for parallel computing concepts
- ⚠️ Current single-core performance (~17s) is acceptable for interactive work

---

## Alternative Optimizations (Easier Wins)

### 1. Vectorize Inner Loops
Current generator-level loop could be simplified:
```python
# Instead of looping through generators
for idx, gen in region_gens.iterrows():
    profiles[gen_idx] = cf

# Use vectorized assignment
gen_indices = region_gens.index
profiles[gen_indices] = cf.values[:, np.newaxis]  # Broadcast to all generators
```
**Benefit:** ~20% speedup in inner loop (negligible overall)

### 2. Cache NUTS-2 Assignments
```python
# Pre-compute NUTS-2 → generator mapping
nuts2_to_gens = wind_gens.groupby('nuts2').groups
for nuts2_id in nuts2_regions:
    gen_indices = nuts2_to_gens.get(nuts2_id, [])
    # ... process all generators at once
```
**Benefit:** ~10% speedup (avoid repeated filtering)

### 3. Use Numba JIT Compilation
```python
from numba import jit

@jit(nopython=True)
def wind_power_curve_jit(wind_speed):
    # ... same logic but compiled to machine code
```
**Benefit:** ~2x speedup for power curve calculation
**Cost:** Low - only add `@jit` decorator

---

## Conclusion

**Primary Recommendation:** Multiprocessing offers **significant benefits** (2-3x speedup) for profile generation but adds moderate complexity.

**Secondary Recommendation:** Before implementing multiprocessing, consider:
1. **Numba JIT** for power curve functions (easy win)
2. **Vectorization** improvements (small win, low risk)
3. Profile actual performance on target hardware

**Decision Point:**
- If runtime < 30 seconds: **Skip multiprocessing** (not worth complexity)
- If runtime > 30 seconds: **Implement multiprocessing** (good ROI)
- If runtime > 5 minutes: **Definitely implement** (critical for usability)

Current estimated runtime (~17s) is **borderline**. Recommend starting with simpler optimizations first.
