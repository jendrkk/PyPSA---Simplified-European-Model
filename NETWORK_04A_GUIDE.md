# Complete Guide to network_04A.ipynb - Testing All Clustering Strategies

## 🎯 Purpose of This Notebook

This notebook implements **5 alternative clustering strategies** to ensure better geographic coverage of peripheral European countries (Eastern Europe, Scandinavia, Iberia) compared to the default load-weighted approach.

## 🔧 Critical Fix Applied

**Bug Found and Fixed**: In `network_clust.py` line 593:
```python
# BEFORE (WRONG):
prefix = x.name[0] + x.name[1] + " "  # Fails when sub_network is integer!

# AFTER (CORRECT):
prefix = x.name[0] + str(x.name[1]) + " "  # Converts sub_network to string
```

This was causing clusters from different countries to be merged because the prefix wasn't being created correctly.

---

## 📋 Step-by-Step Testing Guide

### Step 1: Setup and Load Network (Cells 2-4)
**What it does**: Loads the simplified network and calculates bus weights

**Run these cells:**
- Cell 2: Import libraries and set JOIN/FLOAT_ parameters
- Cell 3: Load network
- Cell 4: Calculate load statistics

**Expected output**: 
```
Total countries: 28
Total buses: ~4000
Total load: ~3000 TWh
```

---

### Step 2: Strategy 1 - Per-Country Minimum Allocation (Cells 7)

**Goal**: Guarantee every country gets at least 3 clusters before distributing the rest by load

**How it works:**
1. Give each country `min_per_country` clusters (default: 3)
2. Distribute remaining clusters proportionally by load
3. Ensures peripheral countries aren't ignored

**Run Cell 7 and expect:**
```
Strategy 1: Per-Country Minimum Allocation
Target clusters: 250
Total clusters allocated: 250
```

**What to check:**
- All 28 countries should appear in the output
- Large countries (DE, FR, IT) should have 15-25 clusters
- Small countries (LU, MT, CY) should have exactly 3 clusters
- ✅ **NO errors about country disagreement**

---

### Step 3: Strategy 2 - Hybrid Geographic-Load Weighting (Cell 10)

**Goal**: Balance geographic spread with demand using α parameter

**How it works:**
- α = 0.0: Pure load-weighted (baseline)
- α = 0.5: Equal balance
- α = 1.0: Pure geographic (ignores load)

**Run Cell 10 and expect:**
- 5 different configurations tested (α = 0.0, 0.3, 0.5, 0.7, 1.0)
- Each should complete without errors
- Higher α values should give more clusters to peripheral countries

**What to check:**
- Compare cluster distributions across α values
- Notice how Eastern Europe gets more clusters as α increases
- All networks should preserve total load

---

### Step 4: Strategy 3 - Equal Geographic Distribution (Cell 12)

**Goal**: Treat all countries equally, ignoring load completely

**How it works:**
- Divide 250 clusters equally across all (country, sub_network) pairs
- Gives ~2-3 clusters per group on average

**Run Cell 12 and expect:**
```
Strategy 3: Equal Geographic Distribution
Mean per country: ~9 clusters
Std per country: ~5-6 clusters
```

**What to check:**
- Very balanced distribution
- Even small countries well-represented
- May oversample low-demand regions

---

### Step 5: Strategy 4 - Focus Weights (Cell 14)

**Goal**: Manually boost specific countries (e.g., PL, RO, ES, SE)

**How it works:**
- Apply multipliers to specific countries (e.g., PL: 2.5x, ES: 2.0x)
- Ensures targeted regions get more representation

**Run Cell 14 and expect:**
```
Strategy 4: Focus Weights
Countries boosted: PL, RO, ES, PT, SE, FI, GR, BG, CZ, HU
```

**What to check:**
- Poland, Romania, Spain should have 10+ clusters each
- Total still equals 250
- Western Europe slightly reduced to make room

---

### Step 6: Strategy 5 - Gurobi Optimization (Cell 20)

**Goal**: Use integer quadratic programming to find optimal distribution

**How it works:**
```
minimize:   Σ (n_c - L_c * N)²
subject to: Σ n_c = N
            1 <= n_c <= N_c
```

**Run Cell 20 and expect:**
```
Strategy 5: Optimization-Based Distribution
Gurobi: Academic license - for non-commercial use only
Optimization successful!
Clusters: 250 allocated across 95 groups
```

**What to check:**
- Gurobi solver succeeds (you have a valid license)
- Focus weights applied (PL: 15%, ES: 12%, etc.)
- Final distribution mathematically optimal

**If Gurobi fails:**
- Check license file exists: `~/gurobi.lic`
- Or use SCIP: Change `solver_name='gurobi'` to `solver_name='scip'`

---

### Step 7: Visual Comparison (Cells 16-17)

**Goal**: Compare all strategies side-by-side

**Run Cells 16-17 to generate:**
1. Geographic maps showing cluster distributions
2. Heatmap comparing clusters per country by strategy
3. Statistics table

**What to look for:**
- **Baseline**: Clusters concentrated in Western Europe
- **Min-3**: Better coverage, but still load-biased
- **Equal**: Perfectly balanced geographically
- **Focus**: Targeted boost to peripheral countries
- **Optimized**: Balanced while respecting load distribution

---

### Step 8: Choose and Save Your Preferred Strategy

**Based on your research question:**

| Research Question | Recommended Strategy |
|-------------------|---------------------|
| Pure economic dispatch | Baseline (load-only) |
| Balanced European model | Strategy 1 (Min-3) or Strategy 5 (Optimized) |
| Policy analysis (all countries equal) | Strategy 3 (Equal) |
| Renewable integration study | Strategy 4 (Focus on ES, PT, SE) |
| Eastern European integration | Strategy 4 (Focus on PL, CZ, RO, HU) |

**To save your chosen network:**
```python
# Example: Save Strategy 5 result
strategy_name = "gurobi"  # or "minalloc", "equal", "focus", "hybrid"
save_path = repo_root / "data" / "networks" / "clustered" / 
            f"C+_sEEN{'_join' if JOIN else ''}{'_f' if FLOAT_ else ''}_cl{n_clusters_target}_{strategy_name}.nc"
n_s5.export_to_netcdf(save_path)
print(f"Saved: {save_path}")
```

---

## ✅ Expected Results After Fixing

After the `str(x.name[1])` fix, **ALL strategies should work without errors!**

**Before fix:**
```
ValueError: In Bus cluster country, the values of attribute country do not agree
```

**After fix:**
```
✅ Clustering complete!
✅ Buses: 4000 → 250
✅ All clusters respect country boundaries
✅ Load perfectly conserved
```

---

## 🐛 Troubleshooting

### Issue: "Gurobi license not found"
**Solution:** 
```bash
# Check if license exists
ls ~/gurobi.lic

# If not, download from https://www.gurobi.com/academia/
# Then set environment variable:
export GRB_LICENSE_FILE=~/gurobi.lic
```

### Issue: "ImportError: No module named linopy"
**Solution:**
```bash
pip install linopy
```

### Issue: Still getting country disagreement errors
**Solution:**
1. Reload the module: `importlib.reload(netclust)`
2. Verify the fix is applied: Check line 593 in `network_clust.py`
3. Restart the Jupyter kernel

### Issue: Fewer clusters than requested (e.g., 119 instead of 250)
**Explanation:** This is EXPECTED and not an error!
- Many sub_networks have very few buses (islands, small regions)
- A sub_network with 1 bus can only create 1 cluster
- Total clusters will be less than target due to physical constraints
- This is the same behavior as the original PyPSA-EUR implementation

---

## 📊 Next Steps After Clustering

Once you have a clustered network:

1. **Add Generators**: Attach power plants from `powerplants.csv`
2. **Add Renewables**: Attach solar/wind with capacity factors
3. **Load Time Series**: Add renewable generation profiles
4. **Run Optimization**: Solve dispatch problem with `n.lopf()`
5. **Analyze Results**: Study flows, prices, emissions

---

## 🎓 Key Learnings

1. **Default k-means concentrates clusters** in high-demand regions due to load weighting
2. **Pre-allocation strategies** ensure geographic diversity
3. **PyPSA's clustering API** automatically handles all component remapping
4. **Integer programming** provides mathematically optimal distribution
5. **Focus weights** allow targeted control for specific countries

---

## 📚 Additional Resources

- PyPSA Documentation: https://pypsa.readthedocs.io/
- PyPSA-EUR Repository: https://github.com/PyPSA/pypsa-eur
- Gurobi Academic License: https://www.gurobi.com/academia/
- SCIP Solver (open-source): https://www.scipopt.org/

---

**Ready to run?** Start from Cell 2 and work through each strategy sequentially!
