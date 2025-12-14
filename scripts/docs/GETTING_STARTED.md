# Geometry Module - Getting Started

Welcome to the Geometry Module for European geographical data! This guide will help you get started quickly.

## 📦 What Was Created

This implementation provides a complete solution for working with European geographical data:

### Core Module (geometry.py)
- 700+ lines of production-ready code
- 13 main functions covering all requirements
- Automatic data caching
- Based on PyPSA-Eur best practices

### Four Required Functions ✅
1. **`join_shapes()`** - Join/union multiple boundaries into one
2. **`download_nuts3_shapes()`** - Download NUTS-3 regions (compatible with all operations)
3. **`point_in_shape()`** - Check if (lat, lon) is in boundary → boolean
4. **`mask_shape()`** - Intersection of boundaries (logical AND)

### Documentation (5 files)
1. **QUICK_REFERENCE.md** - ⭐ START HERE - Quick syntax reference
2. **README_geometry.md** - Complete user guide with examples
3. **SUMMARY.md** - Overview and feature list
4. **ARCHITECTURE.md** - Technical details and diagrams
5. **CHECKLIST.md** - Implementation status

### Examples & Tests (3 files)
1. **examples_geometry.py** - 7 comprehensive usage examples
2. **test_geometry.py** - Automated test suite
3. **bus_filtering.py** - PyPSA integration utilities

### Tutorial
- **notebooks/geometry_tutorial.ipynb** - Interactive Jupyter notebook

## 🚀 Quick Start (3 steps)

### Step 1: Install Dependencies
```bash
pip install -r requirements-dev.txt
```

### Step 2: Run Tests
```bash
cd scripts
python test_geometry.py
```

### Step 3: Try Examples
```bash
python examples_geometry.py
```

## 💡 First Use

```python
from scripts.geometry import (
    download_country_shapes,
    join_shapes,
    point_in_shape,
)

# Download Germany and Poland
countries = download_country_shapes(['DE', 'PL'])

# Join them into one boundary
combined = join_shapes(countries)

# Check if Berlin is in the combined region
berlin_in = point_in_shape(52.5200, 13.4050, combined)
print(f"Berlin in DE+PL: {berlin_in}")  # True
```

## 📚 Where to Look

- **Want quick syntax?** → `QUICK_REFERENCE.md`
- **Need complete guide?** → `README_geometry.md`
- **Want to see examples?** → Run `examples_geometry.py`
- **Interactive learning?** → Open `geometry_tutorial.ipynb`
- **Technical details?** → `ARCHITECTURE.md`
- **Integration with PyPSA?** → `bus_filtering.py`

## ✨ Key Features

### 1. Smart Caching
```python
# First time: downloads from Eurostat (~5-10 seconds)
countries = download_country_shapes(['DE'])

# Second time: loads from cache (<0.1 seconds)
countries = download_country_shapes(['DE'])
```

### 2. Four Core Functions (As Requested)

#### Join Shapes (Union)
```python
# Combine Germany and Poland into single boundary
countries = download_country_shapes(['DE', 'PL'])
combined = join_shapes(countries)
```

#### NUTS-3 Regions
```python
# Download NUTS-3 statistical regions
nuts3 = download_nuts3_shapes(['DE'])
# Compatible with all operations!
combined_nuts3 = join_shapes(nuts3)
```

#### Point in Shape
```python
# Check if coordinate is within boundary
is_in = point_in_shape(lat=52.52, lon=13.40, shape=germany)
# Returns: True or False
```

#### Mask Shape (Intersection)
```python
# Get intersection of two boundaries (AND operation)
intersection = mask_shape(shape1, shape2)
```

### 3. Bonus Features
- Buffer operations (expand/contract by distance)
- Area calculations (km², m², hectares)
- Geometry simplification (for performance)
- EU boundary (one function call)

## 🎯 Common Use Cases

### Filter Power Network Buses by Country
```python
import pandas as pd
from scripts.geometry import download_country_shapes, point_in_shape

# Load buses
buses = pd.read_csv('data/raw/OSM Prebuilt Electricity Network/buses.csv')

# Get Germany boundary
germany = download_country_shapes(['DE'])

# Filter
buses['in_germany'] = buses.apply(
    lambda row: point_in_shape(row['y'], row['x'], germany),
    axis=1
)
german_buses = buses[buses['in_germany']]
```

### Multi-Country Analysis
```python
# Define study region
countries = download_country_shapes(['DE', 'PL', 'CZ'])
study_area = join_shapes(countries)

# Get area
from scripts.geometry import get_shape_area
area = get_shape_area(study_area)
print(f"Study area: {area:,.0f} km²")
```

### Regional Analysis with NUTS-3
```python
# Get NUTS-3 regions
nuts3 = download_nuts3_shapes(['DE'])

# Analyze by region
for idx, region in nuts3.iterrows():
    region_id = region['NUTS_ID']
    region_name = region['NAME_LATN']
    # Your analysis here...
```

## 🔧 Data Sources

All data comes from **Eurostat GISCO** (official EU geographical data):
- Countries: CNTR_RG_01M_2020
- NUTS regions: NUTS_RG_01M_2021

Cached locally in: `data/cache/geometry/`

## ⚡ Performance

| Operation | Speed | Notes |
|-----------|-------|-------|
| First download | ~5-10s | From Eurostat |
| Cached load | <0.1s | From local file |
| join_shapes() | <0.1s | Fast unary_union |
| point_in_shape() | <0.01s | Per point |
| mask_shape() | ~0.1s | Intersection |

## 🧪 Testing

All functions are tested:
```bash
python scripts/test_geometry.py
```

Expected output:
```
Testing imports... ✓ PASS
Testing country download... ✓ PASS
Testing point-in-shape... ✓ PASS
Testing join shapes... ✓ PASS
Testing NUTS-3 download... ✓ PASS
Testing mask/intersection... ✓ PASS

ALL TESTS PASSED (6/6)
```

## 📖 Documentation Structure

```
scripts/
├── geometry.py                 # Core module
│
├── GETTING_STARTED.md         # ⭐ This file
├── QUICK_REFERENCE.md         # Quick syntax guide
├── README_geometry.md         # Complete documentation
├── SUMMARY.md                 # Feature overview
├── ARCHITECTURE.md            # Technical details
├── CHECKLIST.md               # Implementation status
│
├── examples_geometry.py       # Usage examples
├── test_geometry.py           # Test suite
└── bus_filtering.py           # PyPSA integration
```

## 🤔 Troubleshooting

### Downloads fail?
- Check internet connection
- Eurostat servers may be temporarily down
- Use cached data if available

### Import errors?
```bash
pip install geopandas shapely requests pandas
```

### Need help?
1. Check `QUICK_REFERENCE.md` for syntax
2. Review `examples_geometry.py` for patterns
3. Run `test_geometry.py` to verify installation
4. Read `README_geometry.md` for detailed docs

## 🎓 Learning Path

1. **Start**: Read this file (GETTING_STARTED.md)
2. **Quick syntax**: Check QUICK_REFERENCE.md
3. **Try it**: Run examples_geometry.py
4. **Test it**: Run test_geometry.py
5. **Interactive**: Open geometry_tutorial.ipynb
6. **Deep dive**: Read README_geometry.md
7. **Integration**: Study bus_filtering.py

## ✅ What's Implemented

All four required functions:
- ✅ Joining shapes (union operation)
- ✅ Downloading NUTS-3 regions
- ✅ Point-in-polygon checking
- ✅ Boundary intersection (masking)

Plus additional features:
- ✅ Automatic data caching
- ✅ Country shape downloads
- ✅ Buffer operations
- ✅ Area calculations
- ✅ Geometry simplification
- ✅ EU boundary helper
- ✅ PyPSA integration utilities

## 🚀 Next Steps

Choose your path:

**For quick start:**
1. Open `QUICK_REFERENCE.md`
2. Run `examples_geometry.py`
3. Start coding!

**For thorough learning:**
1. Read `README_geometry.md`
2. Work through `geometry_tutorial.ipynb`
3. Study `bus_filtering.py`

**For integration:**
1. Check `bus_filtering.py` for patterns
2. Adapt to your use case
3. Refer to `QUICK_REFERENCE.md` as needed

## 💬 Questions?

- **Syntax?** → QUICK_REFERENCE.md
- **How to use?** → README_geometry.md
- **Examples?** → examples_geometry.py
- **Technical?** → ARCHITECTURE.md
- **PyPSA integration?** → bus_filtering.py

## 🎉 You're Ready!

The module is fully functional and production-ready. All requirements are met, and the code follows best practices from PyPSA-Eur.

**Start with:**
```bash
python scripts/examples_geometry.py
```

Happy coding! 🚀
