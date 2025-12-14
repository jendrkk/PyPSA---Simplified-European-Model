# Geometry Module - Summary

## Overview

A comprehensive Python module for working with European geographical data, designed for PyPSA energy system modeling and analysis. The module provides efficient access to country boundaries and NUTS-3 regions with automatic caching and powerful geometry operations.

## Files Created

### Core Module
- **`geometry.py`** - Main module with all geometry functions
  - 700+ lines of well-documented code
  - Based on PyPSA-Eur best practices
  - Automatic data caching
  - Efficient geometry operations

### Documentation
- **`README_geometry.md`** - Complete user guide
  - Installation instructions
  - API reference
  - Usage examples
  - Troubleshooting guide

### Examples & Tests
- **`examples_geometry.py`** - 7 comprehensive examples demonstrating all features
- **`test_geometry.py`** - Automated test suite
- **`bus_filtering.py`** - Practical utilities for filtering power network buses

### Configuration
- **`requirements-dev.txt`** - Updated with geometry dependencies

## Key Features Implemented

### ✅ 1. Data Download & Caching
- Downloads geographical data from Eurostat GISCO API
- Automatic local caching (no re-download needed)
- Force re-download option available
- Fast access to cached data

### ✅ 2. Joining Shapes (Union Operation)
```python
# Join Germany and Poland into single boundary
countries = download_country_shapes(['DE', 'PL'])
combined = join_shapes(countries)
```

### ✅ 3. NUTS-3 Region Support
```python
# Download NUTS-3 regions (compatible with country shapes)
nuts3 = download_nuts3_shapes(['DE', 'FR'])
# Can use same operations as country shapes
combined_nuts3 = join_shapes(nuts3)
```

### ✅ 4. Point-in-Polygon Checking
```python
# Check if Berlin is in Germany
is_in_germany = point_in_shape(lat=52.5200, lon=13.4050, shape=germany)
# Returns: True
```

### ✅ 5. Boundary Intersection (Masking)
```python
# Get intersection of two boundaries (logical AND)
overlap = mask_shape(shape1, shape2)
```

## Additional Utilities

Beyond the core requirements, the module includes:

- **Buffer operations**: Expand/contract boundaries by distance
- **Area calculation**: Get area in km², m², or hectares  
- **Simplification**: Reduce geometry complexity for performance
- **EU boundary**: One-line access to unified EU27 shape

## Performance Optimizations

1. **Local Caching**: Data downloaded once and reused
2. **Efficient Projections**: Uses EPSG:3035 for accurate distance/area calculations
3. **Shapely Operations**: Fast unary_union for combining geometries
4. **Optional Simplification**: Reduce vertices for complex shapes

## Data Sources

All data from **Eurostat GISCO** (official EU geographical data):
- Countries: CNTR_RG_01M_2020 (1:1 million scale)
- NUTS regions: NUTS_RG_01M_2021 (1:1 million scale)

## Integration with PyPSA

Practical integration demonstrated in `bus_filtering.py`:

```python
# Filter power network buses by country
buses = load_buses()
buses_germany = filter_buses_by_countries(buses, ['DE'])

# Assign NUTS-3 regions to buses
buses_with_regions = assign_nuts3_to_buses(buses, countries=['DE'])

# Filter buses by custom boundary
from geometry import buffer_shape
germany_buffer = buffer_shape(germany, distance_km=50)
buses_near_germany = filter_buses_by_boundary(buses, germany_buffer)
```

## Code Quality

- **Type hints**: Full type annotations for better IDE support
- **Docstrings**: Google-style docstrings with examples
- **Error handling**: Proper exception handling and logging
- **PEP 8 compliant**: Clean, readable code
- **Tested**: Automated test suite included

## Quick Start

```bash
# Install dependencies
pip install -r requirements-dev.txt

# Run tests
python scripts/test_geometry.py

# Run examples
python scripts/examples_geometry.py

# Try bus filtering
python scripts/bus_filtering.py
```

## Example Usage

```python
from scripts.geometry import (
    download_country_shapes,
    download_nuts3_shapes,
    join_shapes,
    point_in_shape,
    mask_shape,
)

# Download countries
germany = download_country_shapes(['DE'])
poland = download_country_shapes(['PL'])

# Join them
combined = join_shapes([germany, poland])

# Check if Berlin is in the combined region
berlin_in = point_in_shape(52.5200, 13.4050, combined)
print(f"Berlin in DE+PL: {berlin_in}")  # True

# Download NUTS-3 regions
nuts3 = download_nuts3_shapes(['DE'])
print(f"German NUTS-3 regions: {len(nuts3)}")

# Find which region contains Berlin
for idx, row in nuts3.iterrows():
    if point_in_shape(52.5200, 13.4050, row['geometry']):
        print(f"Berlin is in: {row['NUTS_ID']}")
        break
```

## Function Reference

### Download Functions
- `download_country_shapes()` - Get country boundaries
- `download_nuts3_shapes()` - Get NUTS-3 regions
- `get_european_union_shape()` - Get unified EU boundary

### Core Operations (Required Features)
- `join_shapes()` - ✅ Join/union multiple geometries
- `point_in_shape()` - ✅ Check if point is in boundary
- `mask_shape()` - ✅ Intersection of boundaries

### Utility Functions
- `buffer_shape()` - Create buffer around shape
- `get_shape_area()` - Calculate area
- `simplify_shape()` - Reduce complexity

## Technical Details

### Coordinate Systems
- **Input/Output**: EPSG:4326 (WGS84 - standard lat/lon)
- **Internal calculations**: EPSG:3035 (ETRS89/LAEA Europe)
  - More accurate for European areas and distances
  - Results converted back to EPSG:4326

### Caching Location
```
data/cache/geometry/
├── countries.geojson      # All European countries
└── nuts_regions.geojson   # NUTS-3 regions
```

### Dependencies
- `geopandas>=0.12.0` - Geospatial data handling
- `shapely>=2.0.0` - Geometric operations
- `requests>=2.28.0` - HTTP downloads
- `pandas>=1.3.0` - Data manipulation

## Inspiration from PyPSA-Eur

This module follows patterns from the [PyPSA-Eur](https://github.com/pypsa/pypsa-eur) project:

1. **Geometry handling** from `build_shapes.py`
2. **Caching strategy** from `_helpers.py`
3. **Data sources** from Eurostat GISCO
4. **CRS handling** (EPSG:4326 for I/O, EPSG:3035 for calculations)
5. **Simplification methods** for performance

## Testing

Run the test suite:
```bash
cd scripts
python test_geometry.py
```

Tests cover:
- ✅ Module imports
- ✅ Country download & caching
- ✅ Point-in-polygon checks
- ✅ Shape joining
- ✅ NUTS-3 download
- ✅ Intersection operations

## Next Steps

The module is ready to use! Possible extensions:

1. **Async downloads** for better performance
2. **Spatial indexing** for faster point queries
3. **More NUTS levels** (NUTS-0, NUTS-1, NUTS-2)
4. **EEZ support** (Exclusive Economic Zones)
5. **Custom region definitions**

## Support

For issues or questions:
1. Check `README_geometry.md` for detailed documentation
2. Run `examples_geometry.py` to see all features
3. Review `bus_filtering.py` for practical integration examples

## License

MIT License (consistent with PyPSA-Eur)
