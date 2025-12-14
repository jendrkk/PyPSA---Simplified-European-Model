# Geometry Module - Quick Reference

## Installation
```bash
pip install geopandas shapely requests pandas
```

## Import
```python
from scripts.geometry import (
    download_country_shapes,
    download_nuts3_shapes,
    join_shapes,
    point_in_shape,
    mask_shape,
    buffer_shape,
    get_shape_area,
)
```

## Core Functions

### 1. Download Countries
```python
# Single country
germany = download_country_shapes(['DE'])

# Multiple countries
countries = download_country_shapes(['DE', 'PL', 'FR'])

# All European countries
all_countries = download_country_shapes()
```

### 2. Download NUTS-3 Regions
```python
# Regions for specific countries
nuts3 = download_nuts3_shapes(['DE', 'FR'])

# All NUTS-3 regions
all_nuts3 = download_nuts3_shapes()
```

### 3. Join Shapes (Union)
```python
# Join multiple countries into one boundary
countries = download_country_shapes(['DE', 'PL'])
combined = join_shapes(countries)

# Join NUTS-3 regions
nuts3_regions = download_nuts3_shapes(['DE'])
combined_region = join_shapes(nuts3_regions)
```

### 4. Point in Shape (Boolean Check)
```python
# Check if Berlin is in Germany
is_in = point_in_shape(lat=52.52, lon=13.40, shape=germany)
# Returns: True or False

# Check multiple points
cities = [(52.52, 13.40), (48.86, 2.35)]  # Berlin, Paris
for lat, lon in cities:
    in_germany = point_in_shape(lat, lon, germany)
    print(f"({lat}, {lon}): {in_germany}")
```

### 5. Mask Shape (Intersection)
```python
# Get intersection of two boundaries
overlap = mask_shape(shape1, shape2)

# Example: Find area within both Germany and custom boundary
from shapely.geometry import Point
buffer = Point(13.40, 52.52).buffer(1.0)
intersection = mask_shape(germany, buffer)
```

## Utility Functions

### Buffer (Expand/Contract)
```python
# Expand by 50km
expanded = buffer_shape(germany, distance_km=50)

# Contract by 10km
contracted = buffer_shape(germany, distance_km=-10)
```

### Calculate Area
```python
# Area in km²
area_km2 = get_shape_area(germany, unit='km2')

# Area in hectares
area_ha = get_shape_area(germany, unit='ha')

# Area in m²
area_m2 = get_shape_area(germany, unit='m2')
```

### Simplify Geometry
```python
# Reduce complexity for faster operations
simplified = simplify_shape(germany, tolerance=0.01)
```

### Get EU Boundary
```python
# Unified shape of all EU27 member states
eu = get_european_union_shape()
```

## Common Patterns

### Pattern 1: Filter Points by Country
```python
import pandas as pd

# Load your data
data = pd.read_csv('points.csv')  # lat, lon columns

# Get country boundary
germany = download_country_shapes(['DE'])

# Filter
data['in_germany'] = data.apply(
    lambda row: point_in_shape(row['lat'], row['lon'], germany),
    axis=1
)
german_points = data[data['in_germany']]
```

### Pattern 2: Multi-Country Analysis
```python
# Define region of interest
countries = download_country_shapes(['DE', 'PL', 'CZ'])
study_area = join_shapes(countries)

# Calculate total area
total_area = get_shape_area(study_area)
print(f"Study area: {total_area:,.0f} km²")
```

### Pattern 3: Regional Aggregation
```python
# Get NUTS-3 regions
nuts3 = download_nuts3_shapes(['DE'])

# For each region
for idx, region in nuts3.iterrows():
    # Calculate metrics for this region
    region_area = get_shape_area(region.geometry)
    print(f"{region['NUTS_ID']}: {region_area:.0f} km²")
```

### Pattern 4: Proximity Analysis
```python
# Find areas within 100km of Berlin
from shapely.geometry import Point

berlin = Point(13.40, 52.52)
buffer_100km = berlin.buffer(1.0)  # ~100km

# Get countries
countries = download_country_shapes()

# Check which countries intersect
for idx, country in countries.iterrows():
    intersection = mask_shape(country, buffer_100km)
    if not intersection.is_empty:
        print(f"{country['country']} is within 100km of Berlin")
```

### Pattern 5: PyPSA Integration
```python
# Filter buses by country
buses = pd.read_csv('buses.csv')  # x, y columns
germany = download_country_shapes(['DE'])

buses['in_germany'] = buses.apply(
    lambda row: point_in_shape(row['y'], row['x'], germany),
    axis=1
)
german_buses = buses[buses['in_germany']]
```

## Coordinate Systems

- **Input/Output**: EPSG:4326 (WGS84 - standard lat/lon)
- **Internal calculations**: EPSG:3035 (for accurate European areas)
- Conversions handled automatically

## Data Caching

First time:
- Downloads from Eurostat GISCO (~5-10 seconds)
- Saves to `data/cache/geometry/`

Subsequent times:
- Loads from cache (<0.1 seconds)
- No internet required

Force re-download:
```python
shapes = download_country_shapes(['DE'], force_download=True)
```

## Performance Tips

1. **Use caching**: Data is cached automatically
2. **Simplify geometries**: Use `simplify_shape()` for faster operations
3. **Batch operations**: Process multiple points together
4. **Filter early**: Download only needed countries

## Troubleshooting

### Download fails
- Check internet connection
- Eurostat servers may be busy - wait and retry
- Use cached data if available

### Memory issues
- Use `simplify_shape()` to reduce complexity
- Process in batches
- Filter to specific countries

### Projection warnings
- Ignore - module handles CRS automatically
- All operations use correct projections

## Examples

Run included examples:
```bash
# All examples
python scripts/examples_geometry.py

# Tests
python scripts/test_geometry.py

# PyPSA integration
python scripts/bus_filtering.py

# Interactive tutorial
jupyter notebook notebooks/geometry_tutorial.ipynb
```

## Common Country Codes (ISO 2-letter)

| Code | Country       | Code | Country      |
|------|--------------|------|--------------|
| AT   | Austria      | IT   | Italy        |
| BE   | Belgium      | LU   | Luxembourg   |
| BG   | Bulgaria     | NL   | Netherlands  |
| CH   | Switzerland  | NO   | Norway       |
| CZ   | Czechia      | PL   | Poland       |
| DE   | Germany      | PT   | Portugal     |
| DK   | Denmark      | RO   | Romania      |
| ES   | Spain        | SE   | Sweden       |
| FI   | Finland      | SI   | Slovenia     |
| FR   | France       | SK   | Slovakia     |
| GB   | UK           | UA   | Ukraine      |
| GR   | Greece       | XK   | Kosovo       |
| HR   | Croatia      | ...  | and more     |

## File Locations

```
scripts/
├── geometry.py              # Main module
├── README_geometry.md       # Full documentation
├── examples_geometry.py     # Usage examples
├── test_geometry.py         # Tests
└── bus_filtering.py         # PyPSA utilities

data/cache/geometry/
├── countries.geojson        # Cached countries
└── nuts_regions.geojson     # Cached NUTS-3

notebooks/
└── geometry_tutorial.ipynb  # Interactive tutorial
```

## Dependencies

```
geopandas>=0.12.0
shapely>=2.0.0
requests>=2.28.0
pandas>=1.3.0
```

## Support

- Full docs: `README_geometry.md`
- Examples: `examples_geometry.py`
- Tutorial: `notebooks/geometry_tutorial.ipynb`
- Architecture: `ARCHITECTURE.md`
