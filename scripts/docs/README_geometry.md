# Geometry Module

A Python module for working with European geographical data, providing easy access to country boundaries and NUTS-3 regions with efficient caching and geometry operations.

## Features

### 1. **Data Download & Caching**
- Automatically downloads geographical data from Eurostat GISCO
- Local caching to avoid repeated downloads
- Fast access to cached data

### 2. **Supported Data**
- **Country shapes**: All European countries (ISO 2-letter codes)
- **NUTS-3 regions**: Nomenclature of Territorial Units for Statistics (level 3)
- **European Union**: Unified boundary of EU27 member states

### 3. **Core Operations**

#### Join/Union Shapes
Combine multiple geometries into a single unified boundary:
```python
from scripts.geometry import download_country_shapes, join_shapes

# Download and join Germany and Poland
countries = download_country_shapes(['DE', 'PL'])
combined = join_shapes(countries)
```

#### Point-in-Polygon Checks
Test if a coordinate point falls within a boundary:
```python
from scripts.geometry import point_in_shape

# Check if Berlin is in Germany
is_in_germany = point_in_shape(lat=52.5200, lon=13.4050, shape=germany)
# Returns: True
```

#### Intersection/Masking
Compute the overlap between two boundaries (logical AND):
```python
from scripts.geometry import mask_shape

# Get intersection of two regions
overlap = mask_shape(shape1, shape2)
```

#### Additional Utilities
- **Buffer**: Expand or contract boundaries by a distance in kilometers
- **Area calculation**: Get area in km², m², or hectares
- **Simplification**: Reduce geometry complexity for faster operations

## Installation

### Requirements
```bash
pip install geopandas>=0.12.0 shapely>=2.0.0 requests>=2.28.0 pandas>=1.3.0
```

Or install all development requirements:
```bash
pip install -r requirements-dev.txt
```

## Quick Start

### Basic Usage

```python
from scripts.geometry import (
    download_country_shapes,
    download_nuts3_shapes,
    join_shapes,
    point_in_shape,
    mask_shape,
)

# 1. Download country shapes
germany = download_country_shapes(['DE'])
france = download_country_shapes(['FR'])

# 2. Download NUTS-3 regions
nuts3 = download_nuts3_shapes(['DE', 'FR'])

# 3. Join multiple countries
benelux = download_country_shapes(['BE', 'NL', 'LU'])
benelux_shape = join_shapes(benelux)

# 4. Check if a point is within a boundary
berlin_coords = (52.5200, 13.4050)  # (lat, lon)
is_in_germany = point_in_shape(*berlin_coords, germany)

# 5. Compute intersection of two boundaries
overlap = mask_shape(germany, france)  # Should be empty (no border overlap)
```

### Working with NUTS-3 Regions

```python
# Download NUTS-3 regions for Germany
nuts3_de = download_nuts3_shapes(['DE'])

print(f"Number of regions: {len(nuts3_de)}")

# Find which region contains a point
for idx, row in nuts3_de.iterrows():
    if point_in_shape(52.5200, 13.4050, row['geometry']):
        print(f"Berlin is in: {row['NUTS_ID']} - {row['NAME_LATN']}")
        break
```

### Advanced Features

```python
from scripts.geometry import buffer_shape, get_shape_area, simplify_shape

# Create a 50km buffer around a country
luxembourg = download_country_shapes(['LU'])
buffered = buffer_shape(luxembourg, distance_km=50)

# Calculate area
area_km2 = get_shape_area(luxembourg, unit='km2')
print(f"Luxembourg area: {area_km2:.0f} km²")

# Simplify complex geometries for faster operations
norway = download_country_shapes(['NO'])
norway_simple = simplify_shape(norway, tolerance=0.01)
```

### Get European Union Boundary

```python
from scripts.geometry import get_european_union_shape

# Get unified EU boundary (all EU27 member states)
eu = get_european_union_shape()

# Check if a point is in the EU
is_in_eu = point_in_shape(52.5200, 13.4050, eu)
```

## Examples

Run the complete examples script:
```bash
python scripts/examples_geometry.py
```

This will demonstrate:
1. Downloading and joining country shapes
2. Point-in-polygon checks for various cities
3. Working with NUTS-3 regions
4. Computing boundary intersections
5. Creating buffers
6. Getting the EU boundary
7. Simplifying complex geometries

## Data Sources

All geographical data is sourced from **Eurostat GISCO** (Geographic Information System of the Commission):
- Countries: [CNTR_RG_01M_2020](https://gisco-services.ec.europa.eu/distribution/v2/countries/)
- NUTS regions: [NUTS_RG_01M_2021](https://gisco-services.ec.europa.eu/distribution/v2/nuts/)

Scale: 1:1 million (01M) - suitable for most analysis purposes while keeping file sizes manageable.

## Coordinate Systems

- **Input/Output CRS**: EPSG:4326 (WGS84 - standard lat/lon coordinates)
- **Distance Calculations**: EPSG:3035 (ETRS89-extended / LAEA Europe)
  - Used internally for accurate area and distance calculations
  - All results are converted back to EPSG:4326

## Caching

Downloaded data is cached in:
```
data/cache/geometry/
├── countries.geojson
└── nuts_regions.geojson
```

To force re-download:
```python
shapes = download_country_shapes(['DE'], force_download=True)
```

## Function Reference

### Download Functions

#### `download_country_shapes(countries=None, cache_dir=None, force_download=False)`
Download country boundaries for Europe.

**Parameters:**
- `countries` (list, optional): ISO 2-letter country codes. If None, returns all European countries.
- `cache_dir` (Path, optional): Custom cache directory
- `force_download` (bool): Re-download even if cached

**Returns:** `gpd.GeoDataFrame` with country geometries

#### `download_nuts3_shapes(countries=None, cache_dir=None, force_download=False)`
Download NUTS-3 region boundaries.

**Parameters:**
- `countries` (list, optional): Filter by country codes
- `cache_dir` (Path, optional): Custom cache directory
- `force_download` (bool): Re-download even if cached

**Returns:** `gpd.GeoDataFrame` with NUTS-3 geometries

### Geometry Operations

#### `join_shapes(shapes, dissolve=True)`
Join multiple shapes into a unified boundary.

**Parameters:**
- `shapes`: GeoDataFrame or list of geometries
- `dissolve` (bool): If True, removes internal boundaries

**Returns:** `Polygon` or `MultiPolygon`

#### `point_in_shape(lat, lon, shape, crs='EPSG:4326')`
Check if a point is inside a shape.

**Parameters:**
- `lat` (float): Latitude
- `lon` (float): Longitude
- `shape`: Boundary to check (Polygon, MultiPolygon, or GeoDataFrame)
- `crs` (str): Coordinate reference system

**Returns:** `bool`

#### `mask_shape(shape1, shape2, return_gdf=False)`
Compute intersection of two boundaries.

**Parameters:**
- `shape1`: First boundary
- `shape2`: Second boundary (mask)
- `return_gdf` (bool): Return as GeoDataFrame

**Returns:** `Polygon`, `MultiPolygon`, or `GeoDataFrame`

### Utility Functions

#### `buffer_shape(shape, distance_km, return_gdf=False)`
Create buffer around a shape.

**Parameters:**
- `shape`: Boundary to buffer
- `distance_km` (float): Distance in kilometers (+ to expand, - to contract)
- `return_gdf` (bool): Return as GeoDataFrame

**Returns:** Buffered geometry

#### `get_shape_area(shape, unit='km2')`
Calculate area of a shape.

**Parameters:**
- `shape`: Boundary to measure
- `unit` (str): 'km2', 'm2', or 'ha'

**Returns:** `float` - Area in specified units

#### `simplify_shape(shape, tolerance=0.01, return_gdf=False)`
Simplify geometry by reducing vertices.

**Parameters:**
- `shape`: Boundary to simplify
- `tolerance` (float): Simplification tolerance (degrees)
- `return_gdf` (bool): Return as GeoDataFrame

**Returns:** Simplified geometry

#### `get_european_union_shape(cache_dir=None, force_download=False)`
Get unified EU27 boundary.

**Returns:** `Polygon` representing the European Union

## Integration with PyPSA

This module is designed to work seamlessly with PyPSA energy system models. Example integration:

```python
from scripts.geometry import download_country_shapes, point_in_shape
import pandas as pd

# Load your power plant data
plants = pd.read_csv('data/raw/powerplants.csv')

# Get country boundaries
germany = download_country_shapes(['DE'])

# Filter plants by location
german_plants = plants[
    plants.apply(
        lambda row: point_in_shape(row['lat'], row['lon'], germany),
        axis=1
    )
]
```

## Performance Tips

1. **Use caching**: Downloaded data is automatically cached. Don't force re-download unless necessary.
2. **Simplify geometries**: For visualization or when high precision isn't needed:
   ```python
   shapes_simple = simplify_shape(shapes, tolerance=0.01)
   ```
3. **Batch operations**: When checking many points, convert to GeoDataFrame first for vectorized operations.

## Troubleshooting

### Download fails
- Check internet connection
- Eurostat GISCO servers may be temporarily unavailable - try again later
- Use cached data if available

### Memory issues with many regions
- Use `simplify_shape()` to reduce geometry complexity
- Process regions in batches
- Filter to specific countries before operations

### Projection warnings
- The module handles CRS conversions automatically
- All public functions return data in EPSG:4326
- Distance calculations are done in EPSG:3035 for accuracy

## Contributing

Based on patterns from [PyPSA-Eur](https://github.com/pypsa/pypsa-eur) project.

## License

This module follows the MIT license, consistent with PyPSA-Eur.
