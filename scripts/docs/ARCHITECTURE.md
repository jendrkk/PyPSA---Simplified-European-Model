# Geometry Module Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         GEOMETRY MODULE                              │
│                        (geometry.py)                                 │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    │             │             │
         ┌──────────▼──────┐  ┌──▼───────┐  ┌──▼──────────┐
         │  Data Download  │  │  Core     │  │  Utilities  │
         │   & Caching     │  │Operations │  │             │
         └─────────────────┘  └───────────┘  └─────────────┘
                │                   │               │
        ┌───────┴────────┐  ┌───────┴────────┐    │
        │                │  │                │    │
  ┌─────▼──────┐  ┌─────▼──────┐  │        │    │
  │  Countries │  │   NUTS-3    │  │        │    │
  └────────────┘  └─────────────┘  │        │    │
        │                │          │        │    │
        │    Eurostat    │          │        │    │
        │     GISCO      │          │        │    │
        │      API       │          │        │    │
        └────────┬───────┘          │        │    │
                 │                  │        │    │
           ┌─────▼─────┐            │        │    │
           │  Local    │            │        │    │
           │  Cache    │            │        │    │
           └───────────┘            │        │    │
                                    │        │    │
                            ┌───────▼────────┼────▼────────┐
                            │ join_shapes()  │  buffer()   │
                            │ point_in_shape│  area()     │
                            │ mask_shape()  │  simplify() │
                            └───────────────┴─────────────┘


DATA FLOW
═════════

1. USER REQUEST
   └─> download_country_shapes(['DE', 'PL'])
        │
        ├─> Check cache: data/cache/geometry/countries.geojson
        │   │
        │   ├─> Found? Return cached data ✓
        │   │
        │   └─> Not found? Download from Eurostat GISCO
        │       └─> Save to cache
        │           └─> Return data
        │
        └─> Filter by countries ['DE', 'PL']
            └─> Return GeoDataFrame


2. GEOMETRY OPERATIONS
   └─> join_shapes([shape1, shape2])
        │
        └─> unary_union([geoms])
            └─> Return unified Polygon/MultiPolygon


3. POINT CHECK
   └─> point_in_shape(52.52, 13.40, germany)
        │
        ├─> Create Point(lon, lat)
        ├─> Check CRS compatibility
        └─> Execute point.within(shape)
            └─> Return True/False


4. INTERSECTION
   └─> mask_shape(shape1, shape2)
        │
        ├─> Extract geometries
        ├─> Ensure same CRS
        └─> Compute intersection
            └─> Return intersected geometry


MODULE DEPENDENCIES
═══════════════════

geometry.py
    │
    ├── geopandas      (GeoDataFrame operations)
    ├── shapely        (Geometry objects & operations)
    ├── pandas         (Data manipulation)
    ├── requests       (HTTP downloads)
    └── pathlib        (File system operations)


FILE STRUCTURE
══════════════

scripts/
├── geometry.py              # Core module (700+ lines)
├── README_geometry.md       # Documentation
├── SUMMARY.md              # This summary
├── examples_geometry.py    # 7 example use cases
├── test_geometry.py        # Automated tests
└── bus_filtering.py        # PyPSA integration utilities

data/
└── cache/
    └── geometry/
        ├── countries.geojson      # Cached country shapes
        └── nuts_regions.geojson   # Cached NUTS-3 regions


COORDINATE SYSTEMS
══════════════════

Input/Output:  EPSG:4326 (WGS84)
               └─> Standard latitude/longitude
               └─> Compatible with most GIS data

Internal Calc: EPSG:3035 (ETRS89-extended / LAEA Europe)
               └─> Used for area calculations
               └─> Used for distance/buffer operations
               └─> More accurate for European data
               └─> Results converted back to EPSG:4326


FOUR CORE FUNCTIONS (as requested)
═══════════════════════════════════

1. join_shapes()
   ├─ Purpose: Union multiple boundaries into one
   ├─ Input: List of geometries or GeoDataFrame
   ├─ Output: Single Polygon/MultiPolygon
   └─ Use case: Combine Germany + Poland → DE+PL boundary

2. download_nuts3_shapes()
   ├─ Purpose: Get NUTS-3 regional boundaries
   ├─ Input: Country codes (optional filter)
   ├─ Output: GeoDataFrame with NUTS-3 regions
   └─ Use case: Regional energy analysis

3. point_in_shape()
   ├─ Purpose: Check if coordinate is within boundary
   ├─ Input: lat, lon, shape
   ├─ Output: Boolean (True/False)
   └─ Use case: "Is this power plant in Germany?"

4. mask_shape()
   ├─ Purpose: Intersection of two boundaries (AND)
   ├─ Input: Two shapes
   ├─ Output: Overlapping geometry
   └─ Use case: Find regions within buffer zone


USAGE PATTERNS
══════════════

Pattern 1: Filter Network Elements by Country
───────────────────────────────────────────────
from geometry import download_country_shapes, point_in_shape

germany = download_country_shapes(['DE'])
german_buses = [
    bus for bus in buses 
    if point_in_shape(bus.lat, bus.lon, germany)
]


Pattern 2: Regional Aggregation
────────────────────────────────
from geometry import download_nuts3_shapes

nuts3 = download_nuts3_shapes(['DE'])
for region in nuts3.itertuples():
    regional_load = calculate_load_in(region.geometry)


Pattern 3: Multi-Country Analysis
──────────────────────────────────
from geometry import download_country_shapes, join_shapes

countries = download_country_shapes(['DE', 'PL', 'CZ'])
study_area = join_shapes(countries)


Pattern 4: Spatial Masking
───────────────────────────
from geometry import mask_shape, buffer_shape

# Find offshore wind potential near coast
country = download_country_shapes(['DE'])
coastal_buffer = buffer_shape(country, 50)  # 50km
offshore_area = mask_shape(eez, coastal_buffer)


PERFORMANCE CHARACTERISTICS
════════════════════════════

Operation              Speed    Notes
────────────────────────────────────────────────────
First download         ~5-10s   Downloads from Eurostat
Cached load            <0.1s    Loads from local file
join_shapes()          <0.1s    Fast unary_union
point_in_shape()       <0.01s   Single point check
point_in_shape() x1000 ~1-2s    Batch operations
mask_shape()           ~0.1s    Intersection calculation
buffer_shape()         ~0.2s    Requires CRS transform
simplify_shape()       ~0.1s    Reduces vertices


DESIGN PRINCIPLES
═════════════════

1. Smart Caching
   └─> Download once, use forever
   └─> Automatic cache management
   └─> Optional force re-download

2. Correct Projections
   └─> EPSG:4326 for I/O (compatibility)
   └─> EPSG:3035 for calculations (accuracy)
   └─> Automatic conversions

3. Type Flexibility
   └─> Accept Polygon, MultiPolygon, or GeoDataFrame
   └─> Consistent return types
   └─> Optional return formats

4. Documentation
   └─> Every function has docstring
   └─> Examples in docstrings
   └─> Separate examples file

5. Error Handling
   └─> Informative error messages
   └─> Logging for debugging
   └─> Graceful failures


TESTING STRATEGY
════════════════

test_geometry.py includes:

✓ Import tests
✓ Download & caching
✓ Point-in-polygon accuracy
✓ Shape joining correctness
✓ NUTS-3 region support
✓ Intersection operations

Run: python scripts/test_geometry.py


INTEGRATION EXAMPLES
════════════════════

Example 1: Filter PyPSA buses by country
────────────────────────────────────────
import pandas as pd
from geometry import download_country_shapes, point_in_shape

buses = pd.read_csv('data/raw/OSM Prebuilt Electricity Network/buses.csv')
germany = download_country_shapes(['DE'])

buses['in_germany'] = buses.apply(
    lambda row: point_in_shape(row['y'], row['x'], germany),
    axis=1
)
german_buses = buses[buses['in_germany']]


Example 2: Aggregate by NUTS-3 region
──────────────────────────────────────
from geometry import download_nuts3_shapes, point_in_shape

nuts3 = download_nuts3_shapes(['DE'])

for idx, region in nuts3.iterrows():
    region_buses = buses[
        buses.apply(
            lambda row: point_in_shape(row['y'], row['x'], region.geometry),
            axis=1
        )
    ]
    print(f"{region['NUTS_ID']}: {len(region_buses)} buses")


FUTURE ENHANCEMENTS
═══════════════════

Potential additions:
├─ Async downloads for parallel processing
├─ Spatial indexing (rtree) for faster queries
├─ Support for NUTS-0, NUTS-1, NUTS-2 levels
├─ Exclusive Economic Zones (EEZ) support
├─ Custom region definitions from GeoJSON
└─ Batch point checking optimization
```
