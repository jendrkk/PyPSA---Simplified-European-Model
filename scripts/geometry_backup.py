"""
Geometry utilities for working with European geographical data.

This module provides functions to:
- Download and cache country and NUTS-3 region shapes
- Join/union multiple geometries
- Check if points are within boundaries
- Compute intersections of boundaries

Based on practices from PyPSA-Eur project.
"""

import logging
import sys
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple, Union

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Polygon, MultiPolygon, Point, LineString, LinearRing, MultiLineString
from shapely.ops import unary_union, polygonize, polygonize_full
from scipy.spatial import Voronoi
import numpy as np
import datetime

# Add src to path for data_prep import
def find_repo_root(start_path: Path, max_up: int = 6) -> Path:
    """Find repository root by searching upward for README.md or .git"""
    p = start_path.resolve()
    for _ in range(max_up):
        if (p / 'README.md').exists() or (p / '.git').exists():
            return p
        if p.parent == p:
            break
        p = p.parent
    return start_path.resolve()

repo_root = find_repo_root(Path(__file__).parent)
src_path = repo_root / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(1, str(src_path))

try:
    from pypsa_simplified import data_prep
except ImportError:
    data_prep = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
GEO_CRS = "EPSG:4326"  # WGS84 - standard lat/lon
DISTANCE_CRS = "EPSG:3035"  # ETRS89-extended / LAEA Europe for area calculations

# Default cache directory
DEFAULT_CACHE_DIR = Path(__file__).parent.parent / "data" / "cache" / "geometry"

# Data sources
NUTS_URL = "https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/NUTS_RG_01M_2021_4326.geojson"
COUNTRIES_URL = "https://gisco-services.ec.europa.eu/distribution/v2/countries/geojson/CNTR_RG_01M_2020_4326.geojson"

# European country codes (ISO 2-letter)
EUROPE_COUNTRIES = [
    "AL", "AT", "BA", "BE", "BG", "CH", "CZ", "DE", "DK", 
    "EE", 'EL', "ES", "FI", "FR", "HR", "HU", "IE", "IT",
    "LT", "LU", "LV", "ME", "MK", "NL", "NO", "PL", "PT",
    "RO", "RS", "SE", "SI", "SK", "XK", "UA", 'UK', "MD"
]

# EU27 (actually EU26 -> we remove Cyprus since it isn't connected to Europe with any link) member states (as of 2023)
EU27 = [
    'AT', 'BE', 'BG', 'HR', 'CZ', 'DK', 'EE', 'EL', 'FI', 'FR',
    'DE', 'GR', 'HU', 'IE', 'IT', 'LV', 'LT', 'LU', 'MT', 'NL',
    'PL', 'PT', 'RO', 'SK', 'SI', 'ES', 'SE'
]

EUROPE_EXTREME_POINTS = Polygon([(-12,72),(40.3,72),(40.3,34),(-12,34)])

def _ensure_cache_dir(cache_dir: Optional[Path] = None) -> Path:
    """
    Ensure cache directory exists and return its path.
    
    Parameters
    ----------
    cache_dir : Path, optional
        Custom cache directory. If None, uses default.
        
    Returns
    -------
    Path
        Path to cache directory
    """
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _download_file(url: str, output_path: Path, force: bool = False) -> Path:
    """
    Download a file from URL to output path with caching.
    
    Parameters
    ----------
    url : str
        URL to download from
    output_path : Path
        Where to save the file
    force : bool
        If True, download even if file exists
        
    Returns
    -------
    Path
        Path to downloaded file
    """
    if output_path.exists() and not force:
        logger.info(f"Using cached file: {output_path}")
        return output_path
    
    logger.info(f"Downloading from {url}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    logger.info(f"Downloaded to {output_path}")
    return output_path


def download_country_shapes(
    countries: Optional[List[str]] = None,
    cache_dir: Optional[Path] = None,
    force_download: bool = False,
    clip_to_continental: bool = True
) -> gpd.GeoDataFrame:
    """
    Download country shapes for Europe from Eurostat GISCO.
    
    If the data is already cached locally, it will be loaded from cache
    unless force_download is True.
    
    Parameters
    ----------
    countries : list of str, optional
        List of ISO 2-letter country codes. If None, returns all European countries.
    cache_dir : Path, optional
        Directory for caching downloaded data
    force_download : bool
        If True, re-download even if cached
    clip_to_continental : bool
        If True, clip shapes to continental Europe (EUROPE_EXTREME_POINTS)
        
    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with country geometries in EPSG:4326
        
    Examples
    --------
    >>> # Download all European countries
    >>> countries = download_country_shapes()
    >>> 
    >>> # Download specific countries
    >>> de_pl = download_country_shapes(['DE', 'PL'])
    """
    cache_dir = _ensure_cache_dir(cache_dir)
    cache_file = cache_dir / "countries.geojson"
    
    # Download or use cached file
    if not cache_file.exists() or force_download:
        _download_file(COUNTRIES_URL, cache_file, force=force_download)
    
    # Load and filter
    gdf = gpd.read_file(cache_file)
    
    # Standardize country codes
    if 'CNTR_ID' in gdf.columns:
        gdf = gdf.rename(columns={'CNTR_ID': 'country'})
    elif 'ISO2' in gdf.columns:
        gdf = gdf.rename(columns={'ISO2': 'country'})
    
    # Filter for Europe
    if countries is None:
        countries = EUROPE_COUNTRIES
    
    gdf = gdf[gdf['country'].isin(countries)].copy()
    
    # Ensure CRS
    if gdf.crs != GEO_CRS:
        gdf = gdf.to_crs(GEO_CRS)
    
    # Clip to continental Europe if requested
    if clip_to_continental:
        gdf['geometry'] = gdf.geometry.intersection(EUROPE_EXTREME_POINTS)
        # Remove empty geometries
        gdf = gdf[~gdf.geometry.is_empty].copy()
        logger.info(f"Clipped to continental Europe: {len(gdf)} country shapes remain")
    else:
        logger.info(f"Loaded {len(gdf)} country shapes")
    
    return gdf


def download_nuts3_shapes(
    countries: Optional[List[str]] = None,
    cache_dir: Optional[Path] = None,
    force_download: bool = False,
    clip_to_continental: bool = True
) -> gpd.GeoDataFrame:
    """
    Download NUTS-3 region shapes for Europe from Eurostat GISCO.
    
    If the data is already cached locally, it will be loaded from cache
    unless force_download is True.
    
    NUTS (Nomenclature of Territorial Units for Statistics) is a standard
    for referencing administrative divisions in Europe.
    
    Parameters
    ----------
    countries : list of str, optional
        List of ISO 2-letter country codes to filter by. If None, returns all regions.
    cache_dir : Path, optional
        Directory for caching downloaded data
    force_download : bool
        If True, re-download even if cached
    clip_to_continental : bool
        If True, clip shapes to continental Europe (EUROPE_EXTREME_POINTS)
        
    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with NUTS-3 geometries in EPSG:4326
        Columns include: NUTS_ID, CNTR_CODE, NAME_LATN, LEVL_CODE, geometry
        
    Examples
    --------
    >>> # Download all NUTS-3 regions
    >>> nuts3 = download_nuts3_shapes()
    >>> 
    >>> # Download NUTS-3 for Germany and Poland
    >>> nuts3_de_pl = download_nuts3_shapes(['DE', 'PL'])
    >>> 
    >>> # Get only NUTS-3 level (level 3)
    >>> nuts3_only = nuts3[nuts3['LEVL_CODE'] == 3]
    """
    cache_dir = _ensure_cache_dir(cache_dir)
    cache_file = cache_dir / "nuts_regions.geojson"
    
    # Download or use cached file
    if not cache_file.exists() or force_download:
        _download_file(NUTS_URL, cache_file, force=force_download)
    
    # Load data
    gdf = gpd.read_file(cache_file)
    
    # Filter for NUTS-3 level only (LEVL_CODE == 3)
    if 'LEVL_CODE' in gdf.columns:
        gdf = gdf[gdf['LEVL_CODE'] == 3].copy()
    
    # Standardize country codes (EL -> GR, UK -> GB)
    if 'CNTR_CODE' in gdf.columns:
        gdf.loc[gdf['CNTR_CODE'] == 'EL', 'CNTR_CODE'] = 'GR'
        gdf.loc[gdf['CNTR_CODE'] == 'UK', 'CNTR_CODE'] = 'GB'
    
    if 'NUTS_ID' in gdf.columns:
        gdf['NUTS_ID'] = gdf['NUTS_ID'].str.replace('EL', 'GR').str.replace('UK', 'GB')
    
    # Filter by countries if specified
    if countries is not None:
        gdf = gdf[gdf['CNTR_CODE'].isin(countries)].copy()
    
    # Ensure CRS
    if gdf.crs != GEO_CRS:
        gdf = gdf.to_crs(GEO_CRS)
    
    # Clip to continental Europe if requested
    if clip_to_continental:
        gdf['geometry'] = gdf.geometry.intersection(EUROPE_EXTREME_POINTS)
        # Remove empty geometries
        gdf = gdf[~gdf.geometry.is_empty].copy()
        logger.info(f"Clipped to continental Europe: {len(gdf)} NUTS-3 region shapes remain")
    else:
        logger.info(f"Loaded {len(gdf)} NUTS-3 region shapes")
    
    return gdf


def join_shapes(
    shapes: Union[gpd.GeoDataFrame, List[Union[Polygon, MultiPolygon]]],
    dissolve: bool = True
) -> Union[Polygon, MultiPolygon]:
    """
    Join multiple shapes together to create a unified boundary.
    
    This performs a union operation on all geometries. For example,
    joining shapes of Germany and Poland results in a single boundary
    as if they were one country/region.
    
    Parameters
    ----------
    shapes : GeoDataFrame or list of geometries
        Either a GeoDataFrame with geometry column or a list of Shapely geometries
    dissolve : bool
        If True, dissolves internal boundaries (default). If False, keeps them.
        
    Returns
    -------
    Polygon or MultiPolygon
        Unified geometry representing the joined shapes
        
    Examples
    --------
    >>> # Join country shapes
    >>> countries = download_country_shapes(['DE', 'PL'])
    >>> combined = join_shapes(countries)
    >>> 
    >>> # Join specific geometries
    >>> geom1 = countries[countries['country'] == 'DE'].geometry.iloc[0]
    >>> geom2 = countries[countries['country'] == 'PL'].geometry.iloc[0]
    >>> combined = join_shapes([geom1, geom2])
    """
    if isinstance(shapes, gpd.GeoDataFrame):
        geoms = shapes.geometry.tolist()
    else:
        geoms = shapes
    
    if len(geoms) == 0:
        raise ValueError("No geometries to join")
    
    # Use unary_union for efficient joining
    unified = unary_union(geoms)
    
    logger.info(f"Joined {len(geoms)} shapes into unified boundary")
    return unified


def point_in_shape(
    lat: float = None,
    lon: float = None,
    point: Point = None,
    shape: Union[Polygon, MultiPolygon, gpd.GeoDataFrame] = None,
    crs: str = GEO_CRS
) -> bool:
    """
    Check if a point (latitude, longitude) is inside a shape.
    
    Parameters
    ----------
    lat : float
        Latitude of the point
    lon : float
        Longitude of the point
    shape : Polygon, MultiPolygon, or GeoDataFrame
        The boundary to check against. If GeoDataFrame, uses union of all geometries.
    crs : str
        Coordinate reference system of the input coordinates (default: EPSG:4326)
        
    Returns
    -------
    bool
        True if point is inside the shape, False otherwise
        
    Examples
    --------
    >>> # Check if Berlin is in Germany
    >>> germany = download_country_shapes(['DE'])
    >>> berlin_lat, berlin_lon = 52.5200, 13.4050
    >>> is_in_germany = point_in_shape(berlin_lat, berlin_lon, germany)
    >>> print(is_in_germany)  # True
    >>> 
    >>> # Check if a point is in a specific NUTS-3 region
    >>> nuts3 = download_nuts3_shapes(['DE'])
    >>> berlin_region = nuts3[nuts3['NUTS_ID'] == 'DE300']
    >>> is_in_berlin = point_in_shape(52.5200, 13.4050, berlin_region)
    """
    # Create point
    if not isinstance(point, Point):
        try:
            point = Point(lon, lat)
        except Exception as e:
            try:
                point = Point(float(lon), float(lat))
            except Exception as e2:
                logger.error(f"Error creating Point from lat/lon: {e}, {e2}")
                return False
    
    # Extract geometry if GeoDataFrame
    if isinstance(shape, gpd.GeoDataFrame):
        if len(shape) == 0 or shape is None:
            logger.warning("Empty GeoDataFrame provided for shape.")
            return False
        # Create GeoSeries with proper CRS
        geom = gpd.GeoSeries([point], crs=crs)
        shape_union = shape.geometry.union_all()
        # Ensure same CRS
        if shape.crs and crs != str(shape.crs):
            geom = geom.to_crs(shape.crs)
        return geom.iloc[0].within(shape_union)
    
    # Direct geometry check
    return point.within(shape)


def mask_shape(
    shape1: Union[Polygon, MultiPolygon, gpd.GeoDataFrame],
    shape2: Union[Polygon, MultiPolygon, gpd.GeoDataFrame],
    return_gdf: bool = False,
    verbose: bool = True
) -> Union[Polygon, MultiPolygon, gpd.GeoDataFrame]:
    """
    Compute the intersection of two boundaries (logical AND operation).
    
    This creates a new boundary that represents only the area where both
    input boundaries overlap. Useful for masking one region with another.
    
    Parameters
    ----------
    shape1 : Polygon, MultiPolygon, or GeoDataFrame
        First boundary
    shape2 : Polygon, MultiPolygon, or GeoDataFrame
        Second boundary (mask)
    return_gdf : bool
        If True, return as GeoDataFrame. Otherwise return geometry.
        
    Returns
    -------
    Polygon, MultiPolygon, or GeoDataFrame
        Intersection of the two boundaries
        
    Examples
    --------
    >>> # Get overlap between Germany and a custom boundary
    >>> germany = download_country_shapes(['DE'])
    >>> custom_boundary = ...  # some other boundary
    >>> overlap = mask_shape(germany, custom_boundary)
    >>> 
    >>> # Find NUTS-3 regions within a buffer around a city
    >>> from shapely.geometry import Point
    >>> berlin = Point(13.4050, 52.5200).buffer(1)  # 1 degree buffer
    >>> nuts3 = download_nuts3_shapes(['DE'])
    >>> regions_near_berlin = mask_shape(nuts3, berlin, return_gdf=True)
    """
    # Extract geometries
    if isinstance(shape1, gpd.GeoDataFrame):
        geom1 = shape1.geometry.union_all()
        crs1 = shape1.crs
    else:
        geom1 = shape1
        crs1 = GEO_CRS
    
    if isinstance(shape2, gpd.GeoDataFrame):
        geom2 = shape2.geometry.union_all()
        crs2 = shape2.crs
    else:
        geom2 = shape2
        crs2 = GEO_CRS
    
    # Ensure same CRS if both are GeoDataFrames
    if crs1 and crs2 and crs1 != crs2:
        logger.warning(f"Different CRS detected: {crs1} vs {crs2}. Using first CRS.")
        if isinstance(shape2, gpd.GeoDataFrame):
            geom2 = gpd.GeoSeries([geom2], crs=crs2).to_crs(crs1).iloc[0]
    
    # Compute intersection
    intersection = geom1.intersection(geom2)
    if verbose:
        logger.info(f"Computed intersection of boundaries")
    
    if return_gdf:
        return gpd.GeoDataFrame({'geometry': [intersection]}, crs=crs1 or GEO_CRS)
    return intersection


def buffer_shape(
    shape: Union[Polygon, MultiPolygon, gpd.GeoDataFrame],
    distance_km: float,
    return_gdf: bool = False
) -> Union[Polygon, MultiPolygon, gpd.GeoDataFrame]:
    """
    Create a buffer around a shape with a specified distance in kilometers.
    
    This is useful for creating expanded boundaries or zones around regions.
    Uses ETRS89-extended / LAEA Europe projection for accurate distance calculations.
    
    Parameters
    ----------
    shape : Polygon, MultiPolygon, or GeoDataFrame
        The boundary to buffer
    distance_km : float
        Buffer distance in kilometers (positive for expansion, negative for contraction)
    return_gdf : bool
        If True, return as GeoDataFrame. Otherwise return geometry.
        
    Returns
    -------
    Polygon, MultiPolygon, or GeoDataFrame
        Buffered boundary in EPSG:4326
        
    Examples
    --------
    >>> # Create a 50km buffer around Germany
    >>> germany = download_country_shapes(['DE'])
    >>> germany_buffer = buffer_shape(germany, distance_km=50)
    >>> 
    >>> # Contract a boundary by 10km
    >>> contracted = buffer_shape(germany, distance_km=-10)
    """
    # Extract geometry and CRS
    if isinstance(shape, gpd.GeoDataFrame):
        gdf = shape.copy()
        original_crs = gdf.crs or GEO_CRS
    else:
        gdf = gpd.GeoDataFrame({'geometry': [shape]}, crs=GEO_CRS)
        original_crs = GEO_CRS
    
    # Convert to distance CRS for accurate buffering
    gdf_dist = gdf.to_crs(DISTANCE_CRS)
    
    # Buffer (distance in meters)
    gdf_dist['geometry'] = gdf_dist.geometry.buffer(distance_km * 1000)
    
    # Convert back to original CRS
    gdf_buffered = gdf_dist.to_crs(original_crs)
    
    logger.info(f"Created {distance_km}km buffer around shape")
    
    if return_gdf:
        return gdf_buffered
    return gdf_buffered.geometry.iloc[0] if len(gdf_buffered) == 1 else gdf_buffered.geometry.union_all()


def get_shape_area(
    shape: Union[Polygon, MultiPolygon, gpd.GeoDataFrame],
    unit: str = 'km2'
) -> float:
    """
    Calculate the area of a shape in specified units.
    
    Uses ETRS89-extended / LAEA Europe projection for accurate area calculations.
    
    Parameters
    ----------
    shape : Polygon, MultiPolygon, or GeoDataFrame
        The boundary to measure
    unit : str
        Unit for area: 'km2' (default), 'm2', or 'ha' (hectares)
        
    Returns
    -------
    float
        Area in specified units
        
    Examples
    --------
    >>> # Get area of Germany
    >>> germany = download_country_shapes(['DE'])
    >>> area_km2 = get_shape_area(germany, unit='km2')
    >>> print(f"Germany area: {area_km2:.0f} km²")
    """
    # Extract geometry
    if isinstance(shape, gpd.GeoDataFrame):
        gdf = shape.copy()
        original_crs = gdf.crs or GEO_CRS
    else:
        gdf = gpd.GeoDataFrame({'geometry': [shape]}, crs=GEO_CRS)
        original_crs = GEO_CRS
    
    # Convert to distance CRS for accurate area calculation
    gdf_dist = gdf.to_crs(DISTANCE_CRS)
    
    # Calculate area in m²
    area_m2 = gdf_dist.geometry.area.sum()
    
    # Convert to requested unit
    if unit == 'km2':
        area = area_m2 / 1e6
    elif unit == 'ha':
        area = area_m2 / 1e4
    elif unit == 'm2':
        area = area_m2
    else:
        raise ValueError(f"Unknown unit: {unit}. Use 'km2', 'm2', or 'ha'")
    
    return area


def simplify_shape(
    shape: Union[Polygon, MultiPolygon, gpd.GeoDataFrame],
    tolerance: float = 0.01,
    return_gdf: bool = False
) -> Union[Polygon, MultiPolygon, gpd.GeoDataFrame]:
    """
    Simplify a shape by reducing the number of vertices.
    
    This can significantly speed up operations on complex geometries while
    maintaining the overall shape. Uses the Douglas-Peucker algorithm.
    
    Parameters
    ----------
    shape : Polygon, MultiPolygon, or GeoDataFrame
        The boundary to simplify
    tolerance : float
        Tolerance parameter (in degrees if EPSG:4326). Larger values = more simplification.
        Default 0.01 ≈ 1km at mid-latitudes
    return_gdf : bool
        If True, return as GeoDataFrame. Otherwise return geometry.
        
    Returns
    -------
    Polygon, MultiPolygon, or GeoDataFrame
        Simplified boundary
        
    Examples
    --------
    >>> # Simplify NUTS-3 regions for faster plotting
    >>> nuts3 = download_nuts3_shapes(['DE'])
    >>> nuts3_simple = simplify_shape(nuts3, tolerance=0.01, return_gdf=True)
    """
    if isinstance(shape, gpd.GeoDataFrame):
        gdf = shape.copy()
        gdf['geometry'] = gdf.geometry.simplify(tolerance)
        logger.info(f"Simplified {len(gdf)} shapes with tolerance={tolerance}")
        return gdf if return_gdf else gdf.geometry.union_all()
    else:
        simplified = shape.simplify(tolerance)
        logger.info(f"Simplified shape with tolerance={tolerance}")
        return gpd.GeoDataFrame({'geometry': [simplified]}, crs=GEO_CRS) if return_gdf else simplified

def fill_from_boundary(boundary):
    """
    boundary can be a closed LineString/LinearRing, a collection (MultiLineString), or
    any iterable of LineStrings. Returns a polygon or MultiPolygon.
    """
    # Single closed ring -> just make a Polygon
    if boundary.geom_type in ("LineString", "LinearRing"):
        # ensure closed: first and last coordinate equal
        coords = list(boundary.coords)
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        return Polygon(coords)

    # If it's a collection of lines, merge and polygonize
    # polygonize only makes polygons from closed loops
    merged = unary_union(boundary)
    polys = list(polygonize(merged))
    if not polys:
        # helpful debug info: polygonize_full returns dangles/cuts that prevented polygonization
        polys_all, dangles, cuts, invalids = polygonize_full(merged)
        # return what polygonize_full found (polys_all is an iterator)
        polys = list(polys_all)
    if len(polys) == 1:
        return polys[0]
    return MultiPolygon(polys)


# Convenience function for common use case
def get_european_union_shape(
    cache_dir: Optional[Path] = None,
    force_download: bool = False,
    clip_to_continental: bool = True
) -> Polygon:
    """
    Get the unified boundary of the European Union.
    
    This downloads all EU member country shapes and joins them into
    a single boundary representing the EU as a whole.
    
    Parameters
    ----------
    cache_dir : Path, optional
        Directory for caching downloaded data
    force_download : bool
        If True, re-download even if cached
    clip_to_continental : bool
        If True, clip shapes to continental Europe (EUROPE_EXTREME_POINTS)
        
    Returns
    -------
    Polygon
        Unified EU boundary
        
    Examples
    --------
    >>> # Get EU boundary
    >>> eu = get_european_union_shape()
    >>> 
    >>> # Check if a point is in the EU
    >>> is_in_eu = point_in_shape(52.5200, 13.4050, eu)
    """
    countries = download_country_shapes(EU27, cache_dir, force_download, clip_to_continental)
    return join_shapes(countries)


def save_shapes_efficiently(
    gdf: gpd.GeoDataFrame,
    output_path: Path,
    compress: bool = True
) -> Path:
    """
    Save GeoDataFrame efficiently with compression.
    
    Uses GeoParquet format for best compression and speed.
    Falls back to compressed GeoJSON if geoparquet not available.
    
    Parameters
    ----------
    gdf : GeoDataFrame
        Data to save
    output_path : Path
        Where to save (extension will be adjusted)
    compress : bool
        Whether to use compression
        
    Returns
    -------
    Path
        Actual output path used
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Try GeoParquet first (best compression + speed)
    try:
        parquet_path = output_path.with_suffix('.parquet')
        gdf.to_parquet(parquet_path, compression='gzip' if compress else None)
        logger.info(f"Saved to {parquet_path} (GeoParquet)")
        return parquet_path
    except Exception:
        # Fallback to GeoJSON
        json_path = output_path.with_suffix('.geojson')
        gdf.to_file(json_path, driver='GeoJSON')
        logger.info(f"Saved to {json_path} (GeoJSON)")
        return json_path


def load_shapes_efficiently(input_path: Path) -> gpd.GeoDataFrame:
    """
    Load GeoDataFrame from parquet or geojson.
    
    Parameters
    ----------
    input_path : Path
        Path to file (with or without extension)
        
    Returns
    -------
    GeoDataFrame
        Loaded data
    """
    input_path = Path(input_path)
    
    # Try parquet first
    parquet_path = input_path.with_suffix('.parquet')
    if parquet_path.exists():
        return gpd.read_parquet(parquet_path)
    
    # Try geojson
    json_path = input_path.with_suffix('.geojson')
    if json_path.exists():
        return gpd.read_file(json_path)
    
    # Try as-is
    if input_path.exists():
        return gpd.read_file(input_path)
    
    raise FileNotFoundError(f"No file found: {input_path}")

def to_point(input: str) -> Point:
    """
    Convert input string "POINT(lat lon)" to Shapely Point.
    
    Parameters
    ----------
    input : str
        String in format "POINT(lat lon)" typical 
        in our data.
        
    Returns
    -------
    Point
        Shapely Point object
    """
    try:
        return Point(input.strip("POINT (").strip(")").split(" "))
    except Exception as e:
        logger.error(f"Error converting to Point: {e}")
        return None

def get_voronoi(
    raw_data: data_prep.RawData,
    countries: Optional[List[str]] = None,
    join: bool = True,
    cache_dir: Optional[Path] = None,
    add_mirror_points: bool = True,
    if_clip_coordiantes: bool = True
) -> Tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """
    Create Voronoi diagram for bus locations within country boundaries.
    
    This creates a Voronoi tessellation where each bus gets a cell representing
    the area closest to it. Cells are clipped to country boundaries.
    
    Parameters
    ----------
    raw_data : data_prep.RawData
        RawData object containing buses data
    countries : list of str, optional
        ISO 2-letter country codes. If None, uses EU27.
    join : bool
        If True, join all countries into one shape before computing Voronoi.
        If False, compute separate Voronoi diagrams for each country.
    cache_dir : Path, optional
        Cache directory for country shapes
    add_mirror_points : bool
        If True, add mirror points outside boundary to ensure bounded Voronoi cells
        
    Returns
    -------
    tuple of (GeoDataFrame, DataFrame)
        - GeoDataFrame with Voronoi cell geometries
        - DataFrame mapping bus_id to Voronoi cell properties
        
    Examples
    --------
    >>> from pypsa_simplified.data_prep import prepare_osm_source, RawData
    >>> data_dict = prepare_osm_source('data/raw/OSM Prebuilt Electricity Network')
    >>> raw_data = RawData(data_dict)
    >>> voronoi_gdf, mapping_df = get_voronoi(raw_data, countries=['DE'])
    """
    if data_prep is None:
        raise ImportError("data_prep module not available. Cannot use get_voronoi.")
    
    if countries is None:
        countries = EU27
    
    # Get buses data
    buses_df = raw_data.data.get('buses')
    if buses_df is None:
        raise ValueError("No buses data in RawData")
    
    # Filter buses by country
    if 'country' in buses_df.columns:
        buses_filtered = buses_df[buses_df['country'].isin(countries)].copy()
    else:
        # If no country column, use geometry check
        logger.warning("Filtering buses by point-in-shape check")
        country_shapes = download_country_shapes(countries, cache_dir)
        combined = join_shapes(country_shapes)
        try:
            mask = buses_df.apply(
            lambda row: point_in_shape(row['geometry'], combined),
            axis=1
            )
        except Exception as e:
            mask = buses_df.apply(
                lambda row: point_in_shape(row['y'], row['x'], combined),
                axis=1
            )
        buses_filtered = buses_df[mask].copy()

    
    if len(buses_filtered) == 0:
        logger.warning(f"No buses found in countries: {countries}")
        return gpd.GeoDataFrame(), pd.DataFrame()
    
    logger.info(f"Creating Voronoi diagram for {len(buses_filtered)} buses in {len(countries)} countries")
    
    # Get country shapes (already clipped to continental if requested)
    country_shapes_gdf = download_country_shapes(countries, cache_dir, clip_to_continental=if_clip_coordiantes)
    combined_shape = join_shapes(country_shapes_gdf)
    
    # Rename country EL to GR for consistency
    country_shapes_gdf.loc[country_shapes_gdf['country'] == 'EL', 'country'] = 'GR'
    country_shapes_gdf.loc[country_shapes_gdf['country'] == 'UK', 'country'] = 'GB'
    
    # Filter out buses that are on the sea (with small tolerance)
    combined_shape = buffer_shape(combined_shape, distance_km=0.1)
    buses_to_drop = []
    for idx, row in buses_filtered.iterrows():
        try:
            pt = to_point(row['geometry'])
            if not point_in_shape(point=pt, shape=combined_shape):
                buses_to_drop.append(idx)
        except Exception as e:
            logger.debug(f"Could not check bus {row['bus_id']}: {e}")
    
    buses_filtered = buses_filtered.drop(buses_to_drop)
    logger.info(f"{len(buses_filtered)} buses remain after sea filtering. Removed {len(buses_to_drop)} buses.")
    
    if join:
        # Single Voronoi diagram for all countries
        shapes = [combined_shape]
        shape_labels = ['combined']
    else:
        # Separate Voronoi diagram for each country
        shapes = []
        shape_labels = []
        for country_code in countries:
            country_gdf = country_shapes_gdf[country_shapes_gdf['country'] == country_code]
            if len(country_gdf) > 0:
                shapes.append(country_gdf.geometry.iloc[0])
                shape_labels.append(country_code)

    logger.info(f"Computing Voronoi for {len(shape_labels)} shapes")
    
    # Compute Voronoi for each shape
    all_voronoi_cells = []
    
    for shape, label in zip(shapes, shape_labels):
        # Get buses within this shape
        if if_clip_coordiantes:
            europes_extremes = Polygon([(-12,72),(40.3,72),(40.3,34),(-12,34)])
            shape = mask_shape(shape, europes_extremes)
        if join:
            buses_in_shape = buses_filtered
        else:
            mask = buses_filtered.apply(
                lambda row: row['country'] == label,
                axis=1
            )
            logger.info(f"Computing Voronoi for {label} with {mask.sum()} buses")
            buses_in_shape = buses_filtered[mask]
        
        if len(buses_in_shape) < 1:
            logger.warning(f"Skipping {label}: need at least 1 bus for Voronoi, found {len(buses_in_shape)}")
            continue
        
        # Extract coordinates
        points = buses_in_shape[['x', 'y']].values
        bus_ids = buses_in_shape['bus_id'].values
        num_real_points = len(points)
        
        # Add mirror points to bound Voronoi cells
        if add_mirror_points:
            bounds = shape.bounds  # (minx, miny, maxx, maxy)
            margin = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) * 0.5
            
            # Create grid of mirror points outside boundary
            mirror_points = []
            grid_spacing = margin / 3
            
            # Add points around the bounding box
            for x in np.arange(bounds[0] - margin, bounds[2] + margin, grid_spacing):
                mirror_points.append([x, bounds[1] - margin])
                mirror_points.append([x, bounds[3] + margin])
            for y in np.arange(bounds[1] - margin, bounds[3] + margin, grid_spacing):
                mirror_points.append([bounds[0] - margin, y])
                mirror_points.append([bounds[2] + margin, y])
            
            points = np.vstack([points, mirror_points])
            logger.info(f"Added {len(mirror_points)} mirror points to ensure bounded Voronoi cells")
        
        # Create Voronoi diagram
        vor = Voronoi(points)
        
        # Track which buses got cells
        buses_with_cells = set()
        
        # Convert Voronoi regions to polygons and clip to shape
        for point_idx in range(num_real_points):  # Only process real points, not mirrors
            region_idx = vor.point_region[point_idx]
            region = vor.regions[region_idx]
            
            if not region or -1 in region:
                # For infinite regions, create a bounded region using the Voronoi ridges
                logger.debug(f"Bus {bus_ids[point_idx]} has infinite Voronoi region, attempting to bound it")
                
                # Find finite Voronoi vertices and create a large polygon
                finite_vertices = [vor.vertices[i] for i in region if i != -1]
                
                if len(finite_vertices) >= 2:
                    # Create a polygon from finite vertices and extend to boundary
                    try:
                        # Use a large buffer around the point and intersect with shape
                        bus_point = Point(points[point_idx])
                        large_buffer = bus_point.buffer(10.0)  # 10 degrees ~ 1000km
                        clipped = large_buffer.intersection(shape)
                        
                        if not clipped.is_empty and get_shape_area(clipped) > 0:
                            buses_with_cells.add(bus_ids[point_idx])
                            all_voronoi_cells.append({
                                'geometry': clipped,
                                'bus_id': bus_ids[point_idx],
                                'bus_x': points[point_idx, 0],
                                'bus_y': points[point_idx, 1],
                                'region': label,
                                'area_km2': get_shape_area(clipped, unit='km2'),
                                'bounded': False  # Mark as unbounded originally
                            })
                            continue
                    except Exception as e:
                        logger.debug(f"Could not create fallback cell for bus {bus_ids[point_idx]}: {e}")
                
                continue
            
            # Create polygon from Voronoi vertices
            polygon_coords = [vor.vertices[i] for i in region]
            try:
                voronoi_poly = Polygon(polygon_coords)
                
                if not voronoi_poly.is_valid:
                    voronoi_poly = voronoi_poly.buffer(0)  # Fix invalid geometries
                
                # Clip to shape boundary
                clipped = voronoi_poly.intersection(shape)
                
                if not clipped.is_empty and get_shape_area(clipped) > 0:
                    buses_with_cells.add(bus_ids[point_idx])
                    all_voronoi_cells.append({
                        'geometry': clipped,
                        'bus_id': bus_ids[point_idx],
                        'bus_x': points[point_idx, 0],
                        'bus_y': points[point_idx, 1],
                        'region': label,
                        'area_km2': get_shape_area(clipped, unit='km2'),
                        'bounded': True
                    })
            except Exception as e:
                logger.debug(f"Could not create Voronoi cell for bus {bus_ids[point_idx]}: {e}")
                continue
        
        # For buses without cells, create fallback cells
        buses_without_cells = set(bus_ids) - buses_with_cells
        if buses_without_cells:
            logger.warning(f"{len(buses_without_cells)} buses did not get Voronoi cells, creating fallback cells")
            
            for bus_id in buses_without_cells:
                idx = np.where(bus_ids == bus_id)[0][0]
                bus_point = Point(points[idx, 0], points[idx, 1])
                
                # Create a small buffer around the bus
                small_buffer = bus_point.buffer(0.1)  # 0.1 degrees ~ 10km
                clipped = small_buffer.intersection(shape)
                
                if not clipped.is_empty:
                    all_voronoi_cells.append({
                        'geometry': clipped,
                        'bus_id': bus_id,
                        'bus_x': points[idx, 0],
                        'bus_y': points[idx, 1],
                        'region': label,
                        'area_km2': get_shape_area(clipped, unit='km2'),
                        'bounded': False  # Fallback cell
                    })
    
    if len(all_voronoi_cells) == 0:
        logger.warning("No valid Voronoi cells created")
        return gpd.GeoDataFrame(), pd.DataFrame()
    
    # Create GeoDataFrame
    voronoi_gdf = gpd.GeoDataFrame(all_voronoi_cells, crs=GEO_CRS)
    
    # Create mapping DataFrame
    mapping_df = pd.DataFrame({
        'bus_id': voronoi_gdf['bus_id'],
        'region': voronoi_gdf['region'],
        'area_km2': voronoi_gdf['area_km2'],
        'bus_x': voronoi_gdf['bus_x'],
        'bus_y': voronoi_gdf['bus_y'],
        'bounded': voronoi_gdf['bounded'],
    })
    
    logger.info(f"Created {len(voronoi_gdf)} Voronoi cells ({(voronoi_gdf['bounded']==True).sum()} bounded, {(voronoi_gdf['bounded']==False).sum()} fallback)")
    
    return voronoi_gdf, mapping_df


if __name__ == "__main__":
    start_time = datetime.datetime.now()
    print("=" * 80)
    print("GEOMETRY MODULE - DATA PREPARATION")
    print("=" * 80)
    print("\nThis will download and save all required geographical data:")
    print("- All European country boundaries")
    print("- All NUTS-3 regions")
    print("- Voronoi diagrams for EU27 buses")
    print("\n" + "=" * 80 + "\n")
    
    cache_dir = _ensure_cache_dir()
    
    # 1. Download all European country shapes
    print("Step 1/4: Downloading all European country boundaries...")
    all_countries = download_country_shapes()
    countries_path = save_shapes_efficiently(all_countries, cache_dir / "all_countries")
    all_countries = download_country_shapes(EUROPE_COUNTRIES)
    countries_path = save_shapes_efficiently(all_countries, cache_dir / "all_EUROPE_countries")
    print(f"   ✓ Saved {len(all_countries)} countries to {countries_path}")
    print(f"   File size: {countries_path.stat().st_size / 1024:.1f} KB\n")
    
    # 2. Download all NUTS-3 regions
    print("Step 2/4: Downloading all NUTS-3 regions...")
    all_nuts3 = download_nuts3_shapes()
    nuts3_path = save_shapes_efficiently(all_nuts3, cache_dir / "all_nuts3")
    print(f"   ✓ Saved {len(all_nuts3)} NUTS-3 regions to {nuts3_path}")
    print(f"   File size: {nuts3_path.stat().st_size / 1024:.1f} KB\n")
    
    # 3. Try to load buses and create Voronoi diagrams
    print("Step 3/4: Creating Voronoi diagrams for EU27 buses...")
    join = input("Join? ([T]/F): ").strip().lower()
    join = (join == '' or join == 't' or join == 'T')
    try:
        import pypsa_simplified.data_prep as dp
        
        # Try to load buses
        osm_dir = Path(__file__).parent.parent / "data" / "raw" / "OSM Prebuilt Electricity Network"
        if osm_dir.exists():
            data_dict = dp.prepare_osm_source(osm_dir)
            raw_data = dp.RawData(data_dict)
            
            if raw_data.data.get('buses') is not None:
                # Create Voronoi for EU27
                _join = "_join" if join else ""
                voronoi_gdf, mapping_df = get_voronoi(raw_data, countries=EU27, join=join, cache_dir=cache_dir)
                
                if len(voronoi_gdf) > 0:
                    # Save Voronoi cells
                    voronoi_path = save_shapes_efficiently(voronoi_gdf, cache_dir / f"voronoi_eu27{_join}")
                    print(f"   ✓ Saved {len(voronoi_gdf)} Voronoi cells to {voronoi_path}")
                    print(f"   File size: {voronoi_path.stat().st_size / 1024:.1f} KB")
                    
                    # Save mapping
                    mapping_path = cache_dir / f"voronoi_eu27_mapping{_join}.csv"
                    mapping_df.to_csv(mapping_path, index=False)
                    print(f"   ✓ Saved bus-to-cell mapping to {mapping_path}")
                    print(f"   File size: {mapping_path.stat().st_size / 1024:.1f} KB\n")
                else:
                    print("   ⚠ No valid Voronoi cells created\n")
            else:
                print("   ⚠ No buses data found in RawData\n")
        else:
            print(f"   ⚠ OSM directory not found: {osm_dir}")
            print("   Skipping Voronoi diagram creation\n")
    except ImportError as e:
        print(f"   ⚠ Could not import data_prep: {e}")
        print("   Skipping Voronoi diagram creation\n")
    except Exception as e:
        print(f"   ⚠ Error creating Voronoi diagrams: {e}")
        print("   Continuing with other steps\n")
    
    # 4. Quick verification tests
    print("Step 4/4: Running verification tests...")
    
    # Test point in shape
    berlin = (52.5200, 13.4050)
    germany = all_countries[all_countries['country'] == 'DE']
    is_in_germany = point_in_shape(*berlin, germany)
    print(f"   ✓ Berlin in Germany: {is_in_germany} (expected True)")
    
    # Test join
    de_pl = all_countries[all_countries['country'].isin(['DE', 'PL'])]
    combined = join_shapes(de_pl)
    area = get_shape_area(combined)
    print(f"   ✓ Germany + Poland area: {area:,.0f} km²")
    
    # Test NUTS-3
    de_nuts3 = all_nuts3[all_nuts3['CNTR_CODE'] == 'DE']
    print(f"   ✓ German NUTS-3 regions: {len(de_nuts3)}")
    
    print("\n" + "=" * 80)
    print("✓ ALL DATA PREPARED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\nData saved to: {cache_dir}")
    print("\nAvailable files:")
    for file in sorted(cache_dir.glob("*")):
        if file.is_file():
            size_kb = file.stat().st_size / 1024
            print(f"  - {file.name} ({size_kb:.1f} KB)")
    print("\n" + "=" * 80)
    print(f"Total time: {datetime.datetime.now() - start_time}")