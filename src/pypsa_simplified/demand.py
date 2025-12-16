import pandas as pd
import numpy as np
import sys
from pathlib import Path
import logging
from typing import Dict
import shapely
from shapely.geometry import Point
import multiprocessing as mp
import psutil

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
src_path = repo_root / 'scripts'
if str(src_path) not in sys.path:
    sys.path.insert(1, str(src_path))

try:
    import geometry as geom
except ImportError:
    raise ImportError("Could not import geometry module from scripts. Ensure you are running within the repository structure.")

repo_root = find_repo_root(Path(__file__).parent)
src_path = repo_root / 'src'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_population_grid(path: str | Path) -> pd.DataFrame:
    """Load population grid data from Eurostat and return as DataFrame."""
    # Load parquet file
    logger.info(f"Loading population grid data from {path}...")
    pop_grid = pd.read_parquet(path)
    logger.info(f"Loaded population grid with {len(pop_grid)} entries.")
    
    return pop_grid

def _compute_optimal_buffer(voronoi_row: pd.Series) -> float:
    """Compute optimal buffer distance for a single coordinate."""
    lat = voronoi_row['bus_y']
    lon = voronoi_row['bus_x']
    pt = Point(lon, lat)
    
    run = 0
    if pt.within(voronoi_row.geometry):
        area = voronoi_row['area_km2']
        increment = 0.001 * area
        buffer = 0.0
        run = 0
        while shapely.within(voronoi_row.geometry, pt.buffer(buffer)):
            buffer += increment
            run += 1
            if run == 1000:
                logger.warning(f"Max iterations reached for buffer calculation at point ({lon}, {lat}).")
                increment *= 2  
        
        return buffer
    else:
        logger.warning(f"Point ({lon}, {lat}) is not within its Voronoi cell.")
        return 0.0
    
def _sqares_in_rect(args: tuple) -> pd.DataFrame:
    """Get population data points within the rectangle."""
    
    idx, voronoi_row, data = args
    rect = geom.to_rect(voronoi_row.geometry)
    lon_min, lat_min, lon_max, lat_max = rect.bounds
    data_in_rect = data[
        (data['lon'] >= lon_min) & (data['lon'] <= lon_max) &
        (data['lat'] >= lat_min) & (data['lat'] <= lat_max)
    ].reset_index(drop=True)
    # Add the index of the voronoi cell for reference
    data_in_rect['voronoi_idx'] = idx
    return data_in_rect
    
def _sum_population_in_shape(args: tuple) -> int:
    """Sum population within a given shape."""
    idx, voronoi_row, data_in_rect = args
    total_population = 0
    shape = voronoi_row.geometry
    for _, point_row in data_in_rect.iterrows():
        pt = Point(point_row['lon'], point_row['lat'])
        if shapely.within(pt, shape):
            total_population += point_row['population']
    voronoi_row['population'] = total_population
    return total_population

def calculate_population_voronoi(pop_path: str | Path, voronoi_path: str | Path, options: Dict = None) -> pd.DataFrame:
    """Calculate population per Voronoi cell."""
    # Load population grid
    pop_grid = get_population_grid(pop_path) # It is expected that this DataFrame has 'lon', 'lat' and 'population' columns 
    logger.info(f"Population grid length: {len(pop_grid)}")
    if options is None:
        options = {}
    
    lon_min = options.get('lon_min', -12.0)
    lon_max = options.get('lon_max', 40.3)
    lat_min = options.get('lat_min', 34.0)
    lat_max = options.get('lat_max', 72.0)
    
    pop_grid = pop_grid[
        (pop_grid['lon'] >= lon_min) & (pop_grid['lon'] <= lon_max) &
        (pop_grid['lat'] >= lat_min) & (pop_grid['lat'] <= lat_max)
    ].reset_index(drop=True)
    logger.info(f"Filtered population grid to bounding box: {len(pop_grid)} entries remain.")
    
    # Load Voronoi geometries
    logger.info(f"Loading Voronoi geometries from {voronoi_path}...")
    voronoi_gdf = geom.load_shapes_efficiently(voronoi_path) # Expected to have 'geometry' column with polygons
    logger.info(f"Loaded {len(voronoi_gdf)} Voronoi cells.")
    # Initialize population column
    voronoi_gdf['population'] = 0  # Initialize population column
    # Population per Voronoi cell via point_in_shape method from geom that takes lon, lat and shape
    logger.info("Calculating population per Voronoi cell...")
    check_args = [(idx, row, pop_grid) for idx, row in voronoi_gdf.iterrows()]
    with mp.Pool() as pool:
        pop_datas = pool.map(_sqares_in_rect, check_args)
        logger.info(f"Using {psutil.cpu_count(logical=False)} CPU cores for parallel processing.")
        logger.info(f"Memory usage: {psutil.virtual_memory().percent}%")
        logger.info(f"CPU usage: {psutil.cpu_percent(interval=1)}%")
    
    logger.info("Summing population within each Voronoi cell...")
    check_args = [(idx, voronoi_gdf.iloc[idx], pop_datas[idx]) for idx in range(len(voronoi_gdf))]
    with mp.Pool() as pool:
        populations = pool.map(_sum_population_in_shape, check_args)
        logger.info(f"Using {psutil.cpu_count(logical=False)} CPU cores for parallel processing.")
        logger.info(f"Memory usage: {psutil.virtual_memory().percent}%")
        logger.info(f"CPU usage: {psutil.cpu_percent(interval=1)}%")
    
    logger.info("Population calculation completed.")
    voronoi_gdf['population'] = populations
    
    cashe_dir = geom.DEFAULT_CACHE_DIR
    voronoi_csv_path = cashe_dir / voronoi_path.name.replace('.parquet', '.csv')
    voronoi_csv = pd.read_csv(voronoi_csv_path)
    voronoi_csv['population'] = voronoi_gdf['population']
    voronoi_csv.to_csv(voronoi_csv_path, index=False)
    logger.info(f"Saved Voronoi population data to {voronoi_csv_path}.")
                
    # Return the whole GeoDataFrame with population column
    return voronoi_csv

if __name__ == "__main__":
    pop_path = repo_root / 'data' / 'processed' / 'jrc_population_nonzero.parquet'
    voronoi_path = geom.DEFAULT_CACHE_DIR / 'voronoi_eu27_join.parquet'
    import datetime as dt
    start_time = dt.datetime.now()
    calculate_population_voronoi(pop_path, voronoi_path)
    # In seconds
    end_time = dt.datetime.now()
    logger.info(f"Total computation time: {end_time - start_time}")
