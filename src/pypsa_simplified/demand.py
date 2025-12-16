import pandas as pd
import numpy as np
import sys
from pathlib import Path
import logging
from typing import Dict

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
    if_fast = options.get('if_fast', True)
    
    pop_grid = pop_grid[
        (pop_grid['lon'] >= lon_min) & (pop_grid['lon'] <= lon_max) &
        (pop_grid['lat'] >= lat_min) & (pop_grid['lat'] <= lat_max)
    ].reset_index(drop=True)
    logger.info(f"Filtered population grid to bounding box: {len(pop_grid)} entries remain.")
    
    '''
    if options.get('if_fast', True):
        logger.info("Using fast method for population calculation.")
        return calculate_population_voronoi_fast(
            pop_grid['lon'].values,
            pop_grid['lat'].values,
            pop_grid['population'].values,
            voronoi_path
        )
    '''
    
    # Load Voronoi geometries
    logger.info(f"Loading Voronoi geometries from {voronoi_path}...")
    voronoi_gdf = geom.load_shapes_efficiently(voronoi_path) # Expected to have 'geometry' column with polygons
    logger.info(f"Loaded {len(voronoi_gdf)} Voronoi cells.")
    # Initialize population column
    voronoi_gdf['population'] = 0  # Initialize population column
    # Population per Voronoi cell via point_in_shape method from geom that takes lon, lat and shape
    for idx, voronoi_row in voronoi_gdf.iterrows():
        logger.info(f"Processing Voronoi cell {idx+1}/{len(voronoi_gdf)} ({(idx+1)/len(voronoi_gdf)*100:.2f}%)...")
        shape = voronoi_row['geometry']
        
        mask = [geom.point_in_shape(pop_grid['lon'].values[i], pop_grid['lat'].values[i], shape) for i in range(len(pop_grid))]
        population_per_cell = pop_grid.loc[mask, 'population'].sum()
        voronoi_gdf.at[idx, 'population'] = population_per_cell
    
    # Return the whole GeoDataFrame with population column
    return voronoi_gdf

