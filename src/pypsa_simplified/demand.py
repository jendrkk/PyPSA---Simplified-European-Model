import pandas as pd
import numpy as np
import data_prep
import eurostat
import sys
from pathlib import Path

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


def get_population_grid(path: str | Path) -> pd.DataFrame:
    """Load population grid data from Eurostat and return as DataFrame."""
    # Load parquet file
    pop_grid = pd.read_parquet(path)
    
    return pop_grid

def calulate_population_voronoi(pop_path: str | Path, voronoi_path: str | Path) -> pd.DataFrame:
    """Calculate population per Voronoi cell."""
    # Load population grid
    pop_grid = get_population_grid(pop_path) # It is expected that this DataFrame has 'lon', 'lat' and 'population' columns 
    
    # Load Voronoi geometries
    voronoi_gdf = geom.load_shapes_efficiently(voronoi_path) # Expected to have 'geometry' column with polygons
    
    # Population per Voronoi cell via point_in_shape method from geom that takes lon, lat and shape
    for idx, voronoi_row in voronoi_gdf.iterrows():
        shape = voronoi_row['geometry']
        mask = geom.point_in_shape(pop_grid['lon'].values, pop_grid['lat'].values, shape)
        population_per_cell = pop_grid.loc[mask, 'population'].sum()
        voronoi_gdf.at[idx, 'population'] = population_per_cell    
    
    # Return the whole GeoDataFrame with population column
    return voronoi_gdf