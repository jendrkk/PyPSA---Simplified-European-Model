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
import threading
import time
import os

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

# Configure logging to stdout so monitor and other logs appear together
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# Useful fast functions
v_startswith = np.vectorize(lambda x, prefix: str(x).startswith(prefix))
v_contains = np.vectorize(lambda x, substring: substring in str(x))
v_first_n = np.vectorize(lambda x, n: str(x)[:n])
v_and = np.vectorize(lambda x, y: x and y)
v_isbool = np.vectorize(lambda x,y: x==y)

def get_population_grid(path: str | Path) -> pd.DataFrame:
    """Load population grid data from Eurostat and return as DataFrame."""
    # Load parquet file
    logger.info(f"Loading population grid data from {path}...")
    pop_grid = pd.read_parquet(path)
    logger.info(f"Loaded population grid with {len(pop_grid)} entries.")
    
    return pop_grid
    
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
    voronoi_gdf['EU_population_share'] = 0.0  # Initialize population share column

    # If cached CSV exists and already contains population information, skip heavy computation
    cashe_dir = geom.DEFAULT_CACHE_DIR
    voronoi_csv_path = cashe_dir / voronoi_path.name.replace('.parquet', '.csv')
    if voronoi_csv_path.exists():
        try:
            existing = pd.read_csv(voronoi_csv_path)
            if 'population' in existing.columns and 'EU_population_share' in existing.columns:
                logger.info(f"Found existing population CSV at {voronoi_csv_path}; skipping computation.")
                return existing
        except Exception:
            logger.info("Existing Voronoi CSV found but failed to read; will recompute.")
    # Population per Voronoi cell via point_in_shape method from geom that takes lon, lat and shape
    logger.info("Calculating population per Voronoi cell...")
    check_args = [(idx, row, pop_grid) for idx, row in voronoi_gdf.iterrows()]

    # Helper to run mapping in pool with fallback to sequential mapping
    def _map_with_fallback(func, args_list):
        try:
            with mp.Pool() as pool:
                result = pool.map(func, args_list)
            logger.info(f"Parallel map completed using {psutil.cpu_count(logical=False)} physical cores.")
            logger.info(f"Memory usage: {psutil.virtual_memory().percent}%")
            logger.info(f"CPU usage: {psutil.cpu_percent(interval=1)}%")
            return result
        except Exception as exc:
            logger.warning(f"Parallel processing failed ({exc}); falling back to sequential loop.")
            results = []
            for a in args_list:
                results.append(func(a))
            return results

    pop_datas = _map_with_fallback(_sqares_in_rect, check_args)
    
    logger.info("Summing population within each Voronoi cell...")
    check_args = [(idx, voronoi_gdf.iloc[idx], pop_datas[idx]) for idx in range(len(voronoi_gdf))]
    populations = _map_with_fallback(_sum_population_in_shape, check_args)
    
    logger.info("Population calculation completed.")
    
    voronoi_gdf['population'] = populations
    voronoi_gdf['EU_population_share'] = voronoi_gdf['population'] / voronoi_gdf['population'].sum()

    # Update or create CSV cache with population values
    try:
        if voronoi_csv_path.exists():
            voronoi_csv = pd.read_csv(voronoi_csv_path)
        else:
            # Create a minimal CSV structure from GeoDataFrame if not present
            voronoi_csv = pd.DataFrame({
                'bus_id': voronoi_gdf.get('bus_id', pd.Series(range(len(voronoi_gdf))))
            })
        voronoi_csv['country'] = voronoi_gdf['country']
        voronoi_csv['population'] = voronoi_gdf['population']
        voronoi_csv['EU_population_share'] = voronoi_gdf['EU_population_share']
        voronoi_csv.to_csv(voronoi_csv_path, index=False)
        logger.info(f"Saved Voronoi population data to {voronoi_csv_path}.")
    except Exception as exc:
        logger.warning(f"Failed to write voronoi CSV cache ({exc}); returning computed DataFrame anyway.")
    
    # We save also the geodataframe as parquet for future use
    voronoi_parquet_path = cashe_dir / voronoi_path.name.replace('.parquet', '_P.parquet')
    try:
        geom.save_shapes_efficiently(voronoi_gdf, voronoi_parquet_path)
        logger.info(f"Saved Voronoi GeoDataFrame with population to {voronoi_parquet_path}.")
    except Exception as exc:
        logger.warning(f"Failed to write voronoi parquet cache ({exc}); continuing.")
    
    # Save also population variable as separate file with bus index and population only
    pop_name = voronoi_path.name.replace('.parquet', '_population.csv')
    pop_path = cashe_dir / pop_name
    try:
        pop_df = voronoi_csv[['bus_id', 'population', 'EU_population_share']]
        pop_df.to_csv(pop_path, index=False)
        logger.info(f"Saved population data to {pop_path}.")
    except Exception:
        logger.warning("Could not write population-only CSV; continuing.")

    # Return the cached/updated CSV DataFrame (keeps original function behavior)
    return voronoi_csv

def execute_CPV(pop_path: str | Path, voronoi_path: str | Path, options: Dict = None):
    """Execute the calculation of population per Voronoi cell."""
    import datetime as dt

    answer = input('Show live system monitoring during run? (y/N): ').strip().lower()
    enable_monitor = answer.startswith('y')

    stop_event = threading.Event()
    monitor_thread = None
    if enable_monitor:
        monitor_thread = threading.Thread(target=monitor_system, args=(stop_event, 10), daemon=True)
        monitor_thread.start()

    start_time = dt.datetime.now()
    try:
        calculate_population_voronoi(pop_path, voronoi_path)
    finally:
        # Ensure we stop the monitor thread if it was started
        if enable_monitor:
            stop_event.set()
            # Give monitor a moment to exit
            monitor_thread.join(timeout=5)
        end_time = dt.datetime.now()
        logger.info(f"Total computation time: {end_time - start_time}")

def load_load_data(path: str | Path) -> pd.DataFrame:
    """Load load data from CSV file."""
    logger.info(f"Loading load data from {path}...")
    load_data_path = repo_root / "data" / "raw" / "time_series_60min_singleindex_filtered.csv"

    # Load the load data
    load_data = pd.read_csv(load_data_path, index_col=0, parse_dates=True)
    # Column 'utc_timestamp' to datetime index and in GMT+1
    load_data.index = pd.to_datetime(load_data.index).tz_convert(60*60)
    
    logger.info(f"Loaded load data with shape {load_data.shape}.")
    return load_data

def filter_load_data(load_data: pd.DataFrame, countries: list[str]) -> pd.DataFrame:
    """Filter load data for specified countries."""
    logger.info(f"Filtering load data for countries...")
    cols = np.array(load_data.columns.tolist())
    country_tags = v_first_n(cols, 2)
    country_tags = country_tags[country_tags != 'GB']
    country_tags = np.unique(country_tags)

    load_cols = [f"{country}_load_actual_entsoe_transparency" if country in countries else "not included" for country in country_tags]

    mask = np.where(v_isbool(v_startswith(load_cols, "not").tolist(), False).tolist())[0].tolist()
    new_load_cols = []
    for i in mask:
        new_load_cols.append(load_cols[i])

    new_load_cols

    cols = cols[v_startswith(cols, "GB")]
    cols = cols[v_contains(cols, "_load_actual")]

    new_load_cols = new_load_cols + cols.tolist()
    
    if len(new_load_cols) == 30:
        logger.info("All country load data included.")
    else:
        logger.info(f"Included load data for {len(new_load_cols)} countries.")

    filtered_load_data = load_data.copy()
    filtered_load_data = filtered_load_data[new_load_cols]
    filtered_load_data.columns = v_first_n(filtered_load_data.columns,2)
    
    return filtered_load_data

def save_load_data(load_data: pd.DataFrame, path: str | Path):
    """Save filtered load data to CSV file."""
    logger.info(f"Saving filtered load data to {path}...")
    load_data.to_csv(path)
    logger.info("Load data saved.")
    
def compute_demand(voronoi_path: str | Path, load_data_path: str | Path, output_path: str | Path, join: bool = False):
    """Compute demand data per Voronoi cell based on population and load data."""
    # Load Voronoi population data
    cashe_dir = geom.DEFAULT_CACHE_DIR
    voronoi_csv_path = cashe_dir / voronoi_path.name.replace('.parquet', '.csv')
    voronoi_pop = pd.read_csv(voronoi_csv_path)
    
    # Load load data
    load_data = load_load_data(load_data_path)
    
    # Filter load data for countries in Voronoi cells
    filtered_load_data = filter_load_data(load_data, geom.EU27)
    
    # Compute demand per Voronoi cell
    demand_data = pd.DataFrame(index=filtered_load_data.index)
    for idx, row in voronoi_pop.iterrows():
        
        country = row['country']
        bus_id = row['bus_id']
        pop_share = row['EU_population_share']
        
        load_col = f"{country}_load_actual_entsoe_transparency"
        if load_col in filtered_load_data.columns:
            demand_data[bus_id] = filtered_load_data[load_col] * pop_share
        else:
            logger.warning(f"Load data for country {country} not found. Skipping bus {bus_id}.")
    
    # Save demand data
    save_load_data(demand_data, output_path)
    

if __name__ == "__main__":
    def monitor_system(stop_event: threading.Event, interval: int = 10) -> None:
        """Background system monitor that prints CPU (per-core) and memory usage every `interval` seconds.

        The monitor stops when `stop_event` is set.
        """
        proc = psutil.Process(os.getpid())
        # Prime cpu_percent measurements
        psutil.cpu_percent(interval=None)
        while not stop_event.is_set():
            # Get per-core CPU usage (1s sampling for accurate per-core values)
            per_core = psutil.cpu_percent(percpu=True, interval=1)
            total = psutil.cpu_percent()
            vm = psutil.virtual_memory()
            rss = proc.memory_info().rss / (1024 ** 2)
            vms = proc.memory_info().vms / (1024 ** 2)
            # Log monitor output so it doesn't get cleared and interleaves with other logs
            lines = []
            lines.append(f"System monitor (pid={proc.pid}) - {time.strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append(f"Total CPU: {total:.1f}%    Logical cores: {psutil.cpu_count(logical=True)}    Physical cores: {psutil.cpu_count(logical=False)}")
            lines.append("Per-core: " + ' '.join(f"{p:5.1f}%" for p in per_core))
            lines.append(f"System memory: {vm.percent:.1f}% (available={vm.available // (1024**2)} MB)")
            lines.append(f"Process memory: RSS={rss:.1f} MB  VMS={vms:.1f} MB")
            lines.append("(Press Ctrl+C to cancel monitoring output; program will continue.)")
            logger.info('\n'.join(lines))
            # Wait for the remainder of the interval, checking stop_event
            for _ in range(interval - 1):
                if stop_event.is_set():
                    break
                time.sleep(1)


    def main():
        """
        Main function to compute population and demand data.
        """
        
        # Ask if we want a joined version or not
        force_download = input("Force recomputation of population data? (y/N): ").strip().lower() == 'y'
        join = input('Compute population data for joined Voronoi cells? (y/N): ').strip().lower() == 'y'
        pop_path = repo_root / 'data' / 'processed' / 'jrc_population_nonzero.parquet'
        voronoi_path = geom.DEFAULT_CACHE_DIR / Path(f"voronoi_eu27{'_join' if join else ''}.parquet")

        # Ensure voronoi_path is a Path
        voronoi_path = Path(voronoi_path)

        # If the voronoi population CSV does not exist in cache, create it first
        cashe_dir = geom.DEFAULT_CACHE_DIR
        pop_csv = cashe_dir / voronoi_path.name.replace('.parquet', '_population.csv')
        if not pop_csv.exists() or force_download:
            execute_CPV(pop_path, voronoi_path)
        else:
            logger.info(f"Population data for {voronoi_path} already exists at {pop_csv}. Skipping computation.")
            
        force_demand = input("Compute demand data now? (y/N): ").strip().lower() == 'y'
        # After population exists, compute demand data and save to processed folder (skip if exists)
        demand_out = repo_root / 'data' / 'processed' / f"demand_{voronoi_path.stem}.csv"
        if not demand_out.exists() or force_demand:
            try:
                (voronoi_path, repo_root / 'data' / 'raw' / 'time_series_60min_singleindex_filtered.csv', demand_out)
                logger.info(f"Demand data saved to {demand_out}.")
            except Exception as exc:
                logger.warning(f"Failed to compute demand data: {exc}")
        else:
            logger.info(f"Demand output {demand_out} already exists. Skipping demand computation.")

    try:
        main()
    except KeyboardInterrupt:
        # Allow user to abort with Ctrl+C gracefully
        logger.info('Interrupted by user.')
