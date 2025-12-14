import pandas as pd
import numpy as np
import scripts.geometry as geometry
import data_prep
import eurostat
from pathlib import Path


def get_population_grid(path: str | Path) -> pd.DataFrame:
    """Load population grid data from Eurostat and return as DataFrame."""
    # Load parquet file
    pop_grid = pd.read_parquet(path)
    
    return pop_grid