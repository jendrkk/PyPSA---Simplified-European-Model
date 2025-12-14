"""
Lightweight data preparation helpers for OSM-derived CSVs.

Provides `prepare_osm_source` which loads common CSVs from the
prebuilt OSM directory and returns a dict of pandas DataFrames.
This is intentionally minimal and robust for notebook usage.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import numpy as np
import pickle
import gzip
import json

class RawData:
    """Container for OSM Prebuilt Electricity Network data as DataFrames."""
    
    def __init__(self, data: Dict[str, pd.DataFrame | None] | None) -> None:
        self.data = data
    
    def get(self, key: str) -> pd.DataFrame | None:
        return self.data.get(key, None)
    
    def set(self, key: str, value: pd.DataFrame | None) -> None:
        self.data[key] = value
    
    def keys(self):
        return self.data.keys()
    
    def items(self):
        return self.data.items()
    
    def data(self) -> Dict[str, pd.DataFrame | None]:
        """Return the internal data dictionary."""
        return self.data
    
    def bus_coords(self) -> pd.DataFrame | None:
        """Return DataFrame of bus coordinates if available."""
        buses = self.data.get("buses", None)
        if buses is None:
            return None
        if 'x' in buses.columns and 'y' in buses.columns:
            return buses[['bus_id', 'x', 'y']]
        return None
    
    def match_bus_id(self, lat: float, lon: float, tol: float = 1e-6) -> str | None:
        """Find bus_id matching given coordinates within tolerance."""
        buses = self.data.get("buses", None)
        if buses is None:
            return None
        
        t = min(len(str(lat).split(".")[1]), 
                len(str(lon).split(".")[1]), 
                len(str(np.format_float_positional(tol, trim='-')).split(".")[1])
                )
        tol = 10**(-t)
        
        for idx, row in buses.iterrows():
            if abs(row['x'] - x) <= tol and abs(row['y'] - y) <= tol:
                return row['bus_id']
        return None
    
    def save(self, output_path: str | Path) -> str:
        """
        Serialize source inputs needed to build a network.

        Attempts gzipped JSON first for portability; if the data contains
        non-JSON-serializable objects (e.g., pandas DataFrames), falls back
        to gzipped pickle (`.pkl.gz`). Returns the written file path.
        """
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)

        data_dict = self.data

        # Try gzipped JSON first for portability
        try:
            with gzip.open(str(p), "wt", encoding="utf-8") as f:
                json.dump(data_dict, f, separators=(",", ":"))
            return str(p)
        except (TypeError, OverflowError):
            # Fallback to gzipped pickle. Be careful not to produce double
            # extensions like `.pkl.pkl.gz` -- normalize the output path.
            s = str(p)
            if s.endswith(".pkl.gz"):
                out = Path(s)
            elif s.endswith(".json.gz"):
                out = Path(s[:-len(".json.gz")] + ".pkl.gz")
            elif s.endswith(".gz"):
                out = Path(s[:-3] + ".pkl.gz")
            else:
                out = Path(s + ".pkl.gz")

            with gzip.open(str(out), "wb") as f:
                pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
            return str(out)
    
    def load(self, input_path: str) -> Dict[str, Any]:
        """Load previously serialized source inputs supporting gzipped JSON and gzipped pickle."""
        # Try JSON text mode first
        try:
            with gzip.open(input_path, "rt", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            # Fallback to pickle binary
            with gzip.open(input_path, "rb") as f:
                data = pickle.load(f)
        self.data = data
        return data
    
def _read_csv_if_exists(path: Path, special_handling: bool = True) -> pd.DataFrame | None:
    if not path.exists():
        return None
    if not special_handling:
        return pd.read_csv(path)
    
    return load_csv_drop_last(path)


def load_csv_drop_last(path, encoding="utf-8"):
    """Read a CSV where the last (true) column may contain commas
    inside a geometry text (e.g. 'LINESTRING (...)').

    This preserves the header-determined number of columns. If a data line
    contains more comma-separated parts than the header, everything from the
    start of the geometry (detected by 'LINESTRING' or 'MULTILINESTRING')
    is packed into the last column as a single string.
    """
    with open(path, "r", encoding=encoding) as f:
        header = f.readline().rstrip("\n")
        cols = header.split(",")
        n_true = len(cols)

        rows = []
        for line in f:
            line = line.rstrip("\n")
            # try to locate geometry start (robust to presence/absence of surrounding quotes)
            geom_idx = line.find('LINESTRING')
            if geom_idx == -1:
                geom_idx = line.find('MULTILINESTRING')

            if geom_idx != -1:
                # find the comma separating the last true column from geometry
                sep = line.rfind(',', 0, geom_idx)
                if sep == -1:
                    # can't find separator before geometry -> conservative split
                    parts = line.split(',', n_true-1)
                    if len(parts) < n_true:
                        parts += [''] * (n_true - len(parts))
                    row = parts[:n_true]
                else:
                    left = line[:sep]
                    last = line[sep+1:]
                    left_parts = left.split(',')
                    # ensure exactly n_true-1 entries on the left side
                    if len(left_parts) > n_true-1:
                        left_parts = left_parts[:n_true-1]
                    while len(left_parts) < n_true-1:
                        left_parts.append('')
                    row = left_parts + [last]
            else:
                # no geometry marker -> split conservatively into n_true columns
                parts = line.split(',', n_true-1)
                if len(parts) < n_true:
                    parts += [''] * (n_true - len(parts))
                row = parts[:n_true]

            rows.append(row)

    return pd.DataFrame(rows, columns=cols)

def shift_right(df: pd.DataFrame) -> pd.DataFrame:
    """Shift all columns of DataFrame `df` to the right by one place,
    filling the leftmost column with df.index.
    """ 
    first_col = df.index
    names = df.columns.tolist()
    new_names = names + [names[-1] + "_2"]
    
    shifted_data = {new_names[0]: first_col}
    for i in range(len(names)):
        shifted_data[new_names[i + 1]] = df[names[i]]
    
    shifted_df = pd.DataFrame(shifted_data)
    shifted_df.index = [i for i in range(len(shifted_df))]
    
    return shifted_df

def prepare_osm_source(osm_dir: str | Path) -> Dict[str, object]:
    """
    Load OSM Prebuilt Electricity Network CSVs into a dictionary.

    Parameters
    - osm_dir: path to the folder containing CSVs like `buses.csv`, `lines.csv`, etc.

    Returns
    - dict where keys are filenames (without extension) and values are DataFrames or None.
    """
    
    osm_dir = Path(osm_dir)
    files = [
        "buses.csv",
        "lines.csv",
        "converters.csv",
        "links.csv",
        "transformers.csv",
        "generators.csv",
        "loads.csv",
        "storage.csv"
    ]
    special_handling_files = [
        "lines.csv",
        "links.csv"
    ]
    out = {}
    for f in files:
        p = osm_dir / f
        df = _read_csv_if_exists(p, special_handling=(f in special_handling_files))
        if df is not None and df.index[0] != 0:
            df = shift_right(df)

        # Try to convert every column to float; if conversion fails, leave it unchanged.
        if df is not None:
            for col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors="raise").astype(float)
                except Exception:
                    # leave column as-is if any value cannot be converted to float
                    pass

        key = Path(f).stem
        out[key] = df

    return out

def prepare_generator_data(gen_data_path: str | Path) -> pd.DataFrame | None:
    """
    Load generator data CSV into a DataFrame.

    Parameters
    - gen_data_path: path to the `generator_data.csv` file.

    Returns
    - DataFrame with generator data or None if file does not exist.
    """
    gen_data_path = Path(gen_data_path)
    if not gen_data_path.exists():
        return None

    df = pd.read_csv(gen_data_path)

    # Try to convert every column to float; if conversion fails, leave it unchanged.
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors="raise").astype(float)
        except Exception:
            # leave column as-is if any value cannot be converted to float
            pass

    return df