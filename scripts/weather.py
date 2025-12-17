"""Weather processing utilities.

This module provides a small function to load, filter and combine
wind-speed and solar radiation CSV files and save the processed
DataFrame to ``data/processed``.

Usage (CLI):
  python scripts/weather.py --ws PATH_TO_WS_CSV --rad PATH_TO_RAD_CSV

The original quick script read CSVs with `skiprows=52`; this module
keeps that behaviour but exposes the paths and date-range as arguments.
"""

from pathlib import Path
import argparse
import logging
import pandas as pd
from typing import Union

logger = logging.getLogger(__name__)

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

def process_weather(rad_path: Union[str, Path], ws_path: Union[str, Path],
					start: str = "2015-01-01", end: str = "2024-12-31",
					skiprows: int = 52) -> pd.DataFrame:
	"""Load, filter and combine radiation and wind-speed CSV files.

	Parameters
	- rad_path, ws_path: paths to CSV files (they must contain a `Date` column)
	- start, end: inclusive date range (YYYY-MM-DD)
	- skiprows: int passed to ``pd.read_csv`` (keeps original behaviour)

	Returns
	- Combined DataFrame indexed by datetime `Date` with prefixed columns.
	"""
	rad = pd.read_csv(rad_path, skiprows=skiprows)
	ws = pd.read_csv(ws_path, skiprows=skiprows)

	if 'Date' not in rad.columns or 'Date' not in ws.columns:
		raise ValueError("Input CSV files must contain a 'Date' column")

	rad = rad.copy()
	ws = ws.copy()
	rad['Date'] = pd.to_datetime(rad['Date'])
	ws['Date'] = pd.to_datetime(ws['Date'])

	rad_filtered = rad[(rad['Date'] >= start) & (rad['Date'] <= end)].copy()
	ws_filtered = ws[(ws['Date'] >= start) & (ws['Date'] <= end)].copy()

	rad_filtered = rad_filtered[~rad_filtered.index.duplicated(keep='first')]
	ws_filtered = ws_filtered[~ws_filtered.index.duplicated(keep='first')]

	ws_filtered.set_index('Date', inplace=True)
	rad_filtered.set_index('Date', inplace=True)

	ws_filtered.columns = [f"ws_{col}" for col in ws_filtered.columns]
	rad_filtered.columns = [f"rad_{col}" for col in rad_filtered.columns]

	df_w = pd.concat([ws_filtered, rad_filtered], axis=1)
	logger.debug("Processed weather DataFrame shape: %s", df_w.shape)
	return df_w


def save_processed(df: pd.DataFrame, out_dir: Union[str, Path] = "data/processed",
				   filename: str = "weather_processed.csv.gz") -> Path:
	"""Save the processed DataFrame to `out_dir/filename`.

	The file is written as a gzipped CSV to avoid additional runtime
	dependencies.
	"""
	out_path = Path(out_dir)
	out_path.mkdir(parents=True, exist_ok=True)
	full = out_path / filename
	df.to_csv(full, compression='gzip', index=True)
	logger.info("Saved processed weather to %s", full)
	return full


def _cli():
    
	p = argparse.ArgumentParser(description="Process weather CSVs and save result")
	p.add_argument("--rad", required=False, help="Path to radiation CSV")
	p.add_argument("--ws", required=False, help="Path to wind-speed CSV")
	p.add_argument("--start", default="2015-01-01", help="Start date (inclusive)")
	p.add_argument("--end", default="2024-12-31", help="End date (inclusive)")
	p.add_argument("--out-dir", default=REPO_ROOT / "data" / "processed" , help="Output directory")
	p.add_argument("--out-file", default="weather_processed.csv.gz", help="Output filename")
	p.add_argument("--skiprows", type=int, default=52, help="Rows to skip when reading CSVs")
	p.add_argument("--verbose", action="store_true", help="Enable debug logging")
	args = p.parse_args()
    
	logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
						format="%(asctime)s %(levelname)s %(message)s")
    
	df = process_weather(Path(REPO_ROOT / "solar.csv"), Path(REPO_ROOT / 'wind.csv'), start=args.start, end=args.end, skiprows=args.skiprows)
	out = save_processed(df, out_dir=args.out_dir, filename=args.out_file)
	print(out)
    

if __name__ == "__main__":
    REPO_ROOT = find_repo_root(Path(__file__).parent) / "data" / "raw" / "weather"
    _cli()