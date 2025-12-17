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
    """Load, filter and combine radiation and wind-speed CSV files."""
    # 确保路径为 Path 对象
    rad_path, ws_path = Path(rad_path), Path(ws_path)
    
    if not rad_path.exists() or not ws_path.exists():
        raise FileNotFoundError(f"Input files not found: {rad_path} or {ws_path}")

    rad = pd.read_csv(rad_path, skiprows=skiprows)
    ws = pd.read_csv(ws_path, skiprows=skiprows)

    if 'Date' not in rad.columns or 'Date' not in ws.columns:
        raise ValueError("Input CSV files must contain a 'Date' column")

    rad['Date'] = pd.to_datetime(rad['Date'])
    ws['Date'] = pd.to_datetime(ws['Date'])

    # 过滤时间段
    rad_filtered = rad[(rad['Date'] >= start) & (rad['Date'] <= end)].copy()
    ws_filtered = ws[(ws['Date'] >= start) & (ws['Date'] <= end)].copy()

    # 去重并设置索引
    rad_filtered = rad_filtered.drop_duplicates(subset='Date').set_index('Date')
    ws_filtered = ws_filtered.drop_duplicates(subset='Date').set_index('Date')

    # 重命名列名以区分风速和辐射
    ws_filtered.columns = [f"ws_{col}" for col in ws_filtered.columns]
    rad_filtered.columns = [f"rad_{col}" for col in rad_filtered.columns]

    df_w = pd.concat([ws_filtered, rad_filtered], axis=1)
    logger.debug("Processed weather DataFrame shape: %s", df_w.shape)
    return df_w

def save_processed(df: pd.DataFrame, out_dir: Union[str, Path],
                   filename: str = "weather_processed.csv.gz") -> Path:
    """Save the processed DataFrame to out_dir/filename."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    full_path = out_path / filename
    df.to_csv(full_path, compression='gzip', index=True)
    logger.info("Saved processed weather to %s", full_path)
    return full_path

def _cli():
    # 在函数内动态获取根目录，避免全局作用域报错
    repo_root = find_repo_root(Path(__file__).parent)
    default_raw_dir = repo_root / "data" / "raw" / "weather"
    default_out_dir = repo_root / "data" / "processed"

    p = argparse.ArgumentParser(description="Process weather CSVs and save result")
    
    # 将动态路径设为默认值，同时允许用户通过命令行覆盖
    p.add_argument("--rad", default=default_raw_dir / "solar.csv", help="Path to radiation CSV")
    p.add_argument("--ws", default=default_raw_dir / "wind.csv", help="Path to wind-speed CSV")
    p.add_argument("--start", default="2015-01-01", help="Start date (inclusive)")
    p.add_argument("--end", default="2024-12-31", help="End date (inclusive)")
    p.add_argument("--out-dir", default=default_out_dir, help="Output directory")
    p.add_argument("--out-file", default="weather_processed.csv.gz", help="Output filename")
    p.add_argument("--skiprows", type=int, default=52, help="Rows to skip when reading CSVs")
    p.add_argument("--verbose", action="store_true", help="Enable debug logging")
    
    args = p.parse_args()
    
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    
    # 使用解析后的参数 args，不再使用硬编码字符串
    try:
        df = process_weather(
            rad_path=args.rad, 
            ws_path=args.ws, 
            start=args.start, 
            end=args.end, 
            skiprows=args.skiprows
        )
        out = save_processed(df, out_dir=args.out_dir, filename=args.out_file)
        print(f"Success! Processed data saved to: {out}")
    except Exception as e:
        logger.error("Failed to process weather data: %s", e)

if __name__ == "__main__":
    _cli()

# You need to make sure this the folder structure 
# Git/
# ├── .git (或 README.md)
# ├── data/
# │   └── raw/
# │       └── weather/
# │           ├── solar.csv
# │           └── wind.csv
# └── scripts/
    # └── weather_test.py