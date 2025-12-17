# Quick Reference: process_load_data()

## Function Signature
```python
def process_load_data(load_df: pd.DataFrame) -> pd.DataFrame
```

## Location
File: `scripts/fetch_load_data.py` (lines 200-385)

## What It Does
Converts combined electricity load data from CET to UTC, applies country-specific handling (UK scaling, IE gap-filling), and fills remaining missing values using time interpolation.

## Usage
```python
from fetch_load_data import process_load_data
import pandas as pd

# Load combined data (output from combine_all)
combined_df = pd.read_csv('combined_load_data.csv', 
                           index_col=['Timestamp', 'Country'],
                           parse_dates=True)

# Process the data
result_df = process_load_data(combined_df)

# Result has no missing Load values
print(result_df)
```

## Input Format
- Multi-index DataFrame: `['Timestamp', 'Country']`
- Column: `'Load'`
- Timestamps in CET (GMT+1)
- Long format (one row per country/timestamp)

## Output Format
- Regular index (0, 1, 2, ...)
- Columns: `['Timestamp', 'Country', 'Load', 'Missing']`
- Timestamps in UTC
- Time range: 2015-01-01 00:00:00 to 2024-12-31 23:00:00 UTC
- Load column: NO NaN values
- Missing column: Boolean tracking originally-missing values

## Special Handling

### UK (GB)
- Scales historical periods using 2020-2021 reference data
- Scaling factors by period:
  - 2021-06-14 to 2022-06-14: ×0.975
  - 2022-06-14 to 2023-06-14: ×0.95
  - 2023-06-14 to 2024-02-28: ×0.91
  - 2024-02-29: ×1.0
  - 2024-03-01 to 2024-06-14: ×0.91
  - 2024-06-14 onwards: ×0.91
- Gaps filled with time interpolation

### Ireland (IE)
- Identifies gaps of 7+ consecutive hours
- Attempts sequential filling strategies:
  1. Same hours previous day (if ≤1 NaN)
  2. Same hours next day (if ≤1 NaN)
  3. Same time previous year (if no 8+ hour gaps)
- Handles leap year mismatches (Feb 28/29)
- Prints status for each fill attempt
- Warns if gap cannot be filled

### Other Countries
- Simple time interpolation: `interpolate(method='time')`
- Fills all gaps linearly in time domain

## Key Features
✓ Timezone-safe (CET → UTC conversion)
✓ Handles leap years correctly
✓ Tracks original missing data
✓ Maintains data integrity
✓ Country-specific logic preserved from notebook
✓ No NaN values in Load column

## Output Guarantee
- All Load values are numeric (no NaN)
- Missing column indicates original status
- Timestamps in UTC timezone
- Complete hourly coverage 2015-2024
- Sorted by Timestamp and Country

## Example Output
```
   Timestamp Country        Load  Missing
0 2015-01-01       AT  1234.5678     False
1 2015-01-01       BE  2345.6789      True
2 2015-01-01       BG   345.6789     False
...
```

## Related Functions
- `combine_all()` - Produces input for this function
- `find_missing_ranges()` - Helper for identifying gaps (used internally)

## Notes
- Existing functions in fetch_load_data.py are NOT modified
- Function is production-ready with type hints and docstrings
- Suitable for PyPSA energy model workflows
