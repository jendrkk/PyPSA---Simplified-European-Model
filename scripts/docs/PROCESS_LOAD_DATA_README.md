# Process Load Data Function Documentation

## Overview

Two new functions have been added to `fetch_load_data.py`:

1. **`find_missing_ranges(df, n_missing=6)`** - Utility function to identify consecutive missing values
2. **`process_load_data(load_df)`** - Main function to process combined load data

## Function: `process_load_data(load_df)`

### Purpose
Converts combined load data from CET timestamps to UTC, applies special handling for UK and Ireland, and fills remaining missing values using time-based interpolation.

### Input
- **load_df**: DataFrame from `combine_all()` with structure:
  - Multi-index: `['Timestamp', 'Country']`
  - Column: `'Load'`
  - Timestamps in CET (GMT+1)
  - Long format (one row per country per timestamp)

### Output
- **DataFrame** with columns:
  - `'Timestamp'`: UTC timezone (from 2015-01-01 00:00:00+00:00 to 2024-12-31 23:00:00+00:00)
  - `'Country'`: ISO2 country codes
  - `'Load'`: Hourly electricity load values (no NaN values after processing)
  - `'Missing'`: Boolean indicating if the original value was missing

### Processing Logic

#### 1. UTC Conversion
- Converts from CET (GMT+1) to UTC timezone
- Preserves the long format (countries in separate rows)

#### 2. Time Range Alignment
- Creates complete hourly time series from 2015-01-01 to 2024-12-31 (UTC)
- Merges original data into this complete range
- Tracks which values were originally missing

#### 3. Special UK Handling
UK data receives specific scaling adjustments for historical periods:
- 2021-06-14 to 2022-06-14: Scale by 0.975
- 2022-06-14 to 2023-06-14: Scale by 0.95
- 2023-06-14 to 2024-02-28: Scale by 0.91
- 2024-02-29 (leap day): Scale by 1.0
- 2024-03-01 to 2024-06-14: Scale by 0.91
- 2024-06-14 onwards: Scale by 0.91

Reference periods use data from 2020-2021.
After scaling, remaining gaps are filled with time interpolation.

#### 4. Special Ireland (IE) Handling
For consecutive missing values of 7+ hours:
1. Try filling from previous day's same hours (if at most 1 NaN)
2. Try filling from next day's same hours (if at most 1 NaN)
3. Try filling from same time last year (if no 8+ hour gaps in that period)
4. Handle leap year mismatches (Feb 29 vs Feb 28)
5. Print warnings if gaps cannot be filled

#### 5. Default Missing Value Handling
For all other countries:
- Use `interpolate(method='time')` to fill gaps
- This performs linear interpolation in time domain

### Usage Example

```python
import pandas as pd
from fetch_load_data import process_load_data

# Load the combined data (from combine_all())
combined_df = pd.read_csv('combined_load_data.csv', index_col=['Timestamp', 'Country'], parse_dates=True)

# Process the data
processed_df = process_load_data(combined_df)

# Result has no missing Load values and includes Missing column
print(processed_df.head())
print(processed_df.isna().sum())  # Should show 0 NaN values in 'Load' column
```

### Important Notes

1. **Data Format**: Input should maintain multi-index structure from `combine_all()`
2. **Timezone Handling**: Input timestamps in CET, output in UTC
3. **No Data Loss**: Original missing values are tracked in the `Missing` column
4. **Leap Year Safe**: Handles Feb 29 properly when interpolating
5. **Country-Specific**: Special UK and IE logic; all others use time interpolation

### Implementation Details

- `find_missing_ranges()` helper identifies gaps of 7+ hours for IE processing
- Uses `pd.DateOffset()` for robust date arithmetic (handles leap years, month boundaries)
- `np.concatenate()` handles length mismatches during leap year edge cases
- Final output is sorted by Timestamp and Country for consistency

## Function: `find_missing_ranges(df, n_missing=6)`

### Purpose
Identifies consecutive missing values in a DataFrame's 'Load' column.

### Parameters
- `df`: DataFrame with 'Load' column
- `n_missing`: Minimum consecutive missing values to report (default: 6)

### Returns
- List of tuples `(start_timestamp, end_timestamp)` for each gap of n_missing or more consecutive NaNs

### Usage
```python
# Used internally by process_load_data for IE handling
missing_ranges = find_missing_ranges(df_subset, n_missing=7)
for start, end in missing_ranges:
    print(f"Gap from {start} to {end}")
```
