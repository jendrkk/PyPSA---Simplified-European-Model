# Output Format Specification

## process_load_data() Output Structure

### DataFrame Shape
- Rows: (Countries) × (Hourly timestamps from 2015-01-01 to 2024-12-31 UTC)
- Columns: 4

### Column Details

#### 1. Timestamp
```
Type: datetime64[ns, UTC]
Range: 2015-01-01 00:00:00+00:00 to 2024-12-31 23:00:00+00:00
Frequency: Hourly
Timezone: UTC
Format: Pandas DatetimeIndex with UTC timezone
```

#### 2. Country
```
Type: object (string)
Values: ISO2 country codes
Examples: 'AT', 'BE', 'BG', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FI', 'FR', 'DE', 'GR', 'HU', 'IE', 'IT', 'LV', 'LT', 'LU', 'MT', 'NL', 'PL', 'PT', 'RO', 'SK', 'SI', 'ES', 'SE', 'UK'
All EU27 + UK
```

#### 3. Load
```
Type: float64
Values: Hourly electricity load in [specified unit, e.g., MW]
Range: Positive values (no negative load)
Missing: NONE (guaranteed no NaN)
Note: All original missing values filled according to country-specific rules
```

#### 4. Missing
```
Type: bool
Values: True or False
True: Value was originally missing from combined_load_data
False: Value was originally present
Usage: Track data quality and distinguish observed vs. interpolated values
```

## Example Output Snippet

```
              Timestamp Country         Load  Missing
0  2015-01-01 00:00:00+00:00       AT  12345.67     False
1  2015-01-01 00:00:00+00:00       BE  23456.78     True
2  2015-01-01 00:00:00+00:00       BG   3456.89     False
3  2015-01-01 00:00:00+00:00       HR   4567.90     False
4  2015-01-01 00:00:00+00:00       CY    567.01     False
...
N  2024-12-31 23:00:00+00:00       UK  45678.90     False
```

## Verification Checks

After calling `process_load_data()`:

```python
result_df = process_load_data(combined_df)

# Check 1: No missing Load values
assert result_df['Load'].isna().sum() == 0, "Load has NaN values!"

# Check 2: Correct timezone
assert str(result_df['Timestamp'].dtype) == 'datetime64[ns, UTC]', "Timestamp not UTC!"

# Check 3: Correct time range
assert result_df['Timestamp'].min() == pd.Timestamp('2015-01-01', tz='UTC'), "Start date wrong!"
assert result_df['Timestamp'].max() == pd.Timestamp('2024-12-31 23:00:00', tz='UTC'), "End date wrong!"

# Check 4: All required columns present
required_cols = {'Timestamp', 'Country', 'Load', 'Missing'}
assert required_cols.issubset(result_df.columns), "Missing columns!"

# Check 5: Correct data types
assert result_df['Load'].dtype == 'float64', "Load not float!"
assert result_df['Missing'].dtype == 'bool', "Missing not bool!"

# Check 6: Complete coverage
unique_countries = result_df['Country'].nunique()
unique_timestamps = result_df['Timestamp'].nunique()
expected_rows = unique_countries * unique_timestamps
assert len(result_df) == expected_rows, "Missing rows!"
```

## Sorting

Output is sorted by:
1. **Timestamp** (ascending, chronological)
2. **Country** (ascending, alphabetical)

Example:
```
2015-01-01 00:00:00+00:00  AT  ...
2015-01-01 00:00:00+00:00  BE  ...
2015-01-01 00:00:00+00:00  BG  ...
2015-01-01 01:00:00+00:00  AT  ...
2015-01-01 01:00:00+00:00  BE  ...
```

## Data Quality Metrics

After processing:

```python
# Check data coverage by country
coverage = result_df.groupby('Country').agg({
    'Load': ['count', 'mean', 'std', 'min', 'max'],
    'Missing': 'sum'
})

# Should show:
# - count: 87,648 hours for 2015-2024 (10 years × 365.25 days/year × 24 hours)
# - Missing: varies by country, but all Load values filled
```

## Load Values by Country (Approximate Ranges)

Typical hourly load values (MW):
- Austria (AT): 5,000 - 12,000
- Belgium (BE): 8,000 - 18,000
- Germany (DE): 30,000 - 80,000
- France (FR): 40,000 - 90,000
- Ireland (IE): 2,000 - 5,000
- UK (GB): 25,000 - 60,000
- Poland (PL): 10,000 - 30,000
- Spain (ES): 15,000 - 50,000
- Italy (IT): 20,000 - 60,000

(Actual values depend on season, time of day, weather, economic activity)

## Memory Footprint

Approximate size:
- 28 countries × 87,648 hours = 2,454,144 rows
- 4 columns × 2,454,144 rows
- Memory: ~200-300 MB (depending on dtype optimization)

## CSV Export

To save output:
```python
result_df.to_csv('processed_load_data.csv', index=False)

# Or with compression
result_df.to_csv('processed_load_data.csv.gz', index=False, compression='gzip')
```

CSV format will be:
```
Timestamp,Country,Load,Missing
2015-01-01 00:00:00+00:00,AT,12345.67,False
2015-01-01 00:00:00+00:00,BE,23456.78,True
...
```

## Database Compatibility

Can be stored in SQL databases:
```python
# SQLAlchemy example
from sqlalchemy import create_engine
engine = create_engine('postgresql://user:password@localhost/db')
result_df.to_sql('electricity_load', engine, if_exists='replace')
```

Column types for database:
- Timestamp: TIMESTAMP WITH TIME ZONE
- Country: VARCHAR(2)
- Load: NUMERIC(12,2)
- Missing: BOOLEAN
