# Data Directory

Place your data files here:

- `train.csv`: Training data with anomaly labels
- `test.csv`: Test data (optional)

## Expected Format

The CSV files should have the following columns:
- `timestamp`: DateTime string (e.g., "2020-01-01 00:00:00")
- `building_id`: Building identifier (integer)
- `meter_reading`: Energy meter reading value (float)
- `anomaly`: Binary label (0 = normal, 1 = anomaly) - required for train.csv

Example:
```csv
timestamp,building_id,meter_reading,anomaly
2020-01-01 00:00:00,1,1234.5,0
2020-01-01 01:00:00,1,1235.2,0
2020-01-01 02:00:00,1,1500.0,1
```
