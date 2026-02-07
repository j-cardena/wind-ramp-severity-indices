# Data Description

## Input Data Requirements

The ramp severity indices require wind power time series data with the following characteristics:

### Format
- **File format**: CSV (comma-separated values)
- **Columns**: 
  - `timestamp`: DateTime or hours from start
  - `power`: Power output (normalized 0-1 or actual MW)

### Normalization
- Power values should ideally be normalized to [0, 1] range
- Normalization: `P_norm = P_actual / P_rated`
- If using actual MW, the code will handle normalization

### Resolution
- Recommended: Hourly data
- Sub-hourly data (e.g., 10-minute) can be used but may require resampling
- The paper validation used hourly data

### Duration
- Minimum: 1 month for meaningful statistics
- Recommended: 1+ years for seasonal analysis
- Paper validation used 5 years (2017-2021)

## Sample Data

The file `sample_ramps.csv` contains pre-detected ramp events for testing:

| Column | Description |
|--------|-------------|
| `ramp_id` | Unique identifier |
| `start_idx` | Start index in original time series |
| `end_idx` | End index |
| `start_hour` | Hour of day (0-23) |
| `direction` | 'up' or 'down' |
| `magnitude` | ΔP (normalized) |
| `duration` | Hours |
| `rate` | ΔP/Δt |
| `power_series` | JSON array of power values |

## Data Privacy

The original wind power data used in the paper is proprietary and provided by NB Power under research agreement. Users should:

1. Obtain their own wind power data from:
   - Utility partners
   - Public datasets (e.g., ERCOT, CAISO)
   - Research databases (e.g., NREL Wind Toolkit)

2. Or use synthetic data for algorithm testing

## Public Data Sources

- **ERCOT**: [http://www.ercot.com/gridinfo/generation](http://www.ercot.com/gridinfo/generation)
- **NREL Wind Toolkit**: [https://www.nrel.gov/grid/wind-toolkit.html](https://www.nrel.gov/grid/wind-toolkit.html)
- **EIA**: [https://www.eia.gov/electricity/data.php](https://www.eia.gov/electricity/data.php)
