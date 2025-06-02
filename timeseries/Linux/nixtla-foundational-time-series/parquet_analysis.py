import pyarrow.parquet as pq
import pandas as pd
pd.set_option("display.max_columns", None)      # Show all columns
pd.set_option("display.width", 0)               # Don't limit display width
pd.set_option("display.max_colwidth", None)     # Show full column content
file_path = "/work/pi_shenoy_umass_edu/hshastri/FMaaS-motivation/timeseries/Linux/nixtla-foundational-time-series/results/smart_hourly/chronos_base/forecast_df.parquet"

# Open Parquet file metadata
pf = pq.ParquetFile(file_path)

# Read the first row group (if available)
first_batch = pf.read_row_group(0).to_pandas()

print("Columns:", first_batch.columns.tolist())
print(first_batch.head())