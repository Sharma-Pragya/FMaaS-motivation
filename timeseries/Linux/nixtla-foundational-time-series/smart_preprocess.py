import os
import pandas as pd

# Directory containing the CSV files
data_dir = "./data/apartment/2014"  # Change if your path is different


all_data = []

# Constants
FREQ = "Hourly"
PANDAS_FREQ = "H"
SEASONALITY = 24
HORIZON = 24

for file in os.listdir(data_dir):
    if file.endswith(".csv"):
        apt_id = os.path.splitext(file)[0]
        file_path = os.path.join(data_dir, file)

        # Load CSV without headers
        df = pd.read_csv(file_path, header=None, names=["ds", "y"])
        df["ds"] = pd.to_datetime(df["ds"])

        # Set datetime index
        df.set_index("ds", inplace=True)

        # Resample to hourly power usage
        df_hourly = df["y"].resample("1H").mean().reset_index()

        # Add metadata columns
        df_hourly["unique_id"] = apt_id
        df_hourly["frequency"] = FREQ
        df_hourly["pandas_frequency"] = PANDAS_FREQ
        df_hourly["seasonality"] = SEASONALITY
        df_hourly["horizon"] = HORIZON

        all_data.append(df_hourly)

# Combine all apartments into one dataframe
df_all = pd.concat(all_data, ignore_index=True)

# Reorder columns
df_all = df_all[["unique_id", "ds", "y", "frequency", "pandas_frequency", "seasonality", "horizon"]]

# Save to Parquet or CSV
df_all.to_parquet("./data/smart_hourly.parquet")

# Quick check
print(df_all.head())