import pandas as pd

# 1. Load the original dataset
file_path = "bmd_weather_data_10000x20.csv"
df = pd.read_csv(file_path)

# 2. Standardize column names (lowercase, trim spaces)
df.columns = df.columns.str.strip().str.lower()

# 3. Handle missing values and duplicates
df = df.drop_duplicates()
df = df.dropna()  # Drops any incomplete rows if present

# 4. Standardize text data formatting
text_cols = ["station_id", "station_name", "division", "season"]
for col in text_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()

# 5. Convert date and time columns to proper datetime format
df["date"] = pd.to_datetime(df["date"])
df["time"] = pd.to_datetime(df["time"], format="%H:%M").dt.time

# 6. Ensure correct numerical data types
float_cols = [
    "latitude",
    "longitude",
    "temp_max_c",
    "temp_min_c",
    "current_temp_c",
    "temp_range_c",
    "rainfall_mm",
    "wind_speed_kmh",
    "surface_pressure_hpa",
    "rainfall_lag1_mm",
]
int_cols = [
    "elevation_m",
    "humidity_morning_pct",
    "humidity_evening_pct",
    "target_rain_tomorrow",
]

df[float_cols] = df[float_cols].apply(pd.to_numeric, downcast="float")
df[int_cols] = df[int_cols].apply(pd.to_numeric, downcast="integer")

# 7. Add derived time features for analysis and modeling
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["day_of_week"] = df["date"].dt.day_name()

# 8. Sort dataset sequentially by date, station, and time
df = df.sort_values(by=["date", "station_id", "time"]).reset_index(drop=True)

# 9. Save the cleaned dataset to a new CSV file
output_file = "bmd_weather_data_cleaned.csv"
df.to_csv(output_file, index=False)

print(f"Data cleaning complete. Cleaned file saved as: {output_file}")
print(f"Final dataset shape: {df.shape}")