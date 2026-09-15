from pathlib import Path
import sys
import pandas as pd
import numpy as np


# Allow Python to import load_data.py
sys.path.append(str(Path(__file__).parent))

from load_data import load_raw


# ---------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------

raw = load_raw("data")

generation = raw["gen1"].copy()
sensor = raw["sensor1"].copy()


# ---------------------------------------------------------
# 2. Inspect raw timestamp strings
# ---------------------------------------------------------

print("=" * 70)
print("RAW GENERATION TIMESTAMPS")
print("=" * 70)

print(generation["DATE_TIME"].head(10).to_string())

print("\nLast values:")
print(generation["DATE_TIME"].tail(10).to_string())


print("\n" + "=" * 70)
print("RAW SENSOR TIMESTAMPS")
print("=" * 70)

print(sensor["DATE_TIME"].head(10).to_string())

print("\nLast values:")
print(sensor["DATE_TIME"].tail(10).to_string())


# ---------------------------------------------------------
# 3. Convert timestamps
# ---------------------------------------------------------

generation["datetime"] = pd.to_datetime(
    generation["DATE_TIME"],
    dayfirst=True,
    errors="coerce"
)

sensor["datetime"] = pd.to_datetime(
    sensor["DATE_TIME"],
    dayfirst=True,
    errors="coerce"
)


# Check failed conversions
bad_generation_dates = generation["datetime"].isna().sum()
bad_sensor_dates = sensor["datetime"].isna().sum()

print("\nInvalid generation dates:", bad_generation_dates)
print("Invalid sensor dates:", bad_sensor_dates)


# Remove invalid timestamps
generation = generation.dropna(subset=["datetime"]).copy()
sensor = sensor.dropna(subset=["datetime"]).copy()


# ---------------------------------------------------------
# 4. Print first and last timestamps
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("CONVERTED TIMESTAMPS")
print("=" * 70)

print("Generation first:", generation["datetime"].min())
print("Generation last :", generation["datetime"].max())

print("Sensor first    :", sensor["datetime"].min())
print("Sensor last     :", sensor["datetime"].max())


# ---------------------------------------------------------
# 5. Sum AC and DC power over all inverters
# ---------------------------------------------------------

plant_power = (
    generation
    .groupby("datetime")[["AC_POWER", "DC_POWER"]]
    .sum()
    .reset_index()
)


plant_power.rename(
    columns={
        "AC_POWER": "ac_power",
        "DC_POWER": "dc_power"
    },
    inplace=True
)


print("\n" + "=" * 70)
print("PLANT LEVEL POWER")
print("=" * 70)

print(plant_power.head())


# ---------------------------------------------------------
# 6. Prepare weather sensor data
# ---------------------------------------------------------

sensor_data = sensor[
    [
        "datetime",
        "AMBIENT_TEMPERATURE",
        "MODULE_TEMPERATURE",
        "IRRADIATION"
    ]
].copy()


sensor_data.rename(
    columns={
        "AMBIENT_TEMPERATURE": "ambient_temp",
        "MODULE_TEMPERATURE": "module_temp",
        "IRRADIATION": "irradiation"
    },
    inplace=True
)


# ---------------------------------------------------------
# 7. Check timestamps existing in only one file
# ---------------------------------------------------------

generation_times = set(plant_power["datetime"])
sensor_times = set(sensor_data["datetime"])


only_generation = generation_times - sensor_times
only_sensor = sensor_times - generation_times


print("\n" + "=" * 70)
print("TIMESTAMP ALIGNMENT")
print("=" * 70)

print(
    "Timestamps only in generation file:",
    len(only_generation)
)

print(
    "Timestamps only in sensor file:",
    len(only_sensor)
)


# ---------------------------------------------------------
# 8. Merge generation and sensor data
# ---------------------------------------------------------

merged = pd.merge(
    plant_power,
    sensor_data,
    on="datetime",
    how="outer",
    indicator=True
)


print("\nMerged rows:", len(merged))

print("\nMerge status:")
print(merged["_merge"].value_counts())


# ---------------------------------------------------------
# 9. Remove merge indicator
# ---------------------------------------------------------

merged.drop(columns=["_merge"], inplace=True)


# ---------------------------------------------------------
# 10. Sort by datetime
# ---------------------------------------------------------

merged.sort_values("datetime", inplace=True)

merged.set_index("datetime", inplace=True)


# ---------------------------------------------------------
# 11. Resample to hourly mean
# ---------------------------------------------------------

hourly = merged.resample("1h").mean()


# Restore datetime as normal column
hourly.reset_index(inplace=True)


# ---------------------------------------------------------
# 12. Check missing values
# ---------------------------------------------------------

missing_before = hourly.isna().any(axis=1).sum()

print("\n" + "=" * 70)
print("HOURLY DATA")
print("=" * 70)

print("Hourly rows:", len(hourly))

print(
    "Rows containing missing values:",
    missing_before
)

print("\nMissing values by column:")
print(hourly.isna().sum())


# ---------------------------------------------------------
# 13. Handle missing values
# ---------------------------------------------------------
#
# For regression we need complete rows.
# Therefore rows containing missing values are removed.
#
# This is preferable to inventing weather/power values.
# ---------------------------------------------------------

hourly = hourly.dropna().copy()


print(
    "\nHourly rows after removing missing rows:",
    len(hourly)
)


# ---------------------------------------------------------
# 14. Select required columns
# ---------------------------------------------------------

hourly = hourly[
    [
        "datetime",
        "ac_power",
        "dc_power",
        "ambient_temp",
        "module_temp",
        "irradiation"
    ]
]


# ---------------------------------------------------------
# 15. Save result
# ---------------------------------------------------------

output_path = Path("data") / "plant1 hourly.csv"

hourly.to_csv(
    output_path,
    index=False
)


# ---------------------------------------------------------
# 16. Final report
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("TASK 1 COMPLETE")
print("=" * 70)

print("Output file:", output_path)
print("Final hourly rows:", len(hourly))
print(
    "Rows with missing values after cleaning:",
    hourly.isna().any(axis=1).sum()
)

print("\nFirst 5 hourly rows:")
print(hourly.head().to_string(index=False))