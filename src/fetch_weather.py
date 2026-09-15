from pathlib import Path

import requests
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# OPEN-METEO SETTINGS
# =========================================================

LATITUDE = 14.82
LONGITUDE = 78.28

START_DATE = "2020-05-15"
END_DATE = "2020-06-17"

TIMEZONE = "Asia/Kolkata"

URL = "https://archive-api.open-meteo.com/v1/archive"


# =========================================================
# OUTPUT DIRECTORIES
# =========================================================

data_dir = Path("data")

figure_dir = Path("results") / "figures"

figure_dir.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# REQUEST PARAMETERS
# =========================================================

params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "start_date": START_DATE,
    "end_date": END_DATE,
    "hourly": "shortwave_radiation,temperature_2m,cloud_cover",
    "timezone": TIMEZONE
}


# =========================================================
# DOWNLOAD DATA
# =========================================================

print("=" * 70)
print("DOWNLOADING OPEN-METEO DATA")
print("=" * 70)

response = requests.get(
    URL,
    params=params,
    timeout=60
)


print("HTTP status:", response.status_code)


response.raise_for_status()


data = response.json()


# =========================================================
# CHECK API RESPONSE
# =========================================================

if "hourly" not in data:
    raise RuntimeError(
        "Open-Meteo response does not contain hourly data."
    )


hourly_data = data["hourly"]


print(
    "Number of downloaded timestamps:",
    len(hourly_data["time"])
)


# =========================================================
# CREATE DATAFRAME
# =========================================================

weather = pd.DataFrame({
    "datetime": pd.to_datetime(
        hourly_data["time"]
    ),

    "sw_radiation": hourly_data[
        "shortwave_radiation"
    ],

    "temp_2m": hourly_data[
        "temperature_2m"
    ],

    "cloud_cover": hourly_data[
        "cloud_cover"
    ]
})


# =========================================================
# SAVE CSV
# =========================================================

output_file = (
    data_dir /
    "plant1 openmeteo.csv"
)


weather.to_csv(
    output_file,
    index=False
)


print("\nSaved:")
print(output_file)

print("\nFirst 5 rows:")
print(weather.head())

print("\nLast 5 rows:")
print(weather.tail())


# =========================================================
# MERGE WITH SENSOR DATA
# =========================================================

sensor_file = (
    data_dir /
    "plant1 hourly.csv"
)

sensor = pd.read_csv(
    sensor_file,
    parse_dates=["datetime"]
)


merged = pd.merge(
    sensor,
    weather,
    on="datetime",
    how="inner"
)


print("\nMerged rows:", len(merged))


# =========================================================
# UNIT INFORMATION
# =========================================================

print("""
IMPORTANT UNIT DIFFERENCE:

Sensor irradiation:
    kW/m²

Open-Meteo shortwave radiation:
    W/m²

Therefore:

    1 kW/m² = 1000 W/m²

For direct comparison:

    Open-Meteo radiation / 1000
""")

merged["sw_radiation_kw_m2"] = (
    merged["sw_radiation"] / 1000.0
)


# =========================================================
# LOCATION VERIFICATION
# THREE DAYS
# =========================================================

days_to_check = [
    "2020-05-20",
    "2020-05-21",
    "2020-05-22"
]


print("\n" + "=" * 70)
print("LOCATION / TIMEZONE VERIFICATION")
print("=" * 70)


for day in days_to_check:

    start = pd.Timestamp(day)

    end = start + pd.Timedelta(days=1)

    one_day = merged[
        (merged["datetime"] >= start) &
        (merged["datetime"] < end)
    ].copy()


    if one_day.empty:
        print("No data for:", day)
        continue


    # Sensor peak
    sensor_peak_index = (
        one_day["irradiation"]
        .idxmax()
    )

    sensor_peak_time = one_day.loc[
        sensor_peak_index,
        "datetime"
    ]


    sensor_peak_value = one_day.loc[
        sensor_peak_index,
        "irradiation"
    ]


    # Open-Meteo peak
    weather_peak_index = (
        one_day["sw_radiation_kw_m2"]
        .idxmax()
    )

    weather_peak_time = one_day.loc[
        weather_peak_index,
        "datetime"
    ]


    weather_peak_value = one_day.loc[
        weather_peak_index,
        "sw_radiation_kw_m2"
    ]


    print("\nDate:", day)

    print(
        "Sensor peak:",
        sensor_peak_time,
        "value =",
        sensor_peak_value
    )

    print(
        "Open-Meteo peak:",
        weather_peak_time,
        "value =",
        weather_peak_value
    )


    # =====================================================
    # PLOT
    # =====================================================

    plt.figure(figsize=(10, 5))

    plt.plot(
        one_day["datetime"],
        one_day["irradiation"],
        label="On-site sensor"
    )

    plt.plot(
        one_day["datetime"],
        one_day["sw_radiation_kw_m2"],
        label="Open-Meteo"
    )

    plt.xlabel("Time")
    plt.ylabel("Radiation (kW/m²)")
    plt.title(
        f"Sensor vs Open-Meteo Radiation - {day}"
    )

    plt.legend()

    plt.grid(True)

    plt.xticks(rotation=45)

    plt.tight_layout()


    filename = (
        figure_dir /
        f"05_sensor_vs_openmeteo_{day}.png"
    )

    plt.savefig(
        filename,
        dpi=300
    )

    plt.show()


print("\nTask 3 complete.")