from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Load hourly data
# ---------------------------------------------------------

input_file = Path("data") / "plant1 hourly.csv"

df = pd.read_csv(
    input_file,
    parse_dates=["datetime"]
)


# Create figure directory
figure_dir = Path("results") / "figures"
figure_dir.mkdir(parents=True, exist_ok=True)


print("=" * 70)
print("TASK 2 - EXPLORATORY DATA ANALYSIS")
print("=" * 70)

print("Rows:", len(df))
print(df.head())


# =========================================================
# PLOT 1
# AC POWER vs IRRADIATION
# =========================================================

plt.figure(figsize=(8, 5))

plt.scatter(
    df["irradiation"],
    df["ac_power"],
    alpha=0.6
)

plt.xlabel("Irradiation (kW/m²)")
plt.ylabel("AC Power (kW)")
plt.title("AC Power vs Irradiation")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    figure_dir / "01_ac_power_vs_irradiation.png",
    dpi=300
)

plt.show()


# =========================================================
# PLOT 2
# MODULE TEMP vs AMBIENT TEMP
# COLORED BY IRRADIATION
# =========================================================

plt.figure(figsize=(8, 5))

scatter = plt.scatter(
    df["ambient_temp"],
    df["module_temp"],
    c=df["irradiation"],
    cmap="viridis",
    alpha=0.7
)

plt.colorbar(scatter, label="Irradiation (kW/m²)")

plt.xlabel("Ambient Temperature (°C)")
plt.ylabel("Module Temperature (°C)")
plt.title("Module Temperature vs Ambient Temperature")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    figure_dir / "02_module_vs_ambient_temperature.png",
    dpi=300
)

plt.show()


# =========================================================
# PLOT 3
# AC POWER vs DC POWER
# =========================================================

plt.figure(figsize=(8, 5))

plt.scatter(
    df["dc_power"],
    df["ac_power"],
    alpha=0.6
)

plt.xlabel("DC Power (kW)")
plt.ylabel("AC Power (kW)")
plt.title("AC Power vs DC Power")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    figure_dir / "03_ac_power_vs_dc_power.png",
    dpi=300
)

plt.show()


# AC/DC ratio
daytime = df[df["dc_power"] > 0].copy()

daytime["ac_dc_ratio"] = (
    daytime["ac_power"] /
    daytime["dc_power"]
)

print("\nAC/DC ratio statistics:")

print(
    daytime["ac_dc_ratio"].describe()
)


# =========================================================
# PLOT 4
# AVERAGE AC POWER BY HOUR
# =========================================================

df["hour"] = df["datetime"].dt.hour

hourly_average = (
    df.groupby("hour")["ac_power"]
    .mean()
)


plt.figure(figsize=(8, 5))

plt.plot(
    hourly_average.index,
    hourly_average.values,
    marker="o"
)

plt.xlabel("Hour of Day")
plt.ylabel("Average AC Power (kW)")
plt.title("Average AC Power by Hour of Day")

plt.xticks(range(24))

plt.grid(True)

plt.tight_layout()

plt.savefig(
    figure_dir / "04_average_ac_power_by_hour.png",
    dpi=300
)

plt.show()


# =========================================================
# PRINT BASIC ANALYSIS
# =========================================================

print("\n" + "=" * 70)
print("TASK 2 ANALYSIS")
print("=" * 70)


print("""
1. AC power vs irradiation:
Solar power should generally increase as irradiation increases.
Therefore, a positive relationship is expected. At very high
irradiation the relationship can flatten because of inverter and
plant capacity limits.
""")


print("""
2. Module temperature vs ambient temperature:
Module temperature should generally increase with ambient temperature.
Higher irradiation can additionally heat the solar panels, so points
with stronger irradiation can have higher module temperatures.
""")


print("""
3. AC power vs DC power:
AC power should generally increase with DC power. AC power is normally
slightly lower because of inverter conversion losses and other system
losses.
""")


print("""
4. Average AC power by hour:
Power should be close to zero during nighttime. It should increase
after sunrise, reach its maximum around the middle of the day, and
then decrease toward sunset.
""")


print("\nAll Task 2 figures saved in:")
print(figure_dir)