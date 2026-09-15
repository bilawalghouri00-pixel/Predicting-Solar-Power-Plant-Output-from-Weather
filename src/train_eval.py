from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# IMPORT OUR REGRESSION FUNCTIONS
# =========================================================

sys.path.append(
    str(Path(__file__).parent)
)

from regression import (
    hypothesis,
    cost,
    fit_normal,
    fit_batch_gd,
    fit_sgd,
    rmse,
    clip_predictions
)


# =========================================================
# PATHS
# =========================================================

DATA_FILE = (
    Path("data") /
    "plant1 openmeteo.csv"
)

HOURLY_FILE = (
    Path("data") /
    "plant1 hourly.csv"
)


RESULTS_DIR = Path("results")

FIGURE_DIR = (
    RESULTS_DIR /
    "figures"
)


RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# LOAD DATA
# =========================================================

sensor = pd.read_csv(
    HOURLY_FILE,
    parse_dates=["datetime"]
)


weather = pd.read_csv(
    DATA_FILE,
    parse_dates=["datetime"]
)


# =========================================================
# MERGE SENSOR + OPEN-METEO
# =========================================================

df = pd.merge(
    sensor,
    weather,
    on="datetime",
    how="inner"
)


# =========================================================
# CONVERT OPEN-METEO RADIATION
#
# Open-Meteo: W/m²
# Sensor: kW/m²
# =========================================================

df["sw_radiation_kw_m2"] = (
    df["sw_radiation"] / 1000.0
)


# =========================================================
# TIME FEATURES
# =========================================================

df["hour"] = (
    df["datetime"].dt.hour
)


df["sin_hour"] = np.sin(
    2 * np.pi * df["hour"] / 24
)


df["cos_hour"] = np.cos(
    2 * np.pi * df["hour"] / 24
)


# =========================================================
# TRAIN / TEST SPLIT
# =========================================================
#
# Train:
# May 15 - June 10
#
# Test:
# June 11 - June 17
#
# NO SHUFFLING
# =========================================================

train_start = pd.Timestamp(
    "2020-05-15 00:00:00"
)

train_end = pd.Timestamp(
    "2020-06-10 23:59:59"
)


test_start = pd.Timestamp(
    "2020-06-11 00:00:00"
)

test_end = pd.Timestamp(
    "2020-06-17 23:59:59"
)


train_df = df[
    (df["datetime"] >= train_start) &
    (df["datetime"] <= train_end)
].copy()


test_df = df[
    (df["datetime"] >= test_start) &
    (df["datetime"] <= test_end)
].copy()


print("=" * 70)
print("DATA SPLIT")
print("=" * 70)

print("Training rows:", len(train_df))
print("Testing rows :", len(test_df))

print(
    "Training first:",
    train_df["datetime"].min()
)

print(
    "Training last:",
    train_df["datetime"].max()
)

print(
    "Testing first:",
    test_df["datetime"].min()
)

print(
    "Testing last:",
    test_df["datetime"].max()
)


# =========================================================
# FEATURE SETS
# =========================================================

SET_A_FEATURES = [
    "irradiation",
    "module_temp",
    "ambient_temp",
    "sin_hour",
    "cos_hour"
]


SET_B_FEATURES = [
    "sw_radiation_kw_m2",
    "temp_2m",
    "cloud_cover",
    "sin_hour",
    "cos_hour"
]


TARGET = "ac_power"


# =========================================================
# SCALE FEATURES
# =========================================================

def scale_train_test(
    train,
    test,
    feature_columns
):

    train_features = (
        train[feature_columns]
        .astype(float)
    )


    test_features = (
        test[feature_columns]
        .astype(float)
    )


    # Training statistics ONLY
    means = train_features.mean()

    stds = train_features.std(
        ddof=0
    )


    # Prevent divide by zero
    stds = stds.replace(
        0,
        1
    )


    train_scaled = (
        train_features - means
    ) / stds


    test_scaled = (
        test_features - means
    ) / stds


    return (
        train_scaled.to_numpy(),
        test_scaled.to_numpy(),
        means,
        stds
    )


# =========================================================
# ADD INTERCEPT
# =========================================================

def add_intercept(X):

    ones = np.ones(
        (X.shape[0], 1)
    )

    return np.hstack(
        [ones, X]
    )


# =========================================================
# PREPARE SET A
# =========================================================

X_A_train_scaled, X_A_test_scaled, A_means, A_stds = (
    scale_train_test(
        train_df,
        test_df,
        SET_A_FEATURES
    )
)


X_A_train = add_intercept(
    X_A_train_scaled
)

X_A_test = add_intercept(
    X_A_test_scaled
)


# =========================================================
# PREPARE SET B
# =========================================================

X_B_train_scaled, X_B_test_scaled, B_means, B_stds = (
    scale_train_test(
        train_df,
        test_df,
        SET_B_FEATURES
    )
)


X_B_train = add_intercept(
    X_B_train_scaled
)

X_B_test = add_intercept(
    X_B_test_scaled
)


# =========================================================
# TARGET
# =========================================================

y_train = (
    train_df[TARGET]
    .to_numpy(dtype=float)
)


y_test = (
    test_df[TARGET]
    .to_numpy(dtype=float)
)


# =========================================================
# NORMAL EQUATION
# =========================================================

print("\n" + "=" * 70)
print("NORMAL EQUATION")
print("=" * 70)


theta_A_normal = fit_normal(
    X_A_train,
    y_train
)


theta_B_normal = fit_normal(
    X_B_train,
    y_train
)


# =========================================================
# LEARNING RATE SEARCH
# =========================================================

batch_alphas = [
    1e-5,
    1e-4,
    1e-3
]


sgd_alphas = [
    1e-4,
    1e-3,
    1e-2
]


# =========================================================
# BATCH GD LEARNING RATE TEST - SET A
# =========================================================

print("\n" + "=" * 70)
print("BATCH GD LEARNING RATE TEST - SET A")
print("=" * 70)


batch_A_histories = {}

for alpha in batch_alphas:

    print(
        "Running alpha =",
        alpha
    )


    theta, history = fit_batch_gd(
        X_A_train,
        y_train,
        alpha,
        500
    )


    batch_A_histories[alpha] = (
        theta,
        history
    )


plt.figure(figsize=(9, 6))


for alpha, (
    theta,
    history
) in batch_A_histories.items():

    plt.plot(
        history,
        label=f"alpha={alpha}"
    )


plt.xlabel("Iteration")
plt.ylabel("J(theta)")
plt.title(
    "Batch Gradient Descent Learning Rates - Set A"
)

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR /
    "06_batch_learning_rates_set_A.png",
    dpi=300
)

plt.show()


# =========================================================
# BATCH GD LEARNING RATE TEST - SET B
# =========================================================

print("\n" + "=" * 70)
print("BATCH GD LEARNING RATE TEST - SET B")
print("=" * 70)


batch_B_histories = {}


for alpha in batch_alphas:

    print(
        "Running alpha =",
        alpha
    )


    theta, history = fit_batch_gd(
        X_B_train,
        y_train,
        alpha,
        500
    )


    batch_B_histories[alpha] = (
        theta,
        history
    )


plt.figure(figsize=(9, 6))


for alpha, (
    theta,
    history
) in batch_B_histories.items():

    plt.plot(
        history,
        label=f"alpha={alpha}"
    )


plt.xlabel("Iteration")
plt.ylabel("J(theta)")
plt.title(
    "Batch Gradient Descent Learning Rates - Set B"
)

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR /
    "07_batch_learning_rates_set_B.png",
    dpi=300
)

plt.show()


# =========================================================
# CHOOSE BATCH ALPHA AUTOMATICALLY
# =========================================================
#
# We select the alpha that gives the lowest final cost,
# provided the curve is decreasing.
# =========================================================

def choose_batch_alpha(histories):

    valid = {}

    for alpha, (
        theta,
        history
    ) in histories.items():

        decreasing = np.all(
            np.diff(history) <= 1e-8
        )

        if decreasing:
            valid[alpha] = history[-1]


    if not valid:

        # fallback
        return min(
            histories,
            key=lambda a: histories[a][1][-1]
        )


    return min(
        valid,
        key=valid.get
    )


batch_alpha_A = choose_batch_alpha(
    batch_A_histories
)

batch_alpha_B = choose_batch_alpha(
    batch_B_histories
)


print(
    "\nChosen Batch alpha Set A:",
    batch_alpha_A
)

print(
    "Chosen Batch alpha Set B:",
    batch_alpha_B
)


# =========================================================
# SGD LEARNING RATE TEST - SET A
# =========================================================

print("\n" + "=" * 70)
print("SGD LEARNING RATE TEST - SET A")
print("=" * 70)


sgd_A_histories = {}


for alpha in sgd_alphas:

    print(
        "Running alpha =",
        alpha
    )


    theta, history = fit_sgd(
        X_A_train,
        y_train,
        alpha,
        50
    )


    sgd_A_histories[alpha] = (
        theta,
        history
    )


plt.figure(figsize=(9, 6))


for alpha, (
    theta,
    history
) in sgd_A_histories.items():

    plt.plot(
        history,
        label=f"alpha={alpha}"
    )


plt.xlabel("Epoch")
plt.ylabel("J(theta)")
plt.title(
    "Stochastic Gradient Descent Learning Rates - Set A"
)

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR /
    "08_sgd_learning_rates_set_A.png",
    dpi=300
)

plt.show()


# =========================================================
# SGD LEARNING RATE TEST - SET B
# =========================================================

print("\n" + "=" * 70)
print("SGD LEARNING RATE TEST - SET B")
print("=" * 70)


sgd_B_histories = {}


for alpha in sgd_alphas:

    print(
        "Running alpha =",
        alpha
    )


    theta, history = fit_sgd(
        X_B_train,
        y_train,
        alpha,
        50
    )


    sgd_B_histories[alpha] = (
        theta,
        history
    )


plt.figure(figsize=(9, 6))


for alpha, (
    theta,
    history
) in sgd_B_histories.items():

    plt.plot(
        history,
        label=f"alpha={alpha}"
    )


plt.xlabel("Epoch")
plt.ylabel("J(theta)")
plt.title(
    "Stochastic Gradient Descent Learning Rates - Set B"
)

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR /
    "09_sgd_learning_rates_set_B.png",
    dpi=300
)

plt.show()


# =========================================================
# CHOOSE SGD ALPHA
# =========================================================

def choose_sgd_alpha(histories):

    return min(
        histories,
        key=lambda a: histories[a][1][-1]
    )


sgd_alpha_A = choose_sgd_alpha(
    sgd_A_histories
)

sgd_alpha_B = choose_sgd_alpha(
    sgd_B_histories
)


print(
    "\nChosen SGD alpha Set A:",
    sgd_alpha_A
)

print(
    "Chosen SGD alpha Set B:",
    sgd_alpha_B
)


# =========================================================
# FINAL FITS - SET A
# =========================================================

theta_A_batch, history_A_batch = fit_batch_gd(
    X_A_train,
    y_train,
    batch_alpha_A,
    500
)


theta_A_sgd, history_A_sgd = fit_sgd(
    X_A_train,
    y_train,
    sgd_alpha_A,
    50
)


# =========================================================
# FINAL FITS - SET B
# =========================================================

theta_B_batch, history_B_batch = fit_batch_gd(
    X_B_train,
    y_train,
    batch_alpha_B,
    500
)


theta_B_sgd, history_B_sgd = fit_sgd(
    X_B_train,
    y_train,
    sgd_alpha_B,
    50
)


# =========================================================
# PREDICTIONS
# =========================================================

pred_A_normal = clip_predictions(
    hypothesis(
        X_A_test,
        theta_A_normal
    )
)


pred_A_batch = clip_predictions(
    hypothesis(
        X_A_test,
        theta_A_batch
    )
)


pred_A_sgd = clip_predictions(
    hypothesis(
        X_A_test,
        theta_A_sgd
    )
)


pred_B_normal = clip_predictions(
    hypothesis(
        X_B_test,
        theta_B_normal
    )
)


pred_B_batch = clip_predictions(
    hypothesis(
        X_B_test,
        theta_B_batch
    )
)


pred_B_sgd = clip_predictions(
    hypothesis(
        X_B_test,
        theta_B_sgd
    )
)


# =========================================================
# DAYTIME MASK
# =========================================================

daytime_mask = (
    test_df["irradiation"].to_numpy()
    > 0
)


# =========================================================
# CALCULATE ALL RMSE VALUES
# =========================================================

results = []


def add_result(
    solver,
    feature_set,
    predictions
):

    all_rmse = rmse(
        y_test,
        predictions
    )


    daytime_rmse = rmse(
        y_test[daytime_mask],
        predictions[daytime_mask]
    )


    results.append({
        "Solver": solver,
        "Features": feature_set,
        "All hours RMSE (kW)": all_rmse,
        "Daytime RMSE (kW)": daytime_rmse
    })


# Set A
add_result(
    "Normal equation",
    "Set A",
    pred_A_normal
)

add_result(
    f"Batch GD (alpha={batch_alpha_A}, iters=500)",
    "Set A",
    pred_A_batch
)

add_result(
    f"Stochastic GD (alpha={sgd_alpha_A}, epochs=50)",
    "Set A",
    pred_A_sgd
)


# Set B
add_result(
    "Normal equation",
    "Set B",
    pred_B_normal
)

add_result(
    f"Batch GD (alpha={batch_alpha_B}, iters=500)",
    "Set B",
    pred_B_batch
)

add_result(
    f"Stochastic GD (alpha={sgd_alpha_B}, epochs=50)",
    "Set B",
    pred_B_sgd
)


results_df = pd.DataFrame(results)


# =========================================================
# SAVE RMSE RESULTS
# =========================================================

results_df.to_csv(
    RESULTS_DIR /
    "rmse_results.csv",
    index=False
)


print("\n" + "=" * 70)
print("RMSE RESULTS")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


# =========================================================
# COMPARE NORMAL EQUATION VS BATCH GD
# =========================================================

max_difference_A = np.max(
    np.abs(
        theta_A_batch -
        theta_A_normal
    )
)


max_difference_B = np.max(
    np.abs(
        theta_B_batch -
        theta_B_normal
    )
)


print("\n" + "=" * 70)
print("NORMAL EQUATION vs BATCH GD")
print("=" * 70)

print(
    "Set A max absolute theta difference:",
    max_difference_A
)

print(
    "Set B max absolute theta difference:",
    max_difference_B
)


# =========================================================
# PRINT THETA SET A
# =========================================================

print("\n" + "=" * 70)
print("SET A THETA")
print("=" * 70)


feature_names_A = [
    "theta0 intercept",
    "theta1 irradiation",
    "theta2 module temperature",
    "theta3 ambient temperature",
    "theta4 sin hour",
    "theta5 cos hour"
]


theta_table = pd.DataFrame({
    "Feature": feature_names_A,
    "Normal Equation": theta_A_normal,
    "Batch GD": theta_A_batch,
    "SGD": theta_A_sgd
})


print(
    theta_table.to_string(
        index=False
    )
)


theta_table.to_csv(
    RESULTS_DIR /
    "theta_results.csv",
    index=False
)


# =========================================================
# FIND LARGEST SET A WEIGHT
# =========================================================

weights = np.abs(
    theta_A_normal[1:]
)

largest_index = np.argmax(
    weights
) + 1


print(
    "\nLargest Set A weight:",
    feature_names_A[largest_index]
)

print(
    "Weight:",
    theta_A_normal[largest_index]
)


# =========================================================
# PLANT PEAK POWER
# =========================================================

peak_hourly_power = (
    train_df["ac_power"].max()
)


print(
    "\nTraining peak hourly AC power:",
    peak_hourly_power,
    "kW"
)


# =========================================================
# SET A VS SET B DAYTIME RMSE
# =========================================================

set_A_day_rmse = rmse(
    y_test[daytime_mask],
    pred_A_normal[daytime_mask]
)


set_B_day_rmse = rmse(
    y_test[daytime_mask],
    pred_B_normal[daytime_mask]
)


difference_kW = (
    set_B_day_rmse -
    set_A_day_rmse
)


percentage_of_peak = (
    difference_kW /
    peak_hourly_power
) * 100


print("\n" + "=" * 70)
print("SET B VS SET A")
print("=" * 70)

print(
    "Set A daytime RMSE:",
    set_A_day_rmse,
    "kW"
)

print(
    "Set B daytime RMSE:",
    set_B_day_rmse,
    "kW"
)

print(
    "Difference:",
    difference_kW,
    "kW"
)

print(
    "Difference as % of plant peak:",
    percentage_of_peak,
    "%"
)


# =========================================================
# ACTUAL VS PREDICTED
# =========================================================

plt.figure(figsize=(12, 6))

plt.plot(
    test_df["datetime"],
    y_test,
    label="Actual AC power"
)

plt.plot(
    test_df["datetime"],
    pred_A_normal,
    label="Set A prediction"
)

plt.plot(
    test_df["datetime"],
    pred_B_normal,
    label="Set B prediction"
)

plt.xlabel("Date")
plt.ylabel("AC Power (kW)")

plt.title(
    "Actual vs Predicted AC Power - Test Week"
)

plt.legend()

plt.grid(True)

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR /
    "10_actual_vs_predicted_test_week.png",
    dpi=300
)

plt.show()


# =========================================================
# RESIDUALS
# =========================================================

residuals = (
    y_test -
    pred_A_normal
)


residual_df = pd.DataFrame({
    "datetime":
        test_df["datetime"],

    "hour":
        test_df["hour"],

    "actual":
        y_test,

    "prediction":
        pred_A_normal,

    "residual":
        residuals
})


# Residual plot
plt.figure(figsize=(9, 6))

plt.scatter(
    residual_df["hour"],
    residual_df["residual"],
    alpha=0.6
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel("Hour of Day")
plt.ylabel("Residual (Actual - Predicted)")

plt.title(
    "Set A Normal Equation Residuals vs Hour"
)

plt.xticks(range(24))

plt.grid(True)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR /
    "11_residuals_vs_hour.png",
    dpi=300
)

plt.show()


# =========================================================
# WORST HOURS
# =========================================================

residual_df["absolute_residual"] = (
    residual_df["residual"].abs()
)


worst_hours = (
    residual_df
    .groupby("hour")["absolute_residual"]
    .mean()
    .sort_values(
        ascending=False
    )
)


print("\n" + "=" * 70)
print("WORST HOURS")
print("=" * 70)

print(
    worst_hours.head(5)
)


# =========================================================
# SAVE MODEL WEIGHTS
# =========================================================

np.save(
    RESULTS_DIR /
    "theta_A_normal.npy",
    theta_A_normal
)

np.save(
    RESULTS_DIR /
    "theta_A_batch.npy",
    theta_A_batch
)

np.save(
    RESULTS_DIR /
    "theta_A_sgd.npy",
    theta_A_sgd
)

np.save(
    RESULTS_DIR /
    "theta_B_normal.npy",
    theta_B_normal
)

np.save(
    RESULTS_DIR /
    "theta_B_batch.npy",
    theta_B_batch
)

np.save(
    RESULTS_DIR /
    "theta_B_sgd.npy",
    theta_B_sgd
)


# =========================================================
# SAVE SET B SCALING VALUES
# =========================================================

np.savez(
    RESULTS_DIR /
    "set_B_scaling.npz",

    means=B_means.to_numpy(),

    stds=B_stds.to_numpy()
)


# =========================================================
# SAVE SET A SCALING VALUES
# =========================================================

np.savez(
    RESULTS_DIR /
    "set_A_scaling.npz",

    means=A_means.to_numpy(),

    stds=A_stds.to_numpy()
)


# =========================================================
# SAVE TEST PREDICTIONS
# =========================================================

prediction_output = test_df[
    [
        "datetime",
        "hour",
        "irradiation"
    ]
].copy()


prediction_output["actual_ac_power"] = y_test

prediction_output["set_A_normal"] = pred_A_normal

prediction_output["set_A_batch"] = pred_A_batch

prediction_output["set_A_sgd"] = pred_A_sgd

prediction_output["set_B_normal"] = pred_B_normal

prediction_output["set_B_batch"] = pred_B_batch

prediction_output["set_B_sgd"] = pred_B_sgd


prediction_output.to_csv(
    RESULTS_DIR /
    "test_predictions.csv",
    index=False
)


# =========================================================
# SAVE ANALYSIS
# =========================================================

analysis_text = f"""
# Task 5 Analysis

## 5.1 Set A Normal Equation

Theta values:

{theta_table.to_string(index=False)}

The largest absolute weight is:

{feature_names_A[largest_index]}

with weight:

{theta_A_normal[largest_index]:.6f}

The signs should be interpreted using the physical relationship
between solar irradiation, temperature and AC power.

## 5.2 Set B vs Set A

Set A daytime RMSE:
{set_A_day_rmse:.4f} kW

Set B daytime RMSE:
{set_B_day_rmse:.4f} kW

Difference:
{difference_kW:.4f} kW

Difference as percentage of plant peak:
{percentage_of_peak:.2f} %

## 5.3 Solver comparison

Normal equation calculates the least-squares solution directly.

Batch gradient descent repeatedly updates all parameters using the
complete training dataset.

Stochastic gradient descent updates parameters one training example
at a time.

For this relatively small dataset, the normal equation is convenient.
For very large datasets such as 10 million rows, gradient descent is
more practical because explicitly forming and inverting X-transpose-X
can become expensive.

## 5.4 Batch vs SGD cost curves

Batch gradient descent normally produces a smoother cost curve because
every update uses the entire training dataset.

SGD is noisier because every parameter update is based on one training
row.

## 5.5 Residuals

The residual is:

actual - predicted

The worst hours can be identified from the saved residual analysis.
Large errors can occur around sunrise and sunset because the relationship
between weather variables and plant output changes rapidly during those
periods. Cloud variation, shading, inverter behavior and sensor/API
differences can also contribute.
"""


with open(
    RESULTS_DIR / "analysis.md",
    "w",
    encoding="utf-8"
) as f:

    f.write(
        analysis_text
    )


print("\n" + "=" * 70)
print("TASK 4 AND TASK 5 COMPLETE")
print("=" * 70)

print("Results saved in:")
print(RESULTS_DIR)