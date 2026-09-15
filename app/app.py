from pathlib import Path

import numpy as np
import streamlit as st


# =========================================================
# PATH
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = (
    BASE_DIR /
    "results"
)


# =========================================================
# LOAD SAVED MODEL
# =========================================================

theta = np.load(
    RESULTS_DIR /
    "theta_B_normal.npy"
)


scaling = np.load(
    RESULTS_DIR /
    "set_B_scaling.npz"
)


means = scaling["means"]

stds = scaling["stds"]


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="Solar Power Predictor",
    page_icon="☀️",
    layout="centered"
)


# =========================================================
# TITLE
# =========================================================

st.title(
    "☀️ Solar Power Plant AC Power Predictor"
)


st.write(
    """
This application predicts hourly AC power using the
Set B model trained with public weather data.
"""
)


# =========================================================
# INPUTS
# =========================================================

st.header(
    "Weather Information"
)


hour = st.slider(
    "Hour of day",
    min_value=0,
    max_value=23,
    value=12
)


shortwave_radiation = st.number_input(
    "Shortwave radiation (W/m²)",
    min_value=0.0,
    max_value=1500.0,
    value=500.0
)


temperature_2m = st.number_input(
    "Temperature 2m (°C)",
    min_value=-20.0,
    max_value=60.0,
    value=30.0
)


cloud_cover = st.number_input(
    "Cloud cover (%)",
    min_value=0.0,
    max_value=100.0,
    value=20.0
)


# =========================================================
# PREDICTION BUTTON
# =========================================================

if st.button(
    "Predict AC Power"
):


    # -----------------------------------------------------
    # Convert radiation
    # W/m² -> kW/m²
    # -----------------------------------------------------

    sw_radiation_kw_m2 = (
        shortwave_radiation /
        1000.0
    )


    # -----------------------------------------------------
    # Time features
    # -----------------------------------------------------

    sin_hour = np.sin(
        2 * np.pi * hour / 24
    )


    cos_hour = np.cos(
        2 * np.pi * hour / 24
    )


    # -----------------------------------------------------
    # Original Set B feature order:
    #
    # 1. sw radiation
    # 2. temp 2m
    # 3. cloud cover
    # 4. sin hour
    # 5. cos hour
    # -----------------------------------------------------

    raw_features = np.array([
        sw_radiation_kw_m2,
        temperature_2m,
        cloud_cover,
        sin_hour,
        cos_hour
    ])


    # -----------------------------------------------------
    # SCALE USING TRAINING MEAN AND STD
    # -----------------------------------------------------

    scaled_features = (
        raw_features - means
    ) / stds


    # -----------------------------------------------------
    # ADD INTERCEPT
    # -----------------------------------------------------

    X = np.concatenate([
        [1.0],
        scaled_features
    ])


    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    prediction = (
        X @ theta
    )


    # -----------------------------------------------------
    # PHYSICAL CONSTRAINT
    # -----------------------------------------------------

    prediction = max(
        0.0,
        prediction
    )


    # -----------------------------------------------------
    # DISPLAY
    # -----------------------------------------------------

    st.success(
        f"Predicted AC Power: {prediction:.2f} kW"
    )


# =========================================================
# MODEL INFORMATION
# =========================================================

st.divider()

st.subheader(
    "Model Information"
)


st.write(
    """
Model: Linear Regression

Feature set: Set B

Features:
- Shortwave radiation
- Temperature at 2m
- Cloud cover
- Sin(hour)
- Cos(hour)

Solver: Normal Equation
"""
)