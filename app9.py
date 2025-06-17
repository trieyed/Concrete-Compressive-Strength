import streamlit as st
import numpy as np
import joblib


# Load models and scaler
baseline_model = joblib.load("baseline_regression_rfr.pkl")
tuned_model = joblib.load("tuned_regression_rfr .pkl")
scaler = joblib.load("scaler.pkl")

# Feature order and updated min/max ranges from your dataset
original_feature_ranges = {
    "cement ": (50, 700, "float"),
    "blast_furnace_slag": (0, 500, "float"),
    "fly_ash": (0, 300, "float"),
    "water": (100, 500, "float"),
    "superplasticizer": (0, 50, "float"),
    "coarse_aggregate": (500, 1500, "float"),
    "fine_aggregate": (400, 1400, "float"),
    "age": (1, 500, "int")
}

feature_ranges = {}

for feature, (min_val, max_val, dtype) in original_feature_ranges.items():
    feature_ranges[feature] = (min_val, max_val, dtype)

feature_names = list(feature_ranges.keys())


# Set page config
st.set_page_config(page_title="Concrete Strength Predictor", layout="centered")

# UI - Header
st.title("🔩 Concrete Compressive Strength Predictor")
st.markdown("Enter the mix properties below and select a model to predict the **Concrete Compressive Strength (MPa)**.")

# Sidebar for model selection
model_choice = st.sidebar.radio("Select Model", ("Baseline Model", "Tuned Model"))
model = baseline_model if model_choice == "Baseline Model" else tuned_model

with st.form("prediction_form"):
    input_values = []

    for feature, (min_val, max_val, dtype) in original_feature_ranges.items():
        label = feature.replace("_", " ").title() + " (kg/m³)" if feature != "age" else "Age (days)"
        if dtype == "float":
            value = st.slider(
                label,
                min_value=float(min_val),
                max_value=float(max_val),
                value=float((min_val + max_val) / 2),
                step=0.1
            )
        else:  # int
            value = st.slider(
                label,
                min_value=int(min_val),
                max_value=int(max_val),
                value=int((min_val + max_val) / 2),
                step=1
            )
        input_values.append(value)

    submit = st.form_submit_button("Predict Strength")

# Prediction logic
if submit:
    input_array = np.array([input_values])
    scaled_input = scaler.transform(input_array)
    prediction = model.predict(scaled_input)[0]

    st.success(f"🧱 Predicted Concrete Compressive Strength: **{prediction:.2f} MPa**")