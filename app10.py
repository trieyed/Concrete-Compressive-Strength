import streamlit as st
import numpy as np
import joblib

# Load models and scalers
baseline_model = joblib.load("baseline_regression_rfr.pkl")
tuned_model = joblib.load("tuned_regression_rfr.pkl")
x_scaler = joblib.load("x_scaler.pkl")
y_scaler = joblib.load("y_scaler.pkl")

# Set page configuration
st.set_page_config(page_title="Concrete Strength Predictor", layout="centered")

# App title
st.title("🧱 Concrete Compressive Strength Predictor")
st.markdown("Use this app to predict **Concrete Compressive Strength (MPa)** based on mix design inputs.")

# Sidebar model selection
model_choice = st.sidebar.radio("Select Model", ("Baseline Model", "Tuned Model"))
model = baseline_model if model_choice == "Baseline Model" else tuned_model

# Form for inputs
with st.form("input_form"):
    st.markdown("### 🔧 Enter Mix Design Parameters")

    cement = st.number_input("Cement (kg/m³)", min_value=0.0, step=1.0)
    blast_furnace_slag = st.number_input("Blast Furnace Slag (kg/m³)", min_value=0.0, step=1.0)
    fly_ash = st.number_input("Fly Ash (kg/m³)", min_value=0.0, step=1.0)
    water = st.number_input("Water (kg/m³)", min_value=0.0, step=1.0)
    superplasticizer = st.number_input("Superplasticizer (kg/m³)", min_value=0.0, step=0.1)
    coarse_aggregate = st.number_input("Coarse Aggregate (kg/m³)", min_value=0.0, step=1.0)
    fine_aggregate = st.number_input("Fine Aggregate (kg/m³)", min_value=0.0, step=1.0)
    age = st.number_input("Age (days)", min_value=1, step=1)

    submitted = st.form_submit_button("Predict")

# Prediction
if submitted:
    input_features = np.array([[cement, blast_furnace_slag, fly_ash, water,
                                superplasticizer, coarse_aggregate, fine_aggregate, age]])

    # Apply feature scaling
    scaled_input = x_scaler.transform(input_features)

    # Make prediction
    scaled_prediction = model.predict(scaled_input).reshape(-1, 1)

    # Inverse transform prediction to original scale
    predicted_strength = y_scaler.inverse_transform(scaled_prediction).flatten()[0]

    st.success(f"🔮 Predicted Concrete Compressive Strength: **{predicted_strength:.2f} MPa**")
