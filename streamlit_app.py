import streamlit as st
import numpy as np
import joblib

def set_bg_image():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://images.unsplash.com/photo-1517352495258-82e54c35e06e?q=80&w=3270&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg_image()

# Load model, scaler, and label encoder
model = joblib.load("crop_model.pkl")
scaler = joblib.load("crop_scaler.pkl")
label_encoder = joblib.load("crop_label_encoder.pkl")

# App title
st.title("Crop Recommendation System")

# Input fields
N = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=50)
P = st.number_input("Phosphorus (P)", min_value=0, max_value=200, value=50)
K = st.number_input("Potassium (K)", min_value=0, max_value=200, value=50)
temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=6.5)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=300.0, value=100.0)

# Predict button
if st.button("Predict Crop"):
    # Create input array
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    scaled_input = scaler.transform(input_data)
    pred_encoded = model.predict(scaled_input)
    crop_name = label_encoder.inverse_transform(pred_encoded)[0]
    
    # Show result
    st.success(f"Recommended Crop: **{crop_name}**")
