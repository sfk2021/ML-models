import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("height_weight_model.pkl")

# Title
st.title("Height to Weight Prediction")

st.write("Enter a height value and predict weight.")

# User Input
height = st.number_input(
    "Enter Height",
    min_value=0.0,
    value=1.50,
    step=0.01
)

# Prediction Button
if st.button("Predict Weight"):

    prediction = model.predict([[height]])

    st.success(
        f"Predicted Weight: {prediction[0]:.2f}"
    )