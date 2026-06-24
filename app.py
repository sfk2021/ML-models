import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Page settings
st.set_page_config(
    page_title="FitPredict AI",
    page_icon="📏",
    layout="wide"
)

# Load model
model = joblib.load("height_weight_model.pkl")

# Title
st.title("📏 FitPredict AI")
st.subheader("AI-Powered Height to Weight Prediction")

st.write("Enter your height and get a predicted weight instantly.")

# Height input
height = st.slider(
    "Select Height (meters)",
    min_value=1.0,
    max_value=2.5,
    value=1.70,
    step=0.01
)

# Prediction button
if st.button("Predict Weight"):

    weight = model.predict([[height]])[0]

    st.success(f"Predicted Weight: {weight:.2f} kg")

    # Create chart data
    heights = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4]
    weights = model.predict(pd.DataFrame(heights, columns=["Height"]))

    chart_df = pd.DataFrame({
        "Height": heights,
        "Weight": weights
    })

    st.subheader("📊 Height vs Predicted Weight")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(chart_df["Height"], chart_df["Weight"], marker="o")
    ax.scatter(height, weight, s=100)
    ax.set_xlabel("Height (m)")
    ax.set_ylabel("Weight (kg)")
    ax.set_title("Height vs Weight Prediction")

    st.pyplot(fig)

    st.info(
        f"For a height of {height:.2f} m, the model predicts "
        f"a weight of {weight:.2f} kg."
    )