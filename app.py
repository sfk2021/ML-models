import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load model
model = joblib.load("height_weight_model.pkl")

# Set page configuration
st.set_page_config(
    page_title="FitPredictAI",
    page_icon="📏",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
.main-title {
    text-align: center;
    font-size: 50px;
    font-weight: bold;
    background: linear-gradient(90deg,#00c6ff,#0072ff,#7b2ff7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    text-align:center;
    font-size:18px;
    color:#666666;
}

.prediction-box {
    padding:20px;
    border-radius:15px;
    background-color:#e8f4ff;
    text-align:center;
    font-size:30px;
    font-weight:bold;
    color:#0072ff;
}

.accuracy-box {
    padding:10px;
    border-radius:10px;
    background-color:#f0f8ff;
    text-align:center;
    font-size:20px;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown(
    '<div class="main-title">📏 FitPredictAI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-Powered Height to Weight Predictor</div>',
    unsafe_allow_html=True
)

st.write("")

# Accuracy (replace with your actual score)
accuracy = 96.5

st.markdown(
    f'<div class="accuracy-box">🎯 Model Accuracy: {accuracy}%</div>',
    unsafe_allow_html=True
)

st.write("")

# Height Input (+ and - buttons)
height = st.number_input(
    "Select Height (meters)",
    min_value=1.00,
    max_value=2.50,
    value=1.70,
    step=0.01
)

# Blue Predict Button
st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #0072ff;
    color: white;
    font-size: 18px;
    border-radius: 10px;
    padding: 10px 24px;
}
</style>
""", unsafe_allow_html=True)

if st.button("Predict Weight"):

    prediction = model.predict([[height]])[0]

    st.markdown(
        f'<div class="prediction-box">⚖️ {prediction:.2f} kg</div>',
        unsafe_allow_html=True
    )

    st.write("")

    st.subheader("📊 Height vs Weight Trend")

    heights = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4]
    weights = model.predict(pd.DataFrame(heights, columns=["Height"]))

    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(heights, weights, marker="o")
    ax.scatter([height], [prediction], s=150)
    ax.set_xlabel("Height (m)")
    ax.set_ylabel("Weight (kg)")
    ax.set_title("Height vs Predicted Weight")

    st.pyplot(fig)

    st.success(
        f"For a height of {height:.2f} m, the predicted weight is {prediction:.2f} kg."
    )