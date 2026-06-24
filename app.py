import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Page Config
st.set_page_config(
    page_title="FitPredictAI",
    page_icon="📏",
    layout="centered"
)

# Load model
model = joblib.load("height_weight_model.pkl")

# Custom CSS
st.markdown("""
<style>

.main-title {
    text-align:center;
    font-size:52px;
    font-weight:bold;
    background: linear-gradient(90deg,#00c6ff,#0072ff,#7b2ff7);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

.subtitle{
    text-align:center;
    color:gray;
    font-size:18px;
}

.summary-card{
    padding:20px;
    border-radius:15px;
    background:#f0f8ff;
    border-left:6px solid #0072ff;
}

.footer{
    text-align:center;
    color:gray;
    font-size:14px;
}

div.stButton > button:first-child{
    background-color:#0072ff;
    color:white;
    border:none;
    border-radius:10px;
    height:3em;
    width:100%;
    font-size:18px;
}

</style>
""", unsafe_allow_html=True)

# Header
st.markdown(
    '<div class="main-title">📏 FitPredictAI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-Powered Height to Weight Prediction</div>',
    unsafe_allow_html=True
)

st.write("")

# Description
st.markdown("""
### About the Project

FitPredictAI uses a Machine Learning Linear Regression model to estimate
a person's weight based on their height. Enter your height below and
receive an instant prediction.
""")

st.divider()

# Height Input
height = st.number_input(
    "Select Height (meters)",
    min_value=1.00,
    max_value=2.50,
    value=1.70,
    step=0.01
)

# Prediction
if st.button("Predict Weight"):

    prediction = model.predict([[height]])[0]

    # Summary Card
    st.markdown("## Prediction Summary")

    st.markdown(
        f"""
        <div class="summary-card">
        <h3>Prediction Results</h3>
        <p><b>Height Entered:</b> {height:.2f} m</p>
        <p><b>Predicted Weight:</b> {prediction:.2f} kg</p>
        <p><b>Model:</b> Linear Regression</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    # Visualization
    st.markdown("## 📊 Interactive Visualization")

    heights = [1.0,1.2,1.4,1.6,1.8,2.0,2.2,2.4]
    weights = model.predict(
        pd.DataFrame(heights, columns=["Height"])
    )

    fig, ax = plt.subplots(figsize=(8,4))

    ax.plot(
        heights,
        weights,
        marker="o",
        linewidth=2
    )

    ax.scatter(
        [height],
        [prediction],
        s=150
    )

    ax.grid(True)

    ax.set_xlabel("Height (m)")
    ax.set_ylabel("Weight (kg)")
    ax.set_title("Height vs Predicted Weight")

    st.pyplot(fig)

st.divider()

# BMI Calculator
st.markdown("## 🧮 BMI Calculator")

bmi_height = st.number_input(
    "Height for BMI (meters)",
    min_value=1.00,
    max_value=2.50,
    value=1.70,
    key="bmi_height"
)

bmi_weight = st.number_input(
    "Weight for BMI (kg)",
    min_value=1.0,
    value=70.0,
    key="bmi_weight"
)

if st.button("Calculate BMI"):

    bmi = bmi_weight / (bmi_height ** 2)

    st.success(f"Your BMI is {bmi:.2f}")

    if bmi < 18.5:
        st.info("Category: Underweight")
    elif bmi < 25:
        st.info("Category: Normal Weight")
    elif bmi < 30:
        st.warning("Category: Overweight")
    else:
        st.error("Category: Obese")

st.divider()

# AI Model Information
st.markdown("## 🤖 AI Model Information")

st.info("""
**Model Type:** Linear Regression

**Input Feature:** Height

**Output:** Weight

**Machine Learning Library:** Scikit-Learn

**Deployment Platform:** Streamlit
""")

st.divider()

# How It Works
st.markdown("## ⚙️ How It Works")

st.markdown("""
1. Enter your height.

2. The trained Linear Regression model receives the input.

3. The model predicts the corresponding weight.

4. Results are displayed along with a visual trend chart.

5. Use the BMI Calculator for additional health insights.
""")

st.divider()

# Footer
st.markdown(
    """
    <div class="footer">
    Developed with Python, Scikit-Learn and Streamlit
    </div>
    """,
    unsafe_allow_html=True
)