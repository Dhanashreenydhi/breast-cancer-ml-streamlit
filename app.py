import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("xgboost_breast_cancer_model.pkl")

st.set_page_config(
    page_title="Breast Cancer Prediction",
    layout="centered"
)

st.title("Breast Cancer Prediction System")
st.write(
    "This application predicts whether a breast tumor is Benign or Malignant "
    "using a trained XGBoost classification model."
)

st.markdown("---")

# Feature names (30 features from Breast Cancer Wisconsin dataset)
feature_names = [
    "mean radius", "mean texture", "mean perimeter", "mean area",
    "mean smoothness", "mean compactness", "mean concavity",
    "mean concave points", "mean symmetry", "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error",
    "smoothness error", "compactness error", "concavity error",
    "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area",
    "worst smoothness", "worst compactness", "worst concavity",
    "worst concave points", "worst symmetry", "worst fractal dimension"
]

st.subheader("Input Feature Values")

inputs = []
for feature in feature_names:
    value = st.number_input(
        label=feature,
        min_value=0.0,
        format="%.6f"
    )
    inputs.append(value)

input_array = np.array(inputs).reshape(1, -1)

st.markdown("---")

if st.button("Predict"):
    prediction = model.predict(input_array)[0]
    probabilities = model.predict_proba(input_array)[0]

    if prediction == 1:
        st.warning("Prediction Result: Malignant")
    else:
        st.success("Prediction Result: Benign")

    st.write("Prediction Probabilities:")
    st.write(f"Benign: {probabilities[0]:.4f}")
    st.write(f"Malignant: {probabilities[1]:.4f}")
