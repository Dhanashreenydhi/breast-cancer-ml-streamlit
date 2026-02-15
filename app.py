import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)

st.set_page_config(page_title="ML Classification App", layout="centered")

st.title("Breast Cancer Classification using Machine Learning")
st.write("This application evaluates multiple classification models on the Breast Cancer Wisconsin dataset.")


# -------------------------
# Load Models
# -------------------------
models = {
    "Logistic Regression": joblib.load("model/logistic.pkl"),
    "Decision Tree": joblib.load("model/decision_tree.pkl"),
    "KNN": joblib.load("model/knn.pkl"),
    "Naive Bayes": joblib.load("model/naive_bayes.pkl"),
    "Random Forest": joblib.load("model/random_forest.pkl"),
    "XGBoost": joblib.load("model/xgboost.pkl"),
}

# -------------------------
# Model Selection
# -------------------------
selected_model_name = st.selectbox("Select Model", list(models.keys()))
model = models[selected_model_name]

# -------------------------
# File Upload
# -------------------------
uploaded_file = st.file_uploader("Upload Test CSV File", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.write("Preview of Uploaded Data:")
    st.dataframe(data.head())

    if "target" not in data.columns:
        st.error("CSV must contain 'target' column.")
    else:
        X = data.drop("target", axis=1)
        y = data["target"]

        # Scaling for Logistic Regression and KNN
        if selected_model_name in ["Logistic Regression", "KNN"]:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        y_pred = model.predict(X)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X)[:, 1]
            auc = roc_auc_score(y, y_prob)
        else:
            auc = "Not Available"

        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred)
        rec = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        mcc = matthews_corrcoef(y, y_pred)

        st.subheader("Evaluation Metrics")

        st.write(f"Accuracy: {acc:.4f}")
        st.write(f"AUC: {auc}")
        st.write(f"Precision: {prec:.4f}")
        st.write(f"Recall: {rec:.4f}")
        st.write(f"F1 Score: {f1:.4f}")
        st.write(f"MCC: {mcc:.4f}")

        # Confusion Matrix
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y, y_pred)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        st.pyplot(fig)
