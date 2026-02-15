## Breast Cancer Classification using Machine Learning
a. Problem Statement

The objective of this project is to implement and compare multiple machine learning classification models to predict whether a breast tumor is malignant or benign using the Breast Cancer Wisconsin (Diagnostic) dataset. Early detection of breast cancer significantly improves survival rates, and machine learning models can assist in reliable diagnostic prediction.

This project evaluates six classification algorithms and deploys them through an interactive Streamlit web application.

b. Dataset Description

Dataset Name: Breast Cancer Wisconsin (Diagnostic)

Source: UCI Machine Learning Repository (loaded via sklearn)

Number of Instances: 569

Number of Features: 30 numerical features

Target Variable:

0 = Benign

1 = Malignant

The dataset consists of computed features derived from digitized images of fine needle aspirate (FNA) of breast masses.

c. Models Used and Evaluation Metrics

The following six classification models were implemented and evaluated on the same dataset:

Logistic Regression

Decision Tree Classifier

K-Nearest Neighbors (KNN)

Gaussian Naive Bayes

Random Forest (Ensemble Method)

XGBoost (Ensemble Method)

Model Comparison Table

| ML Model            | Accuracy | AUC    | Precision | Recall | F1 Score | MCC    |
| ------------------- | -------- | ------ | --------- | ------ | -------- | ------ |
| Logistic Regression | 0.9825   | 0.9954 | 0.9861    | 0.9861 | 0.9861   | 0.9623 |
| Decision Tree       | 0.9211   | 0.9163 | 0.9565    | 0.9167 | 0.9362   | 0.8341 |
| KNN                 | 0.9561   | 0.9788 | 0.9589    | 0.9722 | 0.9655   | 0.9054 |
| Naive Bayes         | 0.9386   | 0.9878 | 0.9452    | 0.9583 | 0.9517   | 0.8676 |
| Random Forest       | 0.9561   | 0.9937 | 0.9589    | 0.9722 | 0.9655   | 0.9054 |
| XGBoost             | 0.9474   | 0.9944 | 0.9459    | 0.9722 | 0.9589   | 0.8864 |

Model Performance Observations

| ML Model            | Observation                                                                                                                |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| Logistic Regression | Achieved the highest overall accuracy and balanced performance, indicating strong linear separability in the dataset.      |
| Decision Tree       | Simpler model with lower generalization performance compared to ensemble methods.                                          |
| KNN                 | Performed well after feature scaling and showed strong recall and F1-score.                                                |
| Naive Bayes         | Fast and probabilistic classifier; slightly lower performance due to independence assumption between features.             |
| Random Forest       | Robust ensemble method with high AUC and stable performance across metrics.                                                |
| XGBoost             | Achieved very high AUC and recall, making it suitable for medical diagnosis where minimizing false negatives is important. |

