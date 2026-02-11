#!/usr/bin/env python
# coding: utf-8

# In[5]:


# app.py
import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Page Config
st.set_page_config(page_title="ML Model Classifier", layout="wide")

st.title("🤖 Machine Learning Classification App")
st.write("Upload test data to evaluate different classification models.")

# Sidebar for Model Selection
model_options = [
    "Logistic_Regression", "Decision_Tree", "KNN", 
    "Naive_Bayes", "Random_Forest", "XGBoost"
]
selected_model_name = st.sidebar.selectbox("Select Model", model_options)

# Load Model
model_path = f"model/{selected_model_name}.pkl"


if os.path.exists(model_path):
    model = joblib.load(model_path)
    scaler = joblib.load('model/scaler.pkl') # Ensure scaler is also uploaded
else:
    st.error("Model file not found. Please check directory structure.")
    st.stop()

# File Upload
uploaded_file = st.file_uploader("Upload your Test Dataset (CSV)", type=["csv"])
print('senthil 2323')
print(uploaded_file)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("### Preview of Uploaded Data")
    st.dataframe(data.head())

    # Assuming last column is target for evaluation
    # In real deployment, you might let user select target column
    target_col = data.columns[-1]
    X_test = data.drop(columns=[target_col])
    y_test = data[target_col]

    # Preprocessing (Must match training!)
    # Note: In a real app, you would apply the saved scaler here.
    # checking if scaling is needed based on your training logic
    try:
        X_test_scaled = scaler.transform(X_test)
    except Exception as e:
        st.warning(f"Scaling failed (columns might differ). Using raw data. Error: {e}")
        X_test_scaled = X_test

    if st.button("Run Prediction"):
        # Predict
        y_pred = model.predict(X_test_scaled)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        
        st.write(f"### Results for {selected_model_name.replace('_', ' ')}")
        col1, col2 = st.columns(2)
        col1.metric("Accuracy", f"{acc:.4f}")
        
        st.write("### Classification Report")
        st.text(classification_report(y_test, y_pred))

        # Confusion Matrix
        st.write("### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)


# In[ ]:




