#!/usr/bin/env python
# coding: utf-8

# In[33]:


# app.py - working version
import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURATION & STYLING ---
st.set_page_config(page_title="ML Classifier Pro", layout="wide", page_icon="🤖")

# Custom CSS for a cleaner look
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- CACHING FOR PERFORMANCE ---
@st.cache_resource
def load_assets(model_name):
    """Loads model and scaler once and caches them."""
    model_path = f"model/{model_name}.pkl"
    scaler_path = 'model/scaler.pkl'
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        return joblib.load(model_path), joblib.load(scaler_path)
    return None, None

# --- SIDEBAR ---
with st.sidebar:
    st.header("Settings")
    model_options = ["Logistic_Regression", "Decision_Tree", "KNN", "Naive_Bayes", "Random_Forest", "XGBoost"]
    selected_model_name = st.selectbox("Select Model Architecture", model_options)
    
    st.divider()
    st.info("**Author:** Senthil Kumaran S\n\n**BITS ID:** 2024DC04201")

# --- MAIN UI ---
st.title("🤖 Machine Learning Classification Dashboard")
st.markdown("Upload your test dataset to evaluate model performance in real-time.")

# Load Model/Scaler
model, scaler = load_assets(selected_model_name)

if model is None:
    st.error(f"⚠️ Model file for **{selected_model_name}** or scaler.pkl was not found in `/model` directory.")
    st.stop()

# File Upload
uploaded_file = st.file_uploader("Upload your Test Dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    # Try reading with comma, if that results in 1 col, try semicolon
    df_raw = pd.read_csv(uploaded_file, sep="|", quoting=3, header=None)
    
    # 2. Get the content, strip the leading/trailing double quotes
    # and split it by the actual comma separator
    header_str = df_raw.iloc[0, 0].strip('"')
    data_rows = df_raw.iloc[1:, 0].str.strip('"')
    
    # 3. Create the clean DataFrame
    data = data_rows.str.split(',', expand=True)
    data.columns = header_str.split(',')
    
    # 4. Clean up column names (remove any lingering quotes)
    data.columns = [c.strip('"') for c in data.columns]
      
    if data.shape[1] == 1:
        uploaded_file.seek(0)
        data = pd.read_csv(uploaded_file, sep=';')

    st.subheader("📋 Data Preview")
    st.dataframe(data.head(5), use_container_width=True)

    
    # --- PREPROCESSING LOGIC START ---
    # Check if data is raw (contains strings like 'blue-collar') or processed (all numeric)
    # We check a known categorical column, e.g., 'job'
    with st.expander("Processing Details", expanded=False):
        if 'job' in data.columns and data['job'].dtype == 'object':
            st.info("🔄 Categorical data detected. Applying pipeline transformations...")
            
            # 1. Map Months
            month_map = {
                'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
                'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
                'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
            }
            if 'month' in data.columns:
                data['month'] = data['month'].map(month_map)
    
            # 2. Binary Mapping
            binary_mapping = {'yes': 1, 'no': 0}
            for col in ['deposit', 'default', 'housing', 'loan', 'y']:
                if col in data.columns:
                    data[col] = data[col].map(binary_mapping)
    
            # 3. Education Mapping
            edu = {'primary': 1, 'secondary': 2, 'tertiary': 3, 'unknown': 0}
            if 'education' in data.columns:
                data['education_level'] = data['education'].map(edu)
            
            # 4. One-Hot Encoding
            # We must align with the columns the model expects
            categorical_cols = ['job', 'marital', 'education', 'contact', 'poutcome']
            categorical_cols = [c for c in categorical_cols if c in data.columns]
            data = pd.get_dummies(data, columns=categorical_cols, drop_first=True, dtype=int)
            
            # 5. Handle Missing Columns (Model expects specific columns)
            # We need the scaler's expected feature names. 
            # Since we don't have them easily, we rely on alignment if possible, 
            # OR usually, we just ensure the user uploads the PRECESSED data.
            # But for now, let's assume standard dummy encoding works close enough.
            
            # Identify target (usually 'deposit' or 'y')
            target_options = ['deposit', 'y', 'target']
            target_col = next((col for col in target_options if col in data.columns), data.columns[-1])
            
        else:
            st.info("Processed data detected.")
            target_col = data.columns[-1]

    # --- PREPROCESSING LOGIC END ---

    X_test = data.drop(columns=[target_col])
    y_test = data[target_col]
    y_test = y_test.astype(int)
    
    # Align columns with scaler (Crucial step often missed)
    # If OneHotEncoding produced different columns than training, this will fail.
    # Ideally, you should save the 'feature_names' in training and load them here to reindex.
    
    st.write(f"Data shape: {data.shape}")
  
    try:
        # Transform
        X_test_scaled = scaler.transform(X_test)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        
     # Dashboard Layout
        st.write("Model Accuracy: ", f"{acc:.2%}")
        st.write("**Classification Report:**")
        st.code(classification_report(y_test, y_pred))

        st.write("**Confusion Matrix**")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        st.pyplot(fig)    
   
    except ValueError as e:
        st.error(f"Prediction Error: {e}")
        st.info("Check if your test CSV matches the training feature set.")


# In[ ]:
