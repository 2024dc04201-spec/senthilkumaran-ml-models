#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import joblib
import os

# Metrics
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                             recall_score, f1_score, matthews_corrcoef)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Create a directory to save models
if not os.path.exists('model'):
    os.makedirs('model')

os.environ["LOKY_MAX_CPU_COUNT"] = "8"

from xgboost import XGBClassifier

model = XGBClassifier(
    eval_metric="logloss",   # add this instead
    random_state=42
)

# # ---------------------------------------------------------
# # STEP 1: LOAD DATA
# # ---------------------------------------------------------
# # Option A: Use built-in dataset (Matches assignment criteria: >12 features, >500 rows)
# from sklearn.datasets import load_breast_cancer
# data = load_breast_cancer()
# X = pd.DataFrame(data.data, columns=data.feature_names)
# y = pd.Series(data.target)

# Option B: Load from CSV (Uncomment below if using a Kaggle CSV)
df = pd.read_csv('C:/Users/Senkumaran/OneDrive/Documents/Senkumaran/BITS/Semester2/ML/Assignment/Assignment2/bank.csv')  
print(df.columns)

month_map = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
    'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
    'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
}

df['month'] = df['month'].map(month_map)

# Encode binary categorical variables
binary_mapping = {'yes': 1, 'no': 0}
for col in ['deposit', 'default', 'housing', 'loan']:
    if col in df.columns:
        df[col] = df[col].map(binary_mapping)

df.head(10) 



# In[2]:


# df.info()

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# Target distribution
print(df['deposit'].value_counts())

# Education mapping
edu = {
    'primary': 1,
    'secondary': 2,
    'tertiary': 3,
    'unknown': 0
}

if 'education' in df.columns:
    df['education_level'] = df['education'].map(edu)
else:
    print("Column 'education' not found. Available columns:", df.columns)

# Encode target variable
target_col = "deposit"   # adjust if your dataset uses "y" or "subscribed"
df[target_col] = LabelEncoder().fit_transform(df[target_col])

# One-hot encode categorical features
categorical_cols = ['job', 'marital', 'education', 'contact', 'poutcome']
categorical_cols = [col for col in categorical_cols if col in df.columns]

df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)

# Split features and target
X = df.drop(columns=[target_col])
y = df[target_col]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=15
)
# Align train/test columns (important if test set misses some categories)
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)
# from sklearn.preprocessing import StandardScaler


# In[3]:


# ------------------------------
# STEP 2: PREPROCESSING
# ------------------------------
# Split Data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training set shape:", X_train_scaled.shape)
print("Test set shape:", X_test_scaled.shape)


# In[6]:


# Save the scaler for later use in the app
joblib.dump(scaler, 'model/scaler.pkl')

# Save X_test and y_test to a CSV for you to upload to Streamlit later
test_data = X_test.copy()
test_data['target'] = y_test
test_data.to_csv("test_data_sample.csv", index=False, sep=',', encoding='utf-8')
# data.to_csv("test_data_sample_fixed.csv", index=False, sep=',', encoding='utf-8')

print("Test data saved as 'test_data_sample.csv'. Use this to test your Streamlit App.")


# In[5]:


# ---------------------------------------------------------
# STEP 3: DEFINE MODELS
# ---------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "kNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42)
    }

# ---------------------------------------------------------
# STEP 4: TRAIN AND EVALUATE
# ---------------------------------------------------------
results = []

print("\nTraining Models...")
for name, model in models.items():
    # Train
    # Note: Tree based models don't strictly need scaling, but it doesn't hurt.
    # We use scaled data for consistency here.
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else y_pred
    
    # Calculate Metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    mcc = matthews_corrcoef(y_test, y_pred)
    
    results.append({
        "ML Model Name": name,
        "Accuracy": round(acc, 4),
        "AUC": round(auc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1 Score": round(f1, 4),
        "MCC": round(mcc, 4)
    })
    
    # Save Model
    filename = f'model/{name.replace(" ", "_").lower()}.pkl'
    joblib.dump(model, filename)

# ---------------------------------------------------------
# STEP 5: OUTPUT RESULTS FOR README
# ---------------------------------------------------------
results_df = pd.DataFrame(results)
print("\n" + "="*50)
print("COMPARISON TABLE (Copy this to your README.md)")
print("="*50 + '\n')
print(results_df.to_markdown(index=False))

print("\nModels saved successfully in 'model/' folder.")


# In[ ]:




