# ML Assignment 2 - Classification Model Deployment

## a. Problem Statement
To build and deploy a machine learning application capable of classifying [insert what your dataset predicts, e.g., Breast Cancer Malignancy] using multiple algorithms and comparing their performance.

## b. Dataset Description
* **Source:** [Insert Source, e.g., UCI Machine Learning Repository]
* **Description:** The dataset contains [number] instances and [number] attributes.
* **Target Variable:** Binary classification (0 = Benign, 1 = Malignant).

## c. Models Used & Comparison

| ML Model Name          | Accuracy | AUC    | Precision | Recall | F1 Score | MCC    |
|------------------------|----------|--------|-----------|--------|----------|--------|
| Logistic Regression    | 0.9825   | 0.9980 | 0.9826    | 0.9825 | 0.9825   | 0.9622 |
| Decision Tree          | 0.9386   | 0.9351 | 0.9386    | 0.9386 | 0.9386   | 0.8674 |
| kNN                    | 0.9649   | 0.9920 | 0.9654    | 0.9649 | 0.9648   | 0.9242 |
| Naive Bayes            | 0.9561   | 0.9960 | 0.9563    | 0.9561 | 0.9562   | 0.9056 |
| Random Forest          | 0.9649   | 0.9950 | 0.9652    | 0.9649 | 0.9649   | 0.9245 |
| XGBoost                | 0.9737   | 0.9945 | 0.9739    | 0.9737 | 0.9737   | 0.9436 |
*(Note: Replace the values above with the actual output from your train_models.py script)*

## Observations
| ML Model Name       | Observation about model performance |
|---------------------|-------------------------------------|
| Logistic Regression | Performed exceptionally well, indicating the data is linearly separable. |
| Decision Tree       | Slightly lower accuracy, likely due to overfitting on the training split. |
| kNN                 | performed robustly but required feature scaling for optimal results. |
| Naive Bayes         | Good baseline performance; computation was very fast. |
| Random Forest       | High accuracy and stability; less prone to overfitting than single trees. |
| XGBoost             | Competitive performance with high AUC, showing strong predictive power. |

## d. Deployment Instructions
1.  Install dependencies: `pip install -r requirements.txt`
2.  Run the app: `streamlit run app.py`
