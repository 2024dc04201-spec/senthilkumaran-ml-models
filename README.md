# ML Assignment 2 - Classification Model Deployment

## a. Problem Statement
To build and deploy a machine learning application capable of classifying [insert what your dataset predicts, e.g., Breast Cancer Malignancy] using multiple algorithms and comparing their performance.

## b. Dataset Description
* **Source:** [Insert Source, e.g., UCI Machine Learning Repository]
* **Description:** The dataset contains [number] instances and [number] attributes.
* **Target Variable:** Binary classification (0 = Benign, 1 = Malignant).

## c. Models Used & Comparison

| ML Model Name       |   Accuracy |    AUC |   Precision |   Recall |   F1 Score |    MCC |
|:--------------------|-----------:|-------:|------------:|---------:|-----------:|-------:|
| Logistic Regression |     0.8044 | 0.8842 |      0.8055 |   0.8044 |     0.8039 | 0.6087 |
| Decision Tree       |     0.7823 | 0.7816 |      0.7823 |   0.7823 |     0.7822 | 0.5637 |
| kNN                 |     0.7447 | 0.8165 |      0.7453 |   0.7447 |     0.744  | 0.4884 |
| Naive Bayes         |     0.7005 | 0.7865 |      0.7055 |   0.7005 |     0.6967 | 0.4027 |
| Random Forest       |     0.8462 | 0.9159 |      0.8473 |   0.8462 |     0.8463 | 0.6934 |
| XGBoost             |     0.86   | 0.9229 |      0.8606 |   0.86   |     0.86   | 0.7203 |

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
