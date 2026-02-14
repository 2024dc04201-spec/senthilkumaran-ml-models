# ML Assignment 2 - Classification Model Deployment

## Bank Marketing Strategy: Predicting Term Deposit Success
This project analyzes a bank's telemarketing campaign data to predict whether a client will subscribe to a term deposit.

## Objective: 
Develop a high-performance classification model to identify potential customers. Key Focus: Feature Engineering, Model Benchmarking, and Hyperparameter Tuning.

## b. Dataset Description
* **Source:** [Insert Source, e.g., Kaagle Machine Learning Repository]
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

*(Note: Replace the values above with the actual output from your train_models.py script)*

## Observations
| ML Model Name       | Observation about model performance |
|---------------------|-------------------------------------|
| Logistic Regression | An Accuracy of 0.8044 and an AUC of 0.8842 for a simple linear model is impressive. This tells us that a significant portion of your feature set has a strong linear relationship with the target variable. Usually, if Logistic Regression performs this close to complex tree models, your data is relatively "clean" and well-structured. |
| Decision Tree       | Notice the drop-off from Random Forest. The gap in MCC (0.5637 vs 0.6934) suggests the single tree is likely "branching" into noise (overfitting), whereas the Forest averages that noise out. |
| kNN                 | The lower MCC (0.4884) suggests that as the dimensionality of your data increases, the "distance" between neighbors becomes less meaningful. It’s struggling to find clear clusters compared to the boundary-based logic of the trees. |
| Naive Bayes         | This is your floor. While fast, the lower metrics across the board suggest the "independence assumption" of Naive Bayes isn't holding up—meaning your features likely have correlations that this model is ignoring. |
| Random Forest       | Not far behind, its F1 Score (0.8463) indicates a near-perfect balance between Precision and Recall. This suggests the model is robust and handles the non-linear relationships in your data much better than the single Decision Tree. |
| XGBoost             | With the highest AUC (0.9229) and MCC (0.7203), it isn't just getting things right by accident; it’s excellent at distinguishing between classes. The high MCC is particularly important here—it suggests the model performs consistently even if your classes are slightly imbalanced. |

## d. Deployment Instructions
1.  Install dependencies: `pip install -r requirements.txt`
2.  Run the app: `streamlit run app.py`
