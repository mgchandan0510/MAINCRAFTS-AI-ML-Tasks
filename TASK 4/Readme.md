# Classification Models, Evaluation Metrics and Handling Imbalanced Data

## Project Overview

This project demonstrates the development and evaluation of machine learning classification models using the Breast Cancer Wisconsin Dataset from Scikit-learn. The objective is to classify tumors as either malignant or benign and evaluate model performance using various classification metrics.

The project covers the complete machine learning workflow, including data preprocessing, feature scaling, model training, prediction, evaluation, ROC analysis, handling class imbalance, and model comparison.

---

## Objectives

* Understand binary classification problems
* Train and evaluate classification models
* Analyze model performance using multiple evaluation metrics
* Generate ROC Curve and AUC Score
* Handle class imbalance using class weights
* Compare Logistic Regression and Decision Tree classifiers

---

## Dataset

**Dataset:** Breast Cancer Wisconsin Dataset

**Source:** Scikit-learn Built-in Dataset

### Dataset Information

* Total Samples: 569
* Features: 30 numerical features
* Classes: 2

Target Labels:

* 0 → Malignant
* 1 → Benign

---

## Technologies Used

* Python
* Jupyter Notebook
* Pandas
* NumPy
* Matplotlib
* Scikit-learn

---

## Machine Learning Workflow

### 1. Data Loading

* Load Breast Cancer Wisconsin Dataset
* Create feature matrix and target variable

### 2. Data Preprocessing

* Train-test split (80:20)
* Stratified sampling
* Feature scaling using StandardScaler

### 3. Model Training

* Logistic Regression
* Decision Tree Classifier

### 4. Model Evaluation

* Confusion Matrix
* Precision
* Recall
* F1-Score
* ROC Curve
* AUC Score

### 5. Handling Imbalanced Data

* Balanced class weights using:

  ```python
  class_weight='balanced'
  ```

### 6. Model Comparison

* Logistic Regression vs Decision Tree
* Performance and generalization comparison

---

## Evaluation Metrics

### Confusion Matrix

Provides detailed information about:

* True Positives (TP)
* True Negatives (TN)
* False Positives (FP)
* False Negatives (FN)

### Precision

Measures how many predicted positive cases are actually positive.

### Recall

Measures how many actual positive cases are correctly identified.

### F1-Score

Balances Precision and Recall into a single metric.

### ROC Curve

Shows the trade-off between True Positive Rate and False Positive Rate.

### AUC Score

Measures the model's ability to distinguish between classes.

---

## Results

The Logistic Regression model achieved strong classification performance on the Breast Cancer Wisconsin Dataset.

Key observations:

* High Precision and Recall
* Strong F1-Score
* Excellent ROC-AUC performance
* Effective detection of cancer cases
* Better stability and generalization than Decision Tree

Logistic Regression was selected as the final model due to its reliability, interpretability, and overall predictive performance.

---

## Project Structure

```text
├── task4_CM,EM&HID.ipynb
├── CLASSIFICATION MODELS, EVALUATION METRICS AND HANDLING IMBALANCED DATA.pdf
├── README.md
```

---

## Future Improvements

* Hyperparameter tuning
* Feature selection
* Cross-validation
* Ensemble learning techniques
* Random Forest Classifier
* XGBoost Classifier
* Support Vector Machine (SVM)

---

## Conclusion

This project successfully demonstrates the implementation of classification models for breast cancer prediction. Various evaluation metrics were used to assess model performance beyond accuracy, highlighting the importance of Precision, Recall, F1-Score, ROC Curve, and AUC Score. The study also emphasizes the significance of handling class imbalance when building reliable machine learning models for real-world applications.

---
# Author 

Chandan Kumar
