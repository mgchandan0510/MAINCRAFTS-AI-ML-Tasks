# AI & ML Task 3

# Model Validation, Overfitting Control and Hyperparameter Tuning

This project focuses on implementing model validation, overfitting detection, cross-validation, and hyperparameter tuning using the California Housing Dataset from Scikit-learn. The project demonstrates important Machine Learning workflows including data preprocessing, feature scaling, model training, model evaluation, overfitting analysis, cross-validation, and parameter optimization.

Different regression models were implemented and evaluated using RMSE and R² Score to improve prediction accuracy and model generalization.

---

# Project Overview

The workflow followed in this project includes:

* Data Loading
* Data Preprocessing
* Feature Scaling
* Train-Test Split
* Model Training
* Overfitting Detection
* Cross-Validation
* Hyperparameter Tuning
* Model Evaluation
* Data Visualization

---

# Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Jupyter Notebook

---

# Dataset

The project uses the California Housing Dataset available in Scikit-learn.

## Dataset Loader

```python
from sklearn.datasets import fetch_california_housing
```

---

# Features in Dataset

* Median Income
* House Age
* Average Rooms
* Average Bedrooms
* Population
* Occupancy
* Latitude
* Longitude

---

# Target Variable

* Median House Value

---

# Machine Learning Workflow

# Step 1: Import Required Libraries

The required Python libraries for machine learning and evaluation were imported.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV
)

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import (
    mean_squared_error,
    r2_score
)
```

---

# Step 2: Load the Dataset

The California Housing Dataset was loaded and converted into a Pandas DataFrame.

```python
data = fetch_california_housing(as_frame=True)

df = pd.concat(
    [data.data, data.target.rename("HousePrice")],
    axis=1
)

df.head()
```

---

# Step 3: Data Preprocessing

Input features and target variables were separated.

```python
X = df.drop("HousePrice", axis=1)
y = df["HousePrice"]
```

Feature scaling was performed using `StandardScaler()`.

```python
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
```

---

# Step 4: Train-Test Split

The dataset was divided into training and testing sets using an 80:20 ratio.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)
```

---

# Step 5: Model Training

The following regression models were implemented:

* Linear Regression
* Ridge Regression
* Decision Tree Regressor

## Linear Regression

```python
lr = LinearRegression()

lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)
```

## Ridge Regression

```python
ridge = Ridge(alpha=1.0)

ridge.fit(X_train, y_train)

ridge_pred = ridge.predict(X_test)
```

---

# Step 6: Overfitting Detection

A Decision Tree Regressor was trained to analyze overfitting behavior.

```python
tree = DecisionTreeRegressor(random_state=42)

tree.fit(X_train, y_train)

train_pred = tree.predict(X_train)
test_pred = tree.predict(X_test)
```

Training and testing RMSE values were compared to identify overfitting.

```python
train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
```

---

# Step 7: Cross-Validation

5-fold cross-validation was performed using `cross_val_score()`.

```python
cv_scores = cross_val_score(
    tree,
    X_scaled,
    y,
    scoring='neg_root_mean_squared_error',
    cv=5
)

cv_rmse = -cv_scores.mean()
```

---

# Step 8: Hyperparameter Tuning

Hyperparameter tuning was performed using `GridSearchCV()`.

```python
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10]
}

grid = GridSearchCV(
    DecisionTreeRegressor(random_state=42),
    param_grid,
    scoring='neg_root_mean_squared_error',
    cv=5
)

grid.fit(X_train, y_train)
```

---

# Step 9: Model Evaluation

The optimized model was evaluated using RMSE and R² Score.

```python
best_tree = grid.best_estimator_

y_pred = best_tree.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
```

---

# Step 10: Visualization

## RMSE Comparison Plot

```python
plt.figure(figsize=(8,5))

plt.bar(results_df["Model"], results_df["RMSE"])

plt.title("Model RMSE Comparison")

plt.show()
```

## R² Score Comparison Plot

```python
plt.figure(figsize=(8,5))

plt.bar(results_df["Model"], results_df["R2 Score"])

plt.title("Model R2 Score Comparison")

plt.show()
```

---

# Model Performance

| Model                   | RMSE     | R² Score |
| ----------------------- | -------- | -------- |
| Linear Regression       | 0.745581 | 0.575788 |
| Ridge Regression        | 0.745554 | 0.575819 |
| Optimized Decision Tree | 0.645430 | 0.682099 |

---

# Visualizations

The following visualizations were generated during the project:

* Training vs Testing RMSE Output
* Cross-Validation Output
* GridSearchCV Best Parameters
* RMSE Comparison Plot
* R² Score Comparison Plot

---

# Results

The optimized Decision Tree model achieved better prediction accuracy and generalization performance after hyperparameter tuning.

## Key Observations

* Overfitting was detected using training and testing RMSE comparison.
* Cross-validation improved model reliability.
* Hyperparameter tuning reduced overfitting.
* The optimized model achieved lower RMSE and higher R² Score.

---

# Future Improvements

The model performance can be further improved using:

* Feature Engineering
* Advanced Hyperparameter Tuning
* Ensemble Learning Methods
* Outlier Detection
* Feature Selection

Advanced algorithms that can be explored:

* Random Forest Regressor
* Gradient Boosting
* XGBoost
* AdaBoost Regressor

---

# How to Run the Project

## Install Required Libraries

```bash
pip install pandas numpy matplotlib scikit-learn jupyter
```

## Run Jupyter Notebook

```bash
jupyter notebook
```

Open the notebook file:

```bash
task3_model_validation_tuning.ipynb
```

---

# Project Structure

```text
Task-3-Model-Validation/
│
├── task3_model_validation_tuning.ipynb
├── MODEL VALIDATION, OVERFITTING CONTROL AND HYPERPARAMETER TUNING.pdf
├── graphs/
│   ├── overfitting_output.png
│   ├── cross_validation_output.png
│   ├── gridsearch_output.png
│   └── model_comparison_graph.png
└── README.md
```

---

# Conclusion

This project successfully demonstrated model validation, overfitting detection, cross-validation, and hyperparameter tuning using the California Housing Dataset.

The project provided practical understanding of:

* Model Validation
* Overfitting Detection
* Cross-Validation
* Hyperparameter Tuning
* Regression Analysis
* Machine Learning Evaluation Techniques

The results showed that GridSearchCV and cross-validation significantly improved model generalization and reduced overfitting, resulting in more reliable predictions on unseen data.

---

# Author

Chandan Kumar
