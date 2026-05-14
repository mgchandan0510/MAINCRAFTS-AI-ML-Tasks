# AI & ML Task 1  
## Linear Regression Model for House Price Prediction

This project focuses on building and evaluating a **Linear Regression Model** using the **California Housing Dataset** from Scikit-learn. The project demonstrates the complete Machine Learning workflow including data preprocessing, exploratory data analysis, model training, prediction, evaluation, and visualization.

The model predicts house prices based on housing features such as income, house age, rooms, population, occupancy, and geographical location.

---

# Project Overview

The workflow followed in this project includes:

- Data Loading
- Exploratory Data Analysis (EDA)
- Data Preprocessing
- Feature Selection
- Train-Test Split
- Model Training
- Prediction
- Model Evaluation
- Data Visualization

---

# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

---

# Dataset

The project uses the **California Housing Dataset** available in Scikit-learn.

## Dataset Loader

```python
from sklearn.datasets import fetch_california_housing
```

## Features in Dataset

- Median Income
- House Age
- Average Rooms
- Average Bedrooms
- Population
- Occupancy
- Latitude
- Longitude

## Target Variable

- Median House Value

---

# Machine Learning Workflow

## Step 1: Import Required Libraries

The required Python libraries for data analysis, visualization, and machine learning were imported.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
```

---

## Step 2: Load the Dataset

The California Housing Dataset was loaded and converted into a Pandas DataFrame.

```python
data = fetch_california_housing(as_frame=True)
df = pd.concat([data.data, data.target.rename("MedHouseVal")], axis=1)

df.head()
```

---

## Step 3: Exploratory Data Analysis (EDA)

EDA was performed to understand the structure and characteristics of the dataset.

### Operations Performed

- Missing Value Checking
- Statistical Summary
- Correlation Analysis
- Distribution Visualization

```python
df.info()
df.describe()
df.isnull().sum()
```

---

## Step 4: Correlation Analysis

A heatmap was generated to visualize correlations between features.

```python
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
```

---

## Step 5: Feature Selection

Input features and target variables were separated.

```python
X = df.drop(columns='MedHouseVal')
y = df['MedHouseVal']
```

---

## Step 6: Train-Test Split

The dataset was divided into training and testing sets using an 80:20 ratio.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

---

## Step 7: Model Training

A Linear Regression model was trained using the training dataset.

```python
model = LinearRegression()
model.fit(X_train, y_train)
```

---

## Step 8: Prediction

Predictions were generated using the testing dataset.

```python
y_pred = model.predict(X_test)
```

---

## Step 9: Model Evaluation

The model performance was evaluated using regression metrics.

```python
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)
```

---

## Step 10: Visualization of Results

### Actual vs Predicted Plot

```python
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted House Prices")
plt.show()
```

### Residual Distribution Plot

```python
residuals = y_test - y_pred

sns.histplot(residuals, kde=True)
plt.title("Residual Distribution")
plt.show()
```

---

# Model Performance

| Metric | Value |
|--------|--------|
| Mean Absolute Error (MAE) | 0.5332 |
| Root Mean Squared Error (RMSE) | 0.7456 |
| R² Score | 0.5758 |

---

# Visualizations

The following visualizations were generated during the project:

- Correlation Heatmap
- Actual vs Predicted Scatter Plot
- Residual Distribution Plot
- Evaluation Metrics Output

---

# Results

The Linear Regression model successfully predicted house prices with moderate accuracy.

## Key Observations

- Median income strongly influenced house prices.
- The dataset contained no missing values.
- The model explained approximately 57% of the variance in house prices.

---

# Future Improvements

The performance of the model can be improved using:

- Feature Engineering
- Feature Scaling
- Hyperparameter Tuning
- Outlier Removal

Advanced algorithms that can be explored:

- Random Forest Regressor
- Decision Tree Regressor
- Gradient Boosting
- XGBoost

---

# How to Run the Project

## Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

## Run Jupyter Notebook

```bash
jupyter notebook
```

Open the notebook file:

```bash
task1_ml_linear_regression.ipynb
```

---

# Project Structure

```text
Task-1-Linear-Regression/
│
├── task1_ml_linear_regression.ipynb
├── report.pdf
├── graphs/
│   ├── correlation_heatmap.png
│   ├── actual_vs_predicted.png
│   ├── residual_plot.png
│   └── metrics_output.png
└── README.md
```

---

# Conclusion

This project successfully demonstrated the implementation of a Linear Regression model for house price prediction using the California Housing Dataset.

The project provided practical understanding of:

- Regression Analysis
- Data Visualization
- Machine Learning Workflow
- Model Evaluation Techniques
- Python-based Data Science Tools

The results showed that Linear Regression can effectively predict house prices with moderate accuracy and serves as a strong foundation for advanced Machine Learning models.

---

# Author

Chandan Kumar

