# ❤️ Heart Disease Prediction Analysis

This repository provides a comprehensive analysis and predictive modeling for heart disease using machine learning techniques. The data is based on a merged and cleaned heart disease dataset, which includes various medical and demographic attributes.

## 📊 Dataset Overview

The main dataset (`cleaned_merged_heart_dataset.csv`) contains 1,888 entries and 14 columns:

| Column     | Description                                   | Type    |
|------------|-----------------------------------------------|---------|
| age        | Age of the patient                            | int64   |
| sex        | Sex (1 = male; 0 = female)                    | int64   |
| cp         | Chest pain type (0–3)                         | int64   |
| trestbps   | Resting blood pressure                        | int64   |
| chol       | Serum cholesterol (mg/dl)                     | int64   |
| fbs        | Fasting blood sugar (>120 mg/dl) (1 = true)   | int64   |
| restecg    | Resting electrocardiographic results (0–2)    | int64   |
| thalachh   | Maximum heart rate achieved                   | int64   |
| exang      | Exercise induced angina (1 = yes; 0 = no)     | int64   |
| oldpeak    | ST depression induced by exercise             | float64 |
| slope      | Slope of peak exercise ST segment (0–2)       | int64   |
| ca         | Number of major vessels colored by fluoroscopy | int64   |
| thal       | Thalassemia (1 = normal; 2 = fixed defect; 3 = reversible defect) | int64   |
| target     | Diagnosis of heart disease (1 = present, 0 = absent) | int64   |

There are **no missing values** in the dataset. ✅

## 🚀 Project Workflow

1. **Data Import and Inspection**
    - Data is loaded using pandas.
    - Initial inspection using `.head()`, `.info()`, and `.isnull().sum()` confirms data integrity.

2. **Data Preprocessing**
    - Features are standardized using `StandardScaler` to improve model performance.

3. **Model Building**
    - The notebook demonstrates splitting the dataset and training a Logistic Regression model for heart disease prediction.

## 📝 Example Code Snippets

### Importing Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

### Loading and Inspecting Data

```python
df = pd.read_csv('/content/cleaned_merged_heart_dataset.csv')
print(df.head())
print(df.info())
print(df.isnull().sum())
```

### Data Standardization

```python
from sklearn.preprocessing import StandardScaler
X = df.drop('target', axis=1)
y = df['target']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
display(X_scaled.head())
```

## 🧠 Model Training

You can use scikit-learn's Logistic Regression or other classifiers to build and evaluate models for predicting heart disease.

## ⚙️ Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

Install required packages using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## ▶️ Usage

Clone the repository and run the notebook `HeartDiseasePrediction.ipynb` in your preferred environment (e.g., Jupyter Notebook, Google Colab).

## 📄 License

This project is licensed under the MIT License.

## ✨ Author

Kundan Kumar  
GitHub: [kundankumar-35](https://github.com/kundankumar-35)