# 📊 Bengaluru House Price Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)](https://www.python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-%23FF377E?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)

A **Data Science Regression Project** to predict house prices in Bengaluru using machine learning. Built as part of Data Analysis Programming Lab (Python).

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Setup](#setup)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Demo](#demo)
- [Topics Covered](#topics-covered)

## 📈 Overview
This project demonstrates a complete **end-to-end ML pipeline** for house price prediction:
1. **Data Cleaning**: Handle missing values, feature engineering (BHK, price/sqft)
2. **Exploratory Data Analysis (EDA)**: Visualizations, outlier detection
3. **Feature Engineering**: Dimensionality reduction, one-hot encoding
4. **Model Building**: Linear Regression with cross-validation & GridSearchCV
5. **Outlier Removal**: Using business logic, standard deviation, domain rules
6. **Model Export**: Pickle file + JSON for production use

**Accuracy**: ~80-85% on cross-validation.

## 📦 Dataset
- **Source**: [Kaggle - Bengaluru House Price Data](https://www.kaggle.com/amitabhajoy/bengaluru-house-price-data)
- **Size**: ~13k records
- **Features**: `area_type`, `location`, `size`, `total_sqft`, `bath`, `price`
- **Target**: `price` (Lakhs INR)

## 🔧 Features Implemented
- BHK parsing from `size`
- `total_sqft` range averaging
- `price_per_sqft` calculation
- Location dimensionality reduction (>10 occurrences)
- Outlier removal:
  - Min 300 sqft/BHK
  - Price/sqft ±1 std per location
  - BHK-wise price consistency
  - Bath ≤ BHK + 2
- One-hot encoding for locations

## ⚙️ Setup
1. **Clone the repo**:
   ```bash
   git clone https://github.com/<your-username>/Data-Analysis-Programming-Lab-Python-.git
   cd Data-Analysis-Programming-Lab-Python-
   ```

2. **Virtual Environment**:
   ```bash
   python -m venv env
   # Windows:
   env\\Scripts\\activate
   # macOS/Linux:
   source env/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install jupyter pandas numpy matplotlib seaborn scikit-learn
   ```

4. **Download Dataset**:
   Place `bengaluru_house_prices.csv` in the root directory.

## 🚀 Usage
1. Launch Jupyter:
   ```bash
   jupyter notebook
   ```
2. Open `lab.ipynb` and run all cells sequentially.
3. Trained model saved as `banglore_home_prices_model.pickle`
4. Column mapping: `columns.json`

### 🧪 Predict Example
```python
import pickle
import json
import numpy as np

with open('banglore_home_prices_model.pickle', 'rb') as f:
    lr_model = pickle.load(f)
with open('columns.json', 'r') as f:
    cols = json.load(f)['data_columns']

# Example prediction
def predict_price(location, sqft, bath, bhk):
    loc_index = np.where(np.array(cols) == location.lower())[0][0]
    x = np.zeros(len(cols))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
    return lr_model.predict([x])[0]

print(predict_price('1st Phase JP Nagar', 1000, 2, 2))  # ~81.36 Lakhs
```

## 📊 Model Performance
| Model              | CV Score (5-fold) |
|--------------------|-------------------|
| Linear Regression  | **0.83** ↑        |
| Lasso              | 0.66              |
| Decision Tree      | 0.76              |

**Test Score**: 0.82

## 📚 Topics Covered
- **Data Cleaning**: NA handling, parsing, unit conversion
- **EDA**: Heatmaps, histograms, scatter plots
- **Feature Engineering**: New features, encoding
- **Outliers**: Statistical + Domain-based
- **ML Pipeline**: Train-test split, CV, GridSearchCV
- **Model Deployment**: Pickle export, JSON config
- **Visualization**: Seaborn, Matplotlib

## 🛠️ Improvements
- Add Ridge/Lasso hyperparameter tuning
- Clustering for location grouping
- Ensemble methods (XGBoost)
- Web app for predictions (Streamlit/Flask)

---

## 📌 License
MIT License - Free to use/adapt with attribution.

---

**Made by [Sanduni Fernando](https://github.com/sanduf01)**  
[Download CV](https://raw.githubusercontent.com/sanduf01/my-portfolio/main/portfolio/public/Sanduni%20Fernando%20CV.pdf) | [LinkedIn](https://linkedin.com/in/sanduf01)