# Telecom Customer Churn Prediction

## Project Overview

A simple Machine Learning project that predicts whether a telecom customer is likely to churn using Logistic Regression. Customer churn occurs when customers stop doing business with a company. Predicting churn helps businesses take proactive retention steps.

## Dataset

The project uses the standard Telco Customer Churn dataset (`telecom_churn.csv`). Each row represents an individual customer, and the dataset includes:
- **Demographics:** Gender, SeniorCitizen, Partner, Dependents
- **Subscribed Services:** PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
- **Account Details:** Contract type, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges, Tenure
- **Target Variable:** `Churn` (Value is either `Yes` or `No`)

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Logistic Regression
- Joblib
- Streamlit

## Machine Learning Workflow

```text
Dataset (telecom_churn.csv)
       ↓
Data Cleaning (Convert TotalCharges, impute median, drop customerID)
       ↓
Feature Preprocessing (StandardScaler for numerical, OneHotEncoder for categorical)
       ↓
Train/Test Split (80% train, 20% test stratified)
       ↓
Logistic Regression (Pipeline training)
       ↓
Evaluation (Accuracy, Classification Report, Confusion Matrix)
       ↓
Save Model (churn_model_pipeline.pkl via Joblib)
       ↓
Streamlit Web App (Interactive real-time prediction)
```

## Project Structure

```text
Telecom_Churn_Prediction/
│
├── telecom_churn.csv
├── train_model.py
├── app.py
├── churn_model_pipeline.pkl
├── requirements.txt
└── README.md
```

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python train_model.py
```

This trains the Logistic Regression pipeline and saves it as `churn_model_pipeline.pkl`.

### 3. Run the Streamlit Web Application

```bash
streamlit run app.py
```

*(Or run `python -m streamlit run app.py` if streamlit is not in your system PATH)*

## Model

- **Algorithm:** Logistic Regression
- **Preprocessing:** Scikit-Learn `ColumnTransformer` (StandardScaler for continuous variables, OneHotEncoder for categorical variables)
- **Serialization:** Full end-to-end Pipeline saved with `joblib` so that user inputs are automatically scaled and encoded without manual preprocessing code.

## Evaluation

Model evaluation on the 20% unseen test set (1,409 customers):

- **Accuracy:** `80.55%` (0.8055)

### Classification Report

```text
              precision    recall  f1-score   support

           0       0.85      0.89      0.87      1035
           1       0.66      0.56      0.60       374

    accuracy                           0.81      1409
   macro avg       0.75      0.73      0.74      1409
weighted avg       0.80      0.81      0.80      1409
```

### Confusion Matrix

```text
[[926  109]
 [165  209]]
```

- **True Negatives (Correctly predicted non-churn):** 926
- **False Positives (Predicted churn, actually stayed):** 109
- **False Negatives (Predicted stay, actually churned):** 165
- **True Positives (Correctly predicted churn):** 209
