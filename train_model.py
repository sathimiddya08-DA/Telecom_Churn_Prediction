import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# 1. Load data
df = pd.read_csv("telecom_churn.csv")

# 2. Clean TotalCharges
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# 3. Handle missing values
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# 4. Drop unnecessary column
df = df.drop("customerID", axis=1)

# 5. Features and target
X = df.drop("Churn", axis=1)
y = df["Churn"].map({"No": 0, "Yes": 1})

# 6. Categorical and numerical features
categorical = X.select_dtypes(include=["object", "string"]).columns.tolist()
numerical = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

# 7. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 8. Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ]
)

# 9. Final pipeline with Logistic Regression
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000, random_state=42)),
])

# 10. Train the model
pipeline.fit(X_train, y_train)

# 11. Model Evaluation
y_pred = pipeline.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 12. Save the trained pipeline
joblib.dump(pipeline, "churn_model_pipeline.pkl")

print("\nModel trained successfully!")
print("Model saved as churn_model_pipeline.pkl")
