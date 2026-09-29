import streamlit as st
import pandas as pd
import joblib

# Set page configuration
st.set_page_config(
    page_title="Telecom Customer Churn Prediction",
    page_icon="📞",
    layout="wide"
)

# App Title & Description
st.title("Telecom Customer Churn Prediction")
st.write(
    "Enter customer demographic and account details below to predict "
    "whether the customer is likely to churn or stay."
)

# Load the trained model pipeline
@st.cache_resource
def load_pipeline():
    return joblib.load("churn_model_pipeline.pkl")

pipeline = load_pipeline()

# Collect User Inputs in organized columns
st.header("Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Demographics")
    gender = st.selectbox("Gender", ["Female", "Male"])
    SeniorCitizen = st.selectbox("Senior Citizen (0 = No, 1 = Yes)", [0, 1])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["No", "Yes"])

with col2:
    st.subheader("Services Subscribed")
    PhoneService = st.selectbox("Phone Service", ["No", "Yes"])
    MultipleLines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    DeviceProtection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    TechSupport = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    StreamingTV = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

with col3:
    st.subheader("Account & Billing")
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12, step=1)
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
    PaymentMethod = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )
    MonthlyCharges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0, step=1.0)
    TotalCharges = st.number_input("Total Charges ($)", min_value=0.0, value=600.0, step=10.0)

# Prediction Button
st.markdown("---")
if st.button("Predict Churn"):
    # Create input DataFrame matching training feature names
    input_data = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }])

    # Predict using the saved pipeline
    prediction = pipeline.predict(input_data)[0]
    churn_probability = pipeline.predict_proba(input_data)[0][1]

    # Display results
    st.subheader("Prediction Result")
    if prediction == 1:
        st.error("Prediction: Customer is likely to churn")
    else:
        st.success("Prediction: Customer is likely to stay")

    st.write(f"**Churn Probability:** {churn_probability * 100:.2f}%")
