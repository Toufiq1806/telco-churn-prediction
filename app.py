import streamlit as st
import pandas as pd
import joblib
loaded_model = joblib.load("churn_model.pkl")
def predict_churn(customer_data):
    probability = loaded_model.predict_proba(customer_data)[0, 1]

    if probability >= 0.35:
        prediction = "Likely to Churn"
    else:
        prediction = "Likely to Stay"

    return prediction,probability


st.title("Customer Churn Prediction")
st.write("Enter customer information below.")
tenure = st.number_input("Tenure (months)", min_value=0, value=12)
monthly_charges = st.number_input("Monthly Charges",min_value=0.0,value=50.0)
total_charges = st.number_input("Total Charges",min_value=0.0,value=600.0)
gender = st.selectbox("Gender",["Female", "Male"])
partner = st.selectbox("Partner",["Yes", "No"])
dependents = st.selectbox("Dependents",["Yes", "No"])
internet_service = st.selectbox("Internet Service",["DSL", "Fiber optic", "No"])
contract = st.selectbox("Contract",["Month-to-month", "One year", "Two year"])
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.selectbox("Multiple Lines",["Yes", "No", "No phone service"])
online_security = st.selectbox("Online Security",["Yes", "No", "No internet service"])
online_backup = st.selectbox("Online Backup",["Yes", "No", "No internet service"])
device_protection = st.selectbox("Device Protection",["Yes", "No", "No internet service"])
tech_support = st.selectbox("Tech Support",["Yes", "No", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies",["Yes", "No", "No internet service"])
streaming_tv = st.selectbox("Streaming TV",["Yes", "No", "No internet service"])
paperless_billing = st.selectbox("Paperless Billing",["Yes", "No"])
payment_method = st.selectbox("Payment Method",["Bank transfer (automatic)","Credit card (automatic)","Electronic check","Mailed check"])
customer_data = pd.DataFrame({
    "gender": [gender],
    "Partner": [partner],
    "Dependents": [dependents],
    "PhoneService": [phone_service],
    "MultipleLines": [multiple_lines],
    "InternetService": [internet_service],
    "OnlineSecurity": [online_security],
    "OnlineBackup": [online_backup],
    "DeviceProtection": [device_protection],
    "TechSupport": [tech_support],
    "StreamingMovies": [streaming_movies],
    "StreamingTV": [streaming_tv],
    "Contract": [contract],
    "PaperlessBilling": [paperless_billing],
    "PaymentMethod": [payment_method],
    "TotalCharges": [total_charges],
    "MonthlyCharges": [monthly_charges],
    "tenure": [tenure]
})
if st.button("Predict Churn"):
    prediction, probability =predict_churn(customer_data)
    st.write("Prediction:", prediction)
    st.write("Churn Probability:", f"{probability:.2%}")
