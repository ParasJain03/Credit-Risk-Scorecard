import streamlit as st
import numpy as np
import pickle

st.title("Credit Risk Scorecard — IFRS 9 Demo")

st.write("Enter borrower details to predict Probability of Default and Credit Score.")

# Load trained model
with open("pd_model_final.pkl", "rb") as f:
    model_data = pickle.load(f)

model = model_data["model"]
scaler = model_data["scaler"]

# User Inputs
age = st.slider("Age", 18, 80, 35)
credit_utilization = st.slider("Credit Utilization", 0.0, 1.0, 0.3)
delinquencies = st.number_input("Total Delinquencies", 0, 10, 0)
monthly_income = st.number_input("Monthly Income ($)", 1000, 20000, 5000)

# Predict button
if st.button("Predict Credit Risk"):

    input_data = np.array([[age, credit_utilization, delinquencies, monthly_income]])
    input_scaled = scaler.transform(input_data)

    pd_prob = model.predict_proba(input_scaled)[0][1]

    score = int(600 + (1 - pd_prob) * 200)

    if score >= 700:
        decision = "Approved"
        risk = "Low Risk"
    elif score >= 580:
        decision = "Conditional Approval"
        risk = "Medium Risk"
    else:
        decision = "Declined"
        risk = "High Risk"

    st.subheader("Prediction Result")

    st.write(f"Probability of Default: {pd_prob:.2%}")
    st.write(f"Credit Score: {score}")
    st.write(f"Risk Category: {risk}")
    st.write(f"Decision: {decision}")
