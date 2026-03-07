import streamlit as st
import numpy as np
import pickle

# Function required for loading the pickle model
def prior_correction(p_model, real_prior=0.0668, train_prior=0.5):
    numerator = p_model * (real_prior / train_prior)
    denominator = numerator + ((1 - p_model) * ((1 - real_prior) / (1 - train_prior)))
    return numerator / denominator


st.title("Credit Risk Scorecard — IFRS 9 Demo")

st.write(
    "Enter borrower details to predict Probability of Default (PD), Credit Score, and Loan Decision."
)

# Load model and scaler
with open("pd_model_final.pkl", "rb") as f:
    model_data = pickle.load(f)

model = model_data["model"]
scaler = model_data["scaler"]


# -------------------------------
# User Inputs
# -------------------------------

age = st.slider("Age", 18, 80, 35)

credit_utilization = st.slider(
    "Revolving Credit Utilization",
    0.0, 1.0, 0.30
)

late_30 = st.number_input(
    "30–59 Days Late Count",
    0, 10, 0
)

late_60 = st.number_input(
    "60–89 Days Late Count",
    0, 10, 0
)

late_90 = st.number_input(
    "90+ Days Late Count",
    0, 10, 0
)

monthly_income = st.number_input(
    "Monthly Income ($)",
    1000, 20000, 5000
)

debt_ratio = st.slider(
    "Debt Ratio",
    0.0, 5.0, 0.5
)

open_loans = st.number_input(
    "Open Credit Lines",
    0, 20, 5
)

real_estate_loans = st.number_input(
    "Real Estate Loans",
    0, 10, 1
)

dependents = st.number_input(
    "Number of Dependents",
    0, 10, 1
)


# -------------------------------
# Prediction
# -------------------------------

if st.button("Predict Credit Risk"):

    # Feature engineering
    total_delinquency = late_30 + (2 * late_60) + (3 * late_90)

    input_data = np.array([[
        credit_utilization,
        age,
        late_30,
        debt_ratio,
        monthly_income,
        open_loans,
        late_90,
        real_estate_loans,
        late_60,
        dependents,
        total_delinquency
    ]])

    # Scale features
    input_scaled = scaler.transform(input_data)

    # Predict PD
    pd_model = model.predict_proba(input_scaled)[0][1]

    # Apply prior probability correction
    pd_prob = prior_correction(pd_model)

    # Convert PD to credit score
    score = int(600 + (1 - pd_prob) * 200)

    # Risk decision
    if score >= 700:
        decision = "Approved"
        risk = "Low Risk"
    elif score >= 580:
        decision = "Conditional Approval"
        risk = "Medium Risk"
    else:
        decision = "Declined"
        risk = "High Risk"

    # -------------------------------
    # Display Results
    # -------------------------------

    st.subheader("Prediction Result")

    st.write(f"**Probability of Default:** {pd_prob:.2%}")
    st.write(f"**Credit Score:** {score}")
    st.write(f"**Risk Category:** {risk}")
    st.write(f"**Loan Decision:** {decision}")
