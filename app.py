import streamlit as st
import numpy as np
import pickle
import pandas as pd
import plotly.graph_objects as go


# ------------------------------
# Prior probability correction
# ------------------------------

def prior_correction(p_model, real_prior=0.0668, train_prior=0.5):
    numerator = p_model * (real_prior / train_prior)
    denominator = numerator + ((1 - p_model) * ((1 - real_prior) / (1 - train_prior)))
    return numerator / denominator


# ------------------------------
# Page Config
# ------------------------------

st.set_page_config(page_title="Credit Risk Scorecard", layout="centered")

st.title("🏦 Credit Risk Scorecard — IFRS 9 Demo")

st.write(
"""
Enter borrower details to estimate:

• Probability of Default (PD)  
• Credit Score  
• Loan Decision
"""
)


# ------------------------------
# Load Model
# ------------------------------

@st.cache_resource
def load_model():

    with open("pd_model_final.pkl", "rb") as f:
        model_data = pickle.load(f)

    model = model_data["model"]
    scaler = model_data["scaler"]

    return model, scaler


model, scaler = load_model()


# ------------------------------
# Inputs
# ------------------------------

age = st.slider("Age", 18, 80, 35)

credit_utilization = st.slider(
    "Revolving Credit Utilization",
    0.0, 1.0, 0.3
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

dependents = st.number_input(
    "Number of Dependents",
    0, 10, 1
)


# ------------------------------
# Prediction
# ------------------------------

if st.button("Predict Credit Risk"):

    # Feature engineering

    total_delinquency = late_30 + (2 * late_60) + (3 * late_90)

    income_per_dependent = monthly_income / (dependents + 1)

    dti_ratio = debt_ratio

    credit_burden = credit_utilization * debt_ratio


    # Feature order must match training

    input_data = np.array([[

        total_delinquency,
        credit_utilization,
        late_30,
        age,
        income_per_dependent,
        monthly_income,
        debt_ratio,
        dti_ratio,
        open_loans,
        credit_burden,
        dependents

    ]])

    # Scale features

    input_scaled = scaler.transform(input_data)


    # Predict PD

    pd_model = model.predict_proba(input_scaled)[0][1]

    pd_prob = prior_correction(pd_model)


    # ------------------------------
    # Credit Score
    # ------------------------------

    score = int(300 + (1 - pd_prob) * 550)


    # ------------------------------
    # Risk Decision
    # ------------------------------

    if score >= 720:
        decision = "Approved"
        risk = "Low Risk"

    elif score >= 600:
        decision = "Conditional Approval"
        risk = "Medium Risk"

    else:
        decision = "Declined"
        risk = "High Risk"


    # ------------------------------
    # Results
    # ------------------------------

    st.markdown("---")

    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    col1.metric("Probability of Default", f"{pd_prob:.2%}")
    col2.metric("Credit Score", score)

    st.write(f"Risk Category: **{risk}**")

    st.write(f"Decision: **{decision}**")


    # ------------------------------
    # Risk Gauge
    # ------------------------------

    st.markdown("### Risk Gauge")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pd_prob * 100,
        title={'text': "Probability of Default (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "red"},
            'steps': [
                {'range': [0, 10], 'color': "green"},
                {'range': [10, 25], 'color': "yellow"},
                {'range': [25, 100], 'color': "red"}
            ]
        }
    ))

    st.plotly_chart(fig)


    # ------------------------------
    # Feature Importance
    # ------------------------------

    st.subheader("Top Risk Factors")

    importance = model.coef_[0]

    feature_names = [
        "Total_Delinquency",
        "Credit_Utilization",
        "Late_30",
        "Age",
        "Income_Per_Dependent",
        "Monthly_Income",
        "Debt_Ratio",
        "DTI_Ratio",
        "Open_Loans",
        "Credit_Burden",
        "Dependents"
    ]

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Impact": importance
    }).sort_values("Impact", ascending=False)

    st.bar_chart(importance_df.set_index("Feature"))


    # ------------------------------
    # Portfolio Risk Simulator
    # ------------------------------

    st.subheader("Portfolio Risk Simulation")

    loan_amount = st.slider("Loan Amount ($)", 1000, 100000, 10000)

    lgd = 0.45

    ead = loan_amount

    expected_loss = pd_prob * lgd * ead

    st.write(f"Expected Credit Loss: ${expected_loss:,.2f}")


    # ------------------------------
    # Model Confidence
    # ------------------------------

    confidence = (1 - pd_prob) * 100

    st.metric("Model Confidence", f"{confidence:.1f}%")


# ------------------------------
# Footer
# ------------------------------

st.markdown("---")

st.caption("Model: Logistic Regression | AUC ≈ 0.85 | KS ≈ 0.54")
