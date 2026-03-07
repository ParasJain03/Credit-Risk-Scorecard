![Python](https://img.shields.io/badge/Python-3.10-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![Finance](https://img.shields.io/badge/Domain-Credit%20Risk-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
# Credit Risk Scorecard — IFRS 9 PD Model

**Author:** Paras Jain  
B.Tech CSE-AI | KIET Group of Institutions | 2027
## Quick Demo

Example borrower prediction from the model:

| Feature | Value |
|-------|------|
| Age | 45 |
| Credit Utilization | 15% |
| Total Delinquencies | 0 |
| Monthly Income | $8,000 |

### Model Output

- Predicted PD: **3.7%**
- Credit Score: **609**
- Risk Category: **Medium Risk**
- Decision: **Conditional Approval**

This project builds a **bank-style credit risk model** that predicts borrower default probability (PD), converts it into a **credit score**, and estimates **Expected Credit Loss (ECL)** using the **IFRS-9 framework**.

The goal of this project is to simulate how banks evaluate loan applicants and manage portfolio credit risk.

---

# Project Overview

Banks must estimate the likelihood that a borrower will default on a loan.

Financial institutions calculate three core components:

- **PD — Probability of Default**
- **LGD — Loss Given Default**
- **EAD — Exposure at Default**

Expected Credit Loss is calculated as:


```
ECL = PD × LGD × EAD
```


This project implements the **complete credit risk modeling pipeline used in banking analytics**.

---

# Dataset

Dataset used:

**Give Me Some Credit (Kaggle)**  
https://www.kaggle.com/datasets/brycecf/give-me-some-credit

Dataset details:

- 104,619 loan applicants
- 11 financial variables
- Default rate: **6.68%**

Target variable:


SeriousDlqin2yrs
0 = Good borrower
1 = Default


---

# Project Pipeline

The model follows the standard **credit risk modeling workflow used in banks**.

1. Business Understanding  
2. Exploratory Data Analysis  
3. Data Cleaning & Outlier Handling  
4. Feature Engineering  
5. WoE & Information Value Analysis  
6. PD Model Training (Logistic Regression)  
7. Model Validation (AUC, Gini, KS)  
8. Credit Scorecard Generation  
9. Probability Calibration  
10. IFRS-9 Expected Credit Loss Estimation  

---

# Feature Engineering

Key engineered feature:

**Total_Delinquency**


Total_Delinquency =
(30-day late × 1)

(60-day late × 2)

(90-day late × 3)


Information Value:


IV ≈ 1.27


This became the **strongest predictor of default risk** in the model.

Other engineered features:

- Income per dependent
- Debt-to-income ratio
- Credit burden

---

# Model Development

Algorithm used:

**Logistic Regression**

Reason:

- Standard model used in credit scorecards
- Highly interpretable
- Regulatory-friendly for banking models

Handling class imbalance:

Dataset distribution:

- Good borrowers → **93%**
- Defaults → **7%**

Solution used:

**SMOTE (Synthetic Minority Oversampling)**

This balanced the training dataset to a **1:1 ratio**.

---

# Model Performance

| Metric | Result |
|------|------|
| AUC-ROC | **0.8591** |
| Gini Coefficient | **0.7181** |
| KS Statistic | **0.5590** |
| Recall (defaults detected) | **73%** |

These metrics exceed typical **banking model benchmarks**.

---

# Model Explainability

Model predictions were explained using **SHAP (SHapley Additive Explanations)**.

Top risk drivers:

1. Total Delinquency  
2. Credit Utilization  
3. Age  
4. Number of Credit Lines  

Explainability ensures transparency in credit decisions.

---

# Probability Calibration

SMOTE introduces probability bias.

Initial model prediction:


Average PD ≈ 31.9%


Actual dataset default rate:


6.68%


Solution applied:

**Prior Probability Correction**

After calibration:


Predicted PD ≈ 7.27%


The model probabilities now align closely with real-world default rates.

---

# Credit Scorecard

Predicted probabilities were converted into a **credit score** using the **Points to Double Odds (PDO) method**.

Parameters used:

- Base Score = 600  
- PDO = 20  
- Score Range = **300 – 900**

Risk bands:

- Score ≥ 700 → Low Risk (Approved)  
- Score 580–699 → Medium Risk (Conditional Approval)  
- Score < 580 → High Risk (Declined)

---

# IFRS-9 Expected Credit Loss (ECL)

ECL Formula:


ECL = PD × LGD × EAD


Portfolio results:

- Total Loans → **12,588**
- Total Exposure (EAD) → **$934M**
- 12-Month ECL → **$23.8M**
- Lifetime ECL → **$87.3M**

IFRS-9 staging:

- Stage 1 (Low Risk) → 72.8% loans  
- Stage 2 (Medium Risk) → 19.7% loans  
- Stage 3 (High Risk) → 7.5% loans  

Although Stage-3 loans are only **7.5% of the portfolio**, they generate **73% of expected losses**.

---

# Visualizations

## Exploratory Data Analysis
![EDA](fig1_eda.png)

## WoE / Information Value Analysis
![WoE](fig2_woe.png)

## Model Validation
![Model](fig3_model_performance.png)

## SHAP Explainability
![SHAP](fig4_shap.png)

## Probability Calibration
![Calibration](fig5_calibration.png)

## IFRS-9 Expected Credit Loss
![ECL](fig6_ecl.png)

---

# Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Imbalanced-learn  
- SHAP  
- Matplotlib  
- Seaborn  
- Google Colab  

---



# Repository Structure

```
Credit-Risk-Scorecard

app.py
requirements.txt
pd_model_final.pkl

Credit_Risk_Scorecard_ParasJain.ipynb

fig1_eda.png
fig2_woe.png
fig3_model_performance.png
fig4_shap.png
fig5_calibration.png
fig6_ecl.png

README.md
```

---

# Key Takeaways

- Feature engineering significantly improves predictive power  
- Class imbalance strongly affects probability estimates  
- Calibration is necessary after SMOTE  
- Credit scorecards convert ML models into business decisions  
- IFRS-9 provides forward-looking risk provisions  

---

# Author

**Paras Jain**

B.Tech CSE-AI  
KIET Group of Institutions  

Email: parasjainparas310@gmail.com

---

# Future Improvements

- Add macroeconomic stress scenarios  
- Deploy model using a Streamlit dashboard  
- Integrate into a full financial risk analytics portfolio
