import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load calibrated model and feature columns
model = joblib.load('calibrated_random_forest_model_v6.pkl')
feature_cols = pd.read_csv('features_stronger_v6.csv').columns

# Streamlit app
st.title("Insurance Claim Risk Predictor (Calibrated Model V6)")
st.write("Enter customer details to predict claim likelihood")
st.info("Note: Dataset has ~15–20% claims, realistic for insurance data.")

# Input fields
age = st.slider("Age", 18, 80, 30)
location = st.selectbox("Location", ["Nairobi", "Kisumu", "Mombasa", "Eldoret", "Nakuru"])
policy_type = st.selectbox("Policy Type", ["Auto", "Home", "Health"])
gender = st.selectbox("Gender", ["Male", "Female"])
income_bracket = st.selectbox("Income Bracket", ["Low", "Medium", "High"])
policy_duration = st.slider("Policy Duration (Months)", 6, 60, 12)
coverage_amount = st.number_input("Coverage Amount (KES)", 50000, 5000000, 100000)
premium_paid = st.number_input("Premium Paid (KES)", 5000, 100000, 10000)
vehicle_age = st.slider("Vehicle Age (Years)", 0, 15, 5)
late_payments = st.slider("Late Payments", 0, 5, 0)
previous_claims = st.slider("Previous Claims", 0, 5, 0)

# Feature engineering
input_data = pd.DataFrame({
    'age': [age],
    'location': [{'Nairobi': 0, 'Kisumu': 1, 'Mombasa': 2, 'Eldoret': 3, 'Nakuru': 4}[location]],
    'policy_type': [{'Auto': 0, 'Home': 1, 'Health': 2}[policy_type]],
    'gender': [{'Male': 0, 'Female': 1}[gender]],
    'income_bracket': [{'Low': 0, 'Medium': 1, 'High': 2}[income_bracket]],
    'policy_duration_months': [policy_duration],
    'coverage_amount': [coverage_amount],
    'premium_paid': [premium_paid],
    'vehicle_age': [vehicle_age],
    'late_payments': [late_payments],
    'previous_claims': [previous_claims],
    'claim_frequency': [previous_claims / (policy_duration / 12 + 0.01)],
    'premium_to_coverage': [premium_paid / (coverage_amount + 0.01)],
    'has_late_payments': [1 if late_payments > 0 else 0],
    'high_risk_location': [1 if location in ['Kisumu', 'Mombasa'] else 0],
    'young_and_risky': [1 if (age < 30 and location in ['Kisumu', 'Mombasa']) else 0],
    'high_claim_risk': [1 if (previous_claims > 4 and late_payments > 4) else 0],
    'underinsured': [1 if (premium_paid / (coverage_amount + 0.01) < 0.01) else 0],
    'age_risk': [1 if age < 30 else 0],
    'claim_severity': [previous_claims * 100000 / (policy_duration + 0.01)],
    'claims_and_location': [previous_claims * (1 if location in ['Kisumu', 'Mombasa'] else 0)]
})

# Predict
if st.button("Predict Risk"):
    risk_score = model.predict_proba(input_data)[:, 1][0]
    st.write(f"**Claim Likelihood**: {risk_score:.2%}")
    if risk_score > 0.4:
        st.error("High Risk: Consider higher premium or additional scrutiny.")
    elif risk_score > 0.2:
        st.warning("Moderate Risk: Monitor payment behavior.")
    else:
        st.success("Low Risk: Standard underwriting applies.")

    # Feature importance plot
    original_model = joblib.load('random_forest_model_stronger_v6.pkl')
    feature_importance = pd.Series(original_model.feature_importances_, index=feature_cols).sort_values()
    plt.figure(figsize=(8, 6))
    sns.barplot(x=feature_importance.values, y=feature_importance.index)
    plt.title('Feature Importance')
    st.pyplot(plt)
