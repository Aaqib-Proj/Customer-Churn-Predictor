import streamlit as st
import pandas as pd
import joblib

# Load the exported assets
model = joblib.load('churn_rf_model.pkl')
scaler = joblib.load('scaler.pkl')
model_columns = joblib.load('model_columns.pkl')

st.set_page_config(page_title="Churn Predictor", layout="centered")

st.title("📊 Customer Churn Predictor")
st.write("Enter the customer's details below to predict their likelihood of canceling the service.")

st.divider()

# --- 1. User Inputs ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Demographics")
    gender = st.selectbox("Gender", ["Male", "Female"])
    partner = st.selectbox("Has Partner?", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
    
    st.subheader("Services & Billing")
    phone_service = st.selectbox("Phone Service?", ["Yes", "No"])
    paperless = st.selectbox("Paperless Billing?", ["Yes", "No"])

with col2:
    st.subheader("Account Details")
    tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=12)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    
    st.subheader("Financials")
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=600.0)

st.divider()

# --- 2. Predict Button & Logic ---
if st.button("Predict Churn Risk", type="primary", use_container_width=True):
    
    # Bundle the inputs into a dictionary
    input_data = {
        'gender': gender, 'Partner': partner, 'Dependents': dependents,
        'PhoneService': phone_service, 'PaperlessBilling': paperless,
        'tenure': tenure, 'Contract': contract, 'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges
    }
    
    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Apply the exact same feature engineering from Step 2
    input_df['MonthlyCharges_per_Tenure'] = input_df['MonthlyCharges'] / (input_df['tenure'] + 1)
    
    def categorize_tenure(months):
        if months <= 12: return '0-1 Year'
        elif months <= 24: return '1-2 Years'
        elif months <= 48: return '2-4 Years'
        else: return '4+ Years'
    input_df['Tenure_Group'] = input_df['tenure'].apply(categorize_tenure)
    
    # Map binary columns to 1s and 0s
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        input_df[col] = input_df[col].apply(lambda x: 1 if x in ['Yes', 'Male'] else 0)
        
    # One-Hot Encode (dummy variables)
    input_df = pd.get_dummies(input_df)
    
    # Reindex to ensure it has the exact same columns as the training data, filling missing with 0
    input_df = input_df.reindex(columns=model_columns, fill_value=0)
    
    # Apply the scaler to the numerical columns
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'MonthlyCharges_per_Tenure']
    input_df[num_cols] = scaler.transform(input_df[num_cols])
    
    # Make the prediction
    probability = model.predict_proba(input_df)[0][1] * 100
    
    # --- 3. Display Results ---
    st.subheader("Prediction Results")
    
    if probability >= 70:
        st.error(f"🚨 High Risk: {probability:.1f}% chance of churning.")
    elif probability >= 40:
        st.warning(f"⚠️ Medium Risk: {probability:.1f}% chance of churning.")
    else:
        st.success(f"✅ Low Risk: {probability:.1f}% chance of churning. Customer is likely to stay.")