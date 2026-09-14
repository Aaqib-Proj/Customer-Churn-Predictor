import streamlit as st
import pandas as pd
import joblib
import warnings
from datetime import datetime

# Suppress version warnings for a clean user experience
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# 1. PAGE SETUP
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Risk Analyzer",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------------------------------------------------------
# 2. HIGH-CONTRAST, THEME-SAFE, INTUITIVE CSS
# -----------------------------------------------------------------------------
# Fully compatible with Dark Mode and Light Mode with high-contrast text and crisp headers.
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, sans-serif;
    }

    /* Main Section Headers - Vibrant Sky Blue for 100% Visibility */
    .card-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #38bdf8 !important;
        margin-bottom: 0.85rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        border-bottom: 1px solid #334155;
        padding-bottom: 0.5rem;
        letter-spacing: -0.01em;
    }

    /* Subheadings for Right Panel */
    .section-subtitle {
        font-size: 1rem;
        font-weight: 700;
        color: #38bdf8 !important;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }

    /* Risk Score Display Box */
    .risk-score-box {
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin-bottom: 1.25rem;
        border: 2px solid;
    }
    .risk-score-box.high {
        background: rgba(239, 68, 68, 0.15);
        border-color: #ef4444;
        color: #fca5a5;
    }
    .risk-score-box.medium {
        background: rgba(245, 158, 11, 0.15);
        border-color: #f59e0b;
        color: #fde68a;
    }
    .risk-score-box.low {
        background: rgba(16, 185, 129, 0.15);
        border-color: #10b981;
        color: #86efac;
    }

    .risk-score-label {
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.25rem;
        opacity: 0.95;
    }
    .risk-score-number {
        font-size: 3.5rem;
        font-weight: 800;
        line-height: 1.1;
        letter-spacing: -0.03em;
        color: #ffffff !important;
    }
    .risk-score-status {
        display: inline-block;
        font-size: 0.92rem;
        font-weight: 700;
        padding: 0.35rem 1.1rem;
        border-radius: 9999px;
        margin-top: 0.6rem;
        color: #ffffff !important;
        letter-spacing: 0.03em;
    }
    .badge-high { background: #dc2626; }
    .badge-medium { background: #d97706; }
    .badge-low { background: #059669; }

    .risk-score-summary {
        font-size: 0.9rem;
        margin-top: 0.85rem;
        line-height: 1.45;
        color: #f1f5f9 !important;
    }

    /* Financial Highlights Grid */
    .stat-pill {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 0.85rem 1rem;
        text-align: center;
    }
    .stat-pill-label {
        font-size: 0.75rem;
        color: #94a3b8 !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .stat-pill-value {
        font-size: 1.4rem;
        font-weight: 800;
        color: #ffffff !important;
        margin-top: 0.2rem;
    }

    /* Key Drivers List */
    .driver-item {
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
        padding: 0.75rem 0.9rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        font-size: 0.88rem;
        background: #1e293b;
        border: 1px solid #334155;
    }
    .driver-icon {
        font-size: 1rem;
        line-height: 1.3;
    }
    .driver-text {
        color: #e2e8f0 !important;
        line-height: 1.4;
    }
    .driver-text strong {
        color: #ffffff !important;
        font-weight: 700;
    }

    /* Retention Action Items */
    .action-item {
        padding: 0.85rem 1rem;
        border-radius: 8px;
        border: 1px solid #334155;
        background: #1e293b;
        margin-bottom: 0.65rem;
    }
    .action-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-weight: 700;
        font-size: 0.92rem;
        color: #ffffff !important;
    }
    .action-body {
        font-size: 0.82rem;
        color: #cbd5e1 !important;
        margin-top: 0.3rem;
        line-height: 1.4;
    }
    .action-impact {
        font-size: 0.75rem;
        font-weight: 700;
        color: #34d399 !important;
        background: rgba(52, 211, 153, 0.15);
        border: 1px solid #059669;
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
    }

    /* Streamlit labels high contrast */
    label[data-testid="stWidgetLabel"] p {
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        color: #f1f5f9 !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. LOAD ASSETS
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model_assets():
    rf = joblib.load('churn_rf_model.pkl')
    sc = joblib.load('scaler.pkl')
    cols = joblib.load('model_columns.pkl')
    return rf, sc, cols

try:
    model, scaler, model_columns = load_model_assets()
except Exception as e:
    st.error(f"Unable to load prediction model: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# 4. PREDICTION FUNCTION
# -----------------------------------------------------------------------------
def compute_churn_risk(data):
    df = pd.DataFrame([data])
    
    # Feature Engineering
    df['MonthlyCharges_per_Tenure'] = df['MonthlyCharges'] / (df['tenure'] + 1)
    
    def get_tenure_group(m):
        if m <= 12: return '0-1 Year'
        elif m <= 24: return '1-2 Years'
        elif m <= 48: return '2-4 Years'
        else: return '4+ Years'
    df['Tenure_Group'] = df['tenure'].apply(get_tenure_group)
    
    # Binary Encodings
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        df[col] = df[col].apply(lambda x: 1 if x in ['Yes', 'Male', 1] else 0)
    df['SeniorCitizen'] = int(df['SeniorCitizen'].iloc[0])
    
    # One-hot encoding and column matching
    df = pd.get_dummies(df)
    df = df.reindex(columns=model_columns, fill_value=0)
    
    # Standardize numerical features
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'MonthlyCharges_per_Tenure']
    df[num_cols] = scaler.transform(df[num_cols])
    
    # Prediction
    prob = model.predict_proba(df)[0][1] * 100.0
    return float(prob)

# -----------------------------------------------------------------------------
# 5. PRESETS MANAGEMENT (Simple & Intuitive)
# -----------------------------------------------------------------------------
# Initialize session state for clean defaults
if 'preset' not in st.session_state:
    st.session_state.preset = "Custom"
    st.session_state.tenure = 12
    st.session_state.contract = "Month-to-month"
    st.session_state.payment = "Electronic check"
    st.session_state.monthly = 75.0
    st.session_state.total = 900.0
    st.session_state.internet = "Fiber optic"
    st.session_state.tech_support = "No"
    st.session_state.security = "No"
    st.session_state.backup = "No"
    st.session_state.device = "No"
    st.session_state.streaming_tv = "No"
    st.session_state.streaming_movies = "No"
    st.session_state.phone = "Yes"
    st.session_state.multiple = "No"
    st.session_state.senior = "No"
    st.session_state.partner = "No"
    st.session_state.dependents = "No"
    st.session_state.paperless = "Yes"

def apply_preset(name):
    if name == "high_risk":
        st.session_state.tenure = 2
        st.session_state.contract = "Month-to-month"
        st.session_state.payment = "Electronic check"
        st.session_state.monthly = 95.0
        st.session_state.total = 190.0
        st.session_state.internet = "Fiber optic"
        st.session_state.tech_support = "No"
        st.session_state.security = "No"
        st.session_state.backup = "No"
        st.session_state.device = "No"
        st.session_state.streaming_tv = "Yes"
        st.session_state.streaming_movies = "Yes"
        st.session_state.phone = "Yes"
        st.session_state.multiple = "No"
        st.session_state.senior = "No"
        st.session_state.partner = "No"
        st.session_state.dependents = "No"
        st.session_state.paperless = "Yes"
    elif name == "loyal":
        st.session_state.tenure = 60
        st.session_state.contract = "Two year"
        st.session_state.payment = "Credit card (automatic)"
        st.session_state.monthly = 55.0
        st.session_state.total = 3300.0
        st.session_state.internet = "DSL"
        st.session_state.tech_support = "Yes"
        st.session_state.security = "Yes"
        st.session_state.backup = "Yes"
        st.session_state.device = "Yes"
        st.session_state.streaming_tv = "No"
        st.session_state.streaming_movies = "No"
        st.session_state.phone = "Yes"
        st.session_state.multiple = "Yes"
        st.session_state.senior = "No"
        st.session_state.partner = "Yes"
        st.session_state.dependents = "Yes"
        st.session_state.paperless = "No"
    elif name == "average":
        st.session_state.tenure = 18
        st.session_state.contract = "One year"
        st.session_state.payment = "Bank transfer (automatic)"
        st.session_state.monthly = 65.0
        st.session_state.total = 1170.0
        st.session_state.internet = "DSL"
        st.session_state.tech_support = "No"
        st.session_state.security = "Yes"
        st.session_state.backup = "No"
        st.session_state.device = "No"
        st.session_state.streaming_tv = "No"
        st.session_state.streaming_movies = "No"
        st.session_state.phone = "Yes"
        st.session_state.multiple = "No"
        st.session_state.senior = "No"
        st.session_state.partner = "Yes"
        st.session_state.dependents = "No"
        st.session_state.paperless = "Yes"

# -----------------------------------------------------------------------------
# 6. HEADER & QUICK PRESET BUTTONS
# -----------------------------------------------------------------------------
st.markdown("""
<div style="margin-bottom: 1.25rem;">
    <h1 style="font-size: 1.75rem; font-weight: 800; color: #ffffff !important; margin: 0; letter-spacing: -0.02em;">
        Customer Churn Risk Analyzer
    </h1>
    <p style="font-size: 0.95rem; color: #94a3b8 !important; margin: 0.35rem 0 0 0;">
        Assess whether a customer is at risk of canceling their subscription, understand key causes, and review retention recommendations.
    </p>
</div>
""", unsafe_allow_html=True)

# Preset Bar: 3 Simple One-Click Buttons
p_col1, p_col2, p_col3, p_col4 = st.columns([1.2, 1.2, 1.2, 2.4])
with p_col1:
    if st.button("🚨 Load High Risk Example", use_container_width=True, help="Load a customer archetype likely to cancel"):
        apply_preset("high_risk")
        st.rerun()
with p_col2:
    if st.button("✅ Load Safe Example", use_container_width=True, help="Load a long-term loyal customer"):
        apply_preset("loyal")
        st.rerun()
with p_col3:
    if st.button("⚖️ Load Moderate Example", use_container_width=True, help="Load a mid-tenure average account"):
        apply_preset("average")
        st.rerun()

st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 7. MAIN TWO-COLUMN LAYOUT: SIMPLE, DIRECT, INTUITIVE
# -----------------------------------------------------------------------------
col_inputs, col_results = st.columns([1.2, 1.0], gap="large")

# -----------------------------
# LEFT COLUMN: INPUT PARAMETERS
# -----------------------------
with col_inputs:
    # 1. Customer & Contract
    st.markdown("""
        <div class="card-title">
            <span>📋 Customer & Contract Details</span>
        </div>
    """, unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        tenure = st.slider(
            "Account Age (Months with company)",
            min_value=0, max_value=72, value=int(st.session_state.tenure),
            help="How long this customer has maintained an active account"
        )
        contract = st.selectbox(
            "Contract Term",
            ["Month-to-month", "One year", "Two year"],
            index=["Month-to-month", "One year", "Two year"].index(st.session_state.contract)
        )
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            index=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"].index(st.session_state.payment)
        )

    with c2:
        monthly_charges = st.number_input(
            "Monthly Bill Amount ($)",
            min_value=15.0, max_value=200.0, value=float(st.session_state.monthly), step=5.0
        )
        total_charges = st.number_input(
            "Total Invoiced to Date ($)",
            min_value=0.0, max_value=10000.0, value=float(st.session_state.total), step=50.0
        )
        paperless = st.selectbox(
            "Paperless Invoicing",
            ["Yes", "No"],
            index=0 if st.session_state.paperless == "Yes" else 1
        )

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

    # 2. Services Subscribed
    st.markdown("""
        <div class="card-title">
            <span>🌐 Subscribed Services</span>
        </div>
    """, unsafe_allow_html=True)
    
    s1, s2 = st.columns(2)
    with s1:
        internet_service = st.selectbox(
            "Internet Service",
            ["Fiber optic", "DSL", "No"],
            index=["Fiber optic", "DSL", "No"].index(st.session_state.internet)
        )
        
        has_internet = (internet_service != "No")
        sec_opts = ["No", "Yes"] if has_internet else ["No internet service"]
        
        tech_support = st.selectbox(
            "Tech Support Included",
            sec_opts,
            index=0 if st.session_state.tech_support == "No" or not has_internet else 1
        )
        online_security = st.selectbox(
            "Online Security",
            sec_opts,
            index=0 if st.session_state.security == "No" or not has_internet else 1
        )

    with s2:
        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"],
            index=0 if st.session_state.phone == "Yes" else 1
        )
        online_backup = st.selectbox(
            "Cloud Backup",
            sec_opts,
            index=0 if st.session_state.backup == "No" or not has_internet else 1
        )
        device_protection = st.selectbox(
            "Device Protection",
            sec_opts,
            index=0 if st.session_state.device == "No" or not has_internet else 1
        )

    # Optional / Advanced Demographics Expander (Keeps UI clean!)
    with st.expander("More Customer Details (Demographics & Streaming)"):
        d1, d2 = st.columns(2)
        with d1:
            senior = st.selectbox("Senior Citizen (65+)", ["No", "Yes"], index=0 if st.session_state.senior == "No" else 1)
            partner = st.selectbox("Has Partner", ["No", "Yes"], index=0 if st.session_state.partner == "No" else 1)
            dependents = st.selectbox("Has Dependents", ["No", "Yes"], index=0 if st.session_state.dependents == "No" else 1)
        with d2:
            gender = st.selectbox("Gender", ["Female", "Male"], index=0)
            multiple_lines = st.selectbox("Multiple Phone Lines", ["No", "Yes", "No phone service"] if phone_service == "Yes" else ["No phone service"])
            streaming_tv = st.selectbox("Streaming TV", sec_opts)
            streaming_movies = st.selectbox("Streaming Movies", sec_opts)

# -----------------------------
# BUILD DATA PAYLOAD & PREDICT
# -----------------------------
account_data = {
    'gender': gender,
    'SeniorCitizen': 1 if senior == "Yes" else 0,
    'Partner': partner,
    'Dependents': dependents,
    'tenure': tenure,
    'PhoneService': phone_service,
    'MultipleLines': multiple_lines,
    'InternetService': internet_service,
    'OnlineSecurity': online_security,
    'OnlineBackup': online_backup,
    'DeviceProtection': device_protection,
    'TechSupport': tech_support,
    'StreamingTV': streaming_tv,
    'StreamingMovies': streaming_movies,
    'Contract': contract,
    'PaperlessBilling': paperless,
    'PaymentMethod': payment_method,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges
}

risk_score = compute_churn_risk(account_data)
annual_revenue_at_risk = monthly_charges * 12.0

# Determine classification
if risk_score >= 60.0:
    risk_style = "high"
    risk_badge = "badge-high"
    risk_title = "HIGH CHURN RISK"
    summary_sentence = "This customer shows strong indicators of leaving. Immediate retention action is recommended."
elif risk_score >= 35.0:
    risk_style = "medium"
    risk_badge = "badge-medium"
    risk_title = "MODERATE RISK"
    summary_sentence = "This customer has several warning signs. Reviewing their plan could improve retention."
else:
    risk_style = "low"
    risk_badge = "badge-low"
    risk_title = "LOW RISK (LOYAL)"
    summary_sentence = "This customer is well-retained and satisfied based on their account history."

# -----------------------------
# RIGHT COLUMN: CLEAR RESULTS
# -----------------------------
with col_results:
    # 1. Main Risk Score Card
    st.markdown(f"""
    <div class="risk-score-box {risk_style}">
        <div class="risk-score-label">Predicted Churn Probability</div>
        <div class="risk-score-number">{risk_score:.0f}%</div>
        <div class="risk-score-status {risk_badge}">{risk_title}</div>
        <div class="risk-score-summary">
            {summary_sentence}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. Revenue Summary
    st.markdown(f"""
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; margin-bottom: 1.25rem;">
        <div class="stat-pill">
            <div class="stat-pill-label">Monthly Bill</div>
            <div class="stat-pill-value">${monthly_charges:.2f}</div>
        </div>
        <div class="stat-pill">
            <div class="stat-pill-label">Annual Value at Risk</div>
            <div class="stat-pill-value">${annual_revenue_at_risk:,.2f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 3. Why is this customer at risk? (Plain English Drivers)
    st.markdown("""
        <div class="section-subtitle">
            🔍 Why this score?
        </div>
    """, unsafe_allow_html=True)

    reasons = []
    if contract == "Month-to-month":
        reasons.append(("❌", "<strong>Month-to-month contract</strong> increases likelihood of leaving at any time."))
    else:
        reasons.append(("✅", f"<strong>Committed {contract.lower()} contract</strong> provides strong retention stability."))

    if tenure <= 6:
        reasons.append(("❌", f"<strong>New account ({tenure} months)</strong>: First 6 months carry the highest cancellation rate."))
    elif tenure >= 24:
        reasons.append(("✅", f"<strong>Long-term tenure ({tenure} months)</strong>: Established account with proven loyalty."))

    if payment_method == "Electronic check":
        reasons.append(("❌", "<strong>Manual payment via electronic check</strong> has much higher churn than auto-pay."))
    elif "automatic" in payment_method:
        reasons.append(("✅", "<strong>Automatic payment enabled</strong> reduces friction and missed payments."))

    if internet_service == "Fiber optic" and tech_support == "No":
        reasons.append(("⚠️", "<strong>High-speed Fiber without Tech Support</strong>: Vulnerable to unresolved technical issues."))

    for icon, text in reasons[:4]:
        st.markdown(f"""
        <div class="driver-item">
            <span class="driver-icon">{icon}</span>
            <span class="driver-text">{text}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

    # 4. Recommended Actions
    st.markdown("""
        <div class="section-subtitle">
            💡 Recommended Actions to Retain Customer
        </div>
    """, unsafe_allow_html=True)

    # Calculate concrete potential risk reductions
    actions = []
    if contract == "Month-to-month":
        sim_data = dict(account_data)
        sim_data['Contract'] = 'One year'
        new_prob = compute_churn_risk(sim_data)
        reduction = max(1.0, risk_score - new_prob)
        actions.append((
            "Offer 1-Year Contract with Loyalty Perk",
            "Transitioning from month-to-month to an annual agreement.",
            f"-{reduction:.0f}% risk"
        ))

    if payment_method == "Electronic check":
        sim_data = dict(account_data)
        sim_data['PaymentMethod'] = 'Bank transfer (automatic)'
        new_prob = compute_churn_risk(sim_data)
        reduction = max(1.0, risk_score - new_prob)
        actions.append((
            "Encourage Auto-Pay Enrollment",
            "Offer a one-time $10 credit to set up automatic credit card or ACH payment.",
            f"-{reduction:.0f}% risk"
        ))

    if tech_support == "No" and internet_service != "No":
        sim_data = dict(account_data)
        sim_data['TechSupport'] = 'Yes'
        sim_data['OnlineSecurity'] = 'Yes'
        new_prob = compute_churn_risk(sim_data)
        reduction = max(1.0, risk_score - new_prob)
        actions.append((
            "Add Complimentary Tech Support",
            "Provide 6 months of free priority support and online security protection.",
            f"-{reduction:.0f}% risk"
        ))

    if not actions:
        actions.append((
            "Maintain Standard Relationship Check-in",
            "Customer is currently stable. Send routine satisfaction surveys.",
            "Healthy"
        ))

    for title, desc, benefit in actions:
        st.markdown(f"""
        <div class="action-item">
            <div class="action-header">
                <span>{title}</span>
                <span class="action-impact">{benefit}</span>
            </div>
            <div class="action-body">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    # 5. Quick Export Button
    export_df = pd.DataFrame([{
        "Customer_Tenure_Months": tenure,
        "Contract": contract,
        "Monthly_Charges": monthly_charges,
        "Annual_Value_At_Risk": annual_revenue_at_risk,
        "Churn_Probability_Pct": round(risk_score, 1),
        "Risk_Category": risk_title,
        "Export_Date": datetime.now().strftime("%Y-%m-%d %H:%M")
    }])
    csv_str = export_df.to_csv(index=False)
    
    st.download_button(
        label="📄 Download Assessment Summary (CSV)",
        data=csv_str,
        file_name="churn_risk_summary.csv",
        mime="text/csv",
        use_container_width=True
    )