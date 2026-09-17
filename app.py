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
    page_title="StreamPulse™ | OTT Subscriber Churn Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------------------------------------------------------
# 2. EXECUTIVE LIGHT DESIGN SYSTEM & ENHANCED CSS
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

    /* Global Typography & Canvas Background */
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, sans-serif !important;
        background-color: #f6f8fb !important;
        color: #1e293b !important;
    }

    /* Top Title and Subtitle Styling */
    .app-header {
        margin-bottom: 1.25rem;
        padding-top: 0.25rem;
    }
    .app-subtitle {
        font-size: 0.95rem;
        color: #475569 !important;
        margin-top: 0.35rem;
        line-height: 1.5;
        font-weight: 500;
    }

    /* Section Card Containers (matching reference screenshot) */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff !important;
        border: 1px solid #eaeff5 !important;
        border-radius: 16px !important;
        padding: 1.25rem 1.4rem !important;
        box-shadow: 0 4px 20px -2px rgba(50, 50, 93, 0.04), 0 2px 6px -1px rgba(0, 0, 0, 0.03) !important;
        margin-bottom: 1.15rem !important;
        transition: box-shadow 0.2s ease, border-color 0.2s ease;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: #dbe3ee !important;
        box-shadow: 0 8px 25px -3px rgba(50, 50, 93, 0.07), 0 3px 8px -2px rgba(0, 0, 0, 0.04) !important;
    }

    /* Card Title Headers */
    .card-title {
        font-size: 1rem;
        font-weight: 700;
        color: #1e293b !important;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        letter-spacing: -0.01em;
    }

    /* Streamlit Input & Widget Labels */
    label[data-testid="stWidgetLabel"] p {
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        color: #334155 !important;
        letter-spacing: -0.01em;
    }

    /* Clean Input Box Styling */
    div[data-baseweb="input"], div[data-baseweb="select"] {
        border-radius: 8px !important;
    }

    /* Quick Archetype Preset Buttons (Top Bar) */
    div[data-testid="stHorizontalBlock"] button[data-testid="baseButton-secondary"] {
        background-color: #ffffff !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        font-size: 0.86rem !important;
        padding: 0.55rem 1rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03) !important;
    }
    div[data-testid="stHorizontalBlock"] button[data-testid="baseButton-secondary"]:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06) !important;
    }

    /* Preset 1: High Risk (Red Accent) */
    .btn-preset-dropout button {
        border: 1.5px solid #fca5a5 !important;
        color: #991b1b !important;
        background: #fffafa !important;
    }
    .btn-preset-dropout button:hover {
        border-color: #ef4444 !important;
        background: #fee2e2 !important;
    }

    /* Preset 2: Safe Loyal Viewer (Green Accent) */
    .btn-preset-loyal button {
        border: 1.5px solid #86efac !important;
        color: #166534 !important;
        background: #f8fdf9 !important;
    }
    .btn-preset-loyal button:hover {
        border-color: #10b981 !important;
        background: #dcfce7 !important;
    }

    /* Preset 3: Casual Viewer (Amber/Gold Accent) */
    .btn-preset-casual button {
        border: 1.5px solid #fde68a !important;
        color: #92400e !important;
        background: #fffdf5 !important;
    }
    .btn-preset-casual button:hover {
        border-color: #f59e0b !important;
        background: #fef3c7 !important;
    }

    /* Main Gauge & Prediction Container Card */
    .prediction-card {
        background: #ffffff;
        border: 1px solid #eaeff5;
        border-radius: 16px;
        padding: 1.75rem 1.5rem 1.5rem 1.5rem;
        text-align: center;
        box-shadow: 0 4px 20px -2px rgba(50, 50, 93, 0.04), 0 2px 6px -1px rgba(0, 0, 0, 0.03);
        margin-bottom: 1.15rem;
    }
    .prediction-title {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748b;
        margin-bottom: 0.65rem;
    }

    /* Semi-Circular SVG Gauge Wrap */
    .gauge-wrapper {
        position: relative;
        width: 230px;
        height: 118px;
        margin: 0 auto 0.25rem auto;
    }
    .gauge-percentage {
        position: absolute;
        bottom: 0px;
        left: 0;
        right: 0;
        font-size: 3.25rem;
        font-weight: 800;
        line-height: 1;
        letter-spacing: -0.04em;
        color: #1e293b;
    }

    /* Risk Status Badge Pill */
    .risk-pill {
        display: inline-block;
        font-size: 0.82rem;
        font-weight: 700;
        padding: 0.4rem 1.35rem;
        border-radius: 9999px;
        margin-top: 0.85rem;
        letter-spacing: 0.04em;
        color: #ffffff !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    }
    .risk-pill-high {
        background: #dc2626;
    }
    .risk-pill-medium {
        background: #caa368;
    }
    .risk-pill-low {
        background: #059669;
    }

    .risk-description {
        font-size: 0.88rem;
        color: #475569;
        margin-top: 1rem;
        line-height: 1.45;
        max-width: 92%;
        margin-left: auto;
        margin-right: auto;
        font-weight: 500;
    }

    /* Financial Stat Cards (Side-by-Side Tiles) */
    .kpi-tile-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.85rem;
        margin-bottom: 1.15rem;
    }
    .kpi-tile {
        background: #ffffff;
        border: 1px solid #eaeff5;
        border-radius: 14px;
        padding: 1.15rem 1rem;
        text-align: center;
        box-shadow: 0 4px 16px -2px rgba(50, 50, 93, 0.03), 0 2px 5px -1px rgba(0, 0, 0, 0.02);
    }
    .kpi-tile-label {
        font-size: 0.74rem;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .kpi-tile-value {
        font-size: 1.65rem;
        font-weight: 800;
        color: #0f172a;
        margin-top: 0.35rem;
        letter-spacing: -0.02em;
    }

    /* Section Subheaders */
    .section-subtitle {
        font-size: 0.95rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.65rem;
        display: flex;
        align-items: center;
        gap: 0.45rem;
        letter-spacing: -0.01em;
    }

    /* Key Drivers List Items */
    .driver-card {
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
        padding: 0.8rem 1rem;
        border-radius: 10px;
        margin-bottom: 0.55rem;
        font-size: 0.87rem;
        background: #ffffff;
        border: 1px solid #eaeff5;
        box-shadow: 0 2px 6px rgba(50, 50, 93, 0.02);
        line-height: 1.45;
        color: #334155;
    }
    .driver-card.safe {
        background: #f8fdfa;
        border-color: #d1fae5;
    }
    .driver-card.adverse {
        background: #fef8f8;
        border-color: #fee2e2;
    }
    .driver-card.info {
        background: #f8faff;
        border-color: #e0e7ff;
    }
    .driver-card.warning {
        background: #fffdf5;
        border-color: #fef3c7;
    }

    /* Retention Action Items */
    .action-card {
        background: #ffffff;
        border: 1px solid #eaeff5;
        border-radius: 12px;
        padding: 0.95rem 1.15rem;
        margin-bottom: 0.65rem;
        box-shadow: 0 2px 6px rgba(50, 50, 93, 0.02);
    }
    .action-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-weight: 700;
        font-size: 0.92rem;
        color: #0f172a;
    }
    .action-body {
        font-size: 0.84rem;
        color: #64748b;
        margin-top: 0.35rem;
        line-height: 1.45;
    }
    .action-badge {
        font-size: 0.75rem;
        font-weight: 700;
        color: #15803d;
        background: #dcfce7;
        border: 1px solid #bbf7d0;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
    }
    .action-badge.reduction {
        color: #0369a1;
        background: #e0f2fe;
        border-color: #bae6fd;
    }

    /* Download Buttons Customization */
    div.stDownloadButton button {
        background-color: #ffffff !important;
        border: 1.5px solid #eaeff5 !important;
        color: #334155 !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.02) !important;
        transition: all 0.2s ease !important;
    }
    div.stDownloadButton button:hover {
        border-color: #2563eb !important;
        color: #2563eb !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.08) !important;
    }

    /* Expander Container */
    div[data-testid="stExpander"] {
        background-color: #ffffff !important;
        border: 1px solid #eaeff5 !important;
        border-radius: 14px !important;
        box-shadow: 0 4px 16px -2px rgba(50, 50, 93, 0.03) !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. LOAD OTT ASSETS
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_ott_assets():
    rf = joblib.load('ott_rf_model.pkl')
    sc = joblib.load('ott_scaler.pkl')
    cols = joblib.load('ott_columns.pkl')
    return rf, sc, cols

try:
    model, scaler, model_columns = load_ott_assets()
except Exception as e:
    st.error(f"Unable to load OTT prediction model: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# 4. PREDICTION FUNCTION
# -----------------------------------------------------------------------------
def compute_ott_churn(data):
    df = pd.DataFrame([data])
    
    # Preprocessing
    df['multi_screen'] = 1 if str(df['multi_screen'].iloc[0]).lower() in ['yes', '1'] else 0
    df['mail_subscribed'] = 1 if str(df['mail_subscribed'].iloc[0]).lower() in ['yes', '1'] else 0
    df['daily_avg_mins'] = df['weekly_mins_watched'] / 7.0
    df['mins_per_video'] = df['weekly_mins_watched'] / (df['videos_watched'] + 1)
    df['gender_Male'] = 1 if df['gender'].iloc[0] == 'Male' else 0
    df['gender_Unknown'] = 1 if df['gender'].iloc[0] == 'Unknown' else 0
    
    # Reindex & scale
    df = df.reindex(columns=model_columns, fill_value=0)
    scaled = scaler.transform(df)
    
    # Inference
    prob = model.predict_proba(scaled)[0][1] * 100.0
    return float(prob)

# -----------------------------------------------------------------------------
# 5. PRESETS MANAGEMENT (OTT Archetypes)
# -----------------------------------------------------------------------------
if 'preset' not in st.session_state:
    st.session_state.preset = "Custom"
    st.session_state.days_subscribed = 120
    st.session_state.multi_screen = "No"
    st.session_state.mail_subscribed = "Yes"
    st.session_state.weekly_mins = 260
    st.session_state.videos = 5
    st.session_state.min_daily_mins = 15.0
    st.session_state.max_daily_mins = 60.0
    st.session_state.night_mins = 45
    st.session_state.inactive_days = 3.0
    st.session_state.support_calls = 1
    st.session_state.age = 32
    st.session_state.gender = "Female"
    st.session_state.monthly_fee = 499.0

def apply_ott_preset(name):
    if name == "dropout":  # High risk: complaints, high dormancy, disengagement
        st.session_state.days_subscribed = 35
        st.session_state.multi_screen = "No"
        st.session_state.mail_subscribed = "No"
        st.session_state.weekly_mins = 340
        st.session_state.videos = 2
        st.session_state.min_daily_mins = 5.0
        st.session_state.max_daily_mins = 55.0
        st.session_state.night_mins = 30
        st.session_state.inactive_days = 12.0
        st.session_state.support_calls = 5
        st.session_state.age = 26
        st.session_state.gender = "Female"
        st.session_state.monthly_fee = 499.0
    elif name == "binge":  # Safe & Loyal: low complaints, steady engagement
        st.session_state.days_subscribed = 380
        st.session_state.multi_screen = "Yes"
        st.session_state.mail_subscribed = "Yes"
        st.session_state.weekly_mins = 220
        st.session_state.videos = 7
        st.session_state.min_daily_mins = 18.0
        st.session_state.max_daily_mins = 40.0
        st.session_state.night_mins = 75
        st.session_state.inactive_days = 1.0
        st.session_state.support_calls = 1
        st.session_state.age = 35
        st.session_state.gender = "Male"
        st.session_state.monthly_fee = 649.0
    elif name == "casual":  # Moderate: occasional watching, mild friction
        st.session_state.days_subscribed = 90
        st.session_state.multi_screen = "No"
        st.session_state.mail_subscribed = "Yes"
        st.session_state.weekly_mins = 180
        st.session_state.videos = 3
        st.session_state.min_daily_mins = 10.0
        st.session_state.max_daily_mins = 45.0
        st.session_state.night_mins = 40
        st.session_state.inactive_days = 6.0
        st.session_state.support_calls = 3
        st.session_state.age = 29
        st.session_state.gender = "Female"
        st.session_state.monthly_fee = 299.0

# -----------------------------------------------------------------------------
# 6. HEADER & QUICK PRESET BUTTONS
# -----------------------------------------------------------------------------
st.markdown("""
<div class="app-header">
    <p class="app-subtitle">
        Predict subscriber cancellation risk for streaming services (Netflix, Prime, Disney+), detect viewer disengagement, and trigger automated retention offers.
    </p>
</div>
""", unsafe_allow_html=True)

# Preset Bar: 3 Immediate Streaming Archetype Buttons
p_col1, p_col2, p_col3, p_spacer = st.columns([1.1, 1.1, 1.1, 1.5])

with p_col1:
    st.markdown('<div class="btn-preset-dropout">', unsafe_allow_html=True)
    if st.button("🚨 Load Binge Dropout (High Risk)", key="btn_dropout", use_container_width=True, help="Load an inactive subscriber with unresolved support tickets"):
        apply_ott_preset("dropout")
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

with p_col2:
    st.markdown('<div class="btn-preset-loyal">', unsafe_allow_html=True)
    if st.button("🍿 Load Loyal Viewer (Safe)", key="btn_loyal", use_container_width=True, help="Load a long-term engaged subscriber"):
        apply_ott_preset("binge")
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

with p_col3:
    st.markdown('<div class="btn-preset-casual">', unsafe_allow_html=True)
    if st.button("📱 Load Casual Viewer (Moderate)", key="btn_casual", use_container_width=True, help="Load an average casual viewer with occasional inactivity"):
        apply_ott_preset("casual")
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 7. MAIN TWO-COLUMN LAYOUT
# -----------------------------------------------------------------------------
col_inputs, col_results = st.columns([1.25, 1.0], gap="large")

# -----------------------------
# LEFT COLUMN: INPUT PARAMETERS (WRAPPED IN CLEAN CARDS)
# -----------------------------
with col_inputs:
    # 1. Subscriber Plan & Account Card
    with st.container(border=True):
        st.markdown("""
            <div class="card-title">
                <span>🪪 Subscription & Plan Details</span>
            </div>
        """, unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            days_subscribed = st.slider(
                "Account Age (Days Subscribed)",
                min_value=1, max_value=800, value=int(st.session_state.days_subscribed),
                help="Total consecutive days this user has held an active subscription"
            )
            multi_screen = st.selectbox(
                "Plan Type / Multi-Screen Access",
                ["Single Screen", "Multi-Screen (Family / Premium)"],
                index=0 if st.session_state.multi_screen == "No" else 1,
                help="Single-screen basic plan vs multi-screen family streaming tier"
            )
            multi_screen_val = "yes" if "Multi-Screen" in multi_screen else "no"

        with c2:
            monthly_fee = st.number_input(
                "Monthly Subscription Price (₹)",
                min_value=99.0, max_value=2500.0, value=float(st.session_state.monthly_fee), step=50.0,
                help="Current monthly recurring price tier (e.g. ₹199 Mobile, ₹499 Standard, ₹649 Premium 4K)"
            )
            mail_subscribed = st.selectbox(
                "Subscribed to Newsletter / Push Promotions",
                ["Yes", "No"],
                index=0 if st.session_state.mail_subscribed == "Yes" else 1,
                help="Whether the subscriber opens content recommendations and promo emails"
            )
            mail_subscribed_val = "yes" if mail_subscribed == "Yes" else "no"

    # 2. Viewing & Streaming Engagement Card
    with st.container(border=True):
        st.markdown("""
            <div class="card-title">
                <span>👥 Viewing & Streaming Activity</span>
            </div>
        """, unsafe_allow_html=True)
        
        v1, v2 = st.columns(2)
        with v1:
            weekly_mins = st.slider(
                "Weekly Watch Time (Minutes)",
                min_value=0, max_value=800, value=int(st.session_state.weekly_mins), step=10,
                help=f"Weekly stream time. Current: {st.session_state.weekly_mins} mins (~{st.session_state.weekly_mins/60:.1f} hrs/week)"
            )
            videos_watched = st.slider(
                "Shows / Movies Watched (Weekly)",
                min_value=0, max_value=25, value=int(st.session_state.videos),
                help="Count of individual episodes or full movies streamed this week"
            )
            night_mins = st.slider(
                "Night-Time Streaming (Peak Mins)",
                min_value=0, max_value=150, value=int(st.session_state.night_mins), step=5,
                help="Minutes watched between 10 PM and 4 AM"
            )

        with v2:
            max_daily_mins = st.slider(
                "Peak Single-Day Viewing (Mins)",
                min_value=0.0, max_value=150.0, value=float(st.session_state.max_daily_mins), step=5.0,
                help="Maximum minutes watched on a heavy binge day"
            )
            min_daily_mins = st.slider(
                "Lowest Single-Day Viewing (Mins)",
                min_value=0.0, max_value=60.0, value=float(st.session_state.min_daily_mins), step=2.0,
                help="Minimum minutes watched on an active day"
            )

    # 3. Disengagement & Friction Signals Card
    with st.container(border=True):
        st.markdown("""
            <div class="card-title">
                <span>⚠️ Inactivity & Support Friction Signals</span>
            </div>
        """, unsafe_allow_html=True)
        
        f1, f2 = st.columns(2)
        with f1:
            inactive_days = st.slider(
                "Consecutive Days Inactive (Dormancy)",
                min_value=0.0, max_value=25.0, value=float(st.session_state.inactive_days), step=1.0,
                help="Maximum consecutive days without opening the streaming app"
            )
        with f2:
            support_calls = st.slider(
                "Customer Support Calls / Tickets Logged",
                min_value=0, max_value=9, value=int(st.session_state.support_calls), step=1,
                help="Complaints regarding buffering, billing issues, playback errors"
            )

    # Demographics Expander
    with st.expander("Subscriber Demographics (Age & Gender)"):
        d1, d2 = st.columns(2)
        with d1:
            age = st.slider("Subscriber Age", min_value=18, max_value=80, value=int(st.session_state.age))
        with d2:
            gender = st.selectbox("Gender", ["Female", "Male", "Unknown"], index=["Female", "Male", "Unknown"].index(st.session_state.gender))

# -----------------------------
# BUILD OTT DATA PAYLOAD & PREDICT
# -----------------------------
ott_payload = {
    'age': age,
    'gender': gender,
    'no_of_days_subscribed': days_subscribed,
    'multi_screen': multi_screen_val,
    'mail_subscribed': mail_subscribed_val,
    'weekly_mins_watched': float(weekly_mins),
    'minimum_daily_mins': float(min_daily_mins),
    'maximum_daily_mins': float(max_daily_mins),
    'weekly_max_night_mins': int(night_mins),
    'videos_watched': int(videos_watched),
    'maximum_days_inactive': float(inactive_days),
    'customer_support_calls': int(support_calls)
}

churn_risk_pct = compute_ott_churn(ott_payload)
annual_streaming_arr = monthly_fee * 12.0

# Determine OTT risk category and styling
if churn_risk_pct >= 50.0:
    risk_pill_class = "risk-pill-high"
    risk_title = "HIGH CHURN RISK"
    arc_color = "#ef4444"
    summary_sentence = "High probability of cancellation. Customer exhibits severe platform friction or content disengagement."
elif churn_risk_pct >= 25.0:
    risk_pill_class = "risk-pill-medium"
    risk_title = "MODERATE WATCHLIST"
    arc_color = "#334155"  # Sleek slate/dark blue-gray matching reference screenshot
    summary_sentence = "Moderate churn indicators detected. Re-engaging with new content drops or support outreach recommended."
else:
    risk_pill_class = "risk-pill-low"
    risk_title = "LOW RISK (LOYAL VIEWER)"
    arc_color = "#059669"
    summary_sentence = "Active, satisfied subscriber with strong streaming habits and consistent platform retention."

# SVG Arch calculations:
# Semicircle radius R=85, circumference = pi * 85 ~= 267.04
total_arc_len = 267.0
clamped_prob = min(max(churn_risk_pct, 0.0), 100.0)
dash_offset = total_arc_len * (1.0 - (clamped_prob / 100.0))

# -----------------------------
# RIGHT COLUMN: RESULTS & RETENTION PLAYBOOK
# -----------------------------
with col_results:
    # 1. Main Risk Score Card with Semi-Circular Gauge Meter
    st.markdown(f"""
    <div class="prediction-card">
        <div class="prediction-title">PREDICTED CHURN PROBABILITY</div>
        <div class="gauge-wrapper">
            <svg viewBox="0 0 220 120" style="width: 100%; height: 100%; overflow: visible;">
                <!-- Background Grey Track Arc -->
                <path d="M 25 110 A 85 85 0 0 1 195 110" 
                      fill="none" 
                      stroke="#eaeff5" 
                      stroke-width="14" 
                      stroke-linecap="round" />
                <!-- Dynamic Progress Arc -->
                <path d="M 25 110 A 85 85 0 0 1 195 110" 
                      fill="none" 
                      stroke="{arc_color}" 
                      stroke-width="14" 
                      stroke-linecap="round"
                      stroke-dasharray="{total_arc_len}"
                      stroke-dashoffset="{dash_offset:.1f}"
                      style="transition: stroke-dashoffset 0.8s cubic-bezier(0.4, 0, 0.2, 1);" />
            </svg>
            <div class="gauge-percentage">{churn_risk_pct:.0f}%</div>
        </div>
        <div>
            <div class="risk-pill {risk_pill_class}">{risk_title}</div>
        </div>
        <div class="risk-description">
            {summary_sentence}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. Revenue Summary (Side-by-Side KPI Tiles)
    st.markdown(f"""
    <div class="kpi-tile-grid">
        <div class="kpi-tile">
            <div class="kpi-tile-label">SUBSCRIPTION TIER</div>
            <div class="kpi-tile-value">₹{monthly_fee:,.0f}<span style="font-size:0.85rem; font-weight:500; color:#64748b;">/mo</span></div>
        </div>
        <div class="kpi-tile">
            <div class="kpi-tile-label">ANNUAL VALUE (ARR)</div>
            <div class="kpi-tile-value">₹{annual_streaming_arr:,.0f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 3. Why is this subscriber at risk? (Key Drivers)
    st.markdown("""
        <div class="section-subtitle">
            <span>🔍</span> Why this score?
        </div>
    """, unsafe_allow_html=True)

    reasons = []
    if support_calls >= 4:
        reasons.append(("❌", f"<strong>High Support Escalations ({support_calls} calls)</strong>: Strong churn signal from buffering or playback complaints.", "adverse"))
    elif support_calls <= 1:
        reasons.append(("✅", f"<strong>Low Support Friction ({support_calls} tickets)</strong>: Smooth playback experience with zero open disputes.", "safe"))

    if inactive_days >= 8:
        reasons.append(("❌", f"<strong>High App Inactivity ({int(inactive_days)} days inactive)</strong>: Subscriber has stopped streaming content.", "adverse"))
    elif inactive_days <= 2:
        reasons.append(("✅", f"<strong>Consistent Daily Habit ({int(inactive_days)} day inactive)</strong>: Strong daily engagement habit.", "safe"))

    if weekly_mins < 120:
        reasons.append(("⚠️", f"<strong>Low Weekly Watch Time ({weekly_mins} mins)</strong>: Less than 2 hours streamed per week indicates low perceived value.", "warning"))
    elif weekly_mins >= 350:
        reasons.append(("✅", f"<strong>Heavy Binge Engagement ({weekly_mins} mins)</strong>: High streaming volume indicates deep content consumption.", "safe"))

    if multi_screen_val == "no":
        reasons.append(("ℹ️", "<strong>Single-Screen Subscription</strong>: Lacks multi-device household lock-in.", "info"))
    else:
        reasons.append(("ℹ️", "<strong>Multi-Screen Plan</strong>: Multi-device shared account.", "info"))

    for icon, text, style_cls in reasons[:4]:
        st.markdown(f"""
        <div class="driver-card {style_cls}">
            <span style="font-size: 1.05rem; line-height: 1;">{icon}</span>
            <div style="flex: 1;">{text}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 0.35rem;'></div>", unsafe_allow_html=True)

    # 4. Recommended OTT Retention Actions
    st.markdown("""
        <div class="section-subtitle">
            <span>💡</span> Recommended Actions to Retain Subscriber
        </div>
    """, unsafe_allow_html=True)

    actions = []
    if support_calls >= 3:
        sim = dict(ott_payload)
        sim['customer_support_calls'] = 1
        new_p = compute_ott_churn(sim)
        reduction = max(2.0, churn_risk_pct - new_p)
        actions.append((
            "Priority Support Outreach & Goodwill Credit",
            "Resolve streaming complaints and offer a one-time ₹150 credit to rebuild satisfaction.",
            f"-{reduction:.0f}% risk",
            True
        ))

    if inactive_days >= 5 or weekly_mins < 150:
        sim = dict(ott_payload)
        sim['maximum_days_inactive'] = 2.0
        sim['weekly_mins_watched'] = 300.0
        new_p = compute_ott_churn(sim)
        reduction = max(2.0, churn_risk_pct - new_p)
        actions.append((
            "Push Personalized New-Release Recommendations",
            "Send targeted push notification featuring popular series in their preferred genres.",
            f"-{reduction:.0f}% risk",
            True
        ))

    if days_subscribed <= 60:
        actions.append((
            "Onboarding Re-engagement Campaign",
            "Send curated 'What to Watch Next' email sequence to establish long-term viewing habits.",
            "-10% risk",
            True
        ))

    if not actions:
        actions.append((
            "Standard VIP Retention Track",
            "Subscriber is highly satisfied and active. Deliver routine content previews.",
            "Healthy",
            False
        ))

    for title, desc, benefit, is_reduct in actions[:3]:
        badge_cls = "reduction" if is_reduct else ""
        st.markdown(f"""
        <div class="action-card">
            <div class="action-header">
                <span>{title}</span>
                <span class="action-badge {badge_cls}">{benefit}</span>
            </div>
            <div class="action-body">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    # 5. Professional Report Export Options
    st.markdown("<div style='height: 0.35rem;'></div>", unsafe_allow_html=True)
    
    # Generate OTT CSV Report
    def generate_ott_csv():
        lines = [
            "STREAMING SUBSCRIBER CHURN & RETENTION ASSESSMENT REPORT",
            f"Generated Date,'{datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "Platform,StreamPulse OTT Retention Engine",
            "",
            "1. EXECUTIVE SUBSCRIBER RISK SUMMARY",
            "Metric,Subscriber Metric,Assessment",
            f"Predicted Churn Probability,{churn_risk_pct:.1f}%,{risk_title}",
            f"Monthly Subscription Fee,₹{monthly_fee:,.0f},Active Tier",
            f"Annual Revenue at Risk (ARR),₹{annual_streaming_arr:,.0f},Exposure",
            f"Tenure with Platform,{days_subscribed} Days,Account Age",
            "",
            "2. STREAMING BEHAVIOR & ENGAGEMENT",
            "Behavioral Metric,Logged Metric",
            f"Weekly Minutes Streamed,{weekly_mins} Minutes",
            f"Shows & Movies Streamed / Week,{videos_watched}",
            f"Peak Single-Day Watch Time,{max_daily_mins} Minutes",
            f"Night Streaming Volume,{night_mins} Minutes",
            f"Max Consecutive Inactive Days,{inactive_days} Days",
            f"Customer Support Complaints Logged,{support_calls}",
            f"Multi-Screen Tier,{multi_screen}",
            "",
            "3. KEY RISK DRIVERS (WHY THIS SCORE?)",
            "Indicator,Impact Factor,Details",
        ]
        for item in reasons:
            icon, text = item[0], item[1]
            clean = text.replace('<strong>', '').replace('</strong>', '')
            tag = "Adverse Signal" if "❌" in icon else ("Favorable" if "✅" in icon else "Attention Area")
            lines.append(f"{tag},Viewing Behavior,\"{clean}\"")
        lines.extend([
            "",
            "4. RECOMMENDED RETENTION PLAYBOOK",
            "Retention Strategy,Details,Projected Churn Delta"
        ])
        for item in actions:
            title, desc, benefit = item[0], item[1], item[2]
            lines.append(f"\"{title}\",\"{desc}\",\"{benefit}\"")
        return "\n".join(lines)

    def generate_ott_html():
        badge_bg = "#dc2626" if "HIGH" in risk_title else ("#caa368" if "MODERATE" in risk_title else "#059669")
        drivers_html = "".join([
            f"""<tr><td style='padding:10px 14px; border-bottom:1px solid #eaeff5; font-size:16px;'>{item[0]}</td>
            <td style='padding:10px 14px; border-bottom:1px solid #eaeff5; color:#334155;'>{item[1]}</td></tr>""" for item in reasons
        ])
        actions_html = "".join([
            f"""<div style='background:#f8fafc; border:1px solid #eaeff5; border-left:4px solid #2563eb; padding:14px 18px; margin-bottom:12px; border-radius:8px;'>
                <div style='display:flex; justify-content:space-between; font-weight:700; color:#0f172a; font-size:15px;'>
                    <span>{item[0]}</span><span style='color:#059669; font-weight:700;'>{item[2]}</span>
                </div>
                <div style='font-size:13.5px; color:#64748b; margin-top:5px; line-height:1.45;'>{item[1]}</div>
            </div>""" for item in actions
        ])
        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>StreamPulse OTT Churn Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 30px auto; max-width: 850px; color: #1e293b; background: #f6f8fb; }}
        .card {{ background: #ffffff; border: 1px solid #eaeff5; border-radius: 16px; padding: 36px; box-shadow: 0 4px 20px -2px rgba(50,50,93,0.05); }}
        .header {{ border-bottom: 2px solid #2563eb; padding-bottom: 18px; margin-bottom: 26px; display: flex; justify-content: space-between; align-items: flex-end; }}
        .title {{ font-size: 24px; font-weight: 800; color: #0f172a; margin: 0; }}
        .kpi-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 24px; }}
        .kpi-box {{ background: #ffffff; border-radius: 12px; padding: 18px; border: 1px solid #eaeff5; text-align: center; box-shadow: 0 2px 6px rgba(0,0,0,0.02); }}
        .badge {{ display: inline-block; padding: 6px 14px; border-radius: 9999px; font-size: 12px; font-weight: 700; color: #ffffff; background: {badge_bg}; }}
        .section-title {{ font-size: 16px; font-weight: 700; color: #0f172a; border-bottom: 1px solid #eaeff5; padding-bottom: 10px; margin-top: 28px; margin-bottom: 14px; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 14px; margin-bottom: 20px; }}
        th {{ text-align: left; background: #f8fafc; padding: 10px 14px; border-bottom: 1px solid #e2e8f0; font-weight: 600; color: #475569; }}
        td {{ padding: 10px 14px; border-bottom: 1px solid #f1f5f9; color: #334155; }}
        @media print {{ body {{ background: #ffffff; margin: 0; }} .card {{ border: none; box-shadow: none; padding: 0; }} }}
    </style>
</head>
<body>
    <div class="card">
        <div class="header">
            <div>
                <h1 class="title">🎬 StreamPulse™ Subscriber Churn Dossier</h1>
                <div style="font-size:13px; color:#64748b; margin-top:6px;">OTT Retention Intelligence &bull; Date: {datetime.now().strftime('%B %d, %Y')}</div>
            </div>
            <div><span class="badge">{risk_title}</span></div>
        </div>
        <div class="kpi-grid">
            <div class="kpi-box"><div style="font-size:11px; font-weight:700; color:#64748b; text-transform:uppercase;">Predicted Churn Risk</div><div style="font-size:28px; font-weight:800; color:#0f172a; margin-top:4px;">{churn_risk_pct:.1f}%</div></div>
            <div class="kpi-box"><div style="font-size:11px; font-weight:700; color:#64748b; text-transform:uppercase;">Monthly Subscription</div><div style="font-size:28px; font-weight:800; color:#0f172a; margin-top:4px;">₹{monthly_fee:,.0f}</div></div>
            <div class="kpi-box"><div style="font-size:11px; font-weight:700; color:#64748b; text-transform:uppercase;">Annual Revenue at Risk</div><div style="font-size:28px; font-weight:800; color:#0f172a; margin-top:4px;">₹{annual_streaming_arr:,.0f}</div></div>
        </div>
        <div class="section-title">Streaming &amp; Behavioral Profile</div>
        <table>
            <tr><th>Account Age</th><td>{days_subscribed} Days</td><th>Plan Type</th><td>{multi_screen}</td></tr>
            <tr><th>Weekly Watch Time</th><td>{weekly_mins} Mins</td><th>Shows / Movies Watched</th><td>{videos_watched}</td></tr>
            <tr><th>Consecutive Inactive Days</th><td>{inactive_days} Days</td><th>Customer Support Calls</th><td>{support_calls}</td></tr>
        </table>
        <div class="section-title">Key Risk Drivers</div>
        <table>{drivers_html}</table>
        <div class="section-title">Recommended Retention Playbook</div>
        {actions_html}
    </div>
</body>
</html>"""

    export_col1, export_col2 = st.columns(2)
    with export_col1:
        st.download_button(
            label="📑 Export for Excel (.csv)",
            data=generate_ott_csv(),
            file_name=f"OTT_Retention_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    with export_col2:
        st.download_button(
            label="📊 Export Printable Dossier (.html)",
            data=generate_ott_html(),
            file_name=f"OTT_Executive_Dossier_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
            mime="text/html",
            use_container_width=True
        )