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
# 2. HIGH-CONTRAST, THEME-SAFE, INTUITIVE CSS
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, sans-serif;
    }

    /* Main Section Headers - Vibrant Sky Blue for 100% Visibility in Dark & Light Modes */
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
    st.session_state.monthly_fee = 14.99

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
        st.session_state.monthly_fee = 14.99
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
        st.session_state.monthly_fee = 19.99
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
        st.session_state.monthly_fee = 14.99

# -----------------------------------------------------------------------------
# 6. HEADER & QUICK PRESET BUTTONS
# -----------------------------------------------------------------------------
st.markdown("""
<div style="margin-bottom: 1.25rem;">
    <h1 style="font-size: 1.75rem; font-weight: 800; color: #ffffff !important; margin: 0; letter-spacing: -0.02em;">
        🎬 StreamPulse™ | OTT Subscriber Churn Analyzer
    </h1>
    <p style="font-size: 0.95rem; color: #94a3b8 !important; margin: 0.35rem 0 0 0;">
        Predict subscriber cancellation risk for streaming services (Netflix, Prime, Disney+), detect viewer disengagement, and trigger automated retention offers.
    </p>
</div>
""", unsafe_allow_html=True)

# Preset Bar: 3 Immediate Streaming Archetype Buttons
p_col1, p_col2, p_col3, p_col4 = st.columns([1.2, 1.2, 1.2, 2.4])
with p_col1:
    if st.button("🚨 Load Binge Dropout (High Risk)", use_container_width=True, help="Load an inactive subscriber with unresolved support tickets"):
        apply_ott_preset("dropout")
        st.rerun()
with p_col2:
    if st.button("🍿 Load Loyal Viewer (Safe)", use_container_width=True, help="Load a long-term engaged subscriber"):
        apply_ott_preset("binge")
        st.rerun()
with p_col3:
    if st.button("📱 Load Casual Viewer (Moderate)", use_container_width=True, help="Load an average casual viewer with occasional inactivity"):
        apply_ott_preset("casual")
        st.rerun()

st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 7. MAIN TWO-COLUMN LAYOUT
# -----------------------------------------------------------------------------
col_inputs, col_results = st.columns([1.25, 1.0], gap="large")

# -----------------------------
# LEFT COLUMN: INPUT PARAMETERS
# -----------------------------
with col_inputs:
    # 1. Subscriber Plan & Account
    st.markdown("""
        <div class="card-title">
            <span>📺 Subscription & Plan Details</span>
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
            "Monthly Subscription Price ($)",
            min_value=4.99, max_value=35.0, value=float(st.session_state.monthly_fee), step=1.0,
            help="Current monthly recurring price tier (e.g. $14.99 Standard, $19.99 Premium 4K)"
        )
        mail_subscribed = st.selectbox(
            "Subscribed to Newsletter / Push Promotions",
            ["Yes", "No"],
            index=0 if st.session_state.mail_subscribed == "Yes" else 1,
            help="Whether the subscriber opens content recommendations and promo emails"
        )
        mail_subscribed_val = "yes" if mail_subscribed == "Yes" else "no"

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

    # 2. Viewing & Streaming Engagement
    st.markdown("""
        <div class="card-title">
            <span>🎥 Viewing & Streaming Activity</span>
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

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

    # 3. Disengagement & Friction Signals
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

# Determine OTT risk category
if churn_risk_pct >= 50.0:
    risk_style = "high"
    risk_badge = "badge-high"
    risk_title = "HIGH CHURN RISK"
    summary_sentence = "High probability of cancellation. Customer exhibits severe platform friction or content disengagement."
elif churn_risk_pct >= 25.0:
    risk_style = "medium"
    risk_badge = "badge-medium"
    risk_title = "MODERATE WATCHLIST"
    summary_sentence = "Moderate churn indicators detected. Re-engaging with new content drops or support outreach recommended."
else:
    risk_style = "low"
    risk_badge = "badge-low"
    risk_title = "LOW RISK (LOYAL VIEWER)"
    summary_sentence = "Active, satisfied subscriber with strong streaming habits and consistent platform retention."

# -----------------------------
# RIGHT COLUMN: RESULTS & RETENTION PLAYBOOK
# -----------------------------
with col_results:
    # 1. Main Risk Score Card
    st.markdown(f"""
    <div class="risk-score-box {risk_style}">
        <div class="risk-score-label">Predicted Churn Probability</div>
        <div class="risk-score-number">{churn_risk_pct:.0f}%</div>
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
            <div class="stat-pill-label">Subscription Tier</div>
            <div class="stat-pill-value">${monthly_fee:.2f}<span style="font-size:0.85rem; font-weight:500; color:#94a3b8;">/mo</span></div>
        </div>
        <div class="stat-pill">
            <div class="stat-pill-label">Annual Value (ARR)</div>
            <div class="stat-pill-value">${annual_streaming_arr:,.2f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 3. Why is this subscriber at risk? (Plain English OTT Drivers)
    st.markdown("""
        <div class="section-subtitle">
            🔍 Why this score?
        </div>
    """, unsafe_allow_html=True)

    reasons = []
    if support_calls >= 4:
        reasons.append(("❌", f"<strong>High Support Escalations ({support_calls} calls)</strong>: Strong churn signal from buffering or playback complaints."))
    elif support_calls <= 1:
        reasons.append(("✅", f"<strong>Low Support Friction ({support_calls} tickets)</strong>: Smooth playback experience with zero open disputes."))

    if inactive_days >= 8:
        reasons.append(("❌", f"<strong>High App Inactivity ({int(inactive_days)} days inactive)</strong>: Subscriber has stopped streaming content."))
    elif inactive_days <= 2:
        reasons.append(("✅", f"<strong>Consistent Daily Habit ({int(inactive_days)} day inactive)</strong>: Strong daily engagement habit."))

    if weekly_mins < 120:
        reasons.append(("⚠️", f"<strong>Low Weekly Watch Time ({weekly_mins} mins)</strong>: Less than 2 hours streamed per week indicates low perceived value."))
    elif weekly_mins >= 350:
        reasons.append(("✅", f"<strong>Heavy Binge Engagement ({weekly_mins} mins)</strong>: High streaming volume indicates deep content consumption."))

    if multi_screen_val == "no":
        reasons.append(("ℹ️", "<strong>Single-Screen Subscription</strong>: Lacks multi-device household lock-in."))
    else:
        reasons.append(("ℹ️", "<strong>Multi-Screen Plan</strong>: Multi-device shared account."))

    for icon, text in reasons[:4]:
        st.markdown(f"""
        <div class="driver-item">
            <span class="driver-icon">{icon}</span>
            <span class="driver-text">{text}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

    # 4. Recommended OTT Retention Actions
    st.markdown("""
        <div class="section-subtitle">
            💡 Recommended Actions to Retain Subscriber
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
            "Resolve streaming complaints and offer a one-time $5 credit to rebuild satisfaction.",
            f"-{reduction:.0f}% risk"
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
            f"-{reduction:.0f}% risk"
        ))

    if days_subscribed <= 60:
        actions.append((
            "Onboarding Re-engagement Campaign",
            "Send curated 'What to Watch Next' email sequence to establish long-term viewing habits.",
            "-10% risk"
        ))

    if not actions:
        actions.append((
            "Standard VIP Retention Track",
            "Subscriber is highly satisfied and active. Deliver routine content previews.",
            "Healthy"
        ))

    for title, desc, benefit in actions[:3]:
        st.markdown(f"""
        <div class="action-item">
            <div class="action-header">
                <span>{title}</span>
                <span class="action-impact">{benefit}</span>
            </div>
            <div class="action-body">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    # 5. Professional Report Export Options
    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
    
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
            f"Monthly Subscription Fee,${monthly_fee:.2f},Active Tier",
            f"Annual Revenue at Risk (ARR),${annual_streaming_arr:,.2f},Exposure",
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
        for icon, text in reasons:
            clean = text.replace('<strong>', '').replace('</strong>', '')
            tag = "Adverse Signal" if "❌" in icon else ("Favorable" if "✅" in icon else "Attention Area")
            lines.append(f"{tag},Viewing Behavior,\"{clean}\"")
        lines.extend([
            "",
            "4. RECOMMENDED RETENTION PLAYBOOK",
            "Retention Strategy,Details,Projected Churn Delta"
        ])
        for title, desc, benefit in actions:
            lines.append(f"\"{title}\",\"{desc}\",\"{benefit}\"")
        return "\n".join(lines)

    def generate_ott_html():
        badge_bg = "#dc2626" if "HIGH" in risk_title else ("#d97706" if "MODERATE" in risk_title else "#059669")
        drivers_html = "".join([
            f"""<tr><td style='padding:8px 12px; border-bottom:1px solid #e2e8f0; font-weight:600;'>{icon}</td>
            <td style='padding:8px 12px; border-bottom:1px solid #e2e8f0;'>{text}</td></tr>""" for icon, text in reasons
        ])
        actions_html = "".join([
            f"""<div style='background:#f8fafc; border-left:4px solid #2563eb; padding:12px 16px; margin-bottom:10px; border-radius:4px;'>
                <div style='display:flex; justify-content:space-between; font-weight:700; color:#0f172a;'>
                    <span>{title}</span><span style='color:#059669; font-weight:700;'>{benefit}</span>
                </div>
                <div style='font-size:13px; color:#64748b; margin-top:4px;'>{desc}</div>
            </div>""" for title, desc, benefit in actions
        ])
        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>StreamPulse OTT Churn Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 30px auto; max-width: 850px; color: #1e293b; background: #f8fafc; }}
        .card {{ background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 32px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); }}
        .header {{ border-bottom: 2px solid #2563eb; padding-bottom: 16px; margin-bottom: 24px; display: flex; justify-content: space-between; align-items: flex-end; }}
        .title {{ font-size: 24px; font-weight: 800; color: #0f172a; margin: 0; }}
        .kpi-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 24px; }}
        .kpi-box {{ background: #f8fafc; border-radius: 8px; padding: 16px; border: 1px solid #e2e8f0; text-align: center; }}
        .badge {{ display: inline-block; padding: 4px 12px; border-radius: 9999px; font-size: 12px; font-weight: 700; color: #ffffff; background: {badge_bg}; }}
        .section-title {{ font-size: 16px; font-weight: 700; color: #0f172a; border-bottom: 1px solid #e2e8f0; padding-bottom: 8px; margin-top: 24px; margin-bottom: 12px; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 14px; margin-bottom: 16px; }}
        th {{ text-align: left; background: #f1f5f9; padding: 8px 12px; border-bottom: 1px solid #cbd5e1; font-weight: 600; color: #475569; }}
        td {{ padding: 8px 12px; border-bottom: 1px solid #f1f5f9; color: #334155; }}
        @media print {{ body {{ background: #ffffff; margin: 0; }} .card {{ border: none; box-shadow: none; padding: 0; }} }}
    </style>
</head>
<body>
    <div class="card">
        <div class="header">
            <div>
                <h1 class="title">🎬 StreamPulse™ Subscriber Churn Dossier</h1>
                <div style="font-size:13px; color:#64748b; margin-top:4px;">OTT Retention Intelligence &bull; Date: {datetime.now().strftime('%B %d, %Y')}</div>
            </div>
            <div><span class="badge">{risk_title}</span></div>
        </div>
        <div class="kpi-grid">
            <div class="kpi-box"><div style="font-size:11px; font-weight:700; color:#64748b; text-transform:uppercase;">Predicted Churn Risk</div><div style="font-size:28px; font-weight:800; color:#0f172a;">{churn_risk_pct:.1f}%</div></div>
            <div class="kpi-box"><div style="font-size:11px; font-weight:700; color:#64748b; text-transform:uppercase;">Monthly Subscription</div><div style="font-size:28px; font-weight:800; color:#0f172a;">${monthly_fee:.2f}</div></div>
            <div class="kpi-box"><div style="font-size:11px; font-weight:700; color:#64748b; text-transform:uppercase;">Annual Revenue at Risk</div><div style="font-size:28px; font-weight:800; color:#0f172a;">${annual_streaming_arr:,.2f}</div></div>
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