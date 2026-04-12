# ============================================================
# Customer Churn Prediction Dashboard
# Alexis Aquino | University of Batangas
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Intelligence Platform",
    page_icon="assets/favicon.ico" if os.path.exists("assets/favicon.ico") else None,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── GLOBAL STYLES ────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 2rem 3rem;
    max-width: 1400px;
}

.hero {
    background: linear-gradient(135deg, #0f0a1e 0%, #1a1035 60%, #0f0a1e 100%);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    border: 1px solid #3b1f6e;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(139,92,246,0.1) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-size: 1.85rem;
    font-weight: 600;
    color: #f1f5f9;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.02em;
}
.hero-subtitle {
    font-size: 0.9rem;
    color: #64748b;
    font-weight: 400;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin: 0;
}
.hero-badge {
    display: inline-block;
    background: rgba(139,92,246,0.15);
    border: 1px solid rgba(139,92,246,0.35);
    color: #c4b5fd;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    margin-top: 1rem;
}

.metric-card {
    background: #0f0a1e;
    border: 1px solid #1e1535;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    position: relative;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #7c3aed; }

/* THIS is the fix — was #475569 (too dark), now #a78bfa (visible purple) */
.metric-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: #a78bfa;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.5rem;
}

.metric-value {
    font-size: 2rem;
    font-weight: 600;
    color: #f1f5f9;
    font-family: 'DM Mono', monospace;
    line-height: 1;
}
.metric-value.green  { color: #34d399; }
.metric-value.blue   { color: #c4b5fd; }
.metric-value.yellow { color: #fbbf24; }
.metric-desc {
    font-size: 0.75rem;
    color: #64748b;
    margin-top: 0.4rem;
}

.section-header {
    font-size: 0.72rem;
    font-weight: 500;
    color: #a78bfa;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 1.25rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e1535;
}

.result-high {
    background: rgba(239,68,68,0.08);
    border: 1px solid rgba(239,68,68,0.25);
    border-radius: 12px;
    padding: 1.75rem 2rem;
}
.result-low {
    background: rgba(52,211,153,0.08);
    border: 1px solid rgba(52,211,153,0.25);
    border-radius: 12px;
    padding: 1.75rem 2rem;
}
.result-label {
    font-size: 0.72rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.5rem;
}
.result-label.red   { color: #f87171; }
.result-label.green { color: #34d399; }
.result-prob {
    font-size: 3rem;
    font-weight: 600;
    font-family: 'DM Mono', monospace;
    line-height: 1;
    margin-bottom: 0.4rem;
}
.result-prob.red   { color: #ef4444; }
.result-prob.green { color: #10b981; }
.result-desc {
    font-size: 0.85rem;
    color: #94a3b8;
    margin-bottom: 1.25rem;
}

.action-item {
    background: rgba(26,16,53,0.6);
    border-left: 3px solid #7c3aed;
    padding: 0.6rem 1rem;
    margin-bottom: 0.5rem;
    border-radius: 0 6px 6px 0;
    font-size: 0.83rem;
    color: #cbd5e1;
}
.action-item.urgent { border-left-color: #ef4444; }

.input-panel-title {
    font-size: 0.72rem;
    font-weight: 500;
    color: #a78bfa;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e1535;
}

.journey-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.83rem;
}
.journey-table th {
    text-align: left;
    font-size: 0.7rem;
    font-weight: 500;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding: 0.6rem 1rem;
    border-bottom: 1px solid #1e1535;
}
.journey-table td {
    padding: 0.7rem 1rem;
    color: #cbd5e1;
    border-bottom: 1px solid #0f0a1e;
}
.journey-table tr:last-child td { border-bottom: none; }
.tag-best {
    display: inline-block;
    background: rgba(52,211,153,0.15);
    color: #34d399;
    font-size: 0.65rem;
    font-weight: 500;
    padding: 0.15rem 0.5rem;
    border-radius: 999px;
    margin-left: 0.5rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: #0f0a1e;
    border-radius: 8px;
    padding: 4px;
    border: 1px solid #1e1535;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 6px;
    color: #64748b;
    font-size: 0.8rem;
    font-weight: 500;
    padding: 0.5rem 1.25rem;
}
.stTabs [aria-selected="true"] {
    background: #1e1535 !important;
    color: #f1f5f9 !important;
}

.stButton > button {
    background: #7c3aed;
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
    font-weight: 500;
    padding: 0.6rem 2rem;
    transition: background 0.2s, transform 0.1s;
}
.stButton > button:hover {
    background: #6d28d9;
    transform: translateY(-1px);
}

.stSlider > div > div { background: #1e1535; }
.stSelectbox > div > div {
    background: #0f0a1e;
    border: 1px solid #1e1535;
    border-radius: 8px;
    color: #f1f5f9;
    font-size: 0.83rem;
}

.streamlit-expanderHeader {
    background: #0f0a1e;
    border: 1px solid #1e1535;
    border-radius: 8px;
    font-size: 0.8rem;
    color: #64748b;
}

.main { background-color: #07041a; }
</style>
""", unsafe_allow_html=True)


# ── LOAD MODEL ARTIFACTS ─────────────────────────────────────
@st.cache_resource
def load_model():
    base      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model     = joblib.load(os.path.join(base, 'models', 'final_xgboost_model.pkl'))
    features  = joblib.load(os.path.join(base, 'models', 'feature_list.pkl'))
    threshold = joblib.load(os.path.join(base, 'models', 'optimal_threshold.pkl'))
    return model, features, threshold

model, features, threshold = load_model()


# ── HERO HEADER ──────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <p class="hero-subtitle">University of Batangas &nbsp;·&nbsp; Alexis Aquino &nbsp;·&nbsp; 28-Day ML Project</p>
    <h1 class="hero-title">Churn Intelligence Platform</h1>
    <span class="hero-badge">XGBoost &nbsp;·&nbsp; Recall 96.5% &nbsp;·&nbsp; AUC 0.8037 &nbsp;·&nbsp; Threshold 30%</span>
</div>
""", unsafe_allow_html=True)


# ── TABS ─────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "  Predict Customer  ",
    "  Model Performance  ",
    "  Project Summary  "
])


# ════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ════════════════════════════════════════════════════════════
with tab1:
    st.markdown("<div class='section-header'>Customer Profile</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        st.markdown("<div class='input-panel-title'>Account</div>", unsafe_allow_html=True)
        tenure   = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox("Contract Type", [
            "Month-to-month", "One year", "Two year"
        ])
        payment  = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])

    with col2:
        st.markdown("<div class='input-panel-title'>Billing</div>", unsafe_allow_html=True)
        monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0)
        total_charges   = st.slider("Total Charges ($)", 18.0, 9000.0,
                                    float(tenure * monthly_charges))
        senior_citizen  = st.selectbox("Senior Citizen", ["No", "Yes"])
        paperless       = st.selectbox("Paperless Billing", ["Yes", "No"])

    with col3:
        st.markdown("<div class='input-panel-title'>Services</div>", unsafe_allow_html=True)
        internet         = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        online_security  = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        tech_support     = st.selectbox("Tech Support",    ["Yes", "No", "No internet service"])
        streaming_tv     = st.selectbox("Streaming TV",    ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies",["Yes", "No", "No internet service"])

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("Run Prediction", type="primary")

    # ── BUILD INPUT ───────────────────────────────────────────
    def build_input(tenure, monthly_charges, total_charges, senior_citizen,
                    contract, payment, internet, online_security,
                    tech_support, streaming_tv, streaming_movies, paperless):

        tenure_s  = tenure / 72
        monthly_s = (monthly_charges - 18.0) / (118.75 - 18.0)
        total_s   = (total_charges - 18.8) / (8684.8 - 18.8)

        row = {f: 0 for f in features}

        if 'seniorcitizen'  in row: row['seniorcitizen']  = 1 if senior_citizen == "Yes" else 0
        if 'tenure'         in row: row['tenure']         = tenure_s
        if 'monthlycharges' in row: row['monthlycharges'] = monthly_s
        if 'totalcharges'   in row: row['totalcharges']   = total_s

        if 'contract_One year'  in row and contract == "One year":  row['contract_One year']  = 1
        if 'contract_Two year'  in row and contract == "Two year":  row['contract_Two year']  = 1

        if 'paymentmethod_Electronic check'        in row and payment == "Electronic check":        row['paymentmethod_Electronic check']        = 1
        if 'paymentmethod_Mailed check'            in row and payment == "Mailed check":            row['paymentmethod_Mailed check']            = 1
        if 'paymentmethod_Credit card (automatic)' in row and payment == "Credit card (automatic)": row['paymentmethod_Credit card (automatic)'] = 1

        if 'internetservice_Fiber optic' in row and internet == "Fiber optic": row['internetservice_Fiber optic'] = 1
        if 'internetservice_No'          in row and internet == "No":           row['internetservice_No']          = 1

        if 'onlinesecurity_Yes'                in row and online_security == "Yes":                 row['onlinesecurity_Yes']                = 1
        if 'onlinesecurity_No internet service' in row and online_security == "No internet service": row['onlinesecurity_No internet service'] = 1

        if 'techsupport_Yes'                in row and tech_support == "Yes":                 row['techsupport_Yes']                = 1
        if 'techsupport_No internet service' in row and tech_support == "No internet service": row['techsupport_No internet service'] = 1

        if 'streamingtv_Yes'                in row and streaming_tv == "Yes":                 row['streamingtv_Yes']                = 1
        if 'streamingtv_No internet service' in row and streaming_tv == "No internet service": row['streamingtv_No internet service'] = 1

        if 'streamingmovies_Yes'                in row and streaming_movies == "Yes":                 row['streamingmovies_Yes']                = 1
        if 'streamingmovies_No internet service' in row and streaming_movies == "No internet service": row['streamingmovies_No internet service'] = 1

        if 'paperlessbilling_Yes' in row and paperless == "Yes": row['paperlessbilling_Yes'] = 1

        return pd.DataFrame([row])[features]

    # ── RESULT ───────────────────────────────────────────────
    if predict_btn:
        input_df  = build_input(tenure, monthly_charges, total_charges,
                                senior_citizen, contract, payment, internet,
                                online_security, tech_support, streaming_tv,
                                streaming_movies, paperless)
        prob      = model.predict_proba(input_df)[0][1]
        predicted = int(prob >= threshold)

        st.markdown("---")
        res_col, chart_col = st.columns([1.2, 1], gap="large")

        with res_col:
            if predicted == 1:
                pct_above = (prob - threshold) * 100
                st.markdown(f"""
                <div class="result-high">
                    <div class="result-label red">High Churn Risk Detected</div>
                    <div class="result-prob red">{prob*100:.1f}%</div>
                    <div class="result-desc">
                        Churn probability is {pct_above:.1f}% above the intervention threshold of {threshold*100:.0f}%.
                        Immediate action is recommended.
                    </div>
                    <div class="action-item urgent">Offer annual contract upgrade — 20% discount</div>
                    <div class="action-item urgent">Assign dedicated account manager</div>
                    <div class="action-item">Schedule proactive check-in call within 48 hours</div>
                    <div class="action-item">Review service quality if customer is on Fiber optic</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                pct_below = (threshold - prob) * 100
                st.markdown(f"""
                <div class="result-low">
                    <div class="result-label green">Low Churn Risk</div>
                    <div class="result-prob green">{prob*100:.1f}%</div>
                    <div class="result-desc">
                        Churn probability is {pct_below:.1f}% below the intervention threshold of {threshold*100:.0f}%.
                        No immediate action required.
                    </div>
                    <div class="action-item">Maintain current service quality</div>
                    <div class="action-item">Evaluate upsell or cross-sell opportunities</div>
                    <div class="action-item">Consider enrolling in loyalty rewards program</div>
                </div>
                """, unsafe_allow_html=True)

        with chart_col:
            fig, ax = plt.subplots(figsize=(5, 2.8))
            fig.patch.set_facecolor('#0f172a')
            ax.set_facecolor('#0f172a')

            bar_color = '#ef4444' if predicted == 1 else '#10b981'
            ax.barh([''], [prob],        color=bar_color,  height=0.35, zorder=3)
            ax.barh([''], [1 - prob],    left=[prob], color='#1e293b', height=0.35, zorder=3)
            ax.axvline(x=threshold, color='#fbbf24', linestyle='--',
                       linewidth=1.5, label=f'Threshold ({threshold*100:.0f}%)', zorder=4)

            ax.set_xlim(0, 1)
            ax.set_xlabel('Churn Probability', color='#475569', fontsize=9)
            ax.set_title('Risk Assessment', color='#94a3b8',
                         fontsize=10, fontweight='normal', pad=12)
            ax.tick_params(colors='#475569', labelsize=8)
            ax.spines[:].set_color('#1e293b')
            legend = ax.legend(fontsize=8, facecolor='#1e293b',
                               edgecolor='#334155', labelcolor='#94a3b8')

            # Annotate probability
            ax.text(prob / 2, 0, f'{prob*100:.1f}%',
                    ha='center', va='center', color='white',
                    fontsize=11, fontweight='bold', zorder=5)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with st.expander("View raw input sent to model"):
            st.dataframe(input_df.T.rename(columns={0: 'Value'}),
                         use_container_width=True)


# ════════════════════════════════════════════════════════════
# TAB 2 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-header'>Evaluation Metrics</div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Recall</div>
            <div class="metric-value green">96.5%</div>
            <div class="metric-desc">Of 374 real churners, 361 caught</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Precision</div>
            <div class="metric-value blue">36.2%</div>
            <div class="metric-desc">Of churn alerts, 36% confirmed</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">F1 Score</div>
            <div class="metric-value yellow">52.7%</div>
            <div class="metric-desc">Harmonic mean of precision/recall</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">ROC-AUC</div>
            <div class="metric-value blue">0.8037</div>
            <div class="metric-desc">Good discriminative ability</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    report_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'reports', 'final_evaluation_report.png'
    )
    if os.path.exists(report_path):
        st.markdown("<div class='section-header'>Evaluation Report</div>", unsafe_allow_html=True)
        st.image(report_path, use_container_width=True)
    else:
        st.info("Run the Day 22 evaluation notebook to generate the report chart.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Model Development Journey</div>", unsafe_allow_html=True)

    journey_html = """
    <table class="journey-table">
        <thead>
            <tr>
                <th>Version</th>
                <th>Recall</th>
                <th>Key Change</th>
            </tr>
        </thead>
        <tbody>
            <tr><td>Logistic Regression — Baseline</td><td>56.95%</td><td>Initial benchmark</td></tr>
            <tr><td>XGBoost — Default</td><td>78.61%</td><td>Algorithm switched to XGBoost</td></tr>
            <tr><td>Logistic Regression — Balanced</td><td>79.14%</td><td>class_weight correction applied</td></tr>
            <tr><td>XGBoost — GridSearchCV Tuned (50%)</td><td>82.4%</td><td>Hyperparameter search completed</td></tr>
            <tr><td>XGBoost — Threshold Tuned (30%)</td><td>97.6%</td><td>Decision threshold lowered to 30%</td></tr>
            <tr><td>XGBoost — Final Model <span class="tag-best">Production</span></td><td>96.5%</td><td>Zero-importance features removed</td></tr>
        </tbody>
    </table>
    """
    st.markdown(journey_html, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# TAB 3 — PROJECT SUMMARY
# ════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-header'>Project Overview</div>", unsafe_allow_html=True)

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("""
        <div class="metric-card" style="margin-bottom:1rem">
            <div class="metric-label">Business Problem</div>
            <p style="color:#94a3b8; font-size:0.85rem; margin:0.75rem 0 0 0; line-height:1.7">
                A telecom company is losing customers to competitors. The objective is to identify
                customers likely to churn so the retention team can intervene before they leave —
                reducing revenue loss with targeted, cost-effective action.
            </p>
        </div>
        <div class="metric-card" style="margin-bottom:1rem">
            <div class="metric-label">Dataset</div>
            <p style="color:#94a3b8; font-size:0.85rem; margin:0.75rem 0 0 0; line-height:1.7">
                Kaggle Telco Customer Churn &nbsp;&mdash;&nbsp; 7,032 customers, 20 features, 26.6% churn rate.
                Key finding: contract type is the root cause of churn. Month-to-month customers
                churned at 100% versus 0% for annual contracts.
            </p>
        </div>
        <div class="metric-card">
            <div class="metric-label">Business Impact</div>
            <table style="width:100%; font-size:0.82rem; color:#94a3b8; margin-top:0.75rem; border-collapse:collapse">
                <tr><td style="padding:0.35rem 0; color:#94a3b8; text-align:center";>Churners in test set</td><td style="text-align:right; color:#f1f5f9; font-family:'DM Mono',monospace">374</td></tr>
                <tr><td style="padding:0.35rem 0; color:#94a3b8; text-align:center">Churners caught</td><td style="text-align:right; color:#34d399; font-family:'DM Mono',monospace">361 (96.5%)</td></tr>
                <tr><td style="padding:0.35rem 0; color:#94a3b8; text-align:center">Churners missed</td><td style="text-align:right; color:#f87171; font-family:'DM Mono',monospace">13</td></tr>
                <tr><td style="padding:0.35rem 0; color:#94a3b8; border-top:1px solid #1e293b; text-align:center">Net annual savings</td><td style="text-align:right; color:#60a5fa; font-family:'DM Mono',monospace; border-top:1px solid #1e293b">$130,830</td></tr>
                <tr><td style="padding:0.35rem 0; color:#94a3b8; text-align:center">Cost reduction</td><td style="text-align:right; color:#60a5fa; font-family:'DM Mono',monospace">44.8%</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown("""
        <div class="metric-card" style="margin-bottom:1rem">
            <div class="metric-label">Final Model</div>
            <p style="color:#94a3b8; font-size:0.85rem; margin:0.75rem 0 0.75rem 0; line-height:1.7">
                XGBoost (Extreme Gradient Boosting) selected for its best F1-Score (61.25%)
                and native handling of class imbalance via scale_pos_weight.
            </p>
            <table style="width:100%; font-size:0.82rem; color:#94a3b8; border-collapse:collapse">
                <tr><td style="padding:0.35rem 0; color:#94a3b8; text-align:center">learning_rate</td><td style="text-align:right; font-family:'DM Mono',monospace; color:#f1f5f9">0.01</td></tr>
                <tr><td style="padding:0.35rem 0; color:#94a3b8; text-align:center">max_depth</td><td style="text-align:right; font-family:'DM Mono',monospace; color:#f1f5f9">3</td></tr>
                <tr><td style="padding:0.35rem 0; color:#94a3b8; text-align:center">n_estimators</td><td style="text-align:right; font-family:'DM Mono',monospace; color:#f1f5f9">100</td></tr>
                <tr><td style="padding:0.35rem 0; color:#94a3b8; text-align:center">subsample</td><td style="text-align:right; font-family:'DM Mono',monospace; color:#f1f5f9">0.8</td></tr>
                <tr><td style="padding:0.35rem 0; color:#94a3b8; text-align:center">Decision threshold</td><td style="text-align:right; font-family:'DM Mono',monospace; color:#fbbf24">30%</td></tr>
            </table>
        </div>
        <div class="metric-card">
            <div class="metric-label">Project Timeline</div>
            <table style="width:100%; font-size:0.82rem; color:#94a3b8; margin-top:0.75rem; border-collapse:collapse">
                <tr><td style="padding:0.4rem 0; color:#34d399; text-align:center">Week 1</td><td style="color:#94a3b8">Data Exploration &amp; EDA</td></tr>
                <tr><td style="padding:0.4rem 0; color:#34d399; text-align:center">Week 2</td><td style="color:#94a3b8">Feature Engineering &amp; Modeling</td></tr>
                <tr><td style="padding:0.4rem 0; color:#34d399; text-align:center">Week 3</td><td style="color:#94a3b8">Hyperparameter Tuning &amp; Explanation</td></tr>
                <tr><td style="padding:0.4rem 0; color:#34d399; text-align:center">Week 4</td><td style="color:#94a3b8">Evaluation &amp; Deployment</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <p style="font-size:0.75rem; color:#334155; text-align:center">
        Built by Alexis Aquino &nbsp;&middot;&nbsp; University of Batangas &nbsp;&middot;&nbsp;
        <a href="https://github.com/alexis-aquino/customer-churner"
           style="color:#3b82f6; text-decoration:none">github.com/alexis-aquino/customer-churner</a>
    </p>
    """, unsafe_allow_html=True)