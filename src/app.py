# ============================================================
# Customer Churn Prediction Dashboard
# Alexis Aquino | University of Batangas
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# ── LOAD MODEL ARTIFACTS ─────────────────────────────────────
@st.cache_resource
def load_model():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model     = joblib.load(os.path.join(base, 'models', 'final_xgboost_model.pkl'))
    features  = joblib.load(os.path.join(base, 'models', 'feature_list.pkl'))
    threshold = joblib.load(os.path.join(base, 'models', 'optimal_threshold.pkl'))
    return model, features, threshold

model, features, threshold = load_model()

# ── HEADER ────────────────────────────────────────────────────
st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("**University of Batangas | Alexis Aquino | 28-Day ML Project**")
st.markdown("---")

# ── TABS ─────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🔮 Predict Single Customer",
    "📈 Model Performance",
    "📋 Project Summary"
])

# ════════════════════════════════════════════════════════════
# TAB 1: SINGLE CUSTOMER PREDICTION
# ════════════════════════════════════════════════════════════
with tab1:
    st.header("Predict Churn for a Single Customer")
    st.markdown("Fill in the customer details below and click **Predict**.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📌 Account Info")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox("Contract Type", [
            "Month-to-month", "One year", "Two year"
        ])
        payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])

    with col2:
        st.subheader("💰 Charges")
        monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0)
        total_charges   = st.slider("Total Charges ($)", 18.0, 9000.0,
                                     float(tenure * monthly_charges))
        senior_citizen  = st.selectbox("Senior Citizen", ["No", "Yes"])
        paperless       = st.selectbox("Paperless Billing", ["Yes", "No"])

    with col3:
        st.subheader("🌐 Services")
        internet = st.selectbox("Internet Service", [
            "Fiber optic", "DSL", "No"
        ])
        online_security = st.selectbox("Online Security", [
            "Yes", "No", "No internet service"
        ])
        tech_support    = st.selectbox("Tech Support", [
            "Yes", "No", "No internet service"
        ])
        streaming_tv    = st.selectbox("Streaming TV", [
            "Yes", "No", "No internet service"
        ])
        streaming_movies = st.selectbox("Streaming Movies", [
            "Yes", "No", "No internet service"
        ])

    # ── BUILD INPUT ROW ──────────────────────────────────────
    def build_input(tenure, monthly_charges, total_charges, senior_citizen,
                    contract, payment, internet, online_security,
                    tech_support, streaming_tv, streaming_movies,
                    paperless):

        # Scale numeric features (same MinMaxScaler ranges from training)
        tenure_scaled          = tenure / 72
        monthly_charges_scaled = (monthly_charges - 18.0) / (118.75 - 18.0)
        total_charges_scaled   = (total_charges - 18.8) / (8684.8 - 18.8)

        row = {f: 0 for f in features}

        # Numeric
        if 'seniorcitizen'   in row: row['seniorcitizen']   = 1 if senior_citizen == "Yes" else 0
        if 'tenure'          in row: row['tenure']          = tenure_scaled
        if 'monthlycharges'  in row: row['monthlycharges']  = monthly_charges_scaled
        if 'totalcharges'    in row: row['totalcharges']    = total_charges_scaled

        # Contract
        if 'contract_One year' in row and contract == "One year":
            row['contract_One year'] = 1
        if 'contract_Two year' in row and contract == "Two year":
            row['contract_Two year'] = 1

        # Payment
        if 'paymentmethod_Electronic check' in row and payment == "Electronic check":
            row['paymentmethod_Electronic check'] = 1
        if 'paymentmethod_Mailed check' in row and payment == "Mailed check":
            row['paymentmethod_Mailed check'] = 1
        if 'paymentmethod_Credit card (automatic)' in row and payment == "Credit card (automatic)":
            row['paymentmethod_Credit card (automatic)'] = 1

        # Internet
        if 'internetservice_Fiber optic' in row and internet == "Fiber optic":
            row['internetservice_Fiber optic'] = 1
        if 'internetservice_No' in row and internet == "No":
            row['internetservice_No'] = 1

        # Online security
        if 'onlinesecurity_Yes' in row and online_security == "Yes":
            row['onlinesecurity_Yes'] = 1
        if 'onlinesecurity_No internet service' in row and online_security == "No internet service":
            row['onlinesecurity_No internet service'] = 1

        # Tech support
        if 'techsupport_Yes' in row and tech_support == "Yes":
            row['techsupport_Yes'] = 1
        if 'techsupport_No internet service' in row and tech_support == "No internet service":
            row['techsupport_No internet service'] = 1

        # Streaming TV
        if 'streamingtv_Yes' in row and streaming_tv == "Yes":
            row['streamingtv_Yes'] = 1
        if 'streamingtv_No internet service' in row and streaming_tv == "No internet service":
            row['streamingtv_No internet service'] = 1

        # Streaming movies
        if 'streamingmovies_Yes' in row and streaming_movies == "Yes":
            row['streamingmovies_Yes'] = 1
        if 'streamingmovies_No internet service' in row and streaming_movies == "No internet service":
            row['streamingmovies_No internet service'] = 1

        # Paperless billing
        if 'paperlessbilling_Yes' in row and paperless == "Yes":
            row['paperlessbilling_Yes'] = 1

        return pd.DataFrame([row])[features]

    # ── PREDICT BUTTON ───────────────────────────────────────
    if st.button("🔮 Predict Churn", type="primary"):
        input_df = build_input(
            tenure, monthly_charges, total_charges, senior_citizen,
            contract, payment, internet, online_security,
            tech_support, streaming_tv, streaming_movies, paperless
        )

        prob      = model.predict_proba(input_df)[0][1]
        predicted = int(prob >= threshold)

        st.markdown("---")
        col_result, col_gauge = st.columns(2)

        with col_result:
            if predicted == 1:
                st.error(f"## ⚠️ HIGH CHURN RISK")
                st.metric("Churn Probability", f"{prob*100:.1f}%",
                          delta=f"{(prob - threshold)*100:.1f}% above threshold")
                st.markdown(f"""
                **This customer is likely to churn.**

                📋 **Recommended Actions:**
                - Offer annual contract discount (20% off)
                - Assign dedicated account manager
                - Proactive check-in call within 48 hours
                - Review service issues if on Fiber optic
                """)
            else:
                st.success(f"## ✅ LOW CHURN RISK")
                st.metric("Churn Probability", f"{prob*100:.1f}%",
                          delta=f"{(threshold - prob)*100:.1f}% below threshold")
                st.markdown(f"""
                **This customer is likely to stay.**

                📋 **Recommended Actions:**
                - Maintain current service quality
                - Consider upsell opportunities
                - Enroll in loyalty rewards program
                """)

        with col_gauge:
            # Probability gauge chart
            fig, ax = plt.subplots(figsize=(5, 3))
            colors = ['#2ecc71' if prob < threshold else '#e74c3c']
            ax.barh(['Churn Risk'], [prob], color=colors, height=0.4)
            ax.barh(['Churn Risk'], [1 - prob], left=[prob],
                    color='#ecf0f1', height=0.4)
            ax.axvline(x=threshold, color='orange', linestyle='--',
                       linewidth=2, label=f'Threshold ({threshold*100:.0f}%)')
            ax.set_xlim(0, 1)
            ax.set_xlabel('Probability')
            ax.set_title('Churn Probability', fontweight='bold')
            ax.legend()
            ax.set_yticks([])
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # Show input summary
        with st.expander("📊 View Input Data Sent to Model"):
            st.dataframe(input_df)

# ════════════════════════════════════════════════════════════
# TAB 2: MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════
with tab2:
    st.header("Model Performance Summary")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Recall",    "96.5%", help="% of real churners caught")
    col2.metric("Precision", "36.2%", help="% of churn alerts that are correct")
    col3.metric("F1 Score",  "52.7%", help="Balance of Recall and Precision")
    col4.metric("ROC-AUC",   "0.8037", help="Overall model quality (0.5=random, 1.0=perfect)")

    st.markdown("---")

    # Show the saved evaluation chart if it exists
    report_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'reports', 'final_evaluation_report.png'
    )

    if os.path.exists(report_path):
        st.image(report_path, caption="Final Evaluation Report — 4 Panel Chart")
    else:
        st.info("Run Day 22 notebook first to generate the evaluation chart.")

    st.markdown("---")
    st.subheader("📊 Model Journey")

    journey_data = {
        'Version':   ['LR Original', 'XGBoost Basic', 'LR Balanced',
                      'XGBoost Tuned (50%)', 'XGBoost Tuned (30%)', 'Final Model'],
        'Recall':    ['56.95%', '78.61%', '79.14%', '82.4%', '97.6%', '96.5%'],
        'Key Change':['Baseline', 'XGBoost introduced', 'class_weight added',
                      'GridSearchCV tuning', 'Threshold → 30%', 'Features cleaned']
    }
    st.dataframe(pd.DataFrame(journey_data), use_container_width=True)

# ════════════════════════════════════════════════════════════
# TAB 3: PROJECT SUMMARY
# ════════════════════════════════════════════════════════════
with tab3:
    st.header("Project Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎯 Business Problem")
        st.markdown("""
        A telecom company is losing customers to competitors.
        The goal is to **predict which customers are likely to churn**
        so the retention team can intervene before they leave.

        **Dataset:** Kaggle Telco Customer Churn
        - 7,032 customers
        - 20 features
        - 26.6% churn rate

        **Key Finding:** Contract type is the root cause of churn.
        Month-to-month customers churn at 100% vs 0% for annual contracts.
        """)

        st.subheader("💰 Business Impact")
        st.markdown("""
        | Metric | Value |
        |--------|-------|
        | Churners caught | 361 / 374 (96.5%) |
        | Churners missed | 13 |
        | Net annual savings | $130,830 |
        | Cost reduction | 44.8% |
        """)

    with col2:
        st.subheader("🤖 Final Model")
        st.markdown("""
        **Algorithm:** XGBoost (Extreme Gradient Boosting)

        **Key Parameters:**
        - learning_rate: 0.01
        - max_depth: 3
        - n_estimators: 100
        - Decision threshold: 30%

        **Why XGBoost:**
        - Best F1-Score (61.25%) among all models
        - Handles class imbalance natively
        - Sequential learning from mistakes
        - Feature importance built-in
        """)

        st.subheader("📅 Project Timeline")
        st.markdown("""
        - ✅ Week 1: Data Exploration & EDA
        - ✅ Week 2: Feature Engineering & Modeling
        - ✅ Week 3: Hyperparameter Tuning & Explanation
        - ✅ Week 4: Evaluation & Deployment
        """)

    st.markdown("---")
    st.markdown(
        "**Built by Alexis Aquino** | University of Batangas | "
        "GitHub: [customer-churner](https://github.com/alexis-aquino/customer-churner)"
    )