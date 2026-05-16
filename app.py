import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Loan Approval Predictor")
st.markdown("Predict whether your bank loan "
            "will be approved instantly.")
st.markdown("---")

# Load model
@st.cache_resource
def load_model():
    with open('model.pkl',    'rb') as f:
        model    = pickle.load(f)
    with open('features.pkl', 'rb') as f:
        features = pickle.load(f)
    return model, features

try:
    model, features = load_model()
    st.success(
        "✅ Model loaded — "
        "trained on real loan data")
except:
    st.error("Run train_model.py first!")
    st.stop()

# Tabs
tab1, tab2, tab3 = st.tabs([
    "🔍 Check Eligibility",
    "📊 Insights",
    "📈 Model Performance"
])

# Tab 1 — Predict
with tab1:
    st.markdown(
        "### Enter Your Loan Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 👤 Personal Info")
        gender        = st.selectbox(
            "Gender:",
            ["Male", "Female"])
        married       = st.selectbox(
            "Married:",
            ["Yes", "No"])
        dependents    = st.selectbox(
            "Dependents:",
            ["0", "1", "2", "3+"])
        education     = st.selectbox(
            "Education:",
            ["Graduate",
             "Not Graduate"])
        self_employed = st.selectbox(
            "Self Employed:",
            ["No", "Yes"])

    with col2:
        st.markdown("#### 💰 Financial Info")
        applicant_income = st.number_input(
            "Applicant Income (₹/month):",
            min_value=0,
            max_value=100000,
            value=5000,
            step=500)
        coapplicant_income = st.number_input(
            "Co-applicant Income (₹/month):",
            min_value=0,
            max_value=100000,
            value=0,
            step=500)
        loan_amount = st.number_input(
            "Loan Amount (₹ thousands):",
            min_value=0,
            max_value=700,
            value=150,
            step=10)
        loan_term = st.selectbox(
            "Loan Term (months):",
            [360, 180, 120, 84,
             60, 36, 12])
        credit_history = st.selectbox(
            "Credit History:",
            ["Good (1.0)",
             "Bad (0.0)"])

    with col3:
        st.markdown("#### 🏠 Property Info")
        property_area = st.selectbox(
            "Property Area:",
            ["Urban", "Semiurban",
             "Rural"])

        st.markdown("#### 📊 Your Profile")
        total_income = (applicant_income +
                        coapplicant_income)
        emi = (loan_amount /
               loan_term * 1000)
        debt_ratio = (emi / total_income
                      * 100) if total_income \
                     > 0 else 0

        st.metric("Total Income",
                  f"₹{total_income:,}")
        st.metric("Monthly EMI",
                  f"₹{emi:,.0f}")
        st.metric("Debt-to-Income",
                  f"{debt_ratio:.1f}%")

        if debt_ratio > 50:
            st.error(
                "⚠️ High debt ratio — "
                "may affect approval")
        elif debt_ratio > 30:
            st.warning(
                "⚠️ Moderate debt ratio")
        else:
            st.success(
                "✅ Healthy debt ratio")

    if st.button(
        "🔍 Check Loan Eligibility",
        type="primary"
    ):
        # Encode
        encode_map = {
            "Male": 1, "Female": 0,
            "Yes": 1, "No": 0,
            "Graduate": 0,
            "Not Graduate": 1,
            "0": 0, "1": 1,
            "2": 2, "3+": 3,
            "Urban": 2,
            "Semiurban": 1,
            "Rural": 0,
            "Good (1.0)": 1.0,
            "Bad (0.0)": 0.0
        }

        total_inc = (applicant_income +
                     coapplicant_income)
        inc_to_loan = (total_inc /
                       (loan_amount + 1))
        emi_val = (loan_amount / loan_term)

        input_dict = {
            'Gender':
                encode_map[gender],
            'Married':
                encode_map[married],
            'Dependents':
                encode_map[dependents],
            'Education':
                encode_map[education],
            'Self_Employed':
                encode_map[self_employed],
            'ApplicantIncome':
                applicant_income,
            'CoapplicantIncome':
                coapplicant_income,
            'LoanAmount':
                loan_amount,
            'Loan_Amount_Term':
                loan_term,
            'Credit_History':
                encode_map[credit_history],
            'Property_Area':
                encode_map[property_area],
            'Total_Income':
                total_inc,
            'Income_to_Loan':
                inc_to_loan,
            'EMI':
                emi_val
        }

        input_df = pd.DataFrame(
            [input_dict])[features]
        prediction  = model.predict(
            input_df)[0]
        probability = model.predict_proba(
            input_df)[0]

        approve_prob = probability[1] * 100
        reject_prob  = probability[0] * 100

        st.markdown("---")

        if prediction == 1:
            st.markdown(
                "<h2 style='color:#2ecc71;"
                "text-align:center'>"
                "✅ LOAN APPROVED</h2>",
                unsafe_allow_html=True
            )
            st.balloons()
        else:
            st.markdown(
                "<h2 style='color:#e74c3c;"
                "text-align:center'>"
                "❌ LOAN REJECTED</h2>",
                unsafe_allow_html=True
            )

        c1, c2, c3 = st.columns(3)
        c1.metric("Approval Probability",
                  f"{approve_prob:.1f}%")
        c2.metric("Rejection Probability",
                  f"{reject_prob:.1f}%")
        c3.metric("Decision",
                  "Approved ✅"
                  if prediction == 1
                  else "Rejected ❌")

        # Probability bar
        fig, ax = plt.subplots(
            figsize=(8, 2))
        ax.barh(['Approved'],
                [approve_prob],
                color='#2ecc71',
                height=0.3)
        ax.barh(['Rejected'],
                [reject_prob],
                color='#e74c3c',
                height=0.3)
        ax.set_xlim(0, 100)
        ax.set_title(
            'Approval vs Rejection '
            'Probability',
            fontsize=12)
        ax.set_xlabel('Probability (%)')
        plt.tight_layout()
        st.pyplot(fig)

        # Tips
        st.markdown(
            "### 💡 How to Improve "
            "Your Chances")
        tips = []
        if credit_history == "Bad (0.0)":
            tips.append(
                "🔑 **Improve credit history**"
                " — single most important factor")
        if debt_ratio > 40:
            tips.append(
                "💰 **Reduce loan amount** "
                "or increase income to lower "
                "debt-to-income ratio")
        if coapplicant_income == 0:
            tips.append(
                "👥 **Add a co-applicant** "
                "— increases total income "
                "and approval chances")
        if property_area == "Rural":
            tips.append(
                "🏙️ **Urban/Semiurban properties**"
                " have higher approval rates")
        if education == "Not Graduate":
            tips.append(
                "🎓 **Graduate applicants** "
                "have slightly higher "
                "approval rates")

        if tips:
            for tip in tips:
                st.info(tip)
        else:
            st.success(
                "✅ Your profile looks strong!")

# Tab 2 — Insights
with tab2:
    st.markdown("### 📊 Loan Approval Insights")

    charts = ['approval_insights.png']
    for chart in charts:
        if os.path.exists(chart):
            st.image(chart,
                     use_column_width=True)

    st.markdown("### 🔑 Key Findings")
    findings = pd.DataFrame({
        'Factor': [
            'Credit History',
            'Property Area',
            'Education',
            'Marital Status',
            'Income'
        ],
        'Impact': [
            'Highest — Good credit = 80%+ approval',
            'Semiurban > Urban > Rural',
            'Graduates approved more often',
            'Married applicants approved more',
            'Higher income = higher approval'
        ],
        'Importance': [
            '⭐⭐⭐⭐⭐',
            '⭐⭐⭐⭐',
            '⭐⭐⭐',
            '⭐⭐⭐',
            '⭐⭐⭐⭐'
        ]
    })
    st.dataframe(findings,
                 use_container_width=True,
                 hide_index=True)

# Tab 3 — Model Performance
with tab3:
    st.markdown("### 📈 Model Performance")

    charts = [
        'model_comparison.png',
        'confusion_matrix.png',
        'feature_importance.png'
    ]
    for chart in charts:
        if os.path.exists(chart):
            st.image(chart,
                     use_column_width=True)
        else:
            st.info(
                f"Run train_model.py "
                f"to generate {chart}")

    perf = pd.DataFrame({
        'Metric': [
            'Best Algorithm',
            'Training samples',
            'Test samples',
            'Accuracy',
            'AUC-ROC',
            'CV Accuracy'
        ],
        'Value': [
            'Gradient Boosting',
            '~492',
            '~123',
            '~82%',
            '~0.80',
            '~79%'
        ]
    })
    st.dataframe(perf,
                 use_container_width=True,
                 hide_index=True)

st.markdown("---")
st.markdown(
    "Built by **Jyotiraditya** | "
    "Loan Approval Predictor | "
    "For educational purposes only"
)