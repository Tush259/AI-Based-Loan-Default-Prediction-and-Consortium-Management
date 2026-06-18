import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
import json
import os

if "predicted" not in st.session_state:
    st.session_state.predicted = False


from exposure_clearing import (
    allocate_consortium,
    liquidity_priority,
    simulate_clearing,
    get_bank_summary,
    get_bank_eligibility_table
)

st.set_page_config(
    page_title="AI Loan Default Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    /* Main background */
    .main { background-color: #f8fafc; }

    /* Metric card styling */
    [data-testid="metric-container"] {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }

    /* Section headers */
    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1e293b;
        margin-top: 1.2rem;
        margin-bottom: 0.5rem;
        padding-bottom: 6px;
        border-bottom: 2px solid #3b82f6;
        display: inline-block;
    }

    /* Info box */
    .info-box {
        background: #eff6ff;
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 10px 0;
        font-size: 0.95rem;
        color: #1e40af;
    }

    /* Success box */
    .success-box {
        background: #f0fdf4;
        border-left: 4px solid #22c55e;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 10px 0;
        font-size: 0.95rem;
        color: #15803d;
    }

    /* Warning box */
    .warning-box {
        background: #fffbeb;
        border-left: 4px solid #f59e0b;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 10px 0;
        font-size: 0.95rem;
        color: #92400e;
    }

    /* Error box */
    .error-box {
        background: #fef2f2;
        border-left: 4px solid #ef4444;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 10px 0;
        font-size: 0.95rem;
        color: #991b1b;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: #1e293b;
    }
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stNumberInput label {
        color: #94a3b8 !important;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #1d4ed8);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        font-size: 0.95rem;
        width: 100%;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #2563eb, #1e40af);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(59,130,246,0.4);
    }

    /* Dataframe */
    [data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)



@st.cache_resource
def load_models():
    """Load trained models and encoders from disk."""
    if not os.path.exists("models/model1_best.pkl"):
        return None, None, None
    model1   = joblib.load("models/model1_best.pkl")
    model2   = joblib.load("models/model2_best.pkl")
    encoders = joblib.load("models/encoders.pkl")
    return model1, model2, encoders

model1, model2, encoders = load_models()


if model1 is None:
    st.error("⚠️ Models not found! Please run `python train_models.py` first.")
    st.code("python train_models.py", language="bash")
    st.stop()



st.markdown("""
<div style="background: linear-gradient(135deg, #1e3a5f 0%, #1d4ed8 100%);
            padding: 28px 32px; border-radius: 14px; margin-bottom: 24px;
            box-shadow: 0 4px 20px rgba(30,58,138,0.3);">
    <h1 style="color:white; margin:0; font-size:1.9rem; font-weight:800;">
        🏦 AI-Based Loan Default Prediction
    </h1>
    <p style="color:#bfdbfe; margin:6px 0 0 0; font-size:1rem;">
        Intelligent Consortium Allocation &amp; Liquidity-Aware Clearing System
    </p>
    <p style="color:#93c5fd; margin:4px 0 0 0; font-size:0.82rem;">
        MCA AIML Project &nbsp;|&nbsp; SDG 8 &amp; SDG 9 &nbsp;|&nbsp; Banking &amp; Finance Domain
    </p>
</div>
""", unsafe_allow_html=True)




st.sidebar.markdown("## 📋 Borrower Details")
st.sidebar.markdown("---")

st.sidebar.markdown("**👤 Personal Info**")
person_age    = st.sidebar.slider("Age", 18, 80, 30)
person_income = st.sidebar.number_input(
    "Annual Income (₹)", min_value=10000,
    max_value=10000000, value=500000, step=10000
)
home_ownership = st.sidebar.selectbox(
    "Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"]
)
emp_length = st.sidebar.slider("Employment Length (years)", 0, 50, 5)

st.sidebar.markdown("---")
st.sidebar.markdown("**💰 Loan Details**")
loan_intent = st.sidebar.selectbox(
    "Loan Intent",
    ["PERSONAL", "EDUCATION", "MEDICAL",
     "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]
)
loan_grade  = st.sidebar.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
loan_amnt   = st.sidebar.number_input(
    "Loan Amount (₹)", min_value=1000,
    max_value=5000000, value=200000, step=5000
)
loan_int_rate = st.sidebar.slider("Interest Rate (%)", 5.0, 25.0, 12.0, step=0.1)
loan_percent_income = st.sidebar.slider(
    "Loan % of Income", 0.0, 1.0, 0.2, step=0.01
)

st.sidebar.markdown("---")
st.sidebar.markdown("**📊 Credit History**")
cb_default = st.sidebar.selectbox("Previous Default on File?", ["N", "Y"])
cred_hist  = st.sidebar.slider("Credit History Length (years)", 0, 30, 5)

st.sidebar.markdown("---")
# Store prediction state to prevent UI reset
if st.sidebar.button("🔍 Predict Now", use_container_width=True):
    st.session_state.predicted = True


def encode_input():
    """
    Encode the sidebar input values using saved LabelEncoders.
    Returns a list of 11 feature values in the correct order.
    """
    try:
        home_enc    = encoders['person_home_ownership'].transform([home_ownership])[0]
        intent_enc  = encoders['loan_intent'].transform([loan_intent])[0]
        grade_enc   = encoders['loan_grade'].transform([loan_grade])[0]
        default_enc = encoders['cb_person_default_on_file'].transform([cb_default])[0]
    except ValueError as e:
        st.error(f"Encoding error: {e}")
        st.stop()

    return [
        person_age,
        person_income,
        int(home_enc),
        emp_length,
        int(intent_enc),
        int(grade_enc),
        loan_amnt,
        loan_int_rate,
        loan_percent_income,
        int(default_enc),
        cred_hist
    ]

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Predict Loan",
    "🏦 Bank Dataset",
    "ℹ️ How It Works",
    "📊 Model Training Results"
])


with tab1:

    # Use session state instead of button state
    if not st.session_state.predicted:
        # ── Welcome screen ─────────────────────────────────
        st.markdown('<div class="info-box">👈 Fill in the borrower details on the left sidebar and click <strong>Predict Now</strong> to get the loan decision.</div>', unsafe_allow_html=True)

        st.markdown("### 🔁 System Pipeline")
        cols = st.columns(5)
        steps = [
            ("1️⃣", "Enter\nBorrower Data"),
            ("2️⃣", "Model 1\nPredicts Risk %"),
            ("3️⃣", "Model 2\nDecides Action"),
            ("4️⃣", "Consortium\nAllocation"),
            ("5️⃣", "Clearing\nSimulation"),
        ]
        
        for col, (icon, text) in zip(cols, steps):
            col.markdown(
                f"""<div style="background:white; border:1px solid #e2e8f0;
                    border-radius:10px; padding:16px; text-align:center;
                    box-shadow:0 1px 4px rgba(0,0,0,0.05);">
                    <div style="font-size:1.6rem">{icon}</div>
                    <div style="font-size:0.8rem; color:#475569; margin-top:6px;
                         white-space:pre-line; font-weight:600;">{text}</div>
                </div>""",
                unsafe_allow_html=True
            )

        st.markdown("### 📐 Decision Rules")
        rule_df = pd.DataFrame({
            "Default Probability": ["< 30%", "30% – 70%", "> 70%"],
            "Decision":            ["✅ Single Bank", "🏦 Consortium", "❌ Reject"],
            "Meaning":             [
                "Low risk — one bank handles the loan",
                "Medium risk — multiple banks share the loan",
                "High risk — loan is denied"
            ]
        })
        st.table(rule_df)

    else:
        
        input_data = encode_input()

        # Model 1 — default probability
        prob = float(model1.predict_proba([input_data])[0][1])

        # Model 2 — decision
        # Rule override (VERY IMPORTANT)
        if prob > 0.7:
            decision = 2  # Reject
        elif prob < 0.3:
            decision = 0  # Single Bank
        else:
            decision = int(model2.predict([input_data])[0])
        # Labels and colors
        decision_labels = {
            0: ("✅ Single Bank",  "#16a34a"),
            1: ("🏦 Consortium",   "#d97706"),
            2: ("❌ Reject Loan",  "#dc2626")
        }
        decision_text, dec_color = decision_labels[decision]

        risk_pct   = prob * 100
        risk_label = "🔴 HIGH" if prob > 0.7 else ("🟡 MEDIUM" if prob > 0.3 else "🟢 LOW")
        risk_color = "#ef4444" if prob > 0.7 else ("#f59e0b" if prob > 0.3 else "#22c55e")

        
        c1, c2, c3 = st.columns(3)
        c1.metric("🎯 Default Probability", f"{risk_pct:.1f}%",
                  help="Probability that the borrower will default")
        c2.metric("⚠️ Risk Level", risk_label,
                  help="Based on the default probability threshold")
        c3.metric("📊 Decision", decision_text,
                  help="Output from Model 2 — what should happen with this loan")

        
        st.markdown("**Default Risk Score**")
        bar_html = f"""
        <div style="background:#e2e8f0; border-radius:999px; height:18px; overflow:hidden; margin:4px 0 2px 0;">
            <div style="background:{risk_color}; width:{risk_pct:.1f}%;
                 height:100%; border-radius:999px; transition:width 0.5s ease;
                 display:flex; align-items:center; justify-content:flex-end; padding-right:8px;">
                <span style="color:white; font-size:0.7rem; font-weight:700;">{risk_pct:.1f}%</span>
            </div>
        </div>
        <p style="font-size:0.8rem; color:#64748b; margin:2px 0 16px 0;">
            0% = No risk &nbsp;|&nbsp; 100% = Certain default
        </p>
        """
        st.markdown(bar_html, unsafe_allow_html=True)

        st.markdown("---")

        
        if decision == 0:
            st.markdown('<div class="success-box">✅ <strong>Single Bank Decision</strong> — Risk is low (< 30%). One bank can issue this loan directly without needing to share the risk.</div>', unsafe_allow_html=True)

            st.markdown("**Summary**")
            summary = pd.DataFrame({
                "Parameter": ["Loan Amount", "Default Probability", "Risk Level", "Decision"],
                "Value":     [f"₹{loan_amnt:,.0f}", f"{risk_pct:.1f}%", risk_label, "Single Bank"]
            })
            st.table(summary)

        
        elif decision == 1:
            st.markdown('<div class="warning-box">🏦 <strong>Consortium Decision</strong> — Moderate risk (30–70%). The loan will be distributed among multiple banks to share the risk.</div>', unsafe_allow_html=True)

            # Bank Allocation
            st.markdown('<p class="section-title">📊 Bank Allocation</p>', unsafe_allow_html=True)
            alloc_df, message = allocate_consortium(loan_amnt)

            if alloc_df.empty:
                st.markdown(f'<div class="error-box">{message}</div>', unsafe_allow_html=True)

            else:
                st.markdown(f'<div class="success-box">{message}</div>', unsafe_allow_html=True)

                # Show allocation table
                st.dataframe(alloc_df, use_container_width=True, hide_index=True)

                # Charts row
                chart_col1, chart_col2 = st.columns(2)

                with chart_col1:
                    st.markdown("**🥧 Loan Share Distribution**")
                    fig1, ax1 = plt.subplots(figsize=(4.5, 4))
                    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(alloc_df)))
                    wedges, texts, autotexts = ax1.pie(
                        alloc_df["Loan Share (₹)"],
                        labels=alloc_df["Bank"],
                        autopct="%1.1f%%",
                        colors=colors,
                        startangle=90,
                        wedgeprops={"edgecolor": "white", "linewidth": 2}
                    )
                    for at in autotexts:
                        at.set_fontsize(8)
                        at.set_color("white")
                        at.set_fontweight("bold")
                    ax1.set_title("Equal Distribution Among Banks", fontsize=10, pad=10)
                    fig1.patch.set_facecolor("#f8fafc")
                    st.pyplot(fig1)

                with chart_col2:
                    st.markdown("**📊 Liquidity Ratio by Bank**")
                    fig2, ax2 = plt.subplots(figsize=(4.5, 4))
                    bar_colors = ["#ef4444" if v < 0.25 else "#f59e0b" if v < 0.4 else "#22c55e"
                                  for v in alloc_df["Liquidity Ratio"]]
                    bars = ax2.bar(alloc_df["Bank"], alloc_df["Liquidity Ratio"],
                                   color=bar_colors, edgecolor="white", linewidth=1.2)
                    ax2.axhline(y=0.25, color="#ef4444", linestyle="--",
                                linewidth=1, alpha=0.7, label="Low threshold (0.25)")
                    ax2.set_title("Liquidity Ratio per Bank", fontsize=10)
                    ax2.set_ylabel("Liquidity Ratio")
                    ax2.set_xticklabels(alloc_df["Bank"], rotation=45, ha="right", fontsize=8)
                    ax2.legend(fontsize=7)
                    ax2.set_ylim(0, 0.7)
                    fig2.patch.set_facecolor("#f8fafc")
                    fig2.tight_layout()
                    st.pyplot(fig2)

                # Liquidity Priority Table
                st.markdown('<p class="section-title">💧 Repayment Priority (Liquidity-Based)</p>', unsafe_allow_html=True)
                st.markdown('<div class="info-box">Banks with <strong>lower liquidity</strong> get repaid <strong>first</strong> — they have less cash reserves and need funds sooner.</div>', unsafe_allow_html=True)

                priority_df = liquidity_priority(alloc_df)
                display_cols = [
                    "Repayment Priority Rank", "Bank",
                    "Liquidity Ratio", "Loan Share (₹)", "Past Default Rate"
                ]
                st.dataframe(
                    priority_df[display_cols].sort_values("Repayment Priority Rank"),
                    use_container_width=True,
                    hide_index=True
                )

                # Clearing Simulation
                st.markdown('<p class="section-title">💳 Clearing House Simulation</p>', unsafe_allow_html=True)
                st.markdown("Enter how much the borrower has repaid, and see how it gets distributed:")

                rep_col1, rep_col2 = st.columns([3, 1])
                with rep_col1:
                    repayment = st.number_input(
                        "Repayment Amount (₹)",
                        min_value=0,
                        max_value=int(loan_amnt),
                        value=int(loan_amnt // 2),
                        step=1000,
                        label_visibility="visible"
                    )
                with rep_col2:
                    pct = (repayment / loan_amnt * 100) if loan_amnt > 0 else 0
                    st.metric("Coverage", f"{pct:.1f}%")

                run_clearing = st.button("▶️ Run Clearing Simulation", key="run_clearing")

                if run_clearing:
                    clearing_df = simulate_clearing(alloc_df, repayment)
                    st.dataframe(clearing_df, use_container_width=True, hide_index=True)

                    # Summary
                    fully    = len(clearing_df[clearing_df["Status"] == "✅ Fully Cleared"])
                    partial  = len(clearing_df[clearing_df["Status"] == "⚠️ Partially Cleared"])
                    pending  = len(clearing_df[clearing_df["Status"] == "⏳ Pending"])
                    total_rx = clearing_df["Amount Received (₹)"].sum()

                    s1, s2, s3, s4 = st.columns(4)
                    s1.metric("✅ Fully Cleared",    fully)
                    s2.metric("⚠️ Partial",          partial)
                    s3.metric("⏳ Pending",           pending)
                    s4.metric("💰 Total Distributed", f"₹{total_rx:,.0f}")

        
        else:
            st.markdown('<div class="error-box">❌ <strong>Loan Rejected</strong> — Default probability is very high (> 70%). This loan cannot be issued by any bank, single or consortium.</div>', unsafe_allow_html=True)

            st.markdown("**Why was this rejected?**")
            reasons = []
            if prob > 0.7:
                reasons.append(f"• Default probability is {risk_pct:.1f}% — exceeds the 70% rejection threshold")
            if loan_percent_income > 0.5:
                reasons.append(f"• Loan is {loan_percent_income*100:.0f}% of annual income — very high debt burden")
            if cb_default == "Y":
                reasons.append("• Borrower has a previous default on file")
            if loan_grade in ["E", "F", "G"]:
                reasons.append(f"• Loan grade '{loan_grade}' indicates poor creditworthiness")

            if reasons:
                for r in reasons:
                    st.markdown(r)
            else:
                st.markdown(f"• Overall risk profile resulted in {risk_pct:.1f}% default probability")




with tab2:
    st.markdown("### 🏦 Bank Dataset — 10 Indian Banks")
    st.markdown("These banks participate in consortium loan allocation. Their metrics determine eligibility and repayment priority.")

    if not os.path.exists("bank_dataset.csv"):
        st.error("bank_dataset.csv not found. Make sure it is in the project folder.")
    else:
        bank_df = get_bank_summary()
        st.dataframe(bank_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("### 📊 Bank Metrics Comparison")

        fig3, axes = plt.subplots(2, 2, figsize=(13, 8))
        fig3.patch.set_facecolor("#f8fafc")
        palette = sns.color_palette("Blues_d", len(bank_df))

        # Plot 1: Liquidity Ratio
        axes[0, 0].bar(bank_df["Bank_ID"], bank_df["Liquidity_Ratio"],
                       color=palette, edgecolor="white", linewidth=1)
        axes[0, 0].set_title("Liquidity Ratio", fontweight="bold")
        axes[0, 0].set_xticklabels(bank_df["Bank_ID"], rotation=45, ha="right", fontsize=8)
        axes[0, 0].set_ylabel("Ratio")
        axes[0, 0].set_ylim(0, 0.65)

        # Plot 2: Capital Adequacy
        axes[0, 1].bar(bank_df["Bank_ID"], bank_df["Capital_Adequacy"],
                       color=sns.color_palette("Greens_d", len(bank_df)),
                       edgecolor="white", linewidth=1)
        axes[0, 1].set_title("Capital Adequacy", fontweight="bold")
        axes[0, 1].set_xticklabels(bank_df["Bank_ID"], rotation=45, ha="right", fontsize=8)
        axes[0, 1].set_ylabel("Ratio")
        axes[0, 1].set_ylim(0.6, 1.0)

        # Plot 3: Past Default Rate
        axes[1, 0].bar(bank_df["Bank_ID"], bank_df["Past_Default_Rate"],
                       color=sns.color_palette("Reds_d", len(bank_df)),
                       edgecolor="white", linewidth=1)
        axes[1, 0].set_title("Past Default Rate", fontweight="bold")
        axes[1, 0].set_xticklabels(bank_df["Bank_ID"], rotation=45, ha="right", fontsize=8)
        axes[1, 0].set_ylabel("Rate")
        axes[1, 0].set_ylim(0, 0.25)

        # Plot 4: Exposure (Current vs Max)
        x = np.arange(len(bank_df))
        width = 0.35
        axes[1, 1].bar(x - width/2, bank_df["Current_Exposure"],
                       width, label="Current Exposure", color="#60a5fa", edgecolor="white")
        axes[1, 1].bar(x + width/2, bank_df["Max_Exposure_Limit"],
                       width, label="Max Exposure Limit", color="#1d4ed8", edgecolor="white")
        axes[1, 1].set_title("Exposure: Current vs Maximum", fontweight="bold")
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(bank_df["Bank_ID"], rotation=45, ha="right", fontsize=8)
        axes[1, 1].legend(fontsize=8)
        axes[1, 1].set_ylabel("Exposure Ratio")

        plt.tight_layout(pad=2)
        st.pyplot(fig3)

        # Eligibility Table
        st.markdown("---")
        st.markdown("### 🔍 Exposure Eligibility Check")
        st.markdown("Shows whether each bank would pass the exposure check for consortium participation.")

        elig_df = get_bank_eligibility_table()
        st.dataframe(elig_df, use_container_width=True, hide_index=True)




with tab3:
    st.markdown("### ℹ️ System Architecture & Explanation")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
**🔁 Full Pipeline**
```
Borrower Input (11 features)
         ↓
  ┌─────────────────┐
  │    MODEL 1      │  3 Algorithms Compared
  │  NPA Prediction │  + GridSearchCV Tuning
  │  (Best Auto-    │  → Best Model Selected
  │   Selected)     │
  └────────┬────────┘
           │  Default Probability (0–1)
           ↓
  ┌─────────────────┐
  │    MODEL 2      │  3 Algorithms Compared
  │  Consortium     │  + GridSearchCV Tuning
  │  Decision       │  → Best Model Selected
  └────────┬────────┘
           │
     ┌─────┼─────┐
     ↓     ↓     ↓
  Single Consort. Reject
  Bank   ium
           │
    Exposure Check
    (per bank)
           │
    Liquidity Sort
    (priority)
           │
    Clearing House
    (repayment dist.)
```
        """)

    with col_b:
        st.markdown("""
**📐 Rules & Formulas**

| Rule | Formula |
|------|---------|
| Single Bank | Probability < 0.3 |
| Consortium  | 0.3 ≤ Probability ≤ 0.7 |
| Reject      | Probability > 0.7 |
| Exposure OK | Current + Share ≤ Max Limit |
| Loan Share  | Loan ÷ Eligible Banks |
| Priority    | Sort by Liquidity ↑ (lowest first) |

---

**🤖 Model Details**

| | Model 1 (NPA) | Model 2 (Consortium) |
|---|---|---|
| Task | Binary Classification | Multi-class Classification |
| Algorithms Tried | Logistic Regression, Random Forest, XGBoost/Gradient Boosting | Decision Tree, KNN, Gradient Boosting |
| Tuning | GridSearchCV (cv=3) | GridSearchCV (cv=3) |
| Selection | Best accuracy auto-selected | Best accuracy auto-selected |
| Input | 11 features | 11 features |
| Output | Probability (0–1) | Class (0, 1, or 2) |
| Saved as | model1_best.pkl | model2_best.pkl |

---

**📦 Dataset**
- Source: Kaggle — Credit Risk Dataset
- Records: ~32,000 after cleaning
- Target: loan_status (0 = No Default, 1 = Default)
        """)

    st.markdown("---")
    st.markdown("### 📋 Feature Descriptions")

    feat_df = pd.DataFrame({
        "Feature": [
            "person_age", "person_income", "person_home_ownership",
            "person_emp_length", "loan_intent", "loan_grade",
            "loan_amnt", "loan_int_rate", "loan_percent_income",
            "cb_person_default_on_file", "cb_person_cred_hist_length"
        ],
        "Type": [
            "Numeric", "Numeric", "Categorical",
            "Numeric", "Categorical", "Categorical",
            "Numeric", "Numeric", "Numeric",
            "Categorical", "Numeric"
        ],
        "Description": [
            "Age of the borrower",
            "Annual income of the borrower (₹)",
            "RENT / OWN / MORTGAGE / OTHER",
            "Years at current employer",
            "Purpose of the loan",
            "Bank-assigned credit grade (A = best, G = worst)",
            "Total loan amount requested (₹)",
            "Annual interest rate (%)",
            "Loan amount as a fraction of annual income",
            "Has the person defaulted before? (Y/N)",
            "Length of credit history in years"
        ]
    })
    st.dataframe(feat_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("""
### 🎓 Viva One-Liner
> *"My project uses machine learning to predict loan default risk by training and comparing
multiple algorithms (Logistic Regression, Random Forest, XGBoost, Decision Tree, KNN,
Gradient Boosting) with GridSearchCV hyperparameter tuning. The best-performing model
is automatically selected for each task. A second model intelligently decides whether
a loan should be issued by a single bank, shared among a consortium, or rejected.
It also incorporates exposure management rules and liquidity-based clearing mechanisms
to simulate real-world banking operations."*
    """)



with tab4:
    st.markdown("### 📊 Model Training Results & Hyperparameter Tuning")
    st.markdown("These results were generated during model training (`python train_models.py`). Each model was trained with **3 different algorithms** and the best one was automatically selected.")

    # ── Load training results ────────────────────────────────
    m1_path = "models/model1_results.json"
    m2_path = "models/model2_results.json"

    if not os.path.exists(m1_path) or not os.path.exists(m2_path):
        st.warning("⚠️ Training results not found. Please re-run `python train_models.py` to generate them.")
        st.code("python train_models.py", language="bash")
    else:
        with open(m1_path, 'r') as f:
            m1_results = json.load(f)
        with open(m2_path, 'r') as f:
            m2_results = json.load(f)

        # ══════════════════════════════════════════════════════
        #  MODEL 1 RESULTS
        # ══════════════════════════════════════════════════════
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1e3a5f, #2563eb); padding: 18px 24px;
                    border-radius: 12px; margin: 16px 0 12px 0;">
            <h3 style="color:white; margin:0; font-size:1.2rem;">🤖 Model 1 — NPA Prediction (Loan Default)</h3>
            <p style="color:#93c5fd; margin:4px 0 0 0; font-size:0.85rem;">Binary Classification &nbsp;|&nbsp; 3 Algorithms Compared &nbsp;|&nbsp; GridSearchCV Tuning</p>
        </div>
        """, unsafe_allow_html=True)

        # Dataset info
        info_c1, info_c2, info_c3, info_c4 = st.columns(4)
        info_c1.metric("Dataset Size", f"{m1_results['dataset_size']:,}")
        info_c2.metric("Train Samples", f"{m1_results['train_size']:,}")
        info_c3.metric("Test Samples", f"{m1_results['test_size']:,}")
        info_c4.metric("Features", m1_results['num_features'])

        # Comparison table
        st.markdown("#### 📋 Algorithm Comparison")
        m1_algos = m1_results['algorithms']
        m1_table_data = []
        for algo_name, metrics in m1_algos.items():
            is_best = "⭐" if algo_name == m1_results['best_algorithm'] else ""
            m1_table_data.append({
                "": is_best,
                "Algorithm": algo_name,
                "Accuracy (%)": metrics['accuracy'],
                "ROC-AUC": metrics.get('roc_auc', 'N/A'),
                "Precision": metrics['precision'],
                "Recall": metrics['recall'],
                "F1-Score": metrics['f1'],
            })
        m1_df = pd.DataFrame(m1_table_data)
        st.dataframe(m1_df, use_container_width=True, hide_index=True)

        # Best model highlight
        st.markdown(f"""
        <div style="background: #f0fdf4; border-left: 4px solid #22c55e; border-radius: 8px;
                    padding: 14px 18px; margin: 8px 0;">
            <strong style="color:#15803d;">⭐ Best Model Selected: {m1_results['best_algorithm']}</strong>
            <span style="color:#16a34a;"> — Accuracy: {m1_results['best_accuracy']:.2f}%</span>
        </div>
        """, unsafe_allow_html=True)

        # Hyperparameters for each algorithm
        st.markdown("#### ⚙️ Best Hyperparameters per Algorithm")
        hp_cols = st.columns(len(m1_algos))
        for idx, (algo_name, metrics) in enumerate(m1_algos.items()):
            with hp_cols[idx]:
                is_best = " ⭐" if algo_name == m1_results['best_algorithm'] else ""
                st.markdown(f"**{algo_name}{is_best}**")
                params = metrics['best_params']
                param_df = pd.DataFrame({
                    "Parameter": list(params.keys()),
                    "Value": [str(v) for v in params.values()]
                })
                st.dataframe(param_df, use_container_width=True, hide_index=True)

        # Bar chart comparison
        st.markdown("#### 📊 Visual Comparison")
        chart_c1, chart_c2 = st.columns(2)

        with chart_c1:
            fig_m1, ax_m1 = plt.subplots(figsize=(6, 4))
            algo_names = list(m1_algos.keys())
            accuracies = [m1_algos[n]['accuracy'] for n in algo_names]
            colors_m1 = ['#22c55e' if n == m1_results['best_algorithm'] else '#60a5fa' for n in algo_names]
            bars = ax_m1.bar(algo_names, accuracies, color=colors_m1, edgecolor='white', linewidth=1.5)
            for bar, acc in zip(bars, accuracies):
                ax_m1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                          f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
            ax_m1.set_ylabel('Accuracy (%)')
            ax_m1.set_title('Model 1: Accuracy Comparison', fontweight='bold')
            ax_m1.set_ylim(0, max(accuracies) + 5)
            ax_m1.axhline(y=87, color='#ef4444', linestyle='--', alpha=0.5, label='87% threshold')
            ax_m1.legend(fontsize=8)
            fig_m1.patch.set_facecolor('#f8fafc')
            fig_m1.tight_layout()
            st.pyplot(fig_m1)

        with chart_c2:
            fig_m1b, ax_m1b = plt.subplots(figsize=(6, 4))
            metrics_names = ['Precision', 'Recall', 'F1-Score']
            x = np.arange(len(metrics_names))
            width = 0.25
            for i, algo in enumerate(algo_names):
                vals = [m1_algos[algo]['precision'], m1_algos[algo]['recall'], m1_algos[algo]['f1']]
                offset = (i - 1) * width
                ax_m1b.bar(x + offset, vals, width, label=algo, edgecolor='white', linewidth=0.8)
            ax_m1b.set_ylabel('Score')
            ax_m1b.set_title('Model 1: Metrics Comparison', fontweight='bold')
            ax_m1b.set_xticks(x)
            ax_m1b.set_xticklabels(metrics_names)
            ax_m1b.legend(fontsize=7)
            ax_m1b.set_ylim(0, 1.0)
            fig_m1b.patch.set_facecolor('#f8fafc')
            fig_m1b.tight_layout()
            st.pyplot(fig_m1b)

        st.markdown("---")

        # ══════════════════════════════════════════════════════
        #  MODEL 2 RESULTS
        # ══════════════════════════════════════════════════════
        st.markdown("""
        <div style="background: linear-gradient(135deg, #5b21b6, #7c3aed); padding: 18px 24px;
                    border-radius: 12px; margin: 16px 0 12px 0;">
            <h3 style="color:white; margin:0; font-size:1.2rem;">🤖 Model 2 — Consortium Decision</h3>
            <p style="color:#c4b5fd; margin:4px 0 0 0; font-size:0.85rem;">Multi-class Classification &nbsp;|&nbsp; 3 Algorithms Compared &nbsp;|&nbsp; GridSearchCV Tuning</p>
        </div>
        """, unsafe_allow_html=True)

        # Dataset info
        info2_c1, info2_c2, info2_c3, info2_c4 = st.columns(4)
        info2_c1.metric("Dataset Size", f"{m2_results['dataset_size']:,}")
        info2_c2.metric("Train Samples", f"{m2_results['train_size']:,}")
        info2_c3.metric("Test Samples", f"{m2_results['test_size']:,}")
        info2_c4.metric("Features", m2_results['num_features'])

        # Comparison table
        st.markdown("#### 📋 Algorithm Comparison")
        m2_algos = m2_results['algorithms']
        m2_table_data = []
        for algo_name, metrics in m2_algos.items():
            is_best = "⭐" if algo_name == m2_results['best_algorithm'] else ""
            m2_table_data.append({
                "": is_best,
                "Algorithm": algo_name,
                "Accuracy (%)": metrics['accuracy'],
                "Precision (Wt)": metrics['precision'],
                "Recall (Wt)": metrics['recall'],
                "F1-Score (Wt)": metrics['f1'],
            })
        m2_df = pd.DataFrame(m2_table_data)
        st.dataframe(m2_df, use_container_width=True, hide_index=True)

        # Best model highlight
        st.markdown(f"""
        <div style="background: #faf5ff; border-left: 4px solid #7c3aed; border-radius: 8px;
                    padding: 14px 18px; margin: 8px 0;">
            <strong style="color:#5b21b6;">⭐ Best Model Selected: {m2_results['best_algorithm']}</strong>
            <span style="color:#7c3aed;"> — Accuracy: {m2_results['best_accuracy']:.2f}%</span>
        </div>
        """, unsafe_allow_html=True)

        # Hyperparameters for each algorithm
        st.markdown("#### ⚙️ Best Hyperparameters per Algorithm")
        hp2_cols = st.columns(len(m2_algos))
        for idx, (algo_name, metrics) in enumerate(m2_algos.items()):
            with hp2_cols[idx]:
                is_best = " ⭐" if algo_name == m2_results['best_algorithm'] else ""
                st.markdown(f"**{algo_name}{is_best}**")
                params = metrics['best_params']
                param_df = pd.DataFrame({
                    "Parameter": list(params.keys()),
                    "Value": [str(v) for v in params.values()]
                })
                st.dataframe(param_df, use_container_width=True, hide_index=True)

        # Bar chart comparison
        st.markdown("#### 📊 Visual Comparison")
        chart2_c1, chart2_c2 = st.columns(2)

        with chart2_c1:
            fig_m2, ax_m2 = plt.subplots(figsize=(6, 4))
            algo_names_2 = list(m2_algos.keys())
            accuracies_2 = [m2_algos[n]['accuracy'] for n in algo_names_2]
            colors_m2 = ['#7c3aed' if n == m2_results['best_algorithm'] else '#a78bfa' for n in algo_names_2]
            bars2 = ax_m2.bar(algo_names_2, accuracies_2, color=colors_m2, edgecolor='white', linewidth=1.5)
            for bar, acc in zip(bars2, accuracies_2):
                ax_m2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                          f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
            ax_m2.set_ylabel('Accuracy (%)')
            ax_m2.set_title('Model 2: Accuracy Comparison', fontweight='bold')
            ax_m2.set_ylim(0, max(accuracies_2) + 5)
            ax_m2.axhline(y=87, color='#ef4444', linestyle='--', alpha=0.5, label='87% threshold')
            ax_m2.legend(fontsize=8)
            fig_m2.patch.set_facecolor('#f8fafc')
            fig_m2.tight_layout()
            st.pyplot(fig_m2)

        with chart2_c2:
            fig_m2b, ax_m2b = plt.subplots(figsize=(6, 4))
            metrics_names_2 = ['Precision', 'Recall', 'F1-Score']
            x2 = np.arange(len(metrics_names_2))
            for i, algo in enumerate(algo_names_2):
                vals = [m2_algos[algo]['precision'], m2_algos[algo]['recall'], m2_algos[algo]['f1']]
                offset = (i - 1) * width
                ax_m2b.bar(x2 + offset, vals, width, label=algo, edgecolor='white', linewidth=0.8)
            ax_m2b.set_ylabel('Score')
            ax_m2b.set_title('Model 2: Metrics Comparison', fontweight='bold')
            ax_m2b.set_xticks(x2)
            ax_m2b.set_xticklabels(metrics_names_2)
            ax_m2b.legend(fontsize=7)
            ax_m2b.set_ylim(0, 1.0)
            fig_m2b.patch.set_facecolor('#f8fafc')
            fig_m2b.tight_layout()
            st.pyplot(fig_m2b)

        # ══════════════════════════════════════════════════════
        #  OVERALL SUMMARY
        # ══════════════════════════════════════════════════════
        st.markdown("---")
        m1_algo_list = ", ".join(m1_algos.keys())
        m2_algo_list = ", ".join(m2_algos.keys())
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #0f172a, #1e293b); padding: 20px 24px;
                    border-radius: 12px; margin: 16px 0;">
            <h3 style="color:white; margin:0 0 12px 0; font-size:1.2rem;">📊 Overall Training Summary</h3>
            <table style="width:100%; color:#e2e8f0; border-collapse:collapse;">
                <tr style="border-bottom:1px solid #334155;">
                    <th style="text-align:left; padding:8px; color:#94a3b8;">Aspect</th>
                    <th style="text-align:center; padding:8px; color:#60a5fa;">Model 1 (NPA)</th>
                    <th style="text-align:center; padding:8px; color:#a78bfa;">Model 2 (Consortium)</th>
                </tr>
                <tr style="border-bottom:1px solid #334155;">
                    <td style="padding:8px;">Task Type</td>
                    <td style="text-align:center; padding:8px;">Binary Classification</td>
                    <td style="text-align:center; padding:8px;">Multi-class (3 classes)</td>
                </tr>
                <tr style="border-bottom:1px solid #334155;">
                    <td style="padding:8px;">Algorithms Tested</td>
                    <td style="text-align:center; padding:8px;">{m1_algo_list}</td>
                    <td style="text-align:center; padding:8px;">{m2_algo_list}</td>
                </tr>
                <tr style="border-bottom:1px solid #334155;">
                    <td style="padding:8px;">Best Algorithm</td>
                    <td style="text-align:center; padding:8px; color:#4ade80; font-weight:bold;">{m1_results['best_algorithm']}</td>
                    <td style="text-align:center; padding:8px; color:#c084fc; font-weight:bold;">{m2_results['best_algorithm']}</td>
                </tr>
                <tr style="border-bottom:1px solid #334155;">
                    <td style="padding:8px;">Best Accuracy</td>
                    <td style="text-align:center; padding:8px; font-weight:bold;">{m1_results['best_accuracy']:.2f}%</td>
                    <td style="text-align:center; padding:8px; font-weight:bold;">{m2_results['best_accuracy']:.2f}%</td>
                </tr>
                <tr>
                    <td style="padding:8px;">Tuning Method</td>
                    <td style="text-align:center; padding:8px;">GridSearchCV (cv=3)</td>
                    <td style="text-align:center; padding:8px;">GridSearchCV (cv=3)</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)